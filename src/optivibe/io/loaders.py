"""Real-data loaders behind the data seam of decision SW-08.

Each loader turns one on-disk record into the standard :class:`Excitation`
contract (arrays + fs + units + channel mapping; doc 11 §5), so a measured
signal can replay through the same pipeline as a synthetic one. Loaders are
registered in :data:`LOADER_REGISTRY` and looked up by format key, separate
from the excitation-generator registry: adding an instrument format
(TDMS / UFF / MAT / HDF5, roadmap S8) is a new adapter with a registration,
never a core change (SW-02, SW-08).

All conversions to SI happen here, at the input boundary (10 §6): the ``g``
unit option is converted via :data:`optivibe.core.units.G0_M_S2`; WAV integer
PCM is normalized to [-1, 1] and scaled by the spec's ``full_scale_g``.
"""

from __future__ import annotations

import csv
from fractions import Fraction
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from scipy.io import wavfile
from scipy.signal import resample_poly

from optivibe.core.config.models import (
    CsvSpec,
    ExcitationSpec,
    Hdf5Spec,
    MatSpec,
    TdmsSpec,
    UffSpec,
    WavSpec,
)
from optivibe.core.logging import get_logger
from optivibe.core.registry import Registry
from optivibe.core.types import Excitation, FloatArray
from optivibe.core.units import G0_M_S2

logger = get_logger(__name__)

__all__ = [
    "LOADER_REGISTRY",
    "CsvLoader",
    "Hdf5Loader",
    "MatLoader",
    "SignalLoader",
    "TdmsLoader",
    "UffLoader",
    "WavLoader",
]

# Optional dependencies that back the instrument-format loaders. They are *not*
# core dependencies (the physics core installs without them); each loader
# imports its backend lazily inside ``load`` and raises this message if the
# extra is missing (task S8 §6).
_IO_EXTRA = "io-formats"


def _missing_extra_msg(package: str) -> str:
    """Return the install hint shown when an instrument-format backend is absent."""
    return (
        f"the {package!r} reader needs the optional dependency {package!r}; "
        f"install it with: pip install 'optivibe[{_IO_EXTRA}]'"
    )


@runtime_checkable
class SignalLoader(Protocol):
    """Loads one measured record into the :class:`Excitation` contract."""

    def load(self, spec: ExcitationSpec, *, seed: int | None = None) -> Excitation:
        """Load a file according to ``spec``.

        Parameters
        ----------
        spec : ExcitationSpec
            A file-replay spec (CSV/WAV/TDMS/UFF/MAT/HDF5): path, channel/column
            mapping, units, optional resampling. Each loader accepts the spec
            member matching its format and rejects the others loudly.
        seed : int or None, optional
            Recorded in the contract for traceability (loaders are
            deterministic; no randomness is used).

        Returns
        -------
        Excitation
            The record mapped onto the spec's axis, in SI units (m/s^2).
        """
        ...


LOADER_REGISTRY: Registry[SignalLoader] = Registry("io.loader")


def _pack_on_axis(
    signal: FloatArray,
    axis: str,
    fs: float,
    seed: int | None,
    meta: dict[str, object],
) -> Excitation:
    """Place ``signal`` on the requested axis, zeros elsewhere."""
    zeros = np.zeros_like(signal)
    channels: dict[str, FloatArray] = {"x": zeros, "y": zeros.copy(), "z": zeros.copy()}
    channels[axis] = signal
    return Excitation(
        a_x=channels["x"], a_y=channels["y"], a_z=channels["z"], fs=fs, seed=seed, meta=meta
    )


def _resample(signal: FloatArray, fs_in: float, fs_out: float) -> FloatArray:
    """Polyphase resampling ``fs_in -> fs_out`` with a rational rate ratio.

    Parameters
    ----------
    signal : numpy.ndarray
        Input samples.
    fs_in, fs_out : float
        Source and target sampling rates, Hz.

    Returns
    -------
    numpy.ndarray
        Resampled signal.
    """
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    out = resample_poly(signal, ratio.numerator, ratio.denominator)
    return np.asarray(out, dtype=np.float64)


def _resolve_column(name_or_index: int | str, header: list[str] | None, what: str) -> int:
    """Map a column given by header name or 0-based index to an index."""
    if isinstance(name_or_index, int):
        return name_or_index
    if header is None:
        msg = f"{what} given by name {name_or_index!r} but the CSV file has no header row"
        raise ValueError(msg)
    try:
        return header.index(name_or_index)
    except ValueError as exc:
        msg = f"{what} {name_or_index!r} not found in CSV header {header}"
        raise ValueError(msg) from exc


def _read_csv_table(
    path: Path, delimiter: str, skiprows: int
) -> tuple[list[str] | None, FloatArray]:
    """Read a CSV file into an optional header and a 2-D float table.

    The first non-skipped row is treated as a header if any of its fields is
    not parseable as a float.
    """
    with path.open(newline="") as fh:
        rows = [row for row in csv.reader(fh, delimiter=delimiter) if row]
    rows = rows[skiprows:]
    if not rows:
        msg = f"CSV file {path} contains no data rows"
        raise ValueError(msg)

    def _is_float_row(row: list[str]) -> bool:
        try:
            [float(cell) for cell in row]
        except ValueError:
            return False
        return True

    header: list[str] | None = None
    if not _is_float_row(rows[0]):
        header = [cell.strip() for cell in rows[0]]
        rows = rows[1:]
    if not rows:
        msg = f"CSV file {path} has a header but no data rows"
        raise ValueError(msg)
    table = np.asarray([[float(cell) for cell in row] for row in rows], dtype=np.float64)
    return header, table


@LOADER_REGISTRY.register("csv")
class CsvLoader:
    """CSV record loader: column mapping, fs from a time column, units to SI."""

    def load(self, spec: ExcitationSpec, *, seed: int | None = None) -> Excitation:
        """Load a CSV acceleration record (see :class:`CsvSpec`)."""
        if not isinstance(spec, CsvSpec):
            msg = f"CsvLoader expects CsvSpec, got {type(spec).__name__}"
            raise TypeError(msg)
        path = Path(spec.path)
        header, table = _read_csv_table(path, spec.delimiter, spec.skiprows)

        data_col = _resolve_column(spec.column, header, "data column")
        signal = np.ascontiguousarray(table[:, data_col], dtype=np.float64)

        if spec.time_column is not None:
            time_col = _resolve_column(spec.time_column, header, "time column")
            t = table[:, time_col]
            dt = np.diff(t)
            if dt.size == 0 or np.any(dt <= 0.0):
                msg = "time column must be strictly increasing with >= 2 samples"
                raise ValueError(msg)
            fs = 1.0 / float(np.median(dt))
        else:
            assert spec.fs_hz is not None
            fs = spec.fs_hz

        if spec.units == "g":
            signal = signal * G0_M_S2

        if spec.resample_hz is not None and spec.resample_hz != fs:
            signal = _resample(signal, fs, spec.resample_hz)
            fs = spec.resample_hz

        logger.info("csv loaded: %s, %d samples @ %.6g Hz", path, signal.size, fs)
        meta: dict[str, object] = {
            "generator": "csv",
            "path": str(path),
            "axis": spec.axis,
            "source_units": spec.units,
        }
        return _pack_on_axis(signal, spec.axis, fs, seed, meta)


def _normalize_wav(data: np.ndarray) -> FloatArray:
    """Normalize WAV samples of any PCM dtype to float64 in [-1, 1]."""
    if data.dtype == np.int16:
        return np.asarray(data, dtype=np.float64) / 32768.0
    if data.dtype == np.int32:
        return np.asarray(data, dtype=np.float64) / 2147483648.0
    if data.dtype == np.uint8:
        return (np.asarray(data, dtype=np.float64) - 128.0) / 128.0
    if data.dtype in (np.float32, np.float64):
        return np.asarray(data, dtype=np.float64)
    msg = f"unsupported WAV sample dtype: {data.dtype}"
    raise ValueError(msg)


@LOADER_REGISTRY.register("wav")
class WavLoader:
    """WAV record loader: PCM normalization, channel pick, full-scale mapping."""

    def load(self, spec: ExcitationSpec, *, seed: int | None = None) -> Excitation:
        """Load a WAV acceleration record (see :class:`WavSpec`)."""
        if not isinstance(spec, WavSpec):
            msg = f"WavLoader expects WavSpec, got {type(spec).__name__}"
            raise TypeError(msg)
        path = Path(spec.path)
        fs_int, raw = wavfile.read(path)
        fs = float(fs_int)
        data = np.atleast_2d(np.asarray(raw))
        if raw.ndim == 1:
            data = data.T  # (n, 1)
        n_channels = data.shape[1]
        if spec.channel >= n_channels:
            msg = f"channel {spec.channel} out of range: file has {n_channels} channel(s)"
            raise ValueError(msg)
        signal = _normalize_wav(data[:, spec.channel]) * (spec.full_scale_g * G0_M_S2)

        if spec.resample_hz is not None and spec.resample_hz != fs:
            signal = _resample(signal, fs, spec.resample_hz)
            fs = spec.resample_hz

        logger.info("wav loaded: %s, %d samples @ %.6g Hz", path, signal.size, fs)
        meta: dict[str, object] = {
            "generator": "wav",
            "path": str(path),
            "axis": spec.axis,
            "channel": spec.channel,
            "full_scale_g": spec.full_scale_g,
        }
        return _pack_on_axis(signal, spec.axis, fs, seed, meta)


# ---------------------------------------------------------------------------
# Instrument-format helpers (TDMS / UFF / MAT / HDF5, seam SW-08, S8)
#
# These formats carry their own sampling and unit metadata. The helpers below
# centralize the cross-cutting concerns so each loader stays a thin adapter:
# resolving the channel's physical unit (auto-detected or explicit, with a loud
# error on conflict), converting to SI acceleration, selecting a channel from a
# 2-D record, and the shared resample/log/pack tail.
# ---------------------------------------------------------------------------

# Accepted spellings for each physical unit. The voltage family collapses to a
# single dimension ``"V"`` (raw transducer output assumed to be in volts; the
# numeric scale to acceleration comes from the spec's ``sensitivity``).
_UNIT_ALIASES: dict[str, frozenset[str]] = {
    "g": frozenset({"g", "g0", "gn", "gs", "g-force", "gravity"}),
    "m/s^2": frozenset(
        {"m/s^2", "m/s2", "m/s²", "ms^-2", "ms-2", "m/sec^2", "meter/second^2", "meterpersecond^2"}
    ),
    "V": frozenset({"v", "volt", "volts", "mv", "millivolt", "millivolts"}),
}


def _canonical_unit(label: object) -> str | None:
    """Map a unit string to a canonical ``'g'`` | ``'m/s^2'`` | ``'V'``.

    Parameters
    ----------
    label : object
        Unit string from a file or spec (case- and space-insensitive). ``bytes``
        are decoded; ``None`` and unrecognized labels return ``None``.

    Returns
    -------
    str or None
        The canonical unit, or ``None`` if ``label`` is absent or unrecognized.
    """
    if label is None:
        return None
    if isinstance(label, bytes):
        label = label.decode(errors="replace")
    key = str(label).strip().lower().replace(" ", "")
    if not key:
        return None
    for canon, aliases in _UNIT_ALIASES.items():
        if key in aliases:
            return canon
    return None


def _resolve_units(spec_units: str, file_label: object, *, source: str) -> str:
    """Resolve the channel's unit to a canonical name, failing loudly on conflict.

    Parameters
    ----------
    spec_units : {'g', 'm/s^2', 'V', 'auto'}
        The unit requested by the spec. ``'auto'`` defers to the file's label.
    file_label : object
        The unit label embedded in the file, if any.
    source : str
        Human-readable record identifier for error messages.

    Returns
    -------
    str
        Canonical unit ``'g'`` | ``'m/s^2'`` | ``'V'``.

    Raises
    ------
    ValueError
        If ``'auto'`` is requested but the file has no recognized label, or if an
        explicit unit conflicts with a recognized file label (10 §7).
    """
    file_canon = _canonical_unit(file_label)
    if spec_units == "auto":
        if file_canon is None:
            msg = (
                f"{source}: units='auto' but the file has no recognized unit label "
                f"(found {file_label!r}); set `units` explicitly"
            )
            raise ValueError(msg)
        return file_canon
    spec_canon = _canonical_unit(spec_units)
    if spec_canon is None:  # pragma: no cover - guarded by the spec's Literal
        msg = f"{source}: unsupported unit {spec_units!r}"
        raise ValueError(msg)
    if file_canon is not None and file_canon != spec_canon:
        msg = (
            f"{source}: spec says units={spec_units!r} but the file is labeled "
            f"{file_label!r}; refusing to guess (10 §7)"
        )
        raise ValueError(msg)
    return spec_canon


def _to_si_acceleration(
    signal: FloatArray, unit: str, sensitivity: float | None, sensitivity_unit: str
) -> FloatArray:
    """Convert a stored channel in ``unit`` to acceleration in m/s^2.

    Parameters
    ----------
    signal : numpy.ndarray
        Stored samples in ``unit``.
    unit : {'g', 'm/s^2', 'V'}
        Canonical unit of ``signal`` (see :func:`_resolve_units`).
    sensitivity : float or None
        Accelerometer sensitivity; required when ``unit == 'V'``.
    sensitivity_unit : {'mV/g', 'V/g', 'mV/(m/s^2)', 'V/(m/s^2)'}
        Units of ``sensitivity``.

    Returns
    -------
    numpy.ndarray
        Acceleration in m/s^2.

    Raises
    ------
    ValueError
        If ``unit == 'V'`` but no ``sensitivity`` was given.
    """
    if unit == "m/s^2":
        return signal
    if unit == "g":
        return signal * G0_M_S2
    # unit == "V": divide by sensitivity (V per accel-unit), then to m/s^2.
    if sensitivity is None:
        msg = "voltage record needs `sensitivity` to convert to acceleration"
        raise ValueError(msg)
    volts_per_unit = sensitivity * (1e-3 if sensitivity_unit.startswith("mV") else 1.0)
    accel = signal / volts_per_unit
    if sensitivity_unit.endswith("/g"):
        accel = accel * G0_M_S2
    return np.asarray(accel, dtype=np.float64)


def _select_column(data: npt.ArrayLike, column: int | None, what: str) -> FloatArray:
    """Select one channel from a 1-D or 2-D record as a 1-D float64 array.

    A 2-D record is oriented time-major (the longer axis is taken as time), so a
    transposed ``(channels, samples)`` layout is handled transparently; ``column``
    then selects among the channels.

    Parameters
    ----------
    data : array_like
        The stored array (1-D, or 2-D as samples x channels).
    column : int or None
        Channel index for 2-D data (the first channel if ``None``).
    what : str
        Human-readable record identifier for error messages.

    Returns
    -------
    numpy.ndarray
        The selected channel, 1-D float64.

    Raises
    ------
    ValueError
        If the array is not 1-D/2-D, or the column is out of range.
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        if column not in (None, 0):
            msg = f"{what} is 1-D but column={column} was requested"
            raise ValueError(msg)
        return np.ascontiguousarray(arr)
    if arr.ndim != 2:
        msg = f"{what} must be 1-D or 2-D, got {arr.ndim}-D"
        raise ValueError(msg)
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T  # orient time-major: samples along axis 0
    col = 0 if column is None else column
    if col >= arr.shape[1]:
        msg = f"{what}: column {col} out of range ({arr.shape[1]} channel(s))"
        raise ValueError(msg)
    return np.ascontiguousarray(arr[:, col], dtype=np.float64)


def _pick_channel(names: list[str], selector: int | str, what: str) -> int:
    """Resolve a channel given by 0-based index or name to an index.

    Parameters
    ----------
    names : list of str
        Channel names in order.
    selector : int or str
        A 0-based index or a channel name.
    what : str
        Human-readable record identifier for error messages.

    Returns
    -------
    int
        The resolved 0-based index.

    Raises
    ------
    ValueError
        If the index is out of range or the name is not found.
    """
    if isinstance(selector, int):
        if not 0 <= selector < len(names):
            msg = f"{what}: index {selector} out of range ({len(names)} channel(s))"
            raise ValueError(msg)
        return selector
    try:
        return names.index(selector)
    except ValueError as exc:
        msg = f"{what}: channel {selector!r} not found in {names}"
        raise ValueError(msg) from exc


def _scalar_rate(value: npt.ArrayLike, source: str) -> float:
    """Read a scalar sampling rate from a possibly-array metadata value."""
    fs = float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
    if not np.isfinite(fs) or fs <= 0.0:
        msg = f"{source}: sampling rate must be positive and finite, got {fs!r}"
        raise ValueError(msg)
    return fs


def _finish(
    signal: FloatArray,
    axis: str,
    fs: float,
    resample_hz: float | None,
    seed: int | None,
    meta: dict[str, object],
    kind: str,
) -> Excitation:
    """Resample if requested, log, and pack the channel onto its axis."""
    if resample_hz is not None and resample_hz != fs:
        signal = _resample(signal, fs, resample_hz)
        fs = resample_hz
    signal = np.ascontiguousarray(signal, dtype=np.float64)
    logger.info("%s loaded: %s, %d samples @ %.6g Hz", kind, meta.get("path", "?"), signal.size, fs)
    return _pack_on_axis(signal, axis, fs, seed, meta)


@LOADER_REGISTRY.register("tdms")
class TdmsLoader:
    """NI TDMS record loader (optional dependency: ``nptdms``)."""

    def load(self, spec: ExcitationSpec, *, seed: int | None = None) -> Excitation:
        """Load one channel from an NI TDMS file (see :class:`TdmsSpec`)."""
        if not isinstance(spec, TdmsSpec):
            msg = f"TdmsLoader expects TdmsSpec, got {type(spec).__name__}"
            raise TypeError(msg)
        try:
            from nptdms import TdmsFile
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(_missing_extra_msg("nptdms")) from exc

        path = Path(spec.path)
        tdms = TdmsFile.read(str(path))
        groups = tdms.groups()
        if not groups:
            msg = f"TDMS file {path} has no channel groups"
            raise ValueError(msg)
        if spec.group is None:
            group = groups[0]
        else:
            group_names = [g.name for g in groups]
            group = groups[_pick_channel(group_names, spec.group, f"TDMS group in {path}")]
        channels = group.channels()
        if not channels:
            msg = f"TDMS group {group.name!r} in {path} has no channels"
            raise ValueError(msg)
        idx = _pick_channel(
            [c.name for c in channels], spec.channel, f"TDMS channel in group {group.name!r}"
        )
        channel = channels[idx]
        data = np.asarray(channel[:], dtype=np.float64)
        props = dict(channel.properties)

        if spec.fs_hz is not None:
            fs = spec.fs_hz
        else:
            inc = props.get("wf_increment")
            if inc is None or float(inc) <= 0.0:
                msg = f"TDMS channel {channel.name!r} has no usable wf_increment; set fs_hz"
                raise ValueError(msg)
            fs = 1.0 / float(inc)

        unit = _resolve_units(spec.units, props.get("unit_string"), source=f"TDMS {path}")
        signal = _to_si_acceleration(data, unit, spec.sensitivity, spec.sensitivity_unit)
        meta: dict[str, object] = {
            "generator": "tdms",
            "path": str(path),
            "axis": spec.axis,
            "group": group.name,
            "channel": channel.name,
            "source_units": unit,
        }
        return _finish(signal, spec.axis, fs, spec.resample_hz, seed, meta, "tdms")


@LOADER_REGISTRY.register("uff")
class UffLoader:
    """UFF/UNV dataset-58 record loader (optional dependency: ``pyuff``)."""

    def load(self, spec: ExcitationSpec, *, seed: int | None = None) -> Excitation:
        """Load one dataset-58 function from a UFF/UNV file (see :class:`UffSpec`)."""
        if not isinstance(spec, UffSpec):
            msg = f"UffLoader expects UffSpec, got {type(spec).__name__}"
            raise TypeError(msg)
        try:
            import pyuff
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(_missing_extra_msg("pyuff")) from exc

        path = Path(spec.path)
        uff = pyuff.UFF(str(path))
        set_types = list(uff.get_set_types())
        type58 = [i for i, t in enumerate(set_types) if int(t) == 58]
        if not type58:
            msg = f"UFF file {path} has no dataset-58 (function) records"
            raise ValueError(msg)
        if spec.dataset_index >= len(type58):
            msg = (
                f"UFF dataset_index {spec.dataset_index} out of range "
                f"({len(type58)} dataset-58 record(s) in {path})"
            )
            raise ValueError(msg)
        rec = uff.read_sets(type58[spec.dataset_index])
        data = np.real(np.asarray(rec["data"], dtype=np.float64))

        if spec.fs_hz is not None:
            fs = spec.fs_hz
        else:
            inc = rec.get("abscissa_inc")
            if not inc or float(inc) <= 0.0:
                msg = f"UFF record in {path} has no usable abscissa_inc; set fs_hz"
                raise ValueError(msg)
            fs = 1.0 / float(inc)

        unit = _resolve_units(spec.units, rec.get("ordinate_axis_units_lab"), source=f"UFF {path}")
        signal = _to_si_acceleration(data, unit, spec.sensitivity, spec.sensitivity_unit)
        meta: dict[str, object] = {
            "generator": "uff",
            "path": str(path),
            "axis": spec.axis,
            "dataset_index": spec.dataset_index,
            "source_units": unit,
        }
        return _finish(signal, spec.axis, fs, spec.resample_hz, seed, meta, "uff")


@LOADER_REGISTRY.register("mat")
class MatLoader:
    """MATLAB ``.mat`` (v4/v5/v7) record loader (uses core ``scipy.io``)."""

    def load(self, spec: ExcitationSpec, *, seed: int | None = None) -> Excitation:
        """Load an acceleration variable from a ``.mat`` file (see :class:`MatSpec`)."""
        if not isinstance(spec, MatSpec):
            msg = f"MatLoader expects MatSpec, got {type(spec).__name__}"
            raise TypeError(msg)
        from scipy.io import loadmat  # scipy is a core dependency

        path = Path(spec.path)
        try:
            mat = loadmat(str(path))
        except NotImplementedError as exc:
            msg = (
                f"{path} looks like a MATLAB v7.3 file (HDF5-based); "
                f"use the 'hdf5' loader or re-save it as v7"
            )
            raise ValueError(msg) from exc

        if spec.data_key not in mat:
            keys = [k for k in mat if not k.startswith("__")]
            msg = f"variable {spec.data_key!r} not found in {path}; variables: {keys}"
            raise ValueError(msg)
        signal = _select_column(mat[spec.data_key], spec.column, f"MAT variable {spec.data_key!r}")

        if spec.fs_hz is not None:
            fs = spec.fs_hz
        else:
            assert spec.fs_key is not None  # one of fs_hz/fs_key is set (model validator)
            if spec.fs_key not in mat:
                msg = f"fs_key {spec.fs_key!r} not found in {path}"
                raise ValueError(msg)
            fs = _scalar_rate(mat[spec.fs_key], f"MAT fs_key {spec.fs_key!r}")

        if spec.units == "auto":
            msg = "MAT files carry no unit label; set `units` explicitly (not 'auto')"
            raise ValueError(msg)
        unit = _resolve_units(spec.units, None, source=f"MAT {path}")
        signal = _to_si_acceleration(signal, unit, spec.sensitivity, spec.sensitivity_unit)
        meta: dict[str, object] = {
            "generator": "mat",
            "path": str(path),
            "axis": spec.axis,
            "data_key": spec.data_key,
            "source_units": unit,
        }
        return _finish(signal, spec.axis, fs, spec.resample_hz, seed, meta, "mat")


@LOADER_REGISTRY.register("hdf5")
class Hdf5Loader:
    """HDF5 (``.h5`` / ``.hdf5``) record loader (optional dependency: ``h5py``)."""

    def load(self, spec: ExcitationSpec, *, seed: int | None = None) -> Excitation:
        """Load an acceleration dataset from an HDF5 file (see :class:`Hdf5Spec`)."""
        if not isinstance(spec, Hdf5Spec):
            msg = f"Hdf5Loader expects Hdf5Spec, got {type(spec).__name__}"
            raise TypeError(msg)
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(_missing_extra_msg("h5py")) from exc

        path = Path(spec.path)
        with h5py.File(str(path), "r") as handle:
            if spec.dataset not in handle:
                msg = f"dataset {spec.dataset!r} not found in {path}"
                raise ValueError(msg)
            dataset = handle[spec.dataset]
            raw = np.asarray(dataset[()], dtype=np.float64)
            attrs = dict(dataset.attrs)

        signal = _select_column(raw, spec.column, f"HDF5 dataset {spec.dataset!r}")

        if spec.fs_hz is not None:
            fs = spec.fs_hz
        else:
            assert spec.fs_attr is not None  # one of fs_hz/fs_attr is set (model validator)
            if spec.fs_attr not in attrs:
                msg = f"fs_attr {spec.fs_attr!r} not found on dataset {spec.dataset!r} in {path}"
                raise ValueError(msg)
            fs = _scalar_rate(attrs[spec.fs_attr], f"HDF5 fs_attr {spec.fs_attr!r}")

        file_label = attrs.get(spec.units_attr) if spec.units_attr is not None else None
        unit = _resolve_units(spec.units, file_label, source=f"HDF5 {path}")
        signal = _to_si_acceleration(signal, unit, spec.sensitivity, spec.sensitivity_unit)
        meta: dict[str, object] = {
            "generator": "hdf5",
            "path": str(path),
            "axis": spec.axis,
            "dataset": spec.dataset,
            "source_units": unit,
        }
        return _finish(signal, spec.axis, fs, spec.resample_hz, seed, meta, "hdf5")
