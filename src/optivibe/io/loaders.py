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
from scipy.io import wavfile
from scipy.signal import resample_poly

from optivibe.core.config.models import CsvSpec, WavSpec
from optivibe.core.logging import get_logger
from optivibe.core.registry import Registry
from optivibe.core.types import Excitation, FloatArray
from optivibe.core.units import G0_M_S2

logger = get_logger(__name__)

__all__ = ["LOADER_REGISTRY", "CsvLoader", "SignalLoader", "WavLoader"]


@runtime_checkable
class SignalLoader(Protocol):
    """Loads one measured record into the :class:`Excitation` contract."""

    def load(self, spec: CsvSpec | WavSpec, *, seed: int | None = None) -> Excitation:
        """Load a file according to ``spec``.

        Parameters
        ----------
        spec : CsvSpec or WavSpec
            File path, channel/column mapping, units, optional resampling.
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

    def load(self, spec: CsvSpec | WavSpec, *, seed: int | None = None) -> Excitation:
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

    def load(self, spec: CsvSpec | WavSpec, *, seed: int | None = None) -> Excitation:
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
