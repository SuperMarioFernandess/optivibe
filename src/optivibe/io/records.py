"""Instrument-output records: photocurrent captures for the S-02 analyzer.

The replay loaders (:mod:`optivibe.io.loaders`) feed the *input* seam: measured
acceleration mapped onto the :class:`~optivibe.core.types.Excitation` contract.
This module feeds the *output* seam of decision SW-08 (roles distinguished
there): a recorded **photocurrent** of the real instrument (doc 20 §5, role
S-02 / O-SW-03) is read into the standard
:class:`~optivibe.core.types.DetectorOutput` contract so the same inverse chain
that serves the forward simulation can process it (doc 17 §7).

Record descriptors are pydantic models (a discriminated union over the priority
formats of plan 20 §5: TDMS, HDF5, CSV). The mandatory record metadata of plan
20 §5 is enforced here: the sampling rate ``fs`` (explicit or from the file),
the channel units (``"A"`` or ``"V"`` with the transimpedance ``r_f_ohm``
required for volt records), the reference-accelerometer sensitivity when a
paired reference channel is declared, and a non-empty ``timestamp``.

All unit conversions happen at this input boundary (doc 10 §6): volt records
are divided by ``R_f`` so the contract always carries amperes; the optional
paired reference channel is converted to SI acceleration with the *same*
helpers the replay loaders use (single implementation of the unit rules).
Readers are registered in :data:`RECORD_REGISTRY` (09 §6): adding an instrument
format is a new adapter with a registration, never a core change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from optivibe.core.logging import get_logger
from optivibe.core.registry import Registry
from optivibe.core.types import DetectorOutput, FloatArray
from optivibe.io.loaders import (
    _missing_extra_msg,
    _pick_channel,
    _read_csv_table,
    _resample,
    _resolve_column,
    _resolve_units,
    _scalar_rate,
    _select_column,
    _to_si_acceleration,
)

logger = get_logger(__name__)

__all__ = [
    "RECORD_REGISTRY",
    "CsvRecordReader",
    "CsvRecordSpec",
    "Hdf5RecordReader",
    "Hdf5RecordSpec",
    "InstrumentRecord",
    "RecordReader",
    "RecordSpec",
    "ReferenceChannelSpec",
    "TdmsRecordReader",
    "TdmsRecordSpec",
    "read_record",
]


class _Frozen(BaseModel):
    """Immutable, strictly-validated base (mirror of the config base)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ReferenceChannelSpec(_Frozen):
    """Paired reference-accelerometer channel of a photocurrent record (20 §5).

    Attributes
    ----------
    channel : int or str
        Format-specific selector *inside the same file*: a CSV column (index or
        header name), a TDMS channel (index or name, same group as the signal),
        or an HDF5 dataset path (str) / column of the signal dataset (int).
    units : {"auto", "g", "m/s^2", "V"}
        Stored units of the reference channel. ``"auto"`` reads the label
        embedded in the file (TDMS ``unit_string``; HDF5 ``units`` attribute of
        the reference dataset) and fails loudly where no label exists (CSV,
        HDF5 column selection) -- doc 10 §7, no guessing.
    sensitivity : float or None
        Reference-accelerometer sensitivity from its calibration certificate
        (mandatory metadata, 20 §5); required when ``units == "V"``.
    sensitivity_unit : {"mV/g", "V/g", "mV/(m/s^2)", "V/(m/s^2)"}
        Units of ``sensitivity``.
    """

    channel: int | str
    units: Literal["auto", "g", "m/s^2", "V"] = "auto"
    sensitivity: float | None = Field(default=None, gt=0.0)
    sensitivity_unit: Literal["mV/g", "V/g", "mV/(m/s^2)", "V/(m/s^2)"] = "mV/g"


class _RecordBase(_Frozen):
    """Fields shared by every photocurrent-record descriptor (20 §5).

    Attributes
    ----------
    path : str
        Path to the record data file.
    units : {"A", "V"}
        Units of the stored photocurrent channel. Volt records require
        ``r_f_ohm`` (the transimpedance) so the contract can carry amperes.
    r_f_ohm : float or None
        Transimpedance ``R_f`` of the front end, Ohm (mandatory for ``"V"``).
    fs_hz : float or None
        Explicit sampling rate, Hz; each format documents its file fallback.
    resample_hz : float or None
        Optional polyphase resampling target, Hz.
    timestamp : str
        Record timestamp (mandatory metadata, 20 §5); free-form ISO-8601
        recommended, must be non-empty.
    reference : ReferenceChannelSpec or None
        Optional paired reference-accelerometer channel in the same file.
    """

    path: str = Field(description="Path to the record data file")
    units: Literal["A", "V"] = Field(description="Photocurrent channel units")
    r_f_ohm: float | None = Field(
        default=None, gt=0.0, description="Transimpedance R_f, Ohm (required for 'V')"
    )
    fs_hz: float | None = Field(default=None, gt=0.0, description="Sampling rate, Hz")
    resample_hz: float | None = Field(default=None, gt=0.0, description="Resample target, Hz")
    timestamp: str = Field(min_length=1, description="Record timestamp (mandatory, 20 §5)")
    reference: ReferenceChannelSpec | None = None

    @model_validator(mode="after")
    def _check_volt_needs_rf(self) -> _RecordBase:
        if self.units == "V" and self.r_f_ohm is None:
            msg = (
                "a volt photocurrent record needs the transimpedance r_f_ohm to convert "
                "to amperes (mandatory record metadata, plan 20 §5)"
            )
            raise ValueError(msg)
        return self


class CsvRecordSpec(_RecordBase):
    """CSV photocurrent record: column mapping, fs from a time column (20 §5)."""

    format: Literal["csv"] = "csv"
    column: int | str = Field(default=0, description="Signal column (index or header name)")
    time_column: int | str | None = Field(
        default=None, description="Time column for the sampling rate (fs_hz if None)"
    )
    delimiter: str = ","
    skiprows: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_rate(self) -> CsvRecordSpec:
        if self.fs_hz is None and self.time_column is None:
            msg = "csv record needs either fs_hz or time_column"
            raise ValueError(msg)
        return self


class TdmsRecordSpec(_RecordBase):
    """NI TDMS photocurrent record (optional dependency: ``nptdms``)."""

    format: Literal["tdms"] = "tdms"
    group: int | str | None = Field(default=None, description="Channel group (first if None)")
    channel: int | str = Field(default=0, description="Signal channel (index or name)")
    # fs fallback: the channel's ``wf_increment`` property when fs_hz is None.


class Hdf5RecordSpec(_RecordBase):
    """HDF5 (``.h5``/``.hdf5``) photocurrent record (optional dependency: ``h5py``)."""

    format: Literal["hdf5"] = "hdf5"
    dataset: str = Field(description="Path of the signal dataset inside the file")
    column: int | None = Field(
        default=None, ge=0, description="Column to read for 2-D data (the first if None)"
    )
    fs_attr: str | None = Field(
        default=None, description="Dataset attribute holding the scalar sampling rate"
    )

    @model_validator(mode="after")
    def _check_rate(self) -> Hdf5RecordSpec:
        if self.fs_hz is None and self.fs_attr is None:
            msg = "hdf5 record needs either fs_hz or fs_attr"
            raise ValueError(msg)
        return self


RecordSpec = Annotated[
    CsvRecordSpec | TdmsRecordSpec | Hdf5RecordSpec,
    Field(discriminator="format"),
]
"""Discriminated union of the photocurrent-record descriptors (20 §5 formats)."""


@dataclass(frozen=True)
class InstrumentRecord:
    """A loaded instrument-output record (role S-02).

    Parameters
    ----------
    detector : DetectorOutput
        The photocurrent record on the standard detector contract: samples in
        **amperes** (volt records are divided by ``R_f`` at this boundary,
        doc 10 §6), ``dc_level`` estimated as the sample mean (real front ends
        are AC-coupled, doc 07 §1.4, so the mean is the residual pedestal --
        consistent with :func:`optivibe.dsp.calibration.detector_ac_current`),
        and traceability metadata in ``noise`` (``model="instrument_record"``,
        so the metadata-driven NEA of the forward tract correctly reports
        "no synthetic noise floor" for real records).
    reference_accel : numpy.ndarray or None
        The paired reference channel converted to SI acceleration, m/s^2
        (same length and rate as the signal), or ``None``.
    meta : Mapping[str, object]
        Record metadata: path, format, timestamp, source units, ``r_f_ohm``,
        native sampling rate.
    """

    detector: DetectorOutput
    reference_accel: FloatArray | None = None
    meta: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reference_accel is not None:
            ref = np.ascontiguousarray(self.reference_accel, dtype=np.float64)
            if ref.shape != self.detector.samples.shape:
                msg = (
                    f"reference channel length {ref.size} != signal length "
                    f"{self.detector.n_samples}"
                )
                raise ValueError(msg)
            object.__setattr__(self, "reference_accel", ref)


@runtime_checkable
class RecordReader(Protocol):
    """Loads one photocurrent record into :class:`InstrumentRecord`."""

    def load(self, spec: RecordSpec) -> InstrumentRecord:
        """Load a record according to ``spec``.

        Parameters
        ----------
        spec : RecordSpec
            A photocurrent-record descriptor (CSV/TDMS/HDF5). Each reader
            accepts the spec member matching its format and rejects the others
            loudly.

        Returns
        -------
        InstrumentRecord
            The record in amperes with its optional SI reference channel.
        """
        ...


RECORD_REGISTRY: Registry[RecordReader] = Registry("io.record")


def _finish_record(
    signal_native: FloatArray,
    reference_native: FloatArray | None,
    reference_label: object,
    fs: float,
    spec: CsvRecordSpec | TdmsRecordSpec | Hdf5RecordSpec,
    extra_meta: dict[str, object],
) -> InstrumentRecord:
    """Convert to amperes/SI, resample, and pack the record (shared tail)."""
    signal = np.ascontiguousarray(signal_native, dtype=np.float64)
    if spec.units == "V":
        assert spec.r_f_ohm is not None  # enforced by the descriptor validator
        signal = signal / spec.r_f_ohm

    reference: FloatArray | None = None
    if spec.reference is not None:
        if reference_native is None:  # pragma: no cover - readers always pair them
            msg = f"{spec.path}: reference declared but no channel was read"
            raise ValueError(msg)
        ref_spec = spec.reference
        unit = _resolve_units(ref_spec.units, reference_label, source=f"reference in {spec.path}")
        reference = _to_si_acceleration(
            np.ascontiguousarray(reference_native, dtype=np.float64),
            unit,
            ref_spec.sensitivity,
            ref_spec.sensitivity_unit,
        )

    fs_native = fs
    if spec.resample_hz is not None and spec.resample_hz != fs:
        signal = _resample(signal, fs, spec.resample_hz)
        if reference is not None:
            reference = _resample(reference, fs, spec.resample_hz)
        fs = spec.resample_hz

    signal = np.ascontiguousarray(signal, dtype=np.float64)
    dc_level = float(np.mean(signal))
    meta: dict[str, object] = {
        "loader": f"record:{spec.format}",
        "path": spec.path,
        "timestamp": spec.timestamp,
        "source_units": spec.units,
        "r_f_ohm": spec.r_f_ohm,
        "fs_native_hz": fs_native,
        **extra_meta,
    }
    detector = DetectorOutput(
        samples=signal,
        fs=fs,
        dc_level=dc_level,
        units="A",
        noise={"model": "instrument_record", **meta},
    )
    logger.info(
        "record loaded: %s (%s), %d samples @ %.6g Hz, reference=%s",
        spec.path,
        spec.format,
        signal.size,
        fs,
        reference is not None,
    )
    return InstrumentRecord(detector=detector, reference_accel=reference, meta=meta)


@RECORD_REGISTRY.register("csv")
class CsvRecordReader:
    """CSV photocurrent-record reader (see :class:`CsvRecordSpec`)."""

    def load(self, spec: RecordSpec) -> InstrumentRecord:
        """Load a CSV photocurrent record."""
        if not isinstance(spec, CsvRecordSpec):
            msg = f"CsvRecordReader expects CsvRecordSpec, got {type(spec).__name__}"
            raise TypeError(msg)
        path = Path(spec.path)
        header, table = _read_csv_table(path, spec.delimiter, spec.skiprows)

        data_col = _resolve_column(spec.column, header, "record column")
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
            assert spec.fs_hz is not None  # one of fs_hz/time_column is set (validator)
            fs = spec.fs_hz

        reference: FloatArray | None = None
        if spec.reference is not None:
            ref_col = _resolve_column(spec.reference.channel, header, "reference column")
            reference = np.ascontiguousarray(table[:, ref_col], dtype=np.float64)
        # CSV carries no unit labels: reference units must be explicit (10 §7).
        return _finish_record(signal, reference, None, fs, spec, {"column": spec.column})


@RECORD_REGISTRY.register("tdms")
class TdmsRecordReader:
    """NI TDMS photocurrent-record reader (optional dependency: ``nptdms``)."""

    def load(self, spec: RecordSpec) -> InstrumentRecord:
        """Load one channel (plus optional reference) from an NI TDMS file."""
        if not isinstance(spec, TdmsRecordSpec):
            msg = f"TdmsRecordReader expects TdmsRecordSpec, got {type(spec).__name__}"
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
        names = [c.name for c in channels]
        idx = _pick_channel(names, spec.channel, f"TDMS channel in group {group.name!r}")
        channel = channels[idx]
        signal = np.asarray(channel[:], dtype=np.float64)
        props = dict(channel.properties)

        if spec.fs_hz is not None:
            fs = spec.fs_hz
        else:
            inc = props.get("wf_increment")
            if inc is None or float(inc) <= 0.0:
                msg = f"TDMS channel {channel.name!r} has no usable wf_increment; set fs_hz"
                raise ValueError(msg)
            fs = 1.0 / float(inc)

        reference: FloatArray | None = None
        ref_label: object = None
        if spec.reference is not None:
            ref_idx = _pick_channel(
                names, spec.reference.channel, f"TDMS reference channel in group {group.name!r}"
            )
            ref_channel = channels[ref_idx]
            reference = np.asarray(ref_channel[:], dtype=np.float64)
            ref_label = dict(ref_channel.properties).get("unit_string")

        extra: dict[str, object] = {"group": group.name, "channel": channel.name}
        return _finish_record(signal, reference, ref_label, fs, spec, extra)


@RECORD_REGISTRY.register("hdf5")
class Hdf5RecordReader:
    """HDF5 photocurrent-record reader (optional dependency: ``h5py``)."""

    def load(self, spec: RecordSpec) -> InstrumentRecord:
        """Load a photocurrent dataset (plus optional reference) from HDF5."""
        if not isinstance(spec, Hdf5RecordSpec):
            msg = f"Hdf5RecordReader expects Hdf5RecordSpec, got {type(spec).__name__}"
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

            reference: FloatArray | None = None
            ref_label: object = None
            if spec.reference is not None:
                selector = spec.reference.channel
                if isinstance(selector, str):
                    if selector not in handle:
                        msg = f"reference dataset {selector!r} not found in {path}"
                        raise ValueError(msg)
                    ref_ds = handle[selector]
                    reference = _select_column(
                        np.asarray(ref_ds[()], dtype=np.float64),
                        None,
                        f"HDF5 reference dataset {selector!r}",
                    )
                    ref_label = dict(ref_ds.attrs).get("units")
                else:
                    # An int selects a column of the *signal* dataset: no
                    # per-column unit label exists, so units must be explicit.
                    reference = _select_column(
                        raw, selector, f"HDF5 reference column of {spec.dataset!r}"
                    )

        signal = _select_column(raw, spec.column, f"HDF5 dataset {spec.dataset!r}")

        if spec.fs_hz is not None:
            fs = spec.fs_hz
        else:
            assert spec.fs_attr is not None  # one of fs_hz/fs_attr is set (validator)
            if spec.fs_attr not in attrs:
                msg = f"fs_attr {spec.fs_attr!r} not found on dataset {spec.dataset!r} in {path}"
                raise ValueError(msg)
            fs = _scalar_rate(attrs[spec.fs_attr], f"HDF5 fs_attr {spec.fs_attr!r}")

        extra: dict[str, object] = {"dataset": spec.dataset}
        return _finish_record(signal, reference, ref_label, fs, spec, extra)


def read_record(spec: RecordSpec) -> InstrumentRecord:
    """Read one photocurrent record through the reader registry (09 §6).

    Parameters
    ----------
    spec : RecordSpec
        A validated record descriptor (CSV/TDMS/HDF5).

    Returns
    -------
    InstrumentRecord
        The record in amperes with its optional SI reference channel.
    """
    reader = RECORD_REGISTRY.create(spec.format)
    return reader.load(spec)
