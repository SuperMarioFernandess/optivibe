"""Photocurrent-record reader tests (role S-02; io/records, doc 20 §5).

Reader-level checks with synthetic arrays (no physics needed): unit conversion
V -> A at the boundary, mandatory metadata (timestamp, R_f), reference-channel
conversion and conflicts, format round-trips. The analyzer-level round-trips
against forward runs live in ``test_analysis_instrument.py`` (18 G5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from optivibe.io.records import (
    RECORD_REGISTRY,
    CsvRecordSpec,
    Hdf5RecordSpec,
    InstrumentRecord,
    RecordSpec,
    TdmsRecordSpec,
    read_record,
)

FS = 2000.0
N = 512
TS = "2026-07-04T00:00:00"

_SPEC = TypeAdapter(RecordSpec)


def _current() -> np.ndarray:
    """A deterministic photocurrent-like record (pedestal + tone), A."""
    t = np.arange(N) / FS
    return (1.0e-3 + 2.0e-7 * np.sin(2.0 * np.pi * 60.0 * t)).astype(np.float64)


def _accel() -> np.ndarray:
    """A deterministic reference acceleration, m/s^2."""
    t = np.arange(N) / FS
    return (3.0 * np.sin(2.0 * np.pi * 60.0 * t)).astype(np.float64)


def _write_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    header = ",".join(columns)
    table = np.column_stack(list(columns.values()))
    np.savetxt(path, table, delimiter=",", header=header, comments="", fmt="%.17e")


# --------------------------------------------------------------------------- #
# Descriptor validation (mandatory metadata, 20 §5)
# --------------------------------------------------------------------------- #
def test_timestamp_is_mandatory() -> None:
    with pytest.raises(ValidationError):
        CsvRecordSpec(path="x.csv", units="A", fs_hz=FS, timestamp="")
    with pytest.raises(ValidationError):
        _SPEC.validate_python({"format": "csv", "path": "x.csv", "units": "A", "fs_hz": FS})


def test_volt_record_requires_rf() -> None:
    with pytest.raises(ValidationError, match="r_f_ohm"):
        CsvRecordSpec(path="x.csv", units="V", fs_hz=FS, timestamp=TS)


def test_csv_needs_fs_or_time_column() -> None:
    with pytest.raises(ValidationError, match="fs_hz or time_column"):
        CsvRecordSpec(path="x.csv", units="A", timestamp=TS)


def test_hdf5_needs_fs_or_attr() -> None:
    with pytest.raises(ValidationError, match="fs_hz or fs_attr"):
        Hdf5RecordSpec(path="x.h5", dataset="i", units="A", timestamp=TS)


# --------------------------------------------------------------------------- #
# CSV reader: units, R_f conversion, dc estimate, reference channel
# --------------------------------------------------------------------------- #
def test_csv_record_ampere_roundtrip(tmp_path: Path) -> None:
    """Samples come back in amperes with the dc level = the record mean."""
    path = tmp_path / "rec.csv"
    current = _current()
    t = np.arange(N) / FS
    _write_csv(path, {"time_s": t, "i_pd_a": current})
    spec = _SPEC.validate_python(
        {
            "format": "csv",
            "path": str(path),
            "units": "A",
            "column": "i_pd_a",
            "time_column": "time_s",
            "timestamp": TS,
        }
    )
    record = read_record(spec)
    assert record.detector.units == "A"
    assert record.detector.fs == pytest.approx(FS, rel=1e-9)
    np.testing.assert_allclose(record.detector.samples, current, rtol=0, atol=1e-24)
    assert record.detector.dc_level == pytest.approx(float(np.mean(current)))
    assert record.detector.noise["model"] == "instrument_record"
    assert record.meta["timestamp"] == TS


@pytest.mark.golden
def test_volt_record_matches_ampere_record(tmp_path: Path) -> None:
    """V = I * R_f divided by r_f_ohm at the boundary reproduces the amperes."""
    rf = 2.0e3
    current = _current()
    p_a = tmp_path / "a.csv"
    p_v = tmp_path / "v.csv"
    _write_csv(p_a, {"i": current})
    _write_csv(p_v, {"u": current * rf})
    rec_a = read_record(CsvRecordSpec(path=str(p_a), units="A", column=0, fs_hz=FS, timestamp=TS))
    rec_v = read_record(
        CsvRecordSpec(path=str(p_v), units="V", r_f_ohm=rf, column=0, fs_hz=FS, timestamp=TS)
    )
    np.testing.assert_allclose(rec_v.detector.samples, rec_a.detector.samples, rtol=1e-12)


def test_csv_reference_channel_units_g(tmp_path: Path) -> None:
    """The reference channel converts g -> m/s^2 with the loader unit rules."""
    path = tmp_path / "rec.csv"
    accel = _accel()
    _write_csv(path, {"i": _current(), "ref_g": accel / 9.80665})
    spec = CsvRecordSpec(
        path=str(path),
        units="A",
        column="i",
        fs_hz=FS,
        timestamp=TS,
        reference={"channel": "ref_g", "units": "g"},  # type: ignore[arg-type]
    )
    record = read_record(spec)
    assert record.reference_accel is not None
    np.testing.assert_allclose(record.reference_accel, accel, rtol=1e-9)


def test_csv_reference_auto_units_is_loud(tmp_path: Path) -> None:
    """CSV carries no unit labels: 'auto' reference units refuse to guess (10 §7)."""
    path = tmp_path / "rec.csv"
    _write_csv(path, {"i": _current(), "ref": _accel()})
    spec = CsvRecordSpec(
        path=str(path),
        units="A",
        column="i",
        fs_hz=FS,
        timestamp=TS,
        reference={"channel": "ref"},  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="units"):
        read_record(spec)


def test_reference_voltage_needs_sensitivity(tmp_path: Path) -> None:
    """A volt reference channel without a sensitivity fails loudly (20 §5)."""
    path = tmp_path / "rec.csv"
    _write_csv(path, {"i": _current(), "ref_v": _accel() * 0.01})
    spec = CsvRecordSpec(
        path=str(path),
        units="A",
        column="i",
        fs_hz=FS,
        timestamp=TS,
        reference={"channel": "ref_v", "units": "V"},  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="sensitivity"):
        read_record(spec)


def test_resample_applies_to_both_channels(tmp_path: Path) -> None:
    path = tmp_path / "rec.csv"
    _write_csv(path, {"i": _current(), "ref": _accel()})
    spec = CsvRecordSpec(
        path=str(path),
        units="A",
        column="i",
        fs_hz=FS,
        resample_hz=FS / 2.0,
        timestamp=TS,
        reference={"channel": "ref", "units": "m/s^2"},  # type: ignore[arg-type]
    )
    record = read_record(spec)
    assert record.detector.fs == FS / 2.0
    assert record.reference_accel is not None
    assert record.reference_accel.size == record.detector.n_samples == N // 2


def test_length_mismatch_is_loud() -> None:
    """The InstrumentRecord contract validates channel lengths once (09 §5)."""
    from optivibe.core.types import DetectorOutput

    det = DetectorOutput(samples=np.ones(8), fs=FS, dc_level=1.0, units="A")
    with pytest.raises(ValueError, match="length"):
        InstrumentRecord(detector=det, reference_accel=np.ones(4))


# --------------------------------------------------------------------------- #
# HDF5 / TDMS readers (optional backends)
# --------------------------------------------------------------------------- #
def test_hdf5_record_with_reference_dataset(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "rec.h5"
    current = _current()
    accel = _accel()
    with h5py.File(path, "w") as handle:
        ds = handle.create_dataset("i_pd", data=current)
        ds.attrs["fs"] = FS
        ref = handle.create_dataset("a_ref", data=accel)
        ref.attrs["units"] = "m/s^2"
    spec = Hdf5RecordSpec(
        path=str(path),
        dataset="i_pd",
        units="A",
        fs_attr="fs",
        timestamp=TS,
        reference={"channel": "a_ref"},  # type: ignore[arg-type]  # units: auto from attr
    )
    record = read_record(spec)
    assert record.detector.fs == pytest.approx(FS)
    np.testing.assert_allclose(record.detector.samples, current, rtol=1e-12)
    assert record.reference_accel is not None
    np.testing.assert_allclose(record.reference_accel, accel, rtol=1e-12)


def test_hdf5_reference_column_of_signal_dataset(tmp_path: Path) -> None:
    """An int reference selector picks a column of the signal dataset (explicit units)."""
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "rec.h5"
    table = np.column_stack([_current(), _accel()])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("capture", data=table)
    spec = Hdf5RecordSpec(
        path=str(path),
        dataset="capture",
        column=0,
        units="A",
        fs_hz=FS,
        timestamp=TS,
        reference={"channel": 1, "units": "m/s^2"},  # type: ignore[arg-type]
    )
    record = read_record(spec)
    np.testing.assert_allclose(record.detector.samples, _current(), rtol=1e-12)
    assert record.reference_accel is not None
    np.testing.assert_allclose(record.reference_accel, _accel(), rtol=1e-12)


def _write_tdms(path: Path, channels: dict[str, tuple[np.ndarray, str | None]]) -> None:
    nptdms = pytest.importorskip("nptdms")
    objs = []
    for name, (data, unit) in channels.items():
        props: dict[str, Any] = {"wf_increment": 1.0 / FS}
        if unit is not None:
            props["unit_string"] = unit
        objs.append(nptdms.ChannelObject("capture", name, data, properties=props))
    with nptdms.TdmsWriter(str(path)) as writer:
        writer.write_segment(objs)


def test_tdms_record_fs_from_increment_and_reference_label(tmp_path: Path) -> None:
    path = tmp_path / "rec.tdms"
    current = _current()
    accel = _accel()
    _write_tdms(path, {"i_pd": (current, None), "a_ref": (accel, "m/s^2")})
    spec = TdmsRecordSpec(
        path=str(path),
        units="A",
        channel="i_pd",
        timestamp=TS,
        reference={"channel": "a_ref"},  # type: ignore[arg-type]  # units: auto from label
    )
    record = read_record(spec)
    assert record.detector.fs == pytest.approx(FS)
    np.testing.assert_allclose(record.detector.samples, current, rtol=1e-12)
    assert record.reference_accel is not None
    np.testing.assert_allclose(record.reference_accel, accel, rtol=1e-9)


def test_registry_holds_the_three_priority_formats() -> None:
    """The 20 §5 priority formats are registered readers (09 §6 seam)."""
    for key in ("csv", "tdms", "hdf5"):
        assert RECORD_REGISTRY.get(key) is not None
