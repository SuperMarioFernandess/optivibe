"""Tests for the instrument-format loaders (seam SW-08, S8).

Each format (TDMS / UFF / MAT / HDF5) gets a fixture that *writes* a known
record with its native backend, then a round-trip test that loads it back
through :data:`LOADER_REGISTRY` and checks the SI conversion, sampling rate,
axis placement, and metadata. Tests that need an optional backend skip cleanly
when the ``io-formats`` extra is absent (``pytest.importorskip``); the helper
and error-path tests need no backend and always run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.io import savemat

from optivibe.core.config.models import (
    Hdf5Spec,
    MatSpec,
    SineSpec,
    TdmsSpec,
    UffSpec,
)
from optivibe.core.units import G0_M_S2
from optivibe.io.loaders import (
    LOADER_REGISTRY,
    Hdf5Loader,
    MatLoader,
    TdmsLoader,
    UffLoader,
    _canonical_unit,
    _pick_channel,
    _resolve_units,
    _scalar_rate,
    _select_column,
    _to_si_acceleration,
)

FS = 2000.0
N = 512


def _signal() -> np.ndarray:
    """A deterministic test record (amplitude ~0.5 in its native unit)."""
    t = np.arange(N) / FS
    return (0.5 * np.sin(2.0 * np.pi * 60.0 * t)).astype(np.float64)


# --------------------------------------------------------------------------- #
# Format writers (fixtures)
# --------------------------------------------------------------------------- #
def _write_tdms(path: Path, data: np.ndarray, *, unit: str | None, fs: float = FS) -> None:
    nptdms = pytest.importorskip("nptdms")
    props: dict[str, Any] = {"wf_increment": 1.0 / fs}
    if unit is not None:
        props["unit_string"] = unit
    channel = nptdms.ChannelObject("accel", "x", data, properties=props)
    with nptdms.TdmsWriter(str(path)) as writer:
        writer.write_segment([channel])


def _write_uff(path: Path, data: np.ndarray, *, unit: str, fs: float = FS) -> None:
    pyuff = pytest.importorskip("pyuff")
    dt = 1.0 / fs
    t = np.arange(data.size) * dt
    dataset = {
        "type": 58,
        "func_type": 1,
        "ver_num": 0,
        "load_case_id": 0,
        "rsp_ent_name": "accel",
        "rsp_node": 1,
        "rsp_dir": 1,
        "ref_ent_name": "ref",
        "ref_node": 1,
        "ref_dir": 1,
        "ord_data_type": 4,
        "num_pts": data.size,
        "abscissa_spacing": 1,
        "abscissa_min": 0.0,
        "abscissa_inc": dt,
        "z_axis_value": 0.0,
        "data": data.copy(),
        "x": t,
        "abscissa_spec_data_type": 17,
        "abscissa_len_unit_exp": 0,
        "abscissa_force_unit_exp": 0,
        "abscissa_temp_unit_exp": 0,
        "abscissa_axis_units_lab": "s",
        "ordinate_spec_data_type": 12,
        "ordinate_len_unit_exp": 0,
        "ordinate_force_unit_exp": 0,
        "ordinate_temp_unit_exp": 0,
        "ordinate_axis_units_lab": unit,
        "orddenom_spec_data_type": 0,
        "orddenom_len_unit_exp": 0,
        "orddenom_force_unit_exp": 0,
        "orddenom_temp_unit_exp": 0,
        "orddenom_axis_units_lab": "NONE",
        "z_axis_spec_data_type": 0,
        "z_axis_len_unit_exp": 0,
        "z_axis_force_unit_exp": 0,
        "z_axis_temp_unit_exp": 0,
        "z_axis_units_lab": "NONE",
    }
    pyuff.UFF(str(path)).write_sets([dataset], mode="overwrite")


def _write_hdf5(
    path: Path,
    data: np.ndarray,
    *,
    dataset: str = "/accel/x",
    fs_attr: str | None = "fs_hz",
    units_attr: str | None = "units",
    unit: str = "g",
    fs: float = FS,
) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(str(path), "w") as handle:
        dset = handle.create_dataset(dataset, data=data)
        if fs_attr is not None:
            dset.attrs[fs_attr] = fs
        if units_attr is not None:
            dset.attrs[units_attr] = unit


# --------------------------------------------------------------------------- #
# Round-trips: SI conversion, fs, axis, meta
# --------------------------------------------------------------------------- #
def test_tdms_roundtrip_g_to_si(tmp_path: Path) -> None:
    sig = _signal()
    path = tmp_path / "rec.tdms"
    _write_tdms(path, sig, unit=None)  # no label; units default m/s^2
    spec = TdmsSpec(path=str(path), axis="y")
    exc = LOADER_REGISTRY.create("tdms").load(spec, seed=7)
    assert exc.fs == pytest.approx(FS)
    assert exc.seed == 7
    np.testing.assert_allclose(exc.a_y, sig)  # m/s^2 -> unchanged
    np.testing.assert_array_equal(exc.a_x, 0.0)
    np.testing.assert_array_equal(exc.a_z, 0.0)
    assert exc.meta["generator"] == "tdms"


def test_tdms_units_auto_reads_label(tmp_path: Path) -> None:
    sig = _signal()
    path = tmp_path / "rec.tdms"
    _write_tdms(path, sig, unit="g")
    spec = TdmsSpec(path=str(path), units="auto", axis="x")
    exc = LOADER_REGISTRY.create("tdms").load(spec)
    np.testing.assert_allclose(exc.a_x, sig * G0_M_S2)  # g -> m/s^2


def test_tdms_explicit_unit_conflict_is_loud(tmp_path: Path) -> None:
    path = tmp_path / "rec.tdms"
    _write_tdms(path, _signal(), unit="g")
    spec = TdmsSpec(path=str(path), units="m/s^2")  # file says g -> conflict
    with pytest.raises(ValueError, match="refusing to guess"):
        LOADER_REGISTRY.create("tdms").load(spec)


def test_tdms_fs_override_and_channel_by_name(tmp_path: Path) -> None:
    path = tmp_path / "rec.tdms"
    _write_tdms(path, _signal(), unit="g", fs=1000.0)
    spec = TdmsSpec(path=str(path), channel="x", fs_hz=4000.0, units="g")
    exc = LOADER_REGISTRY.create("tdms").load(spec)
    assert exc.fs == pytest.approx(4000.0)


def test_uff_roundtrip_and_auto(tmp_path: Path) -> None:
    sig = _signal()
    path = tmp_path / "rec.uff"
    _write_uff(path, sig, unit="g")
    spec = UffSpec(path=str(path), units="auto", axis="z")
    exc = LOADER_REGISTRY.create("uff").load(spec)
    assert exc.fs == pytest.approx(FS)
    np.testing.assert_allclose(exc.a_z, sig * G0_M_S2, rtol=1e-6)


def test_mat_roundtrip_and_fs_key(tmp_path: Path) -> None:
    sig = _signal()
    path = tmp_path / "rec.mat"
    savemat(str(path), {"accel": sig, "fs": np.array([[FS]])})
    spec = MatSpec(path=str(path), data_key="accel", fs_key="fs", units="g")
    exc = LOADER_REGISTRY.create("mat").load(spec)
    assert exc.fs == pytest.approx(FS)
    np.testing.assert_allclose(exc.a_x, sig * G0_M_S2)


def test_mat_two_dimensional_column_select(tmp_path: Path) -> None:
    block = np.stack([_signal(), 2.0 * _signal(), 3.0 * _signal()], axis=1)  # (N, 3)
    path = tmp_path / "multi.mat"
    savemat(str(path), {"accel": block})
    spec = MatSpec(path=str(path), data_key="accel", column=1, fs_hz=FS, units="m/s^2")
    exc = LOADER_REGISTRY.create("mat").load(spec)
    np.testing.assert_allclose(exc.a_x, 2.0 * _signal())


def test_mat_auto_units_rejected(tmp_path: Path) -> None:
    path = tmp_path / "rec.mat"
    savemat(str(path), {"accel": _signal()})
    spec = MatSpec(path=str(path), data_key="accel", fs_hz=FS, units="auto")
    with pytest.raises(ValueError, match="no unit label"):
        LOADER_REGISTRY.create("mat").load(spec)


def test_hdf5_roundtrip_auto_and_attrs(tmp_path: Path) -> None:
    sig = _signal()
    path = tmp_path / "rec.h5"
    _write_hdf5(path, sig, unit="g")
    spec = Hdf5Spec(
        path=str(path), dataset="/accel/x", fs_attr="fs_hz", units_attr="units", units="auto"
    )
    exc = LOADER_REGISTRY.create("hdf5").load(spec)
    assert exc.fs == pytest.approx(FS)
    np.testing.assert_allclose(exc.a_x, sig * G0_M_S2)


def test_hdf5_explicit_fs_and_resample(tmp_path: Path) -> None:
    sig = _signal()
    path = tmp_path / "rec.h5"
    _write_hdf5(path, sig, units_attr=None, unit="m/s^2")
    spec = Hdf5Spec(path=str(path), dataset="/accel/x", fs_hz=FS, units="m/s^2", resample_hz=1000.0)
    exc = LOADER_REGISTRY.create("hdf5").load(spec)
    assert exc.fs == pytest.approx(1000.0)
    assert exc.a_x.size == pytest.approx(N // 2, abs=2)


# --------------------------------------------------------------------------- #
# Error paths: wrong spec type, missing keys
# --------------------------------------------------------------------------- #
def test_loaders_reject_foreign_spec() -> None:
    sine = SineSpec(fs_hz=FS, duration_s=0.1, frequency_hz=10.0, amplitude_g=1.0)
    for loader in (TdmsLoader(), UffLoader(), MatLoader(), Hdf5Loader()):
        with pytest.raises(TypeError, match="expects"):
            loader.load(sine)


def test_mat_missing_variable(tmp_path: Path) -> None:
    path = tmp_path / "rec.mat"
    savemat(str(path), {"accel": _signal()})
    spec = MatSpec(path=str(path), data_key="nope", fs_hz=FS, units="g")
    with pytest.raises(ValueError, match="not found"):
        LOADER_REGISTRY.create("mat").load(spec)


def test_hdf5_missing_dataset(tmp_path: Path) -> None:
    path = tmp_path / "rec.h5"
    _write_hdf5(path, _signal())
    spec = Hdf5Spec(path=str(path), dataset="/missing", fs_hz=FS, units="g")
    with pytest.raises(ValueError, match="not found"):
        LOADER_REGISTRY.create("hdf5").load(spec)


def test_mat_v73_points_to_hdf5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "rec.mat"
    savemat(str(path), {"accel": _signal()})

    def _raise(*_a: object, **_k: object) -> None:
        raise NotImplementedError("v7.3")

    monkeypatch.setattr("scipy.io.loadmat", _raise)
    spec = MatSpec(path=str(path), data_key="accel", fs_hz=FS, units="g")
    with pytest.raises(ValueError, match="hdf5"):
        LOADER_REGISTRY.create("mat").load(spec)


# --------------------------------------------------------------------------- #
# Missing-extra path: friendly install hint (simulate absent backend)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("module", "kind", "spec_factory"),
    [
        ("nptdms", "tdms", lambda p: TdmsSpec(path=p, units="g")),
        ("pyuff", "uff", lambda p: UffSpec(path=p, units="g")),
        ("h5py", "hdf5", lambda p: Hdf5Spec(path=p, dataset="/x", fs_hz=FS, units="g")),
    ],
)
def test_missing_backend_hint(
    monkeypatch: pytest.MonkeyPatch,
    module: str,
    kind: str,
    spec_factory: Any,
) -> None:
    monkeypatch.setitem(sys.modules, module, None)  # makes `import module` raise ImportError
    with pytest.raises(ImportError, match="io-formats"):
        LOADER_REGISTRY.create(kind).load(spec_factory("does-not-matter"))


# --------------------------------------------------------------------------- #
# Helper unit tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("g", "g"),
        ("G", "g"),
        (b"g", "g"),
        ("m/s^2", "m/s^2"),
        ("M/S^2", "m/s^2"),
        ("mV", "V"),
        ("Volts", "V"),
        ("", None),
        (None, None),
        ("furlongs", None),
    ],
)
def test_canonical_unit(label: object, expected: str | None) -> None:
    assert _canonical_unit(label) == expected


def test_resolve_units_auto_without_label_errors() -> None:
    with pytest.raises(ValueError, match="no recognized unit label"):
        _resolve_units("auto", None, source="x")


def test_resolve_units_unknown_file_label_does_not_block_explicit() -> None:
    # Unrecognized file label must not override an explicit spec unit.
    assert _resolve_units("m/s^2", "weird", source="x") == "m/s^2"


def test_to_si_acceleration_units() -> None:
    sig = np.array([1.0, -2.0, 3.0])
    np.testing.assert_allclose(_to_si_acceleration(sig, "m/s^2", None, "mV/g"), sig)
    np.testing.assert_allclose(_to_si_acceleration(sig, "g", None, "mV/g"), sig * G0_M_S2)
    # 100 mV/g sensitivity, 1 V in -> 10 g -> 10 * g0 m/s^2
    out = _to_si_acceleration(np.array([1.0]), "V", 100.0, "mV/g")
    np.testing.assert_allclose(out, np.array([10.0 * G0_M_S2]))
    # 1 V/(m/s^2), 2 V in -> 2 m/s^2
    out2 = _to_si_acceleration(np.array([2.0]), "V", 1.0, "V/(m/s^2)")
    np.testing.assert_allclose(out2, np.array([2.0]))


def test_to_si_acceleration_voltage_needs_sensitivity() -> None:
    with pytest.raises(ValueError, match="sensitivity"):
        _to_si_acceleration(np.array([1.0]), "V", None, "mV/g")


def test_select_column_orientations() -> None:
    col = np.arange(10.0)
    np.testing.assert_allclose(_select_column(col, None, "x"), col)  # 1-D
    wide = np.stack([col, col + 100.0])  # (2, 10) -> time-major after transpose
    np.testing.assert_allclose(_select_column(wide, 1, "x"), col + 100.0)
    tall = np.stack([col, col + 100.0], axis=1)  # (10, 2)
    np.testing.assert_allclose(_select_column(tall, 0, "x"), col)


def test_select_column_errors() -> None:
    with pytest.raises(ValueError, match="1-D but column"):
        _select_column(np.arange(5.0), 2, "x")
    with pytest.raises(ValueError, match="out of range"):
        _select_column(np.zeros((10, 2)), 5, "x")
    with pytest.raises(ValueError, match="1-D or 2-D"):
        _select_column(np.zeros((2, 2, 2)), 0, "x")


def test_pick_channel() -> None:
    assert _pick_channel(["a", "b", "c"], 1, "x") == 1
    assert _pick_channel(["a", "b", "c"], "c", "x") == 2
    with pytest.raises(ValueError, match="out of range"):
        _pick_channel(["a"], 3, "x")
    with pytest.raises(ValueError, match="not found"):
        _pick_channel(["a"], "z", "x")


def test_scalar_rate_rejects_nonpositive() -> None:
    assert _scalar_rate(np.array([[100.0]]), "x") == pytest.approx(100.0)
    with pytest.raises(ValueError, match="positive and finite"):
        _scalar_rate(np.array([0.0]), "x")
