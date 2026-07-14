"""Measurements -> twin-configuration tests (task S-13; doc 16 §2a/§3a).

Proves the ingest path on the real ``proto_poc`` base: measured scalars,
ring-down Q, and the OSA spectrum land as subsystem overrides; the produced
composition *resolves* through the standard guards; and the physics-discipline
rules are enforced in code:

* anti-double-count (R-57(v)): a measured RIN replaces the derived ASE floor,
  and next to a measured spectrum the nameplate preset RIN is force-dropped so
  the floor derives from the *measured table*;
* single source of truth (R-57(a)): a scalar linewidth measurement next to a
  measured spectrum is an error, and the override kills any preset scalar;
* override semantics (M-02/M-18): a measured ``q_total`` overrides the
  computable ``Q(L)`` model, whose prediction is recorded for the cross-check;
* the GUM gate (17 §4): parameters without ``u`` stay informational;
* no config slot: ``f1_hz`` (P20-1) and ``dop`` (R-58) are reported, never
  written;
* provenance is a separate artifact next to the composition, never inside it.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from optivibe.cli.main import main as cli_main
from optivibe.core.config.loader import load_constants
from optivibe.core.config.presets import PresetStore, load_system_file
from optivibe.io.characterization import (
    CharacterizationResult,
    MeasuredParameter,
    Provenance,
    load_characterization,
)
from optivibe.io.ingest import apply_measurements, save_provenance


def _sidecar(directory: Path, name: str, body: dict) -> Path:
    path = directory / name
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def _scalar(directory: Path, name: str, parameter: str, value: float, unit: str, u: float) -> Path:
    return _sidecar(
        directory,
        name,
        {
            "kind": "scalar",
            "instrument": "bench",
            "timestamp": "2026-07-10T09:00:00Z",
            "parameter": parameter,
            "value": value,
            "unit": unit,
            "u": u,
        },
    )


def _ringdown(directory: Path, *, q: float = 1661.0, f1: float = 6250.0) -> Path:
    fs = 200_000.0
    t = np.arange(0.0, 0.2, 1.0 / fs)
    y = np.exp(-math.pi * f1 / q * t) * np.cos(2.0 * math.pi * f1 * t)
    np.savetxt(directory / "rd.csv", np.column_stack([t, y]), delimiter=",")
    return _sidecar(
        directory,
        "rd.yaml",
        {
            "kind": "ringdown",
            "instrument": "scope",
            "timestamp": "2026-07-10T09:00:00Z",
            "data_file": "rd.csv",
            "columns": {"x": {"name": "t", "unit": "s"}, "y": {"name": "a", "unit": "arb"}},
        },
    )


def _spectrum(directory: Path) -> Path:
    lam = np.linspace(1450.0, 1650.0, 401)
    sigma = 60.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    psd = np.exp(-0.5 * ((lam - 1550.0) / sigma) ** 2)
    np.savetxt(directory / "sp.csv", np.column_stack([lam, psd]), delimiter=",")
    return _sidecar(
        directory,
        "sp.yaml",
        {
            "kind": "spectrum",
            "instrument": "OSA",
            "timestamp": "2026-07-10T09:00:00Z",
            "data_file": "sp.csv",
            "columns": {"x": {"name": "l", "unit": "nm"}, "y": {"name": "S", "unit": "arb"}},
            "uncertainties": {"wavelength_m": 0.5e-9},
        },
    )


@pytest.fixture()
def proto(config_dir: Path):
    """The proto_poc base composition + store + constants."""
    system = load_system_file(config_dir / "variants" / "proto_poc.yaml")
    store = PresetStore(config_dir)
    constants = load_constants(config_dir / "constants.yaml")
    return system, store, constants


def test_full_ingest_on_proto_poc(tmp_path: Path, proto) -> None:
    """Scalars + ring-down + spectrum land as overrides and the twin resolves."""
    system, store, constants = proto
    results = [
        load_characterization(_scalar(tmp_path, "L.yaml", "length_m", 4.0, "mm", 0.02)),
        load_characterization(_scalar(tmp_path, "gap.yaml", "gap_m", 31.0, "um", 1.0)),
        load_characterization(_ringdown(tmp_path)),
        load_characterization(_spectrum(tmp_path)),
    ]
    report = apply_measurements(system, results, constants=constants)

    source = report.system.source.overrides
    assert source["lineshape"] == "measured"
    assert len(source["spectrum_wavelength_m"]) == 401
    assert source["linewidth_fwhm_m"] is None  # R-57(a)
    assert source["rin_db_hz"] is None  # nameplate dropped (R-57(v) chain)
    assert report.system.cantilever.overrides["length_m"] == pytest.approx(4.0e-3)
    assert report.system.reflector.overrides["gap_m"] == pytest.approx(31.0e-6)
    assert report.system.q_total == pytest.approx(1661.0, rel=0.02)
    # The Q(L) model is recorded next to the measured Q (M-18 cross-check).
    assert report.q_total_model is not None
    assert report.q_total_model == pytest.approx(report.system.q_total, rel=0.05)

    # f1 is reported, never written (P20-1).
    names = {p.name for p in report.informational}
    assert "f1_hz" in names

    # The measured twin resolves through the standard guards (wash-out etc.)
    variant = report.system.resolve(store, constants=constants)
    assert variant.q_total == pytest.approx(1661.0, rel=0.02)
    # And the resolved RIN is the ASE floor of the *measured table*, not the
    # -126 dB/Hz nameplate of the sld preset (anti-double-count chain).
    assert variant.source.rin_db_hz == pytest.approx(-127.5, abs=0.2)


def test_measured_rin_trace_wins_over_table_floor(tmp_path: Path, proto) -> None:
    """A measured RIN trace replaces the floor derived from the spectrum (R-57(v))."""
    system, store, constants = proto
    freq = np.linspace(100.0, 20_000.0, 100)
    np.savetxt(
        tmp_path / "rin.csv", np.column_stack([freq, np.full_like(freq, -121.0)]), delimiter=","
    )
    rin_path = _sidecar(
        tmp_path,
        "rin.yaml",
        {
            "kind": "rin_psd",
            "instrument": "PD+ESA",
            "timestamp": "t",
            "data_file": "rin.csv",
            "columns": {"x": {"name": "f", "unit": "hz"}, "y": {"name": "rin", "unit": "db/hz"}},
            "band": {"f_min_hz": 1_000.0, "f_max_hz": 10_000.0},
            "uncertainties": {"rin_db_hz": 0.5},
        },
    )
    results = [load_characterization(p) for p in (_spectrum(tmp_path), rin_path)]
    report = apply_measurements(system, results, constants=constants)
    assert report.system.source.overrides["rin_db_hz"] == pytest.approx(-121.0)
    variant = report.system.resolve(store, constants=constants)
    assert variant.source.rin_db_hz == pytest.approx(-121.0)


def test_duplicate_parameter_is_a_conflict(tmp_path: Path, proto) -> None:
    """Two measurements of one config field fail loudly (operator must pick)."""
    system, _, constants = proto
    results = [
        load_characterization(_scalar(tmp_path, "a.yaml", "length_m", 4.0, "mm", 0.02)),
        load_characterization(_scalar(tmp_path, "b.yaml", "length_m", 4.1, "mm", 0.02)),
    ]
    with pytest.raises(ValueError, match="measured twice"):
        apply_measurements(system, results, constants=constants)


def test_scalar_linewidth_next_to_spectrum_rejected(tmp_path: Path, proto) -> None:
    """R-57(a): a scalar linewidth cannot coexist with the measured table."""
    system, _, constants = proto
    results = [
        load_characterization(_spectrum(tmp_path)),
        load_characterization(_scalar(tmp_path, "dl.yaml", "linewidth_fwhm_m", 60.0, "nm", 1.0)),
    ]
    with pytest.raises(ValueError, match="single source of truth"):
        apply_measurements(system, results, constants=constants)


def test_two_spectra_rejected(tmp_path: Path, proto) -> None:
    """Two spectrum artifacts are an explicit conflict (R-57)."""
    system, _, constants = proto
    d1 = tmp_path / "one"
    d2 = tmp_path / "two"
    d1.mkdir()
    d2.mkdir()
    results = [load_characterization(_spectrum(d1)), load_characterization(_spectrum(d2))]
    with pytest.raises(ValueError, match="two measured spectra"):
        apply_measurements(system, results, constants=constants)


def test_parameter_without_u_stays_informational(proto) -> None:
    """The GUM gate: u = None never reaches the configuration (17 §4)."""
    system, _, constants = proto
    prov = Provenance(kind="scalar", sidecar="x.yaml", instrument="i", timestamp="t")
    result = CharacterizationResult(
        params=(MeasuredParameter(name="gap_m", value=30.0e-6, u=None, method="guess"),),
        provenance=prov,
    )
    report = apply_measurements(system, [result], constants=constants)
    assert not report.changes
    assert report.informational[0].name == "gap_m"
    assert "reflector.gap_m" in report.model_defaults


def test_dop_has_no_config_slot(tmp_path: Path, proto) -> None:
    """R-58: DOP is measured and reported but never enters the model config."""
    system, _, constants = proto
    results = [load_characterization(_scalar(tmp_path, "dop.yaml", "dop", 0.12, "-", 0.02))]
    report = apply_measurements(system, results, constants=constants)
    assert not report.changes
    assert report.informational[0].name == "dop"


def test_provenance_is_a_separate_artifact(tmp_path: Path, proto) -> None:
    """Provenance lands next to the composition, never inside it (16 §2a)."""
    system, _, constants = proto
    results = [load_characterization(_scalar(tmp_path, "L.yaml", "length_m", 4.0, "mm", 0.02))]
    report = apply_measurements(system, results, constants=constants)
    out = tmp_path / "twin.yaml"
    out.write_text("placeholder", encoding="utf-8")
    prov_path = save_provenance(report, out)
    assert prov_path == tmp_path / "twin.provenance.yaml"
    body = yaml.safe_load(prov_path.read_text())
    assert body["base"] == "proto_poc"
    assert body["measured_fields"][0]["target"] == "cantilever.length_m"
    assert body["measured_fields"][0]["sha256"] is None  # scalar: sidecar-only
    assert "model_or_default_fields" in body


def test_cli_ingest_dry_run_and_write(tmp_path: Path) -> None:
    """The ``optivibe ingest`` command round-trips: dry-run, then write+resolve."""
    _scalar(tmp_path, "L.yaml", "length_m", 4.0, "mm", 0.02)
    rd = _ringdown(tmp_path)
    out = tmp_path / "twin.yaml"
    rc = cli_main(["ingest", str(tmp_path / "L.yaml"), str(rd), "--base", "proto_poc", "--dry-run"])
    assert rc == 0
    assert not out.exists()
    rc = cli_main(
        ["ingest", str(tmp_path / "L.yaml"), str(rd), "--base", "proto_poc", "--out", str(out)]
    )
    assert rc == 0
    assert out.exists() and out.with_suffix(".provenance.yaml").exists()
    written = load_system_file(out)
    assert written.name == "twin"
    assert written.q_total == pytest.approx(1661.0, rel=0.02)
    prov = yaml.safe_load(out.with_suffix(".provenance.yaml").read_text())
    assert prov["composition"] == "twin"


def test_cli_ingest_bad_artifact_fails(tmp_path: Path) -> None:
    """A malformed sidecar makes the command fail with a non-zero exit code."""
    bad = _sidecar(tmp_path, "bad.yaml", {"kind": "scalar", "instrument": "x", "timestamp": "t"})
    rc = cli_main(["ingest", str(bad), "--base", "proto_poc", "--dry-run"])
    assert rc == 2
