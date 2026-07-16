"""Characterization-input-layer tests (task S-13; doc 16 §2a; cards M-15..M-18).

Reader-level checks with synthetic artifacts, pinned to the *base formulas*
per 18 §5(g) rather than to run outputs:

* spectrum (M-15): Gaussian trace -> centroid, numeric FWHM, and the
  noise-equivalent width ``dnu_eff = 1.5053 dnu_FWHM`` (the R-56 Gaussian form
  factor) with the ASE floor ``RIN = 10 log10(2 / dnu_eff)`` -- both evaluated
  through the existing :mod:`optivibe.optics.source` quadratures (R-57);
* ringdown (M-18): ``y = exp(-pi f1 t / Q) cos(2 pi f1 t)`` -> ``Q = pi f1 /
  sigma`` recovered to < 2 %, ``f1`` to within the FFT bin; an undamped tone is
  rejected as "no significant decay";
* profile (M-17): a synthetic circular arc -> Kasa fit recovers ``R_c`` to
  < 1 % with a covariance-propagated ``u``; collinear/non-circular contours are
  rejected;
* rin_psd (M-16 reduce): a flat trace reduces to its level over the declared
  band;
* the unit discipline (declared, never guessed) including the nm-vs-m trap of
  R-57(b), and the GUM gate (17 §4): missing ``u`` on config-bound parameters
  fails loudly.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from optivibe.io.characterization import (
    CHARACTERIZATION_REGISTRY,
    MeasuredParameter,
    load_characterization,
    resolve_sidecar_path,
)

C0 = 299_792_458.0
_GAUSS_SIGMA = 60.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))  # nm, FWHM = 60 nm


def _sidecar(directory: Path, name: str, body: dict) -> Path:
    """Write a sidecar YAML and return its path."""
    path = directory / name
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def _gaussian_spectrum(directory: Path, *, unit: str = "nm") -> Path:
    """A Gaussian OSA trace: lambda0 = 1550 nm, dlam FWHM = 60 nm."""
    lam_nm = np.linspace(1450.0, 1650.0, 401)
    psd = np.exp(-0.5 * ((lam_nm - 1550.0) / _GAUSS_SIGMA) ** 2)
    np.savetxt(directory / "spectrum.csv", np.column_stack([lam_nm, psd]), delimiter=",")
    return _sidecar(
        directory,
        "spectrum.yaml",
        {
            "kind": "spectrum",
            "instrument": "OSA",
            "timestamp": "2026-07-10T09:00:00Z",
            "data_file": "spectrum.csv",
            "columns": {"x": {"name": "lambda", "unit": unit}, "y": {"name": "S", "unit": "arb"}},
            "uncertainties": {"wavelength_m": 0.5e-9},
        },
    )


def _ringdown(directory: Path, *, q: float = 1661.0, f1: float = 6250.0) -> Path:
    """A synthetic free decay ``exp(-pi f1 t / Q) cos(2 pi f1 t)``."""
    fs = 200_000.0
    t = np.arange(0.0, 0.2, 1.0 / fs)
    sigma = math.pi * f1 / q
    y = np.exp(-sigma * t) * np.cos(2.0 * math.pi * f1 * t)
    np.savetxt(directory / "ringdown.csv", np.column_stack([t, y]), delimiter=",")
    return _sidecar(
        directory,
        "ringdown.yaml",
        {
            "kind": "ringdown",
            "instrument": "scope",
            "timestamp": "2026-07-10T09:00:00Z",
            "data_file": "ringdown.csv",
            "columns": {"x": {"name": "t", "unit": "s"}, "y": {"name": "a", "unit": "arb"}},
        },
    )


# --------------------------------------------------------------------------- #
# Spectrum (M-15)
# --------------------------------------------------------------------------- #
def test_spectrum_gaussian_reduction(tmp_path: Path) -> None:
    """Centroid, FWHM, the R-56 form factor and the ASE floor are recovered."""
    result = load_characterization(_gaussian_spectrum(tmp_path))
    lam0 = result.get("wavelength_m")
    fwhm = result.get("spectrum_fwhm_m")
    dnu = result.get("linewidth_eff_hz")
    rin = result.get("rin_db_hz_floor")
    assert lam0 is not None and fwhm is not None and dnu is not None and rin is not None
    assert lam0.value == pytest.approx(1550.0e-9, rel=1.0e-6)
    assert lam0.u == pytest.approx(0.5e-9)
    assert fwhm.value == pytest.approx(60.0e-9, rel=5.0e-3)
    assert fwhm.u == pytest.approx(math.sqrt(2.0) * 0.5e-9)
    # R-56: for a Gaussian, tau_c = sqrt(2 ln 2 / pi) / dnu_FWHM, so
    # dnu_eff = 1 / tau_c = sqrt(pi / (2 ln 2)) * dnu_FWHM ~ 1.5053 dnu_FWHM.
    dnu_fwhm = C0 * 60.0e-9 / (1550.0e-9) ** 2
    form = math.sqrt(math.pi / (2.0 * math.log(2.0)))
    assert dnu.value == pytest.approx(form * dnu_fwhm, rel=2.0e-3)
    assert rin.value == pytest.approx(10.0 * math.log10(2.0 / dnu.value), abs=1.0e-9)
    # The whole table travels with the result for lineshape = "measured".
    assert result.spectrum_wavelength_m is not None and len(result.spectrum_wavelength_m) == 401
    # Derived shape figures carry no u -> informational by construction (17 §4).
    assert dnu.u is None and rin.u is None


def test_spectrum_nm_vs_m_trap(tmp_path: Path) -> None:
    """nm values declared as metres are caught by the sanity window (R-57(b))."""
    with pytest.raises(ValueError, match="sanity window"):
        load_characterization(_gaussian_spectrum(tmp_path, unit="m"))


def test_spectrum_requires_wavelength_uncertainty(tmp_path: Path) -> None:
    """A config-bound centroid without a declared u fails loudly (17 §4)."""
    path = _gaussian_spectrum(tmp_path)
    body = yaml.safe_load(path.read_text())
    body.pop("uncertainties")
    path.write_text(yaml.safe_dump(body))
    with pytest.raises(ValueError, match="wavelength_m"):
        load_characterization(path)


def test_spectrum_unknown_unit_rejected(tmp_path: Path) -> None:
    """Units are declared, never guessed: an unknown unit fails loudly."""
    with pytest.raises(ValueError, match="unknown unit"):
        load_characterization(_gaussian_spectrum(tmp_path, unit="angstrom"))


# --------------------------------------------------------------------------- #
# Ring-down (M-18)
# --------------------------------------------------------------------------- #
def test_ringdown_recovers_q_and_f1(tmp_path: Path) -> None:
    """``Q = pi f1 / sigma`` and the FFT-peak f1 are recovered from the decay."""
    result = load_characterization(_ringdown(tmp_path))
    q = result.get("q_total")
    f1 = result.get("f1_hz")
    assert q is not None and f1 is not None
    assert q.value == pytest.approx(1661.0, rel=0.02)
    assert q.u is not None and 0.0 < q.u < 0.05 * q.value
    assert f1.value == pytest.approx(6250.0, abs=2.0 * (f1.u or 0.0) + 1.0e-6)
    # f1 is a validation metric, not a config input (P20-1): it still carries u.
    assert f1.u is not None and f1.u > 0.0


def test_ringdown_rejects_undamped_tone(tmp_path: Path) -> None:
    """A pure tone has no significant decay and cannot yield Q (M-18 guard)."""
    fs = 200_000.0
    t = np.arange(0.0, 0.2, 1.0 / fs)
    y = np.cos(2.0 * math.pi * 6250.0 * t)
    np.savetxt(tmp_path / "tone.csv", np.column_stack([t, y]), delimiter=",")
    path = _sidecar(
        tmp_path,
        "tone.yaml",
        {
            "kind": "ringdown",
            "instrument": "scope",
            "timestamp": "t",
            "data_file": "tone.csv",
            "columns": {"x": {"name": "t", "unit": "s"}, "y": {"name": "a", "unit": "arb"}},
        },
    )
    with pytest.raises(ValueError, match="no significant decay"):
        load_characterization(path)


def test_ringdown_rejects_nonuniform_grid(tmp_path: Path) -> None:
    """A non-uniform time axis fails loudly (the FFT/Hilbert path needs fs)."""
    t = np.concatenate([np.linspace(0.0, 0.1, 3000), np.linspace(0.11, 0.3, 3000)])
    y = np.exp(-100.0 * t) * np.cos(2.0 * math.pi * 6250.0 * t)
    np.savetxt(tmp_path / "bad.csv", np.column_stack([t, y]), delimiter=",")
    path = _sidecar(
        tmp_path,
        "bad.yaml",
        {
            "kind": "ringdown",
            "instrument": "scope",
            "timestamp": "t",
            "data_file": "bad.csv",
            "columns": {"x": {"name": "t", "unit": "s"}, "y": {"name": "a", "unit": "arb"}},
        },
    )
    with pytest.raises(ValueError, match="uniform"):
        load_characterization(path)


def test_ringdown_time_unit_conversion(tmp_path: Path) -> None:
    """A millisecond time axis converts and yields the same Q (declared units)."""
    fs = 200_000.0
    f1, q_true = 6250.0, 1661.0
    t = np.arange(0.0, 0.2, 1.0 / fs)
    y = np.exp(-math.pi * f1 / q_true * t) * np.cos(2.0 * math.pi * f1 * t)
    np.savetxt(tmp_path / "ms.csv", np.column_stack([t * 1.0e3, y]), delimiter=",")
    path = _sidecar(
        tmp_path,
        "ms.yaml",
        {
            "kind": "ringdown",
            "instrument": "scope",
            "timestamp": "t",
            "data_file": "ms.csv",
            "columns": {"x": {"name": "t", "unit": "ms"}, "y": {"name": "a", "unit": "arb"}},
        },
    )
    result = load_characterization(path)
    q = result.get("q_total")
    assert q is not None and q.value == pytest.approx(q_true, rel=0.02)


# --------------------------------------------------------------------------- #
# Profile (M-17, one azimuth)
# --------------------------------------------------------------------------- #
def _arc(radius_m: float, n: int, noise_m: float, seed: int = 0) -> np.ndarray:
    """A circular crown arc (x, z) with Gaussian noise, SI metres."""
    theta = np.linspace(-0.5, 0.5, n)
    x = radius_m * np.sin(theta)
    z = radius_m * (1.0 - np.cos(theta))
    rng = np.random.default_rng(seed)
    return np.column_stack([x + rng.normal(0.0, noise_m, n), z + rng.normal(0.0, noise_m, n)])


def _profile_sidecar(directory: Path, table_um: np.ndarray) -> Path:
    np.savetxt(directory / "profile.csv", table_um, delimiter=",")
    return _sidecar(
        directory,
        "profile.yaml",
        {
            "kind": "profile",
            "instrument": "microscope",
            "timestamp": "t",
            "data_file": "profile.csv",
            "columns": {"x": {"name": "x", "unit": "um"}, "y": {"name": "z", "unit": "um"}},
        },
    )


def test_profile_circle_fit_recovers_radius(tmp_path: Path) -> None:
    """The Kasa fit recovers R_c to < 1 % with a positive propagated u."""
    path = _profile_sidecar(tmp_path, _arc(150.0e-6, 40, 2.0e-8) * 1.0e6)
    result = load_characterization(path)
    r_c = result.get("curvature_radius_m")
    assert r_c is not None
    assert r_c.value == pytest.approx(150.0e-6, rel=0.01)
    assert r_c.u is not None and 0.0 < r_c.u < 0.05 * r_c.value


def test_profile_rejects_collinear_points(tmp_path: Path) -> None:
    """A straight contour (R_c -> inf) cannot be fitted and fails loudly."""
    x = np.linspace(0.0, 100.0, 30)
    path = _profile_sidecar(tmp_path, np.column_stack([x, 2.0 * x]))
    with pytest.raises(ValueError, match=r"collinear|degenerate"):
        load_characterization(path)


def test_profile_rejects_noncircular_contour(tmp_path: Path) -> None:
    """A grossly non-circular contour trips the residual guard (doc 20 F0-3)."""
    x = np.linspace(-100.0, 100.0, 60)
    z = 0.02 * x**2 / 10.0 + 10.0 * np.sin(x / 5.0)  # wavy, far from a circle
    path = _profile_sidecar(tmp_path, np.column_stack([x, z]))
    with pytest.raises(ValueError, match="not circular"):
        load_characterization(path)


def test_profile_needs_enough_points(tmp_path: Path) -> None:
    """Fewer than 20 contour points is not a usable crown sample (F0-3)."""
    path = _profile_sidecar(tmp_path, _arc(150.0e-6, 10, 0.0) * 1.0e6)
    with pytest.raises(ValueError, match="at least 20"):
        load_characterization(path)


# --------------------------------------------------------------------------- #
# RIN trace (M-16 reduce)
# --------------------------------------------------------------------------- #
def _rin_sidecar(directory: Path, freq: np.ndarray, rin_db: np.ndarray, band: dict) -> Path:
    np.savetxt(directory / "rin.csv", np.column_stack([freq, rin_db]), delimiter=",")
    return _sidecar(
        directory,
        "rin.yaml",
        {
            "kind": "rin_psd",
            "instrument": "PD+ESA",
            "timestamp": "t",
            "data_file": "rin.csv",
            "columns": {"x": {"name": "f", "unit": "hz"}, "y": {"name": "rin", "unit": "db/hz"}},
            "band": band,
            "uncertainties": {"rin_db_hz": 0.5},
        },
    )


def test_rin_psd_reduces_to_band_median(tmp_path: Path) -> None:
    """A flat -120 dB/Hz trace reduces to -120 over the declared band."""
    freq = np.linspace(100.0, 20_000.0, 200)
    path = _rin_sidecar(
        tmp_path,
        freq,
        np.full_like(freq, -120.0),
        {"f_min_hz": 1_000.0, "f_max_hz": 10_000.0},
    )
    result = load_characterization(path)
    rin = result.get("rin_db_hz")
    assert rin is not None
    assert rin.value == pytest.approx(-120.0)
    assert rin.u == pytest.approx(0.5)


def test_rin_psd_needs_points_in_band(tmp_path: Path) -> None:
    """A reduction band containing < 4 trace points fails loudly."""
    freq = np.linspace(100.0, 500.0, 50)
    path = _rin_sidecar(
        tmp_path,
        freq,
        np.full_like(freq, -120.0),
        {"f_min_hz": 10_000.0, "f_max_hz": 20_000.0},
    )
    with pytest.raises(ValueError, match="inside the reduction band"):
        load_characterization(path)


def test_rin_psd_requires_declared_u(tmp_path: Path) -> None:
    """The reduced RIN is config-bound and must carry a declared u (17 §4)."""
    freq = np.linspace(100.0, 20_000.0, 50)
    path = _rin_sidecar(
        tmp_path,
        freq,
        np.full_like(freq, -120.0),
        {"f_min_hz": 1_000.0, "f_max_hz": 10_000.0},
    )
    body = yaml.safe_load(path.read_text())
    body.pop("uncertainties")
    path.write_text(yaml.safe_dump(body))
    with pytest.raises(ValueError, match="rin_db_hz"):
        load_characterization(path)


# --------------------------------------------------------------------------- #
# Scalars + sidecar validation
# --------------------------------------------------------------------------- #
def test_scalar_unit_conversion(tmp_path: Path) -> None:
    """A hand-recorded length in mm converts to SI with its u."""
    path = _sidecar(
        tmp_path,
        "length.yaml",
        {
            "kind": "scalar",
            "instrument": "caliper",
            "timestamp": "t",
            "parameter": "length_m",
            "value": 4.07,
            "unit": "mm",
            "u": 0.02,
        },
    )
    result = load_characterization(path)
    length = result.get("length_m")
    assert length is not None
    assert length.value == pytest.approx(4.07e-3)
    assert length.u == pytest.approx(2.0e-5)
    assert result.provenance.data_file is None and result.provenance.sha256 is None


def test_scalar_requires_uncertainty(tmp_path: Path) -> None:
    """A scalar without u is rejected at the sidecar (GUM gate, 17 §4)."""
    path = _sidecar(
        tmp_path,
        "nou.yaml",
        {
            "kind": "scalar",
            "instrument": "caliper",
            "timestamp": "t",
            "parameter": "length_m",
            "value": 4.0,
            "unit": "mm",
        },
    )
    with pytest.raises(ValueError, match="u"):
        load_characterization(path)


def test_scalar_unknown_parameter_rejected(tmp_path: Path) -> None:
    """Unknown scalar parameter names fail loudly (no guessed mappings)."""
    path = _sidecar(
        tmp_path,
        "who.yaml",
        {
            "kind": "scalar",
            "instrument": "caliper",
            "timestamp": "t",
            "parameter": "mystery_m",
            "value": 1.0,
            "unit": "m",
            "u": 0.1,
        },
    )
    with pytest.raises(ValueError, match="unknown scalar parameter"):
        load_characterization(path)


def test_sidecar_unknown_kind_rejected(tmp_path: Path) -> None:
    """An unknown artifact kind fails at sidecar validation."""
    path = _sidecar(
        tmp_path,
        "kind.yaml",
        {"kind": "hologram", "instrument": "x", "timestamp": "t"},
    )
    with pytest.raises(ValueError, match=r"hologram|kind"):
        load_characterization(path)


def test_provenance_carries_sha_and_instrument(tmp_path: Path) -> None:
    """XY artifacts record the data-file sha256, instrument and timestamp."""
    result = load_characterization(_gaussian_spectrum(tmp_path))
    prov = result.provenance
    assert prov.kind == "spectrum"
    assert prov.data_file == "spectrum.csv"
    assert prov.sha256 is not None and len(prov.sha256) == 64
    assert prov.instrument == "OSA"


def test_csv_tolerates_header_and_separators(tmp_path: Path) -> None:
    """One header line, comments and ;/tab separators are accepted."""
    lines = ["# comment", "f;rin", *(f"{100.0 + i}\t-120.0" for i in range(50))]
    (tmp_path / "rin.csv").write_text("\n".join(lines), encoding="utf-8")
    path = _sidecar(
        tmp_path,
        "rin.yaml",
        {
            "kind": "rin_psd",
            "instrument": "PD",
            "timestamp": "t",
            "data_file": "rin.csv",
            "columns": {"x": {"name": "f", "unit": "hz"}, "y": {"name": "rin", "unit": "db/hz"}},
            "band": {"f_min_hz": 100.0, "f_max_hz": 149.0},
            "uncertainties": {"rin_db_hz": 1.0},
        },
    )
    result = load_characterization(path)
    rin = result.get("rin_db_hz")
    assert rin is not None and rin.value == pytest.approx(-120.0)


def test_registry_lists_all_kinds() -> None:
    """The registry exposes the five artifact kinds (SW-52 pattern)."""
    for kind in ("scalar", "spectrum", "rin_psd", "ringdown", "profile"):
        assert CHARACTERIZATION_REGISTRY.get(kind) is not None


def test_measured_parameter_rejects_negative_u() -> None:
    """The contract enforces u >= 0 (GUM, 17 §4)."""
    with pytest.raises(ValueError, match="u"):
        MeasuredParameter(name="length_m", value=1.0, u=-0.1)


# --------------------------------------------------------------------------- #
# resolve_sidecar_path: the GUI loaders accept the sidecar YAML OR its CSV
# (same-stem convention, doc 16 §2a).
# --------------------------------------------------------------------------- #
def test_resolve_sidecar_path_passes_yaml_through(tmp_path: Path) -> None:
    """A sidecar YAML/YML path is returned unchanged."""
    yaml_path = tmp_path / "a.yaml"
    yaml_path.write_text("kind: scalar\n", encoding="utf-8")
    assert resolve_sidecar_path(yaml_path) == yaml_path
    yml_path = tmp_path / "b.yml"
    yml_path.write_text("kind: scalar\n", encoding="utf-8")
    assert resolve_sidecar_path(yml_path) == yml_path


def test_resolve_sidecar_path_hops_from_csv_to_sidecar(tmp_path: Path) -> None:
    """A CSV path resolves to the same-stem sidecar next to it."""
    (tmp_path / "trace.csv").write_text("1,2\n", encoding="utf-8")
    sidecar = tmp_path / "trace.yaml"
    sidecar.write_text("kind: rin_psd\n", encoding="utf-8")
    assert resolve_sidecar_path(tmp_path / "trace.csv") == sidecar


def test_resolve_sidecar_path_csv_without_sidecar_raises(tmp_path: Path) -> None:
    """A CSV with no same-stem sidecar fails loudly (units are undeclared)."""
    (tmp_path / "orphan.csv").write_text("1,2\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no sidecar found"):
        resolve_sidecar_path(tmp_path / "orphan.csv")


def test_resolve_sidecar_path_unknown_extension_raises(tmp_path: Path) -> None:
    """An unrecognised extension is rejected."""
    with pytest.raises(ValueError, match="unrecognised artifact"):
        resolve_sidecar_path(tmp_path / "trace.txt")


def test_load_characterization_accepts_csv_entry(tmp_path: Path) -> None:
    """End-to-end: picking the CSV loads the artifact via its sidecar."""
    freq = np.linspace(100.0, 20_000.0, 100)
    np.savetxt(
        tmp_path / "rin.csv",
        np.column_stack([freq, np.full_like(freq, -121.5)]),
        delimiter=",",
    )
    _sidecar(
        tmp_path,
        "rin.yaml",
        {
            "kind": "rin_psd",
            "instrument": "ESA",
            "timestamp": "t",
            "data_file": "rin.csv",
            "columns": {
                "x": {"name": "f", "unit": "hz"},
                "y": {"name": "r", "unit": "db/hz"},
            },
            "band": {"f_min_hz": 1_000.0, "f_max_hz": 10_000.0},
            "uncertainties": {"rin_db_hz": 0.5},
        },
    )
    result = load_characterization(tmp_path / "rin.csv")
    assert result.get("rin_db_hz").value == pytest.approx(-121.5)
