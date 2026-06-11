"""Golden tests: constants match doc 01 and reproduce analytical references.

These are the *golden* checks of S0 (doc 10 §10): the YAML constants must equal
the documented reference numbers (doc 01), be internally self-consistent
(``S = pi R^2``, ``I = pi R^4 / 4``, ``c = sqrt(E/rho)``), and reproduce the
documented first-mode scaling law ``f1 ~ 100 / L[mm]^2 kHz`` (doc 08, R-31) for
each variant length, matching the documented headline ``f1`` of A/B/C/D.

Since S2 the mode-frequency formula lives in the mechanics package
(:func:`optivibe.mechanics.first_mode_hz`); the local analytical transcription
below stays as an independent reference and the package function is
cross-checked against it.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from optivibe.core.config import Constants, load_constants, load_variant
from optivibe.mechanics import first_mode_hz

pytestmark = pytest.mark.golden


def _first_mode_hz(c: Constants, length_m: float) -> float:
    """Analytical first natural frequency of the clamped-free fiber cantilever.

    ``f1 = (beta1 L)^2 / (2 pi) * sqrt(E I / (rho S)) / L^2`` (Euler-Bernoulli).
    """
    fiber = c.fiber
    beta1_l = c.universal.beta1_l
    stiffness = math.sqrt(
        fiber.youngs_modulus_pa * fiber.inertia_m4 / (fiber.density_kg_m3 * fiber.area_m2)
    )
    return beta1_l**2 / (2.0 * math.pi) * stiffness / length_m**2


def test_constants_match_doc01(config_dir: Path) -> None:
    c = load_constants(config_dir / "constants.yaml")
    f = c.fiber
    assert f.diameter_m == pytest.approx(125.0e-6)
    assert f.radius_m == pytest.approx(62.5e-6)
    assert f.area_m2 == pytest.approx(1.227e-8)
    assert f.inertia_m4 == pytest.approx(1.198e-17)
    assert f.youngs_modulus_pa == pytest.approx(72.0e9)
    assert f.density_kg_m3 == pytest.approx(2201.0)
    assert f.poisson_ratio == pytest.approx(0.17)
    assert f.bar_velocity_m_s == pytest.approx(5719.5)

    assert c.air.density_kg_m3 == pytest.approx(1.204)
    assert c.air.dynamic_viscosity_pa_s == pytest.approx(1.81e-5)

    u = c.universal
    assert u.g0_m_s2 == pytest.approx(9.80665)
    assert u.beta1_l == pytest.approx(1.8751)
    assert u.beta2_l == pytest.approx(4.6941)
    assert u.phi1_at_tip == pytest.approx(2.000)
    assert u.phi1_dd_at_root == pytest.approx(7.032)

    assert c.tilt_displacement_coupling_per_l == pytest.approx(1.377)


def test_constants_self_consistent(config_dir: Path) -> None:
    f = load_constants(config_dir / "constants.yaml").fiber
    assert f.area_m2 == pytest.approx(math.pi * f.radius_m**2, rel=1e-3)
    assert f.inertia_m4 == pytest.approx(math.pi * f.radius_m**4 / 4.0, rel=1e-3)
    assert f.bar_velocity_m_s == pytest.approx(
        math.sqrt(f.youngs_modulus_pa / f.density_kg_m3), rel=1e-3
    )


def test_first_mode_scaling_law(config_dir: Path) -> None:
    """f1 * L[mm]^2 must equal the documented prefactor ~100 kHz*mm^2."""
    c = load_constants(config_dir / "constants.yaml")
    for length_mm in (1.41, 2.0, 5.0, 4.47):
        length_m = length_mm * 1.0e-3
        f1_khz = first_mode_hz(c, length_m) / 1.0e3
        prefactor = f1_khz * length_mm**2
        assert prefactor == pytest.approx(100.0, rel=2.0e-3)


def test_package_first_mode_matches_local_transcription(config_dir: Path) -> None:
    """The mechanics-package formula equals the independent transcription."""
    c = load_constants(config_dir / "constants.yaml")
    for length_mm in (1.25, 1.41, 2.0, 3.0, 4.47, 5.0):
        length_m = length_mm * 1.0e-3
        assert first_mode_hz(c, length_m) == pytest.approx(_first_mode_hz(c, length_m), rel=1.0e-12)


# Documented headline first-mode frequencies (doc 08 §6), Hz.
EXPECTED_F1_HZ = {"A": 4.0e3, "B": 25.0e3, "C": 50.0e3, "D": 5.0e3}


@pytest.mark.parametrize("name", sorted(EXPECTED_F1_HZ))
def test_variant_first_mode_matches_doc08(name: str, config_dir: Path) -> None:
    c = load_constants(config_dir / "constants.yaml")
    v = load_variant(name, config_dir=config_dir)
    f1 = first_mode_hz(c, v.length_m)
    assert f1 == pytest.approx(EXPECTED_F1_HZ[name], rel=2.0e-2)
