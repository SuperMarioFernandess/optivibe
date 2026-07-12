"""Brownian thermal floor ``NEA_th`` -- golden pins to doc 07 §2 (M-12, gap MG-1).

Every golden here references a closed-form base formula or a documented
reference number of doc 07 (rule 18 §5(g): pin to the base, not to code
output). Physics decisions: R-25 (acceleration mass ``M_a``), R-45 (its unit),
R-52+ (this implementation).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from optivibe.core.config.loader import load_constants, load_variant
from optivibe.core.config.models import Constants
from optivibe.mechanics.cantilever import first_mode_hz
from optivibe.mechanics.damping import q_total_model
from optivibe.mechanics.thermal import (
    acceleration_effective_mass,
    kinetic_effective_mass,
    nea_thermal,
    thermal_force_psd,
)

G0 = 9.80665


@pytest.fixture(scope="module")
def constants(config_dir: Path) -> Constants:
    """Physical constants from the repository config (doc 01 mirror)."""
    return load_constants(config_dir / "constants.yaml")


# --------------------------------------------------------------------------- #
# Effective masses (doc 07 §2.2, R-25/R-45).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_mass_coefficients_closed_form(constants: Constants) -> None:
    """Both mass coefficients follow their closed forms in ``beta_1 L`` (07 §2.2).

    ``m_eff / m_tot = 3 / (beta_1 L)^4 = 0.2427`` and
    ``M_a / m_tot = 3 (beta_1 L)^4 / 64 = 0.5795 ~ 0.58`` (the documented
    rounding). The ratio ``M_a / m_eff = (beta_1 L)^8 / 64 ~ 2.39`` is the
    square of the 1.55x floor overestimate of the naive substitution (R-25).
    """
    length = 1.4e-3
    fiber = constants.fiber
    m_tot = fiber.density_kg_m3 * fiber.area_m2 * length
    beta = constants.universal.beta1_l
    m_eff = kinetic_effective_mass(constants, length)
    m_a = acceleration_effective_mass(constants, length)
    assert m_eff / m_tot == pytest.approx(3.0 / beta**4, rel=1e-12)
    assert m_eff / m_tot == pytest.approx(0.2427, rel=1e-3)  # doc 07 §2.2a
    assert m_a / m_tot == pytest.approx(3.0 * beta**4 / 64.0, rel=1e-12)
    assert m_a / m_tot == pytest.approx(0.58, rel=1e-2)  # doc 07 §2.2b
    # Naive-mass overestimate of the floor: sqrt(M_a / m_eff) = 1.55 (R-25).
    assert math.sqrt(m_a / m_eff) == pytest.approx(1.55, rel=1e-2)


@pytest.mark.golden
def test_golden_acceleration_mass_value_and_unit(constants: Constants) -> None:
    """``M_a(L = 1.4 mm) = 2.19e-8 kg ~ 22 ug`` -- micrograms, not ng (R-45)."""
    m_a = acceleration_effective_mass(constants, 1.4e-3)
    assert m_a == pytest.approx(2.19e-8, rel=1e-2)  # doc 07 §2.2 (SI value)
    assert m_a * 1.0e9 == pytest.approx(21.9, rel=1e-2)  # kg -> ug is x1e9: 21.9 ug, not ng


@pytest.mark.golden
def test_golden_kinetic_mass_reproduces_f1(constants: Constants) -> None:
    """``sqrt(k_eff / m_eff)`` reproduces the Euler-Bernoulli ``omega_1`` (07 §2.2a).

    ``k_eff = 3 E I / L^3`` with ``m_eff = k_eff / omega_1^2`` is an identity by
    construction -- pinned as the cross-check that ties the tip-coordinate
    oscillator used by the FDT to the distributed-beam frequency of doc 02.
    """
    length = 1.4e-3
    fiber = constants.fiber
    k_eff = 3.0 * fiber.youngs_modulus_pa * fiber.inertia_m4 / length**3
    m_eff = kinetic_effective_mass(constants, length)
    f1 = math.sqrt(k_eff / m_eff) / (2.0 * math.pi)
    assert f1 == pytest.approx(first_mode_hz(constants, length), rel=1e-12)


# --------------------------------------------------------------------------- #
# NEA_th: reference numbers and the force-referral identity (doc 07 §2.1/§2.5).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_nea_thermal_reference_14mm(constants: Constants) -> None:
    """Reference floor at ``L = 1.4 mm``: 1.12 ug/sqrt(Hz) at Q = 1950 and
    0.67 at Q = 5490 (doc 07 §2.5; the Q range spans the anchor uncertainty)."""
    length = 1.4e-3
    assert nea_thermal(constants, length, 1950.0) / G0 * 1e6 == pytest.approx(1.12, rel=1e-2)
    assert nea_thermal(constants, length, 5490.0) / G0 * 1e6 == pytest.approx(0.67, rel=1e-2)


@pytest.mark.golden
def test_golden_nea_thermal_scaling_column(constants: Constants) -> None:
    """The doc 07 §4.3 scaling column with the computable Q(L) (M-02 input).

    ``L = 1.0 -> ~2.6, 1.4 -> ~1.1, 2.0 -> 0.57, 3.0 -> 0.34, 4.0 -> 0.25``
    ug/sqrt(Hz) -- the thermal-floor column of the design map, evaluated at
    the model ``Q_total(L)`` (air + anchor + internal).
    """
    column = {1.0e-3: 2.6, 1.4e-3: 1.1, 2.0e-3: 0.57, 3.0e-3: 0.34, 4.0e-3: 0.25}
    for length, expected_ug in column.items():
        q = q_total_model(constants, length)
        value = nea_thermal(constants, length, q) / G0 * 1e6
        assert value == pytest.approx(expected_ug, rel=2.5e-2), f"L = {length}"


@pytest.mark.golden
def test_golden_force_referral_identity(constants: Constants) -> None:
    """``sqrt(S_F) / (0.375 m_tot) == sqrt(4 kB T omega_1 / (Q M_a))`` (07 §2.2).

    The two routes -- Langevin force divided by the equivalent tip force per
    unit acceleration ``F_a/a = (3/8) m_tot = sqrt(M_a m_eff)``, and the closed
    form through ``M_a`` -- must agree exactly (this is how ``M_a`` is defined).
    """
    length, q = 1.4e-3, 1950.0
    fiber = constants.fiber
    m_tot = fiber.density_kg_m3 * fiber.area_m2 * length
    via_force = math.sqrt(thermal_force_psd(constants, length, q)) / (0.375 * m_tot)
    assert via_force == pytest.approx(nea_thermal(constants, length, q), rel=1e-12)
    m_a = acceleration_effective_mass(constants, length)
    m_eff = kinetic_effective_mass(constants, length)
    assert 0.375 * m_tot == pytest.approx(math.sqrt(m_a * m_eff), rel=1e-12)


# --------------------------------------------------------------------------- #
# Limits and scaling exponents (doc 07 §2.1; task M-12 derivation checks).
# --------------------------------------------------------------------------- #
def test_limit_infinite_q_kills_the_floor(constants: Constants) -> None:
    """``Q -> inf  =>  NEA_th -> 0`` (a dissipation-free mode does not fluctuate),
    scaling exactly as ``1/sqrt(Q)``: quadrupling Q halves the floor."""
    length = 1.4e-3
    base = nea_thermal(constants, length, 1.0e3)
    assert nea_thermal(constants, length, 4.0e3) == pytest.approx(base / 2.0, rel=1e-12)
    assert nea_thermal(constants, length, 1.0e12) < 1.0e-4 * base


def test_limit_zero_temperature_kills_the_floor(constants: Constants) -> None:
    """``T -> 0  =>  NEA_th -> 0``, scaling as ``sqrt(T)`` (FDT prefactor)."""
    length, q = 1.4e-3, 1950.0
    cold = constants.model_copy(
        update={"detector": constants.detector.model_copy(update={"temperature_k": 73.25})}
    )
    # T / 4 halves the floor (sqrt scaling; 293 / 4 = 73.25).
    assert nea_thermal(cold, length, q) == pytest.approx(
        nea_thermal(constants, length, q) / 2.0, rel=1e-12
    )


@pytest.mark.golden
def test_golden_length_exponent_is_three_halves_at_fixed_q(constants: Constants) -> None:
    """Fixed-Q length scaling is exactly ``NEA_th ~ L^-3/2`` -- **not** ``L^-2``.

    Derivation: ``omega_1 ~ L^-2`` and ``M_a ~ L`` give
    ``NEA_th ~ sqrt(L^-2 / L) = L^-3/2``. With the computable Q(L) folded in,
    the *effective* exponent over the design window ``1.4-4.0 mm`` is ~ -1.42
    (Q rises then falls over the window, doc 07 §4.3) -- bounded here between
    the fixed-Q -1.5 and -1.2.
    """
    q = 2.0e3
    ratio = nea_thermal(constants, 2.8e-3, q) / nea_thermal(constants, 1.4e-3, q)
    assert ratio == pytest.approx(2.0 ** (-1.5), rel=1e-12)
    l_lo, l_hi = 1.4e-3, 4.0e-3
    v_lo = nea_thermal(constants, l_lo, q_total_model(constants, l_lo))
    v_hi = nea_thermal(constants, l_hi, q_total_model(constants, l_hi))
    exponent = math.log(v_hi / v_lo) / math.log(l_hi / l_lo)
    assert -1.5 < exponent < -1.2


def test_thermal_guards_reject_nonpositive_inputs(constants: Constants) -> None:
    """Loud failures on nonphysical inputs (doc 10 error policy)."""
    for func in (kinetic_effective_mass, acceleration_effective_mass):
        with pytest.raises(ValueError, match="length_m"):
            func(constants, 0.0)
    with pytest.raises(ValueError, match="q_total"):
        nea_thermal(constants, 1.4e-3, 0.0)
    with pytest.raises(ValueError, match="q_total"):
        thermal_force_psd(constants, 1.4e-3, -1.0)


# --------------------------------------------------------------------------- #
# Budget integration: the fourth branch (doc 17 §2 chain; doc 16 M-12 card).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_variant_d_target_is_the_thermal_floor(
    constants: Constants, config_dir: Path
) -> None:
    """Variant D's ``target_nea_ug_rthz = 0.227`` is the thermal floor in air
    at the model Q (R-50 derived the config number from this very formula)."""
    variant = load_variant("D", config_dir)
    floor_ug = nea_thermal(constants, variant.length_m, variant.q_total) / G0 * 1e6
    assert floor_ug == pytest.approx(variant.target_nea_ug_rthz, rel=5e-3)


@pytest.mark.golden
def test_golden_thermal_branch_share_matches_backlog_card(
    constants: Constants, config_dir: Path
) -> None:
    """Doc 16 M-12: the branch moves B/C by < 0.2 % but is essential for A/D.

    Quadrature lift ``hypot(opt, th)/opt - 1``: < 0.2 % for the shot/RIN-limited
    B and C; > 1 % for A and > 10 % for the thermally-targeted D.
    """
    from optivibe.analysis.variant_tools import analytic_point

    lifts: dict[str, float] = {}
    for name in ("A", "B", "C", "D"):
        variant = load_variant(name, config_dir)
        on = analytic_point(variant, constants)
        off = analytic_point(variant, constants, include_thermal=False)
        lifts[name] = on.nea_plateau / off.nea_plateau - 1.0
        # The point-level quadrature identity (the doc 17 §2 chain).
        assert on.nea_plateau == pytest.approx(
            math.hypot(off.nea_plateau, on.nea_thermal), rel=1e-12
        )
    assert lifts["B"] < 2.0e-3
    assert lifts["C"] < 2.0e-3
    assert lifts["A"] > 1.0e-2
    assert lifts["D"] > 1.0e-1
