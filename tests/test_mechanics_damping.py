"""M-02 damping-model tests: golden anchors of docs 02 §5 / 07 §2.3 (R-26/R-31),
limits, the non-monotonic Q(L) shape, and the config wiring (O-SW-06)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from optivibe.core.config import load_constants, load_variant
from optivibe.core.config.models import Constants
from optivibe.core.config.presets import PresetStore
from optivibe.core.config.subsystems import SystemConfig
from optivibe.mechanics.damping import (
    damping_budget,
    hydrodynamic_function,
    knudsen_number,
    q_air,
    q_anchor,
    q_total_model,
    reynolds_number,
)


@pytest.fixture(scope="module")
def constants(config_dir: Path) -> Constants:
    return load_constants(config_dir / "constants.yaml")


def _base_system(length_m: float, **extra: object) -> dict[str, object]:
    """A minimal composition (variant-B shape) with a configurable length."""
    data: dict[str, object] = {
        "name": "T",
        "description": "damping test composition",
        "mode": "offresonance",
        "band": {"f_min_hz": 1.0, "f_max_hz": 10000.0},
        "full_scale_g": 50.0,
        "route": 2,
        "eta_bias": 0.25,
        "vacuum": False,
        "source": {"preset": "sld"},
        "fiber": {"preset": "smf28"},
        "cantilever": {"preset": "silica", "overrides": {"length_m": length_m}},
        "reflector": {"preset": "cyl_rc31"},
        "detector": {"preset": "balanced_24bit"},
    }
    data.update(extra)
    return data


# --------------------------------------------------------------------------- #
# Golden: hydrodynamic function and Q_air anchors (docs 02 §5/§8, 07 §2.3).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_hydrodynamic_function_doc07_values(constants: Constants) -> None:
    """Gamma at the two documented Reynolds numbers (doc 07 §2.3).

    ``Re = 83.3`` (L = 1.4 mm): ``Gamma_i ~ 0.333``, ``Gamma_r ~ 1.31``;
    ``Re = 18.1`` (L = 3 mm): ``Gamma_i ~ 0.769`` (method cross-check of 07).
    """
    gamma_short = hydrodynamic_function(83.3)
    assert gamma_short.imag == pytest.approx(0.333, rel=5e-3)
    assert gamma_short.real == pytest.approx(1.31, rel=5e-3)
    gamma_long = hydrodynamic_function(18.1)
    assert gamma_long.imag == pytest.approx(0.769, rel=5e-3)


@pytest.mark.golden
def test_golden_q_air_matches_doc02_and_doc07(constants: Constants) -> None:
    """``Q_air ~ 2377`` at L = 3 mm (doc 02 §8) and ``~5487`` at 1.4 mm (07 §2.3)."""
    assert q_air(constants, 3.0e-3) == pytest.approx(2377.0, rel=5e-3)
    assert q_air(constants, 1.4e-3) == pytest.approx(5487.0, rel=5e-3)


@pytest.mark.golden
def test_golden_reynolds_numbers_doc07(constants: Constants) -> None:
    """``Re = 18.1`` at 3 mm and ``83.3`` at 1.4 mm (doc 07 §2.3)."""
    from optivibe.mechanics.cantilever import first_mode_hz

    re_3mm = reynolds_number(constants, 2.0 * math.pi * first_mode_hz(constants, 3.0e-3))
    re_14mm = reynolds_number(constants, 2.0 * math.pi * first_mode_hz(constants, 1.4e-3))
    assert re_3mm == pytest.approx(18.1, rel=5e-3)
    assert re_14mm == pytest.approx(83.3, rel=5e-3)


# --------------------------------------------------------------------------- #
# Golden: anchor loss, totals, non-monotonic shape (doc 07 §2.3/§4.3; R-26/R-31).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_q_anchor_doc07(constants: Constants) -> None:
    """``Q_anchor = 2.17 (L/D)^3 ~ 3050`` at L = 1.4 mm (doc 07 §2.3, R-26)."""
    assert q_anchor(constants, 1.4e-3) == pytest.approx(3050.0, rel=5e-3)
    # ~1110 at L = 1 mm (doc 07 §4.3 scaling-table note).
    assert q_anchor(constants, 1.0e-3) == pytest.approx(1111.0, rel=5e-3)


@pytest.mark.golden
def test_golden_q_total_scaling_table_r31(constants: Constants) -> None:
    """``Q_total(L)`` against the doc 07 §4.3 scaling table (R-31).

    Anchors: 1.0 mm -> ~970 (anchor-limited), 1.4 mm -> ~1950 (R-26),
    2.0 mm -> ~2610, 2.5 mm -> ~2480, 3.0 mm -> ~2185, 4.0 mm -> ~1660.
    Tolerance 1 % (the table itself carries 2-3 significant digits).
    """
    anchors = {
        1.0e-3: 970.0,
        1.4e-3: 1950.0,
        2.0e-3: 2610.0,
        2.5e-3: 2480.0,
        3.0e-3: 2185.0,
        4.0e-3: 1660.0,
    }
    for length, expected in anchors.items():
        assert q_total_model(constants, length) == pytest.approx(expected, rel=1e-2), length


@pytest.mark.golden
def test_golden_q_total_nonmonotonic_peak(constants: Constants) -> None:
    """Non-monotonicity of ``Q_total(L)``: peak ~2600 near L ~ 2.0-2.2 mm (R-31).

    Anchor loss ``~(L/D)^3`` dominates at short L, air ``~1/L`` at long L; the
    documented peak ``Q ~ 2600`` sits at ``L ~ 2.0-2.2 mm`` (doc 07 §4.3).
    """
    lengths = np.linspace(1.0e-3, 4.0e-3, 601)
    totals = np.array([q_total_model(constants, float(length)) for length in lengths])
    peak_index = int(np.argmax(totals))
    assert 2.0e-3 <= lengths[peak_index] <= 2.2e-3
    assert totals[peak_index] == pytest.approx(2610.0, rel=2e-2)
    # Interior peak: rises before, falls after (the non-monotonic signature).
    assert totals[0] < totals[peak_index]
    assert totals[-1] < totals[peak_index]


@pytest.mark.golden
def test_golden_q_model_vs_tabulated_variant_numbers(constants: Constants) -> None:
    """The model against the previously tabulated A-D numbers (R-48 audit trail).

    B (2610 at L = 2.0 mm, doc 07 §4.3 peak), C (1950 at 1.41 mm, R-26) and the
    air value of D (1500 at 4.47 mm, doc 08 §5) are reproduced within 1.5 %:
    those numbers came from the same physics and the model only sharpens them.
    A's 1430 does NOT reproduce -- it came from a 1/L scaling of Q_air, which
    over-estimates Q at low Re; the model gives 1297 (-9.3 %). The coordinator
    accepted the model values for the whole family on 2026-07-12 (R-48), so the
    numbers below are the *superseded* ones, kept as the audit trail of that
    decision.
    """
    assert q_total_model(constants, 2.0e-3) == pytest.approx(2610.0, rel=1e-2)
    assert q_total_model(constants, 1.41e-3) == pytest.approx(1950.0, rel=1.5e-2)
    assert q_air(constants, 4.47e-3) == pytest.approx(1500.0, rel=1e-2)
    assert q_total_model(constants, 4.47e-3) == pytest.approx(1472.0, rel=1e-3)
    assert q_total_model(constants, 5.0e-3) == pytest.approx(1297.0, rel=1e-2)
    # The 1/L scaling that produced A's 1430 is the outlier, not the model:
    assert q_total_model(constants, 5.0e-3) < 0.95 * 1430.0


# --------------------------------------------------------------------------- #
# Limits and regime checks (doc 07 §2.3; 11 §7 style).
# --------------------------------------------------------------------------- #
def test_limit_vacuum_removes_air_channel(constants: Constants) -> None:
    """Under vacuum the air channel is removed and Q rises to anchor/internal."""
    budget = damping_budget(constants, 4.47e-3, vacuum=True)
    assert math.isinf(budget["air"])
    expected = 1.0 / (1.0 / budget["anchor"] + 1.0 / budget["structural"] + 1.0 / budget["ted"])
    assert budget["total"] == pytest.approx(expected, rel=1e-12)
    assert budget["total"] > 10.0 * q_total_model(constants, 4.47e-3)


def test_limit_total_below_every_channel(constants: Constants) -> None:
    """Parallel addition: the total is below each individual channel."""
    budget = damping_budget(constants, 2.0e-3)
    for key in ("air", "anchor", "structural", "ted"):
        assert budget["total"] < budget[key]


def test_limit_gamma_inviscid(constants: Constants) -> None:
    """``Re -> inf``: ``Gamma -> 1`` (inviscid added mass, no dissipation)."""
    gamma = hydrodynamic_function(1.0e9)
    assert gamma.real == pytest.approx(1.0, abs=1e-3)
    assert gamma.imag == pytest.approx(0.0, abs=1e-3)


def test_regime_continuum_knudsen(constants: Constants) -> None:
    """Continuum viscous regime: ``Kn = lambda_mfp / R ~ 1.1e-3 << 1`` (07 §2.3)."""
    kn = knudsen_number(constants)
    assert kn == pytest.approx(68.0e-9 / 62.5e-6, rel=1e-12)
    assert kn < 1.0e-2


def test_invalid_inputs_raise(constants: Constants) -> None:
    """Non-positive inputs fail loudly (10 §7)."""
    with pytest.raises(ValueError, match="positive"):
        hydrodynamic_function(0.0)
    with pytest.raises(ValueError, match="positive"):
        reynolds_number(constants, -1.0)
    with pytest.raises(ValueError, match="positive"):
        q_anchor(constants, 0.0)


# --------------------------------------------------------------------------- #
# Config wiring: explicit q_total keeps priority; None computes Q(L) (M-02).
# --------------------------------------------------------------------------- #
def test_resolve_computes_q_when_omitted(config_dir: Path, constants: Constants) -> None:
    """``q_total: null`` resolves to the damping-model value (M-02)."""
    system = SystemConfig.model_validate(_base_system(2.0e-3))
    variant = system.resolve(PresetStore(config_dir), constants=constants)
    assert variant.q_total == pytest.approx(q_total_model(constants, 2.0e-3), rel=1e-12)


def test_resolve_explicit_q_keeps_priority(config_dir: Path, constants: Constants) -> None:
    """An explicit ``q_total`` overrides the model (backward compatibility)."""
    system = SystemConfig.model_validate(_base_system(2.0e-3, q_total=1234.5))
    variant = system.resolve(PresetStore(config_dir), constants=constants)
    assert variant.q_total == 1234.5


def test_resolve_vacuum_flag_reaches_q_model(config_dir: Path, constants: Constants) -> None:
    """``vacuum: true`` removes the air channel from the computed Q (M-02)."""
    system = SystemConfig.model_validate(_base_system(2.0e-3, vacuum=True))
    variant = system.resolve(PresetStore(config_dir), constants=constants)
    assert variant.q_total == pytest.approx(
        q_total_model(constants, 2.0e-3, vacuum=True), rel=1e-12
    )


def test_resolve_without_constants_fails_loudly(config_dir: Path) -> None:
    """Omitting both ``q_total`` and ``constants`` is a loud error (10 §7)."""
    system = SystemConfig.model_validate(_base_system(2.0e-3))
    with pytest.raises(ValueError, match="Q\\(L\\)"):
        system.resolve(PresetStore(config_dir))


def test_builtin_variants_use_the_q_model(config_dir: Path, constants: Constants) -> None:
    """A-D take q_total from the Q(L) model in air (R-48/R-49).

    The family runs in air (``vacuum: false`` everywhere, R-49); each variant
    omits ``q_total`` so the model is the single source of truth. Values:
    A 1297.35 (L = 5.0 mm), B 2606.49 (2.0 mm), C 1969.01 (1.41 mm),
    D 1472.03 (4.47 mm) -- pinned in the resolved-variant golden.
    """
    for name, length in (("A", 5.0e-3), ("B", 2.0e-3), ("C", 1.41e-3), ("D", 4.47e-3)):
        variant = load_variant(name, config_dir=config_dir)
        assert variant.vacuum is False
        assert variant.q_total == pytest.approx(q_total_model(constants, length), rel=1e-12)
