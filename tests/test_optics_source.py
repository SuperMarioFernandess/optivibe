"""M-01 source-linewidth tests: golden anchors of docs 03 §f' / 07 §1.2
(R-13/R-14), limits, applicability boundaries, and the config wiring."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from optivibe.core.config import load_constants
from optivibe.core.config.models import Constants
from optivibe.core.config.presets import PresetStore
from optivibe.core.config.subsystems import SourceConfig, SystemConfig
from optivibe.optics.source import (
    WASHOUT_VISIBILITY_MAX,
    coherence_length_m,
    fringe_visibility,
    linewidth_nu_hz,
    min_gap_for_washout_m,
    rin_ase,
    rin_ase_db_hz,
)

_LAMBDA = 1550.0e-9


@pytest.fixture(scope="module")
def constants(config_dir: Path) -> Constants:
    return load_constants(config_dir / "constants.yaml")


def _base_system(**extra: object) -> dict[str, object]:
    """A minimal composition (variant-B shape) for source-wiring tests."""
    data: dict[str, object] = {
        "name": "T",
        "description": "source test composition",
        "mode": "offresonance",
        "band": {"f_min_hz": 1.0, "f_max_hz": 10000.0},
        "full_scale_g": 50.0,
        "route": 2,
        "eta_bias": 0.25,
        "q_total": 2610.0,
        "vacuum": False,
        "source": {"preset": "sld"},
        "fiber": {"preset": "smf28"},
        "cantilever": {"preset": "silica", "overrides": {"length_m": 2.0e-3}},
        "reflector": {"preset": "cyl_rc31"},
        "detector": {"preset": "balanced_24bit"},
    }
    data.update(extra)
    return data


# --------------------------------------------------------------------------- #
# Golden: ASE RIN floor (doc 07 §1.2).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_rin_ase_doc07_anchor() -> None:
    """``dlam = 60 nm`` @ 1550 nm -> ``dnu = 7.49e12 Hz`` -> ``-125.7 dB/Hz``.

    Doc 07 §1.2: ``RIN_ASE ~ 2/dnu``; the anchor evaluates to 2.67e-13 /Hz,
    i.e. -125.7 dB/Hz -- consistent with the SLD platform figure -126 (R-40).
    """
    delta_nu = linewidth_nu_hz(_LAMBDA, 60.0e-9)
    assert delta_nu == pytest.approx(7.49e12, rel=1e-3)
    assert rin_ase(delta_nu) == pytest.approx(2.67e-13, rel=1e-2)
    assert rin_ase_db_hz(delta_nu) == pytest.approx(-125.7, abs=0.05)


@pytest.mark.golden
def test_golden_rin_ase_is_two_over_dnu() -> None:
    """The adopted convention is exactly ``RIN = 2/dnu`` (doc 07 §1.2)."""
    assert rin_ase(1.0e12) == pytest.approx(2.0e-12, rel=1e-12)


# --------------------------------------------------------------------------- #
# Golden: coherence length and visibility (doc 03 §f' table).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_coherence_length_doc03_table() -> None:
    """``L_c = 0.44 lambda^2 / dlam``: 20/40/60 nm -> 52.9/26.4/17.6 um.

    The code keeps the coefficient exact (``2 ln 2 / pi = 0.4413``, of which
    the base's 0.44 is the rounding), so the tolerance is 1 %.
    """
    table_um = {20.0e-9: 52.9, 40.0e-9: 26.4, 60.0e-9: 17.6}
    for dlam, expected_um in table_um.items():
        assert coherence_length_m(_LAMBDA, dlam) * 1e6 == pytest.approx(expected_um, rel=1e-2)


@pytest.mark.golden
def test_golden_visibility_doc03_table() -> None:
    """``V(A = 30 um)``: 0.41 (20 nm), 0.029 (40 nm), ~0 (60 nm) -- doc 03 §f'."""
    gap = 30.0e-6
    v20 = fringe_visibility(gap, coherence_length_m(_LAMBDA, 20.0e-9))
    v40 = fringe_visibility(gap, coherence_length_m(_LAMBDA, 40.0e-9))
    v60 = fringe_visibility(gap, coherence_length_m(_LAMBDA, 60.0e-9))
    assert v20 == pytest.approx(0.41, abs=0.01)
    assert v40 == pytest.approx(0.029, abs=0.002)
    assert v60 < 1.0e-3


@pytest.mark.golden
def test_golden_washout_gap_factor_r13() -> None:
    """``V < 0.03`` inverts to ``A >= 1.1246 L_c`` (doc 03 §f': ``A >~ 1.12 L_c``).

    Cross-checks: the doc table's minimum gaps (59 um at 20 nm, 30 um at
    40 nm, 20 um at 60 nm) and the exact identity ``V(A_min) = V_max``.
    """
    l_c = coherence_length_m(_LAMBDA, 60.0e-9)
    a_min = min_gap_for_washout_m(l_c)
    assert a_min / l_c == pytest.approx(1.1246, rel=1e-3)
    assert fringe_visibility(a_min, l_c) == pytest.approx(WASHOUT_VISIBILITY_MAX, rel=1e-12)
    table_um = {20.0e-9: 59.0, 40.0e-9: 30.0, 60.0e-9: 20.0}
    for dlam, gap_um in table_um.items():
        computed = min_gap_for_washout_m(coherence_length_m(_LAMBDA, dlam)) * 1e6
        assert computed == pytest.approx(gap_um, rel=2e-2)


# --------------------------------------------------------------------------- #
# Limits and dimensional sanity (11 §7 style).
# --------------------------------------------------------------------------- #
def test_limit_visibility_bounds() -> None:
    """``A -> 0`` gives ``V -> 1``; ``A >> L_c`` gives ``V -> 0``."""
    l_c = coherence_length_m(_LAMBDA, 60.0e-9)
    assert fringe_visibility(0.0, l_c) == 1.0
    assert fringe_visibility(100.0 * l_c, l_c) < 1.0e-12


def test_limit_narrow_line_never_washes_out() -> None:
    """``dlam -> 0``: ``L_c -> inf`` and ``V -> 1`` (monochromatic light)."""
    l_c = coherence_length_m(_LAMBDA, 1.0e-15)
    assert l_c > 1.0
    assert fringe_visibility(40.0e-6, l_c) == pytest.approx(1.0, abs=1e-6)


def test_limit_wide_line_rin_vanishes() -> None:
    """``dnu -> inf``: ``RIN -> 0`` (beat averaging over many modes)."""
    assert rin_ase(1.0e20) < 1.0e-19


def test_gaussian_identity_visibility_form() -> None:
    """``2^{-(2A/L_c)^2}`` equals ``exp(-pi^2 A^2 dlam^2/(lambda^4 ln 2))``.

    The identity of doc 03 §f' holds exactly only with the exact coefficient
    ``L_c = (2 ln 2/pi) lambda^2/dlam`` -- the reason the code keeps it exact.
    """
    dlam, gap = 40.0e-9, 30.0e-6
    lhs = fringe_visibility(gap, coherence_length_m(_LAMBDA, dlam))
    rhs = math.exp(-(math.pi**2) * gap**2 * dlam**2 / (_LAMBDA**4 * math.log(2.0)))
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_invalid_inputs_raise() -> None:
    """Non-physical inputs fail loudly (10 §7), incl. the first-order bound."""
    with pytest.raises(ValueError, match="positive"):
        rin_ase(0.0)
    with pytest.raises(ValueError, match="positive"):
        coherence_length_m(_LAMBDA, -1.0)
    with pytest.raises(ValueError, match="non-negative"):
        fringe_visibility(-1.0e-6, 1.0e-5)
    with pytest.raises(ValueError, match="first-order"):
        linewidth_nu_hz(_LAMBDA, 0.5 * _LAMBDA)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        min_gap_for_washout_m(1.0e-5, visibility_max=1.5)


# --------------------------------------------------------------------------- #
# Config wiring: SourceConfig validators and resolve-time derivations (M-01).
# --------------------------------------------------------------------------- #
def test_source_config_requires_some_noise_input() -> None:
    """Neither RIN nor linewidth -> loud validation error (10 §7)."""
    with pytest.raises(ValueError, match="noise input"):
        SourceConfig(source_kind="SLD", wavelength_m=_LAMBDA, power_w=0.016)


def test_source_config_dfb_needs_explicit_rin() -> None:
    """The ASE relation is thermal/ASE-only: a DFB must state its RIN."""
    with pytest.raises(ValueError, match="DFB"):
        SourceConfig(source_kind="DFB", wavelength_m=_LAMBDA, power_w=0.1, linewidth_fwhm_m=1.0e-14)


def test_resolve_derives_rin_from_linewidth(config_dir: Path) -> None:
    """The ``sld_dl60`` preset resolves to the ASE floor -125.7 dB/Hz (M-01)."""
    system = SystemConfig.model_validate(_base_system(source={"preset": "sld_dl60"}))
    variant = system.resolve(PresetStore(config_dir))
    expected = rin_ase_db_hz(linewidth_nu_hz(_LAMBDA, 60.0e-9))
    assert variant.source.rin_db_hz == pytest.approx(expected, rel=1e-12)
    assert variant.source.rin_db_hz == pytest.approx(-125.73, abs=0.01)


def test_resolve_explicit_rin_keeps_priority(config_dir: Path) -> None:
    """An explicit RIN wins over the derived floor (backward compatibility)."""
    system = SystemConfig.model_validate(
        _base_system(source={"preset": "sld_dl60", "overrides": {"rin_db_hz": -120.0}})
    )
    variant = system.resolve(PresetStore(config_dir))
    assert variant.source.rin_db_hz == -120.0


def test_resolve_washout_guard_fires_for_narrow_sld(config_dir: Path) -> None:
    """Route 2 + dlam = 20 nm at A = 31 um -> V = 0.39 >= 0.03 -> loud error.

    Doc 03 §f' table: a 20-nm SLD needs A >= 59 um; at the platform gap 31 um
    the fringe is NOT washed out and the intensity model does not apply.
    """
    system = SystemConfig.model_validate(
        _base_system(source={"preset": "sld_dl60", "overrides": {"linewidth_fwhm_m": 20.0e-9}})
    )
    with pytest.raises(ValueError, match="wash-out"):
        system.resolve(PresetStore(config_dir))


def test_resolve_washout_guard_ignores_route_1(config_dir: Path) -> None:
    """Route 1 (AR endface) has no wash-out requirement: the guard is silent."""
    system = SystemConfig.model_validate(
        _base_system(
            route=1,
            source={"preset": "sld_dl60", "overrides": {"linewidth_fwhm_m": 20.0e-9}},
        )
    )
    variant = system.resolve(PresetStore(config_dir))
    assert variant.route == 1


def test_resolve_without_linewidth_is_unchanged(config_dir: Path) -> None:
    """Pre-M-01 compositions (no linewidth) resolve exactly as before."""
    system = SystemConfig.model_validate(_base_system())
    variant = system.resolve(PresetStore(config_dir))
    assert variant.source.rin_db_hz == -126.0
