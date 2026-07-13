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
    # Quantized to 6 significant digits at resolve time (R-51).
    assert variant.source.rin_db_hz == pytest.approx(expected, rel=1e-5)
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


# --------------------------------------------------------------------------- #
# M-10: lineshape family -- visibility laws, wash-out tightening, RIN form
# factors (docs 03 §f' / 07 §1.2; Wiener-Khinchin/Siegert derivations).
# --------------------------------------------------------------------------- #
def _sampled_gaussian_spectrum(
    dlam_m: float, *, span_fwhm: float = 6.0, n: int = 8001
) -> tuple[list[float], list[float]]:
    """Densely sampled Gaussian ``S_lambda`` trace of FWHM ``dlam`` @ 1550 nm."""
    import numpy as np

    c = 299792458.0
    dnu = c * dlam_m / _LAMBDA**2
    nu0 = c / _LAMBDA
    lam = np.linspace(_LAMBDA - span_fwhm * dlam_m, _LAMBDA + span_fwhm * dlam_m, n)
    s_nu = np.exp(-4.0 * math.log(2.0) * ((c / lam - nu0) / dnu) ** 2)
    s_lam = s_nu * c / lam**2  # Jacobian back to the wavelength density
    return lam.tolist(), s_lam.tolist()


@pytest.mark.golden
def test_golden_rin_lineshape_form_factors_doc07_table() -> None:
    """kappa = RIN dnu per lineshape reproduces the doc 07 §1.2 table (derived).

    Rectangular 2 (the R-46 convention, 0 dB); Gaussian ``2 sqrt(2 ln 2/pi)``
    = 1.3286 (doc: 1.33, -1.8 dB); Lorentzian ``2/pi`` = 0.6366 (doc: 0.64,
    -5.0 dB). These are now *derived* from ``RIN(0) = 2 tau_c`` -- pinned to
    the closed forms, not to the code output (18 §5(g)).
    """
    from optivibe.optics.source import RIN_KAPPA_BY_LINESHAPE

    assert RIN_KAPPA_BY_LINESHAPE["rectangular"] == 2.0
    kappa_g = RIN_KAPPA_BY_LINESHAPE["gaussian"]
    kappa_l = RIN_KAPPA_BY_LINESHAPE["lorentzian"]
    assert kappa_g == pytest.approx(2.0 * math.sqrt(2.0 * math.log(2.0) / math.pi), rel=1e-12)
    assert kappa_l == pytest.approx(2.0 / math.pi, rel=1e-12)
    # dB offsets against the convention (doc 07 §1.2: -1.8 / -5.0 dB).
    assert 10.0 * math.log10(kappa_g / 2.0) == pytest.approx(-1.78, abs=0.01)
    assert 10.0 * math.log10(kappa_l / 2.0) == pytest.approx(-4.97, abs=0.01)
    # The rin_ase keyword wires the factors; None keeps the R-46 default.
    dnu = linewidth_nu_hz(_LAMBDA, 60.0e-9)
    assert rin_ase(dnu) == rin_ase(dnu, lineshape="rectangular")
    assert rin_ase(dnu, lineshape="gaussian") == pytest.approx(kappa_g / dnu, rel=1e-12)
    assert rin_ase(dnu, lineshape="lorentzian") == pytest.approx(kappa_l / dnu, rel=1e-12)
    with pytest.raises(ValueError, match="lineshape"):
        rin_ase(dnu, lineshape="measured")  # measured goes through rin_ase_measured


@pytest.mark.golden
def test_golden_lorentzian_visibility_closed_form() -> None:
    """Lorentzian visibility: ``2^{-4A/L_c}`` == ``exp(-2 pi A dlam/lambda^2)``.

    Wiener-Khinchin of a Lorentzian FWHM line gives
    ``|gamma| = exp(-pi dnu |tau|)``; expressed through the *project*
    (Gaussian-convention) ``L_c`` this is exactly ``2^{-4A/L_c}`` (M-10).
    """
    from optivibe.optics.source import fringe_visibility_lorentzian

    dlam = 60.0e-9
    l_c = coherence_length_m(_LAMBDA, dlam)
    for a_um in (0.0, 5.0, 17.669, 20.0, 30.0, 60.0):
        a = a_um * 1e-6
        direct = math.exp(-2.0 * math.pi * a * dlam / _LAMBDA**2)
        assert fringe_visibility_lorentzian(a, l_c) == pytest.approx(direct, rel=1e-12)


@pytest.mark.golden
def test_golden_lineshape_crossover_at_coherence_length() -> None:
    """Same FWHM -> same dnu, different V(A); the laws cross at A = L_c, V = 1/16.

    The heart of M-10: an equal-FWHM Gaussian and Lorentzian share ``dnu``
    (hence the default RIN) but not the coherence envelope. Identity:
    ``V_G(L_c) = V_L(L_c) = 2^{-4} = 1/16`` exactly; below ``L_c`` the
    Lorentzian decays faster, above -- slower (heavy tails).
    """
    from optivibe.optics.source import fringe_visibility_lorentzian

    l_c = coherence_length_m(_LAMBDA, 60.0e-9)
    assert fringe_visibility(l_c, l_c) == pytest.approx(1.0 / 16.0, rel=1e-12)
    assert fringe_visibility_lorentzian(l_c, l_c) == pytest.approx(1.0 / 16.0, rel=1e-12)
    below, above = 0.5 * l_c, 1.5 * l_c
    assert fringe_visibility_lorentzian(below, l_c) < fringe_visibility(below, l_c)
    assert fringe_visibility_lorentzian(above, l_c) > fringe_visibility(above, l_c)


@pytest.mark.golden
@pytest.mark.parametrize("v_max", [0.1, 0.03, 0.01])
def test_golden_washout_lorentzian_coefficient_and_identity(v_max: float) -> None:
    """Lorentzian wash-out: ``A >= ln(1/V)/(4 ln 2) L_c``; identity with Gaussian.

    At the documented ``V = 0.03``: ``A_min = 1.2647 L_c`` -- 12.5 % tighter
    than the Gaussian ``1.1246 L_c`` (R-46), because 0.03 < 1/16 lies past
    the ``A = L_c`` crossing. Shape-independent identity at *any* threshold:
    ``A_min^L L_c = (A_min^G)^2`` (both minima are exact inversions of their
    visibility laws).
    """
    from optivibe.optics.source import (
        fringe_visibility_lorentzian,
        min_gap_for_washout_lorentzian_m,
    )

    l_c = coherence_length_m(_LAMBDA, 60.0e-9)
    a_l = min_gap_for_washout_lorentzian_m(l_c, visibility_max=v_max)
    a_g = min_gap_for_washout_m(l_c, visibility_max=v_max)
    assert a_l == pytest.approx(l_c * math.log(1.0 / v_max) / (4.0 * math.log(2.0)), rel=1e-12)
    assert a_l * l_c == pytest.approx(a_g**2, rel=1e-12)
    # The inversion is exact: V at the minimum gap equals the threshold.
    assert fringe_visibility_lorentzian(a_l, l_c) == pytest.approx(v_max, rel=1e-12)
    if v_max == 0.03:
        assert a_l / l_c == pytest.approx(1.2647, abs=1e-4)
        assert a_l / a_g == pytest.approx(1.1246, abs=1e-4)  # +12.5 % tightening


@pytest.mark.golden
def test_golden_lorentzian_worst_case_gap_doc03_design_point() -> None:
    """dlam = 60 nm @ worst-case A = 20 um: Gaussian washed out, Lorentzian NOT.

    Doc 03 §f' design point (nominal A = 30 um, alignment tolerance -10 um):
    Gaussian V(20 um) = 0.029 < 0.03 (marginally OK, the R-46 result), but
    Lorentzian tails give V(20 um) = 0.043 >= 0.03 -- the practical output of
    M-10: heavy-tailed sources need dlam >= 67 nm to cover the same worst
    case. Resolution: measure the actual spectrum (D-03 / E1-P6).
    """
    from optivibe.optics.source import fringe_visibility_lorentzian

    l_c = coherence_length_m(_LAMBDA, 60.0e-9)
    a_worst = 20.0e-6
    v_g = fringe_visibility(a_worst, l_c)
    v_l = fringe_visibility_lorentzian(a_worst, l_c)
    assert v_g == pytest.approx(0.0287, abs=5e-4)
    assert v_g < WASHOUT_VISIBILITY_MAX
    assert v_l == pytest.approx(0.0434, abs=5e-4)
    assert v_l >= WASHOUT_VISIBILITY_MAX
    # dlam that covers the Lorentzian worst case at 20 um: >= 67 nm.
    dlam_needed = math.log(1.0 / WASHOUT_VISIBILITY_MAX) * _LAMBDA**2 / (2.0 * math.pi * a_worst)
    assert dlam_needed * 1e9 == pytest.approx(67.0, abs=0.1)


def test_limit_lorentzian_visibility_bounds_and_guards() -> None:
    """Limits: A = 0 -> V = 1; A -> inf -> V -> 0; guard rails raise."""
    from optivibe.optics.source import (
        fringe_visibility_lorentzian,
        min_gap_for_washout_lorentzian_m,
    )

    l_c = 17.669e-6
    assert fringe_visibility_lorentzian(0.0, l_c) == 1.0
    assert fringe_visibility_lorentzian(100.0 * l_c, l_c) < 1e-100
    with pytest.raises(ValueError):
        fringe_visibility_lorentzian(-1.0e-6, l_c)
    with pytest.raises(ValueError):
        fringe_visibility_lorentzian(1.0e-6, 0.0)
    with pytest.raises(ValueError):
        min_gap_for_washout_lorentzian_m(0.0)
    with pytest.raises(ValueError):
        min_gap_for_washout_lorentzian_m(l_c, visibility_max=1.5)


# --------------------------------------------------------------------------- #
# M-10: measured spectra (the D-03 / E1-P6 bridge).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_measured_gaussian_matches_analytic() -> None:
    """A sampled Gaussian trace reproduces the closed forms: V(A), dnu_eff, RIN.

    Convergence pin: the numeric Wiener-Khinchin quadrature on a dense grid
    must land on the analytic Gaussian law ``V = 2^{-(2A/L_c)^2}``, the
    noise-equivalent linewidth on ``dnu_eff = sqrt(pi/(2 ln 2)) dnu =
    1.5053 dnu`` (Derickson), and the RIN floor on the Gaussian form factor.
    """
    from optivibe.optics.source import (
        effective_linewidth_measured_hz,
        fringe_visibility_measured,
        rin_ase_measured_db_hz,
    )

    dlam = 60.0e-9
    lam, psd = _sampled_gaussian_spectrum(dlam)
    l_c = coherence_length_m(_LAMBDA, dlam)
    dnu = linewidth_nu_hz(_LAMBDA, dlam)
    for a_um in (0.0, 10.0, 20.0, 30.0):
        a = a_um * 1e-6
        assert fringe_visibility_measured(a, lam, psd) == pytest.approx(
            fringe_visibility(a, l_c), abs=2e-5
        )
    assert effective_linewidth_measured_hz(lam, psd) == pytest.approx(
        math.sqrt(math.pi / (2.0 * math.log(2.0))) * dnu, rel=1e-3
    )
    assert rin_ase_measured_db_hz(lam, psd) == pytest.approx(
        rin_ase_db_hz(dnu, lineshape="gaussian"), abs=0.01
    )


@pytest.mark.golden
def test_golden_measured_lorentzian_matches_analytic() -> None:
    """A sampled Lorentzian trace reproduces its heavy-tail closed forms.

    Wide span (+-150 FWHM) bounds the truncated-tail bias; tolerances follow
    the clipped fraction ``~dnu/(pi^2 span)``.
    """
    import numpy as np

    from optivibe.optics.source import (
        coherence_time_measured_s,
        fringe_visibility_lorentzian,
        fringe_visibility_measured,
    )

    c = 299792458.0
    dlam = 20.0e-9
    dnu = c * dlam / _LAMBDA**2
    nu0 = c / _LAMBDA
    # Span limited by nu > 0 (lambda = c/nu must stay positive); +-60 FWHM
    # keeps the Lorentzian tail well inside a positive-frequency window.
    nu = np.linspace(nu0 - 60.0 * dnu, nu0 + 60.0 * dnu, 200001)
    lam = (c / nu)[::-1]
    s_nu = (dnu / (2.0 * math.pi)) / ((nu - nu0) ** 2 + (dnu / 2.0) ** 2)
    s_lam = (s_nu * c / (c / nu) ** 2)[::-1]
    l_c = coherence_length_m(_LAMBDA, dlam)
    # Truncating the tail at +-60 FWHM drops ~2/(pi*60) ~ 1 % of the area,
    # which renormalizes V slightly high; the pin allows for that clipped bias.
    for a_um in (30.0, 60.0):
        a = a_um * 1e-6
        assert fringe_visibility_measured(a, lam.tolist(), s_lam.tolist()) == pytest.approx(
            fringe_visibility_lorentzian(a, l_c), abs=1.5e-2
        )
    # tau_c = 1/(pi dnu) (doc 07 §1.2 table): the squared integrand converges
    # much faster than the transform (the tail carries no tau_c weight).
    assert coherence_time_measured_s(lam.tolist(), s_lam.tolist()) == pytest.approx(
        1.0 / (math.pi * dnu), rel=5e-3
    )


@pytest.mark.golden
def test_golden_measured_two_line_coherence_revival() -> None:
    """Two equal lines split by delta-nu revive coherence at OPD = c/delta-nu.

    ``gamma = cos(pi delta-nu tau) x gamma_envelope``: a null at
    ``OPD = c/(2 delta-nu)`` and a *full revival* to the single-line envelope
    at ``OPD = c/delta-nu``. This is the route-2 risk of structured/rippled
    SLD spectra that only the measured mode captures -- prediction E1-P6
    (doc 19 §3.2) exists to rule it out on the real device.
    """
    import numpy as np

    from optivibe.optics.source import fringe_visibility_measured

    c = 299792458.0
    dlam_line = 10.0e-9
    dnu_line = c * dlam_line / _LAMBDA**2
    delta_nu = 8.0 * dnu_line  # two well-separated lines
    nu0 = c / _LAMBDA
    nu = np.linspace(nu0 - 10.0 * delta_nu, nu0 + 10.0 * delta_nu, 120001)
    line = lambda center: np.exp(-4.0 * math.log(2.0) * ((nu - center) / dnu_line) ** 2)  # noqa: E731
    s_nu = line(nu0 - delta_nu / 2.0) + line(nu0 + delta_nu / 2.0)
    lam = (c / nu)[::-1]
    s_lam = (s_nu * nu**2 / c)[::-1]  # S_lambda = S_nu c/lambda^2 = S_nu nu^2/c
    gap_null = c / (2.0 * delta_nu) / 2.0  # OPD = 2A = c/(2 delta-nu)
    gap_revival = c / delta_nu / 2.0  # OPD = 2A = c/delta-nu
    l_c_line = coherence_length_m(_LAMBDA, dlam_line)
    envelope_at_revival = fringe_visibility(gap_revival, l_c_line)
    v_null = fringe_visibility_measured(gap_null, lam.tolist(), s_lam.tolist())
    v_revival = fringe_visibility_measured(gap_revival, lam.tolist(), s_lam.tolist())
    assert v_null < 5e-3  # cosine null
    assert v_revival == pytest.approx(envelope_at_revival, rel=2e-2)  # full revival
    assert v_revival > 10.0 * v_null


def test_limit_measured_visibility_and_table_guards() -> None:
    """Limits and guard rails of the measured-spectrum path."""
    from optivibe.optics.source import fringe_visibility_measured

    # Dense enough that the quadrature resolves the exp(-i 2 pi nu tau)
    # oscillation at the gaps probed below (Nyquist: dnu tau < 1/2).
    lam, psd = _sampled_gaussian_spectrum(60.0e-9, n=4001)
    assert fringe_visibility_measured(0.0, lam, psd) == pytest.approx(1.0, rel=1e-12)
    # A well past wash-out (L_c = 17.7 um): the fringe is gone.
    assert fringe_visibility_measured(60.0e-6, lam, psd) < 1e-6
    with pytest.raises(ValueError, match="non-negative"):
        fringe_visibility_measured(-1.0e-6, lam, psd)
    with pytest.raises(ValueError, match="increasing"):
        fringe_visibility_measured(0.0, list(reversed(lam)), psd)
    with pytest.raises(ValueError, match="at least 4"):
        fringe_visibility_measured(0.0, lam[:3], psd[:3])
    with pytest.raises(ValueError, match="equal length"):
        fringe_visibility_measured(0.0, lam, psd[:-1])
    with pytest.raises(ValueError, match="non-negative"):
        fringe_visibility_measured(0.0, lam, [-1.0] * len(lam))
    with pytest.raises(ValueError, match="power"):
        fringe_visibility_measured(0.0, lam, [0.0] * len(lam))
    with pytest.raises(ValueError, match="positive"):
        fringe_visibility_measured(0.0, [-1.0e-6, *lam[1:]], psd)
    with pytest.raises(ValueError, match="finite"):
        fringe_visibility_measured(0.0, lam, [math.nan] * len(lam))


# --------------------------------------------------------------------------- #
# M-10 config wiring: lineshape pairing rules and resolve-time behaviour.
# --------------------------------------------------------------------------- #
def test_source_config_lineshape_pairing_rules() -> None:
    """The M-10 validator enforces the lineshape/input pairings loudly (10 §7)."""
    lam, psd = _sampled_gaussian_spectrum(60.0e-9, n=101)
    # Analytic shape without a linewidth: nothing to shape.
    with pytest.raises(ValueError, match="requires linewidth_fwhm_m"):
        SourceConfig(wavelength_m=_LAMBDA, power_w=0.016, rin_db_hz=-126.0, lineshape="lorentzian")
    # A table without lineshape='measured' is dead configuration.
    with pytest.raises(ValueError, match="lineshape = 'measured'"):
        SourceConfig(
            wavelength_m=_LAMBDA,
            power_w=0.016,
            rin_db_hz=-126.0,
            spectrum_wavelength_m=lam,
            spectrum_psd=psd,
        )
    # measured without the table.
    with pytest.raises(ValueError, match="requires both"):
        SourceConfig(wavelength_m=_LAMBDA, power_w=0.016, rin_db_hz=-126.0, lineshape="measured")
    # measured + linewidth: two sources of truth.
    with pytest.raises(ValueError, match="single source of truth"):
        SourceConfig(
            wavelength_m=_LAMBDA,
            power_w=0.016,
            lineshape="measured",
            linewidth_fwhm_m=60.0e-9,
            spectrum_wavelength_m=lam,
            spectrum_psd=psd,
        )
    # nm-vs-m unit slip: the centre wavelength must sit inside the table span.
    with pytest.raises(ValueError, match="units"):
        SourceConfig(
            wavelength_m=_LAMBDA,
            power_w=0.016,
            lineshape="measured",
            spectrum_wavelength_m=[1400.0, 1500.0, 1600.0, 1700.0],
            spectrum_psd=[1.0, 2.0, 2.0, 1.0],
        )
    # A DFB stays DFB: no RIN derivation even from a measured spectrum.
    with pytest.raises(ValueError, match="DFB"):
        SourceConfig(
            source_kind="DFB",
            wavelength_m=_LAMBDA,
            power_w=0.1,
            lineshape="measured",
            spectrum_wavelength_m=lam,
            spectrum_psd=psd,
        )


@pytest.mark.golden
def test_resolve_default_lineshape_keeps_r46_numbers(config_dir: Path) -> None:
    """lineshape omitted -> the R-46 effective-scalar behaviour, bit-identical.

    The contract gate of M-10: without an explicit lineshape the derived RIN
    stays the rectangular-equivalent floor ``2/dnu`` (sld_dl60 -> -125.733
    quantized, R-51) and the wash-out check keeps the Gaussian law. The A-D
    resolved-variant golden (18 §5) is asserted unchanged elsewhere
    (test_config_variants); this pins the derived number itself.
    """
    system = SystemConfig.model_validate(_base_system(source={"preset": "sld_dl60"}))
    variant = system.resolve(PresetStore(config_dir))
    assert variant.source.rin_db_hz == pytest.approx(
        rin_ase_db_hz(linewidth_nu_hz(_LAMBDA, 60.0e-9)), rel=1e-5
    )


@pytest.mark.golden
def test_resolve_gaussian_lineshape_lowers_rin_by_form_factor(config_dir: Path) -> None:
    """Explicit 'gaussian' opts into the derived form factor: -1.78 dB vs default."""
    base = SystemConfig.model_validate(_base_system(source={"preset": "sld_dl60"}))
    shaped = SystemConfig.model_validate(
        _base_system(source={"preset": "sld_dl60", "overrides": {"lineshape": "gaussian"}})
    )
    store = PresetStore(config_dir)
    rin_default = base.resolve(store).source.rin_db_hz
    rin_gauss = shaped.resolve(store).source.rin_db_hz
    assert rin_gauss is not None and rin_default is not None
    assert rin_gauss - rin_default == pytest.approx(
        10.0 * math.log10(math.sqrt(2.0 * math.log(2.0) / math.pi)), abs=2e-3
    )


def test_resolve_lorentzian_washout_gate_tightens(config_dir: Path) -> None:
    """dlam = 60 nm at A = 20 um: gaussian resolves, lorentzian is rejected.

    The doc 03 §f' worst-case design point made executable: the same gap that
    marginally passes under the Gaussian law (V = 0.029) fails under
    Lorentzian tails (V = 0.043) -- the practical M-10 tightening.
    """

    def _system(lineshape: str | None) -> SystemConfig:
        overrides: dict[str, object] = {} if lineshape is None else {"lineshape": lineshape}
        return SystemConfig.model_validate(
            _base_system(
                source={"preset": "sld_dl60", "overrides": overrides},
                reflector={"preset": "cyl_rc31", "overrides": {"gap_m": 20.0e-6}},
            )
        )

    store = PresetStore(config_dir)
    assert _system(None).resolve(store).source.rin_db_hz is not None  # gaussian law: passes
    assert _system("gaussian").resolve(store) is not None
    with pytest.raises(ValueError, match="wash-out"):
        _system("lorentzian").resolve(store)


def test_resolve_measured_spectrum_wires_rin_and_washout(config_dir: Path) -> None:
    """A measured Gaussian table resolves with RIN = 2 tau_c; a narrow one fails.

    The D-03 bridge end-to-end: after phase 0 the OSA trace is pasted into the
    composition and both the RIN floor and the wash-out check run on the real
    spectrum instead of an idealization (E1-P6).
    """
    lam_wide, psd_wide = _sampled_gaussian_spectrum(60.0e-9, n=2001)
    overrides = {
        "lineshape": "measured",
        "linewidth_fwhm_m": None,
        "spectrum_wavelength_m": lam_wide,
        "spectrum_psd": psd_wide,
    }
    system = SystemConfig.model_validate(
        _base_system(source={"preset": "sld_dl60", "overrides": overrides})
    )
    variant = system.resolve(PresetStore(config_dir))
    assert variant.source.rin_db_hz == pytest.approx(
        rin_ase_db_hz(linewidth_nu_hz(_LAMBDA, 60.0e-9), lineshape="gaussian"), abs=0.02
    )
    # A narrow (20 nm) measured spectrum at the platform gap 31 um: V ~ 0.39.
    lam_narrow, psd_narrow = _sampled_gaussian_spectrum(20.0e-9, n=2001)
    narrow = SystemConfig.model_validate(
        _base_system(
            source={
                "preset": "sld_dl60",
                "overrides": {
                    **overrides,
                    "spectrum_wavelength_m": lam_narrow,
                    "spectrum_psd": psd_narrow,
                },
            }
        )
    )
    with pytest.raises(ValueError, match="wash-out"):
        narrow.resolve(PresetStore(config_dir))
