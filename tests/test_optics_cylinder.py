"""S3 optics tests: golden references (docs 03/04/05), limits (11 §7),
integration with the modal mechanics, and property invariants (10 §10).

Golden references follow R-59 (doc 03 v1.6, S-30): the returned beam of a convex
mirror is mismatched to the fiber mode, and the full overlap gives an exact
Gaussian in ``dx_eff`` with ``sigma_eff = (w0/2) sqrt(1 + ((R_c + A)/zR)^2)``.
Every reference is a closed formula of doc 03 §d/§e evaluated *in the test*
(:func:`_closed_forms`) or a value tabulated in doc 03 §d'' -- never the model
output (plan 18 §5(g)). The pre-R-59 references (eta0 ~ 0.25, slope ~ -1.5e5 1/m,
"normalized slope ~ 1/R_c", bias 2.573 um) belonged to the revoked matched-beam
multiplier.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from optivibe.core.config import load_constants, load_variant
from optivibe.core.config.models import Constants
from optivibe.core.types import Excitation, FloatArray, TipState
from optivibe.mechanics import CantileverModel, ModalFrequencyMechanics
from optivibe.optics import (
    OPTICS_REGISTRY,
    CylinderOptics,
    CylinderOpticsModel,
    GaussianBeam,
    eta_parallel_curved,
    eta_parallel_flat,
)
from optivibe.pipeline.orchestrator import run_scenario
from optivibe.viz.optics import plot_eta_map, plot_eta_vs_dx, plot_eta_vs_dy

VARIANTS = ("A", "B", "C", "D")
COUPLING = 1.377  # theta L / delta (doc 01 §0); cross-checked vs constants below


@pytest.fixture(scope="module")
def constants(config_dir: Path) -> Constants:
    return load_constants(config_dir / "constants.yaml")


@pytest.fixture(scope="module")
def model_a(config_dir: Path) -> CylinderOpticsModel:
    """Documented optical platform: R_c = 62.5 um, A = 31 um, bias 2 um."""
    return CylinderOpticsModel.from_config(load_variant("A", config_dir=config_dir))


@pytest.fixture(scope="module")
def model_b(config_dir: Path) -> CylinderOpticsModel:
    """Variant B platform: R_c = 31 um, configured bias 0.992 um.

    The 0.992 um bias was chosen as "scale-equivalent" (sigma ~ R_c); R-59 voids
    that premise (eta0/eta_peak = 0.938 at this bias, doc 03 §d''). The config is
    unchanged here -- its revision is backlog M-25.
    """
    return CylinderOpticsModel.from_config(load_variant("B", config_dir=config_dir))


def _closed_forms(model: CylinderOpticsModel) -> tuple[float, float]:
    """``(eta_peak, sigma_eff)`` from the closed forms of doc 03 §d (R-59).

    ``eta_peak = eta_par^x eta_par^y`` with
    ``eta_par^x = R_c zR / sqrt((A^2 + zR^2)((R_c + A)^2 + zR^2))``,
    ``eta_par^y = 1/sqrt(1 + (A/zR)^2)`` and
    ``sigma_eff = (w0/2) sqrt(1 + ((R_c + A)/zR)^2)``. Only the model's *inputs*
    (lambda, w0, A, R_c) are read; nothing is taken from its outputs.
    """
    w0 = model.beam.waist_radius_m
    z_r = math.pi * w0**2 / model.beam.wavelength_m
    gap, radius = model.gap_m, model.radius_of_curvature_m
    eta_par_x = radius * z_r / math.sqrt((gap**2 + z_r**2) * ((radius + gap) ** 2 + z_r**2))
    eta_par_y = 1.0 / math.sqrt(1.0 + (gap / z_r) ** 2)
    sigma_eff = 0.5 * w0 * math.sqrt(1.0 + ((radius + gap) / z_r) ** 2)
    return eta_par_x * eta_par_y, sigma_eff


def _sine_excitation(
    freq_hz: float, amplitude_g: float, fs: float, duration_s: float, g0: float, axis: str = "x"
) -> Excitation:
    t = np.arange(round(duration_s * fs)) / fs
    signal = amplitude_g * g0 * np.sin(2.0 * np.pi * freq_hz * t)
    zeros = np.zeros_like(signal)
    channels = {"x": zeros.copy(), "y": zeros.copy(), "z": zeros.copy()}
    channels[axis] = signal
    return Excitation(a_x=channels["x"], a_y=channels["y"], a_z=channels["z"], fs=fs)


def _thd(signal: FloatArray, fs: float, fundamental_hz: float, n_harmonics: int = 5) -> float:
    """Total harmonic distortion of a periodic signal via the rFFT bins."""
    n = signal.size
    spectrum = np.abs(np.fft.rfft(signal - np.mean(signal)))
    df = fs / n
    fundamental_bin = round(fundamental_hz / df)
    fund = spectrum[fundamental_bin]
    harmonics = [
        spectrum[k * fundamental_bin]
        for k in range(2, n_harmonics + 2)
        if k * fundamental_bin < spectrum.size
    ]
    return float(np.sqrt(np.sum(np.square(harmonics))) / fund)


# --------------------------------------------------------------------------- #
# Golden: working point and slopes against the documented references (11 §7).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_working_point_platform_a(model_a: CylinderOpticsModel) -> None:
    """eta_peak and eta0 of variant A against doc 03 §d / §d'' (R-59).

    ``eta0 = eta_peak exp(-(Delta x0/sigma_eff)^2)``: 0.4369 and 0.3755 at the
    documented platform (doc 03 §d''; was 0.2396 with the revoked multiplier).
    """
    peak, sigma = _closed_forms(model_a)
    eta0 = peak * math.exp(-((model_a.bias_m / sigma) ** 2))
    assert model_a.eta_peak() == pytest.approx(peak, rel=1.0e-12)
    assert model_a.eta_working_point() == pytest.approx(eta0, rel=1.0e-12)
    assert model_a.eta_peak() == pytest.approx(0.4369, abs=5.0e-5)
    assert model_a.eta_working_point() == pytest.approx(0.3755, abs=5.0e-5)


@pytest.mark.golden
def test_golden_bare_slope_platform_a(model_a: CylinderOpticsModel) -> None:
    """Bare slope ``-2 eta0 Delta x0 / sigma_eff^2 = -5.683e4 1/m`` (docs 03 §e/§d'', 04 §4)."""
    peak, sigma = _closed_forms(model_a)
    bias = model_a.bias_m
    expected = -2.0 * peak * math.exp(-((bias / sigma) ** 2)) * bias / sigma**2
    assert model_a.slope_dx() == pytest.approx(expected, rel=1.0e-10)
    assert model_a.slope_dx() == pytest.approx(-5.683e4, rel=1.0e-4)


@pytest.mark.golden
def test_golden_effective_slope_platform_a(model_a: CylinderOpticsModel) -> None:
    """Effective slope with the tilt multiplier ``1 + 1.377 (R_c + A)/L`` (doc 03 §d'').

    Platform A: -5.829e4 1/m at its own L = 5 mm, and the doc 03 §d'' values at
    other lengths, -6.205e4 (1.4 mm) and -5.926e4 1/m (3 mm). Before R-59 the
    references were -1.6e5 / -1.56e5 1/m (doc 05 §1 "-0.16 1/um", revoked).
    """
    for length, reference in ((5.0e-3, -5.829e4), (1.4e-3, -6.205e4), (3.0e-3, -5.926e4)):
        multiplier = 1.0 + COUPLING * (model_a.radius_of_curvature_m + model_a.gap_m) / length
        slope = model_a.effective_slope_dx(length, COUPLING)
        assert slope == pytest.approx(model_a.slope_dx() * multiplier, rel=1.0e-12)
        assert slope == pytest.approx(reference, rel=1.0e-4)


@pytest.mark.golden
def test_golden_tilt_multiplier(model_a: CylinderOpticsModel) -> None:
    """Tilt enhancement +4 % at L=3 mm, +9 % at L=1.4 mm (doc 05 §1)."""
    assert model_a.tilt_multiplier(3.0e-3, COUPLING) - 1.0 == pytest.approx(0.042, rel=0.1)
    assert model_a.tilt_multiplier(1.4e-3, COUPLING) - 1.0 == pytest.approx(0.091, rel=0.1)


@pytest.mark.golden
def test_golden_slope_scaling_with_radius(model_a: CylinderOpticsModel) -> None:
    """At a fixed eta0/eta_peak the slope peaks at ``R_c* = sqrt(A^2 + zR^2)`` (R-59).

    Doc 03 §e (R-59): at a fixed working-point ratio (rule R-40) the absolute
    slope is proportional to ``eta_par^x / sigma_eff ~ R_c / ((R_c + A)^2 + zR^2)``.
    Halving R_c (62.5 -> 31 um) therefore *lowers* it x0.851, and the optimum is
    ``R_c* ~ 63 um`` at A = 31 um. Revoked: the first-order "~1/R_c" scaling of
    R-42 behind the former -0.32 1/um reference and the x1.31 gain.
    """
    w0 = model_a.beam.waist_radius_m
    z_r = math.pi * w0**2 / model_a.beam.wavelength_m
    gap = model_a.gap_m

    def slope_at_rule(radius: float) -> float:
        untuned = replace(model_a, radius_of_curvature_m=radius, bias_m=0.0)
        tuned = replace(untuned, bias_m=untuned.bias_for_eta_ratio(0.37))
        return abs(tuned.slope_dx())

    def law(radius: float) -> float:
        return radius / ((radius + gap) ** 2 + z_r**2)

    ratio = slope_at_rule(31.0e-6) / slope_at_rule(62.5e-6)
    assert ratio == pytest.approx(law(31.0e-6) / law(62.5e-6), rel=1.0e-10)
    assert ratio == pytest.approx(0.851, abs=5.0e-4)
    r_star = math.sqrt(gap**2 + z_r**2)
    assert r_star == pytest.approx(63.0e-6, abs=0.5e-6)
    best = slope_at_rule(r_star)
    assert best > slope_at_rule(0.8 * r_star)
    assert best > slope_at_rule(1.25 * r_star)


@pytest.mark.golden
def test_golden_normalized_slope_ratio_is_sigma_ratio(
    model_a: CylinderOpticsModel, model_b: CylinderOpticsModel
) -> None:
    """At equal ``t = Delta x0/sigma_eff`` the normalized slope scales as ``1/sigma_eff`` (R-59).

    ``(1/eta0) d eta/d dx = -2 t / sigma_eff``. R-42 claimed it scales *exactly*
    as 1/R_c at the "scale-equivalent" bias ``Delta x0 ~ R_c`` (sigma ~ R_c);
    R-59 revokes that: sigma_eff is not proportional to R_c, and the ratio for
    R_c 31 vs 62.5 um is ``sigma_eff(62.5)/sigma_eff(31) = 1.3097`` (not 2.016).
    The bias of the R_c = 31 um platform with the same eta0/eta_peak as the A
    config is 1.5271 um, not 0.992 um (doc 03 §d'').
    """
    t = model_a.bias_m / model_a.sigma_m
    same_t = replace(model_b, bias_m=t * model_b.sigma_m)
    ratio = (same_t.slope_dx() / same_t.eta_working_point()) / (
        model_a.slope_dx() / model_a.eta_working_point()
    )
    _, sigma_a = _closed_forms(model_a)
    _, sigma_b = _closed_forms(model_b)
    assert ratio == pytest.approx(sigma_a / sigma_b, rel=1.0e-9)
    assert ratio == pytest.approx(1.3097, abs=5.0e-5)
    assert same_t.bias_m == pytest.approx(1.5271e-6, abs=5.0e-11)
    assert abs(ratio - 62.5 / 31.0) > 0.5  # the revoked R-42 statement


@pytest.mark.golden
def test_golden_dy_symmetry_exact(model_a: CylinderOpticsModel) -> None:
    """Pure dy leaves eta exactly unchanged (cylinder symmetry, doc 03 §4)."""
    dy = np.linspace(-5.0e-6, 5.0e-6, 11)
    eta = model_a.eta(dy=dy)
    np.testing.assert_array_equal(eta, np.full_like(eta, model_a.eta_working_point()))


@pytest.mark.golden
def test_golden_cross_axis_residual_quadratic(model_a: CylinderOpticsModel) -> None:
    """The theta_x-mediated cross residual scales x4 when dy doubles (doc 04 §5)."""
    length = 3.0e-3
    tilt = COUPLING / length
    eta0 = model_a.eta_working_point()

    def residual(dy_amp: float) -> float:
        return abs(float(model_a.eta(dy=dy_amp, theta_x=tilt * dy_amp).item()) - eta0)

    small, double = residual(5.0e-9), residual(10.0e-9)
    assert double / small == pytest.approx(4.0, rel=5.0e-3)


@pytest.mark.golden
def test_golden_dz_slope_sign_and_magnitude(model_a: CylinderOpticsModel) -> None:
    """d eta / d dz = -7.958e3 1/m (doc 03 §d''/§e, R-41 + R-59); the dz channel is negligible.

    The three-term structure (R-41) is pinned by V19-L2-07 in
    ``test_model_vv_l2.py``; here the total derivative and the channel ratio:
    at 50 g on variant A (L = 5 mm) the dz contribution to Delta eta is
    ``2.2e-5`` of the target one (doc 04 §4, R-59 rider) -- negligible.
    """
    h = 1.0e-10
    slope_z = float(((model_a.eta(dz=h) - model_a.eta(dz=-h)) / (2.0 * h)).item())
    assert slope_z < 0.0
    assert slope_z == pytest.approx(-7.958e3, rel=1.0e-3)
    # Channel comparison at 50 g on variant A geometry (L = 5 mm):
    constants = load_constants(Path(__file__).resolve().parent.parent / "configs/constants.yaml")
    variant = load_variant("A")
    mech = CantileverModel.from_config(constants, variant)
    accel = 50.0 * constants.universal.g0_m_s2
    dx_amp = mech.h_lat_qs * accel
    dz_amp = mech.axial_compliance * accel
    ratio = abs(slope_z * dz_amp) / abs(model_a.slope_dx() * dx_amp)
    assert ratio == pytest.approx(2.2e-5, abs=5.0e-7)


@pytest.mark.golden
def test_golden_anisotropy(model_a: CylinderOpticsModel, constants: Constants) -> None:
    """Angular-map anisotropy L/(1.377 R_c) ~ 35x at L = 3 mm (doc 04 §6).

    After R-59 this is the ratio of the angular *maps*, not of the sensitivities
    (doc 03 §e, 04 §6 rider); the choice of tolerance metric is backlog M-25, so
    the helper and this pin are unchanged.
    """
    coupling = constants.tilt_displacement_coupling_per_l
    assert coupling == COUPLING
    assert model_a.anisotropy(3.0e-3, coupling) == pytest.approx(
        3.0e-3 / (coupling * 62.5e-6), rel=0.0
    )
    assert model_a.anisotropy(3.0e-3, coupling) == pytest.approx(34.86, rel=1.0e-3)


@pytest.mark.golden
def test_golden_numeric_matches_analytic_slope(model_a: CylinderOpticsModel) -> None:
    """Central-difference slope matches the closed form to 1e-6 (doc 04 §4)."""
    h = 1.0e-10
    numeric = float(((model_a.eta(dx=h) - model_a.eta(dx=-h)) / (2.0 * h)).item())
    assert numeric == pytest.approx(model_a.slope_dx(), rel=1.0e-6)


@pytest.mark.golden
def test_golden_bias_rule_helper(
    model_a: CylinderOpticsModel, model_b: CylinderOpticsModel
) -> None:
    """bias_for_eta_ratio(0.37) = sigma_eff sqrt(-ln 0.37): 5.1267 / 3.9144 um (R-40, R-59).

    The rule eta0 = 0.37 eta_peak (doc 08 R-40/O-05) holds in dimensionless form;
    its absolute values follow sigma_eff (doc 03 §d''; were 2.5734 / 1.2764 um).
    """
    for model, reference in ((model_a, 5.1267e-6), (model_b, 3.9144e-6)):
        bias = model.bias_for_eta_ratio(0.37)
        _, sigma = _closed_forms(model)
        assert bias == pytest.approx(sigma * math.sqrt(-math.log(0.37)), rel=1.0e-12)
        assert bias == pytest.approx(reference, abs=5.0e-11)
        tuned = replace(model, bias_m=bias)
        assert tuned.eta_working_point() / tuned.eta_peak() == pytest.approx(0.37, rel=1.0e-9)


# --------------------------------------------------------------------------- #
# Limits and invariants (11 §7; 10 §10).
# --------------------------------------------------------------------------- #
def test_limit_zero_bias_zero_slope(model_a: CylinderOpticsModel) -> None:
    """Delta x0 -> 0 puts the working point on the peak: slope -> 0 exactly."""
    centered = CylinderOpticsModel(
        beam=model_a.beam,
        gap_m=model_a.gap_m,
        radius_of_curvature_m=model_a.radius_of_curvature_m,
        bias_m=0.0,
    )
    assert centered.slope_dx() == 0.0
    assert centered.eta_working_point() == pytest.approx(centered.eta_peak(), rel=0.0)


def test_limit_curved_reproduces_flat(model_a: CylinderOpticsModel) -> None:
    """R_c -> inf: the curved-plane overlap tends to the flat-mirror form."""
    beam = model_a.beam
    gap = np.linspace(5.0e-6, 60.0e-6, 12)
    curved = eta_parallel_curved(beam, gap, 1.0e3)
    flat = eta_parallel_flat(beam, gap)
    np.testing.assert_allclose(curved, flat, rtol=1.0e-6)


def test_limit_zero_gap(model_a: CylinderOpticsModel) -> None:
    """g -> 0: flat mirror re-couples perfectly; a convex mirror does not.

    Even at zero gap the convex mirror imprints curvature 2/R_c on the returned
    wavefront, so the curved-plane overlap stays < 1: ``R_c / sqrt(R_c^2 + zR^2)``
    = 0.752 at R_c = 62.5 um (doc 03 §d closed form at A = 0); it tends to 1 only
    in the additional limit R_c -> inf (doc 03 §g).
    """
    beam = model_a.beam
    assert float(eta_parallel_flat(beam, 0.0).item()) == pytest.approx(1.0, abs=1.0e-12)
    curved_finite = float(eta_parallel_curved(beam, 0.0, 62.5e-6).item())
    z_r = math.pi * beam.waist_radius_m**2 / beam.wavelength_m
    assert curved_finite == pytest.approx(62.5e-6 / math.hypot(62.5e-6, z_r), rel=1.0e-12)
    assert curved_finite == pytest.approx(0.752, abs=5.0e-4)
    assert float(eta_parallel_curved(beam, 0.0, 1.0e3).item()) == pytest.approx(1.0, rel=1.0e-9)


@settings(max_examples=200, deadline=None)
@given(
    dx=st.floats(-5.0e-6, 5.0e-6),
    dy=st.floats(-5.0e-6, 5.0e-6),
    dz=st.floats(-5.0e-6, 5.0e-6),
    theta_x=st.floats(-5.0e-3, 5.0e-3),
    theta_y=st.floats(-5.0e-3, 5.0e-3),
)
def test_property_eta_bounded(
    dx: float, dy: float, dz: float, theta_x: float, theta_y: float
) -> None:
    """0 <= eta <= 1 for any physically reasonable tip state (10 §10)."""
    model = CylinderOpticsModel(
        beam=GaussianBeam(wavelength_m=1550.0e-9, waist_radius_m=5.2e-6),
        gap_m=31.0e-6,
        radius_of_curvature_m=62.5e-6,
        bias_m=2.0e-6,
    )
    eta = float(model.eta(dx=dx, dy=dy, dz=dz, theta_x=theta_x, theta_y=theta_y).item())
    assert 0.0 <= eta <= 1.0


def test_from_config_rejects_non_cylinder(config_dir: Path) -> None:
    """The cylinder stage refuses other reflector shapes loudly (10 §7)."""
    variant = load_variant("A", config_dir=config_dir)
    bad = variant.model_copy(
        update={"reflector": variant.reflector.model_copy(update={"shape": "sphere"})}
    )
    with pytest.raises(ValueError, match="cylinder"):
        CylinderOpticsModel.from_config(bad)


def test_from_config_validity_guards(config_dir: Path) -> None:
    """Paraxial guards of doc 03 §6: R_c >= 5 w0 and w(A) <= R_c/3."""
    variant = load_variant("A", config_dir=config_dir)
    small_rc = variant.model_copy(
        update={
            "reflector": variant.reflector.model_copy(update={"radius_of_curvature_m": 20.0e-6})
        }
    )
    with pytest.raises(ValueError, match="paraxial"):
        CylinderOpticsModel.from_config(small_rc)
    big_gap = variant.model_copy(
        update={"optics": variant.optics.model_copy(update={"gap_m": 300.0e-6})}
    )
    with pytest.raises(ValueError, match="spot"):
        CylinderOpticsModel.from_config(big_gap)


def test_bias_rule_rejects_bad_ratio(model_a: CylinderOpticsModel) -> None:
    with pytest.raises(ValueError, match="ratio"):
        model_a.bias_for_eta_ratio(0.0)
    with pytest.raises(ValueError, match="ratio"):
        model_a.bias_for_eta_ratio(1.5)


# --------------------------------------------------------------------------- #
# Integration with the modal mechanics (variant B; docs 04/05).
# --------------------------------------------------------------------------- #
def test_integration_small_signal_amplitude(
    constants: Constants, config_dir: Path, model_b: CylinderOpticsModel
) -> None:
    """Delta eta = |effective slope| * H_QS |D(f)| a within 5 % at 0.1 g."""
    variant = load_variant("B", config_dir=config_dir)
    mech_model = CantileverModel.from_config(constants, variant)
    excitation = _sine_excitation(120.0, 0.1, 20000.0, 1.0, constants.universal.g0_m_s2)
    tip = ModalFrequencyMechanics(constants=constants).run(excitation, variant)
    optical = CylinderOptics().run(tip, variant)
    ac = optical.eta - optical.bias
    measured = 0.5 * float(np.max(ac) - np.min(ac))
    coupling = constants.tilt_displacement_coupling_per_l
    accel = 0.1 * constants.universal.g0_m_s2
    gain = abs(mech_model.dynamic_factor(120.0)[0]) * mech_model.h_lat_qs
    expected = abs(model_b.effective_slope_dx(variant.length_m, coupling)) * gain * accel
    assert measured == pytest.approx(expected, rel=5.0e-2)


def test_integration_cross_axis_suppression(
    constants: Constants, config_dir: Path, model_b: CylinderOpticsModel
) -> None:
    """y vs x response follows the closed-form suppression of doc 03 §e (R-59; V19-L2-10).

    Variant B, 10 g / 120 Hz, tip amplitude ``d = |H_lat(f)| a``. Target axis:
    first-order response ``|(d eta/d dx)_eff| d`` with
    ``(d eta/d dx)_eff = -2 eta0 Delta x0/sigma_eff^2 [1 + 1.377 (R_c + A)/L]``.
    Cross axis: first order is zero; only the flat-plane residual through
    ``theta_x = 1.377 dy/L`` remains, ``eta0 (1 - exp[-(2 theta_x/theta0)^2])``
    (angular width theta0/2, R-59).

    The suppression level is a *reported model number*, not a requirement here:
    ~1.04e5 after R-59 (was ~7.4e5 with the revoked multiplier; the former
    threshold 5e5 was an empirical guard on that value). The level the
    cross-axis suppression must reach is decided together with the anisotropy
    metric (backlog M-25; owner decision of S-30, 2026-10-17).
    """
    variant = load_variant("B", config_dir=config_dir)
    mech = ModalFrequencyMechanics(constants=constants)
    stage = CylinderOptics()
    responses = {}
    for axis in ("x", "y"):
        excitation = _sine_excitation(120.0, 10.0, 20000.0, 1.0, constants.universal.g0_m_s2, axis)
        optical = stage.run(mech.run(excitation, variant), variant)
        responses[axis] = float(np.max(np.abs(optical.eta - optical.bias)))

    cantilever = CantileverModel.from_config(constants, variant)
    tip = abs(complex(cantilever.h_lat(120.0)[0])) * 10.0 * constants.universal.g0_m_s2
    coupling = constants.tilt_displacement_coupling_per_l
    peak, sigma = _closed_forms(model_b)
    bias, length = model_b.bias_m, variant.length_m
    eta0 = peak * math.exp(-((bias / sigma) ** 2))
    multiplier = 1.0 + coupling * (model_b.radius_of_curvature_m + model_b.gap_m) / length
    slope_eff = -2.0 * eta0 * bias / sigma**2 * multiplier
    theta0 = model_b.beam.wavelength_m / (math.pi * model_b.beam.waist_radius_m)
    x_expected = abs(slope_eff) * tip
    y_expected = eta0 * (1.0 - math.exp(-((2.0 * coupling * tip / (length * theta0)) ** 2)))
    assert responses["x"] == pytest.approx(x_expected, rel=1.0e-2)
    assert responses["y"] == pytest.approx(y_expected, rel=1.0e-2)
    suppression = responses["x"] / responses["y"]
    assert suppression == pytest.approx(x_expected / y_expected, rel=2.0e-2)


def test_integration_cross_axis_residual_scales_quadratically(
    constants: Constants, config_dir: Path
) -> None:
    """Doubling the y amplitude quadruples the optical residual (doc 04 §5)."""
    variant = load_variant("B", config_dir=config_dir)
    mech = ModalFrequencyMechanics(constants=constants)
    stage = CylinderOptics()

    def residual(amplitude_g: float) -> float:
        excitation = _sine_excitation(
            120.0, amplitude_g, 20000.0, 1.0, constants.universal.g0_m_s2, "y"
        )
        optical = stage.run(mech.run(excitation, variant), variant)
        return float(np.max(np.abs(optical.eta - optical.bias)))

    assert residual(10.0) / residual(5.0) == pytest.approx(4.0, rel=5.0e-2)


def test_integration_thd_grows_past_linearity_boundary(model_b: CylinderOpticsModel) -> None:
    """THD of eta(dx sine) follows the closed-form HD2 and grows monotonically (03 §e, R-59).

    For the Gaussian ``eta(dx_eff)`` (R-59) a sine of amplitude d at the working
    point ``t = Delta x0/sigma_eff`` gives ``HD2 = |4t^2 - 2| d / (8 t sigma_eff)``
    (second-order Taylor, doc 03 §e). At the B config bias the working point sits
    near the peak (eta0/eta_peak = 0.938, t = 0.253, sigma_eff = 3.93 um):
    HD2 = 0.22 % at 10 nm, 2.2 % at 0.1 um and ~11 % at 0.5 um. Before R-59
    (sigma = 1.28 um, revoked) the same drive gave < 0.1 % / ~2.5 %.
    """
    fs, freq, n_cycles = 51200.0, 100.0, 64
    t = np.arange(round(n_cycles * fs / freq)) / fs
    coupling_tilt = 0.0  # pure optical nonlinearity: dx only

    def thd_at(amplitude_m: float) -> float:
        dx = amplitude_m * np.sin(2.0 * np.pi * freq * t)
        tip = TipState(
            dx=dx,
            dy=np.zeros_like(dx),
            dz=np.zeros_like(dx),
            theta_x=np.zeros_like(dx),
            theta_y=coupling_tilt * dx,
            fs=fs,
        )
        eta_x, eta_y = model_b.eta_components(tip.dx, tip.dy, tip.dz, tip.theta_x, tip.theta_y)
        return _thd(eta_x * eta_y, fs, freq)

    levels = [thd_at(a) for a in (10.0e-9, 0.1e-6, 0.5e-6)]
    _, sigma = _closed_forms(model_b)
    t = model_b.bias_m / sigma

    def hd2(amplitude_m: float) -> float:
        return abs(4.0 * t**2 - 2.0) * amplitude_m / (8.0 * t * sigma)

    assert levels[0] == pytest.approx(hd2(10.0e-9), rel=1.0e-2)
    assert levels[1] == pytest.approx(hd2(0.1e-6), rel=1.0e-2)
    # Leading-order Taylor term; the next order is O((d/sigma)^2) ~ 2 % at 0.5 um.
    assert levels[2] == pytest.approx(hd2(0.5e-6), rel=5.0e-2)
    assert levels[2] > 2.0e-2
    assert levels[0] < levels[1] < levels[2]


@pytest.mark.golden
def test_golden_thd_matches_analytic_second_harmonic(model_b: CylinderOpticsModel) -> None:
    """Small-signal HD2 of eta(dx sine) matches the Taylor prediction (plan 18, G4).

    For ``eta(x)`` around the working point, a sine ``x = d sin(wt)`` produces a
    second harmonic with amplitude ratio ``HD2 = |eta''| d / (4 |eta'|)`` (second
    order Taylor term ``eta'' x^2/2`` with ``sin^2 -> cos 2wt / 2``; doc 03 §5).
    The measured ratio from the simulated eta series agrees within 2 % at
    ``d = 50 nm`` (well inside the ~0.5 um small-signal boundary). R-59: for the
    exactly Gaussian ``eta(dx_eff)`` the Taylor ratio has the closed form
    ``|4t^2 - 2| d / (8 t sigma_eff)``, 1.099 % here (was ~0.3 % with the revoked
    multiplier) -- pinned below against the formula, not the model.
    """
    fs, freq, n_cycles = 51200.0, 100.0, 64
    amplitude_m = 50.0e-9
    t = np.arange(round(n_cycles * fs / freq)) / fs
    dx = amplitude_m * np.sin(2.0 * np.pi * freq * t)
    zeros = np.zeros_like(dx)
    eta_x, eta_y = model_b.eta_components(dx, zeros, zeros, zeros, zeros)
    eta = eta_x * eta_y

    n = eta.size
    spectrum = np.abs(np.fft.rfft(eta - np.mean(eta)))
    fundamental_bin = round(freq * n / fs)
    hd2_measured = float(spectrum[2 * fundamental_bin] / spectrum[fundamental_bin])

    # Analytic reference: central finite differences of eta(x) at the working point.
    step = 1.0e-9

    def eta_at(x: float) -> float:
        point = np.array([x], dtype=np.float64)
        zero = np.zeros(1, dtype=np.float64)
        ex, ey = model_b.eta_components(point, zero, zero, zero, zero)
        return float((ex * ey)[0])

    slope = (eta_at(step) - eta_at(-step)) / (2.0 * step)
    curvature = (eta_at(step) - 2.0 * eta_at(0.0) + eta_at(-step)) / step**2
    hd2_analytic = abs(curvature) * amplitude_m / (4.0 * abs(slope))
    assert hd2_measured == pytest.approx(hd2_analytic, rel=0.02)
    _, sigma = _closed_forms(model_b)
    t = model_b.bias_m / sigma
    hd2_closed = abs(4.0 * t**2 - 2.0) * amplitude_m / (8.0 * t * sigma)
    assert hd2_analytic == pytest.approx(hd2_closed, rel=1.0e-3)
    assert hd2_closed == pytest.approx(1.099e-2, abs=5.0e-6)


# --------------------------------------------------------------------------- #
# Regression: registry, scenarios, viz smoke.
# --------------------------------------------------------------------------- #
def test_registry_keys() -> None:
    assert "stub" in OPTICS_REGISTRY
    assert "cylinder" in OPTICS_REGISTRY


def test_scenario_cross_axis_runs(examples_dir: Path) -> None:
    """The cross-axis scenario runs end-to-end; eta stays physical."""
    artifacts = run_scenario(examples_dir / "cross_axis.yaml")
    eta = artifacts.forward.optical.eta
    assert np.all((eta >= 0.0) & (eta <= 1.0))
    residual = np.max(np.abs(eta - artifacts.forward.optical.bias))
    assert residual < 1.0e-6  # quadratic cross residual, ~2.1e-9 (R-59; was ~1.6e-9)


def test_scenario_linearity_ramp_runs(examples_dir: Path) -> None:
    """The linearity-ramp scenario runs end-to-end through the nonlinearity."""
    artifacts = run_scenario(examples_dir / "linearity_ramp.yaml")
    eta = artifacts.forward.optical.eta
    assert np.all((eta >= 0.0) & (eta <= 1.0))
    # The resonant tip swing crosses the documented 0.5 um boundary by design.
    assert float(np.max(np.abs(artifacts.forward.tip.dx))) > 0.5e-6
    assert math.isfinite(artifacts.result.dominant_freqs_hz[0])


def test_viz_optics_figures_build_headless(model_a: CylinderOpticsModel) -> None:
    """The optics figures build without Qt/pyplot (SW-09)."""
    fig_dx = plot_eta_vs_dx(model_a)
    fig_dy = plot_eta_vs_dy(model_a, 3.0e-3, COUPLING)
    fig_map = plot_eta_map(model_a, 3.0e-3, COUPLING, n=31)
    for fig in (fig_dx, fig_dy, fig_map):
        assert fig.get_axes()
