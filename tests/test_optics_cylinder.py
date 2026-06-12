"""S3 optics tests: golden references (docs 03/04/05/08), limits (11 §7),
integration with the modal mechanics, and property invariants (10 §10).

Calibration anchor (journal 2026-06-12): w0 = 5.2 um, Delta x0 = 2 um,
R_c = 62.5 um and the S3-calibrated gap A = 31 um put eta0, eta_peak, the bare
slope (doc 04 §4) and the effective slope (docs 05 §1 / 08) within 5 % of the
documented references simultaneously.
"""

from __future__ import annotations

import math
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
    """Scaled platform: R_c = 31 um, scale-equivalent bias 0.992 um."""
    return CylinderOpticsModel.from_config(load_variant("B", config_dir=config_dir))


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
    """eta0 ~ 0.25 and eta_peak ~ 0.42 at the documented platform (docs 03/08)."""
    assert model_a.eta_working_point() == pytest.approx(0.25, rel=5.0e-2)
    assert model_a.eta_peak() == pytest.approx(0.42, rel=5.0e-2)


@pytest.mark.golden
def test_golden_bare_slope_platform_a(model_a: CylinderOpticsModel) -> None:
    """Bare slope d eta / d dx ~ -1.5e5 1/m at the working point (doc 04 §4)."""
    assert model_a.slope_dx() == pytest.approx(-1.5e5, rel=5.0e-2)


@pytest.mark.golden
def test_golden_effective_slope_platform_a(model_a: CylinderOpticsModel) -> None:
    """Effective slope ~ -1.6e5 1/m at L=1.4 mm and ~ -1.56e5 at 3 mm (doc 05 §1).

    The doc 05/08 reference -0.16 1/um is the *effective* slope including the
    tilt multiplier [1 + 1.377 (R_c + A)/L] at L = 1.4 mm (S3 reinterpretation,
    journal 2026-06-12); the bare value stays at the doc 04 §4 reference.
    """
    assert model_a.effective_slope_dx(1.4e-3, COUPLING) == pytest.approx(-1.6e5, rel=5.0e-2)
    assert model_a.effective_slope_dx(3.0e-3, COUPLING) == pytest.approx(-1.56e5, rel=5.0e-2)


@pytest.mark.golden
def test_golden_tilt_multiplier(model_a: CylinderOpticsModel) -> None:
    """Tilt enhancement +4 % at L=3 mm, +9 % at L=1.4 mm (doc 05 §1)."""
    assert model_a.tilt_multiplier(3.0e-3, COUPLING) - 1.0 == pytest.approx(0.042, rel=0.1)
    assert model_a.tilt_multiplier(1.4e-3, COUPLING) - 1.0 == pytest.approx(0.091, rel=0.1)


@pytest.mark.golden
def test_golden_slope_scaling_with_radius(model_a: CylinderOpticsModel) -> None:
    """First-order slope scaling ~ 1/R_c reproduces the -0.32 1/um reference.

    Doc 08 §2: halving R_c (62.5 -> 31 um) doubles the slope to ~ -0.32 1/um at
    L = 1.4 mm. The reference is the first-order scaling of the effective
    -0.16 1/um; the *normalized* slope (1/eta0) d eta/d dx of the exact model
    scales exactly as 1/R_c at a scale-equivalent bias (Delta x0 ~ R_c), while
    the absolute exact-model slope grows only x1.31 because eta_peak(R_c)
    drops — a recorded knowledge-base discrepancy (journal 2026-06-12).
    """
    scaled = model_a.effective_slope_dx(1.4e-3, COUPLING) * (62.5 / 31.0)
    assert scaled == pytest.approx(-3.2e5, rel=5.0e-2)


@pytest.mark.golden
def test_golden_normalized_slope_scales_inverse_radius(
    model_a: CylinderOpticsModel, model_b: CylinderOpticsModel
) -> None:
    """(1/eta0) d eta/d dx scales exactly as 1/R_c at scale-equivalent bias."""
    ratio = (model_b.slope_dx() / model_b.eta_working_point()) / (
        model_a.slope_dx() / model_a.eta_working_point()
    )
    assert ratio == pytest.approx(62.5 / 31.0, rel=1.0e-6)


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
    """d eta / d dz < 0 with |.| ~ 1e3..2.4e4 1/m; the dz channel is negligible.

    The exact model gives ~ -7.9e3 1/m; the doc 04 §4 illustrative -2e4 1/m is
    not reproduced (and contradicts doc 04's own closed form ~ -1.9e3) — a
    recorded knowledge-base discrepancy (journal 2026-06-12). The negligibility
    conclusion is robust across the window: at 50 g the dz contribution to
    Delta eta stays < 1e-4 of the dx contribution.
    """
    h = 1.0e-10
    slope_z = float(((model_a.eta(dz=h) - model_a.eta(dz=-h)) / (2.0 * h)).item())
    assert slope_z < 0.0
    assert 1.0e3 <= abs(slope_z) <= 2.4e4
    # Channel comparison at 50 g on variant A geometry (L = 5 mm):
    constants = load_constants(Path(__file__).resolve().parent.parent / "configs/constants.yaml")
    variant = load_variant("A")
    mech = CantileverModel.from_config(constants, variant)
    accel = 50.0 * constants.universal.g0_m_s2
    dx_amp = mech.h_lat_qs * accel
    dz_amp = mech.axial_compliance * accel
    ratio = abs(slope_z * dz_amp) / abs(model_a.slope_dx() * dx_amp)
    assert ratio < 1.0e-4


@pytest.mark.golden
def test_golden_anisotropy(model_a: CylinderOpticsModel, constants: Constants) -> None:
    """Angular-map anisotropy L/(1.377 R_c) ~ 35x at L = 3 mm (doc 04 §5)."""
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
def test_golden_bias_rule_helper(model_a: CylinderOpticsModel) -> None:
    """bias_for_eta_ratio(0.37) puts eta0/eta_peak at 0.37 (doc 08 R-40/O-05)."""
    bias = model_a.bias_for_eta_ratio(0.37)
    tuned = CylinderOpticsModel(
        beam=model_a.beam,
        gap_m=model_a.gap_m,
        radius_of_curvature_m=model_a.radius_of_curvature_m,
        bias_m=bias,
    )
    assert tuned.eta_working_point() / tuned.eta_peak() == pytest.approx(0.37, rel=1.0e-9)
    assert bias == pytest.approx(2.573e-6, rel=1.0e-3)


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
    wavefront, so the curved-plane overlap stays < 1; it tends to 1 only in the
    additional limit R_c -> inf (doc 03 §3-§4).
    """
    beam = model_a.beam
    assert float(eta_parallel_flat(beam, 0.0).item()) == pytest.approx(1.0, abs=1.0e-12)
    curved_finite = float(eta_parallel_curved(beam, 0.0, 62.5e-6).item())
    assert 0.0 < curved_finite < 1.0
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


def test_integration_cross_axis_suppression(constants: Constants, config_dir: Path) -> None:
    """A y tone is suppressed ~6 orders relative to the same tone on x (doc 04 §5).

    Measured at 10 g / 120 Hz on variant B: ratio ~ 7.4e5 (the quadratic
    residual grows with amplitude, so the ratio is amplitude-dependent).
    """
    variant = load_variant("B", config_dir=config_dir)
    mech = ModalFrequencyMechanics(constants=constants)
    stage = CylinderOptics()
    responses = {}
    for axis in ("x", "y"):
        excitation = _sine_excitation(120.0, 10.0, 20000.0, 1.0, constants.universal.g0_m_s2, axis)
        optical = stage.run(mech.run(excitation, variant), variant)
        responses[axis] = float(np.max(np.abs(optical.eta - optical.bias)))
    assert responses["x"] / responses["y"] > 5.0e5


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
    """THD of eta(dx sine) is < 1e-3 at 10 nm, > 2 % at 0.5 um, and monotone.

    Doc 03 §5: the small-signal boundary of the working point is w_tip ~ 0.5 um
    (sigma ~ 1.28 um for the R_c = 31 um platform).
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
    assert levels[0] < 1.0e-3
    assert levels[2] > 2.0e-2
    assert levels[0] < levels[1] < levels[2]


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
    assert residual < 1.0e-6  # quadratic cross residual, ~1.6e-9 expected


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
