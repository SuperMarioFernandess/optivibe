"""Tests for the S5 inverse/DSP package (calibration, kinematics, spectra, NEA).

Golden checks against the knowledge base (s_target sign/derivation doc 05 §5.3,
the analytic a -> v -> x kinematics doc 05 §7, the noise-referred NEA), spectral
correctness (amplitude scaling, Parseval, dominant resolution), end-to-end "true
vs recovered" on the full forward chain, hypothesis property tests (calibration
linearity, seed determinism, a -> x -> a round-trip) and the regression that both
DSP keys resolve, the default stays the stub and the prior S1-S4 dominants are
unchanged under the calibrated chain. Tolerances follow doc 11 §7 (inverse <= 2 %
on amplitudes, methods agree <= 3 %, NEA self-check <= 15 %).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from optivibe.core.config.loader import load_constants, load_scenario, load_variant
from optivibe.core.config.models import (
    Constants,
    DspOptions,
    SineSpec,
    StageSelection,
    VariantConfig,
)
from optivibe.core.types import DetectorOutput, Excitation
from optivibe.detector import PhotodiodeDetector, StubDetector
from optivibe.dsp import (
    DSP_REGISTRY,
    INTEGRATOR_REGISTRY,
    StandardDsp,
    amplitude_spectrum,
    analytic_noise_psd,
    calibrate_acceleration,
    cross_axis_suppression,
    dominant_frequencies,
    integrate_frequency,
    integrate_time,
    nea_from_detector,
    nea_spectrum,
    target_sensitivity,
    target_sensitivity_via_multiplier,
    welch_psd,
)
from optivibe.excitation import EXCITATION_REGISTRY
from optivibe.mechanics.modal import ModalFrequencyMechanics
from optivibe.optics.cylinder import CylinderOptics
from optivibe.pipeline.orchestrator import Pipeline

G0 = 9.80665
FS = 5000.0
DURATION = 2.0


# --------------------------------------------------------------------------- #
# Fixtures and forward-chain helpers.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def constants(config_dir: Path) -> Constants:
    """Physical constants bundle."""
    return load_constants(config_dir / "constants.yaml")


@pytest.fixture(scope="module")
def variant_b(config_dir: Path) -> VariantConfig:
    """Variant B preset (general-purpose wideband)."""
    return load_variant("B", config_dir)


@pytest.fixture(scope="module")
def dsp(constants: Constants) -> StandardDsp:
    """Standard DSP stage bound to the constants fixture."""
    return StandardDsp(constants=constants)


def _drive(f0: float, amp_g: float, axis: str = "x") -> Excitation:
    """Generate a single-tone excitation on the given axis."""
    spec = SineSpec(
        kind="sine", axis=axis, fs_hz=FS, duration_s=DURATION, frequency_hz=f0, amplitude_g=amp_g
    )
    return EXCITATION_REGISTRY.create("sine").generate(spec, seed=7)


def _forward(exc: Excitation, variant: VariantConfig, detector: str = "stub") -> DetectorOutput:
    """Run the forward chain modal -> cylinder -> detector for an excitation."""
    optical = CylinderOptics().run(ModalFrequencyMechanics().run(exc, variant), variant)
    if detector == "stub":
        return StubDetector().run(optical, variant)
    return PhotodiodeDetector(scenario_seed=7).run(optical, variant)


def _sine_accel(f0: float, amp: float, n: int = 10000) -> np.ndarray:
    """A clean acceleration tone ``amp * sin(2 pi f0 t)`` at FS, m/s^2."""
    t = np.arange(n) / FS
    return np.asarray(amp * np.sin(2.0 * np.pi * f0 * t), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Golden: calibration / s_target (doc 05 §5.3).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_s_target_is_signed_negative(variant_b: VariantConfig, constants: Constants) -> None:
    """s_target is negative for variant B (slope_eff < 0 carries through, doc 05)."""
    s = target_sensitivity(variant_b, constants)
    assert s < 0.0
    # |s_target| x 50 g matches the ~95 uA datasheet note of B.yaml.
    assert abs(s) * 50.0 * G0 == pytest.approx(95e-6, rel=0.1)


@pytest.mark.golden
def test_s_target_two_derivations_agree(variant_b: VariantConfig, constants: Constants) -> None:
    """The direct and multiplier-form s_target agree to machine precision (doc 05 §5.3)."""
    direct = target_sensitivity(variant_b, constants)
    via_m = target_sensitivity_via_multiplier(variant_b, constants)
    assert via_m == pytest.approx(direct, rel=1e-9)


@pytest.mark.golden
def test_calibration_recovers_acceleration_stub(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """Noiseless stub: calibrated a recovers the applied a in scale and sign (<= 2 %)."""
    exc = _drive(200.0, 1.0)
    accel, s_target = calibrate_acceleration(_forward(exc, variant_b, "stub"), variant_b, constants)
    a_true = np.asarray(exc.a_x, dtype=np.float64)
    assert np.std(accel) == pytest.approx(np.std(a_true), rel=2e-2)
    assert float(np.sign(np.mean(accel * a_true))) == 1.0
    assert s_target < 0.0


# --------------------------------------------------------------------------- #
# Golden: kinematics a -> v -> x (doc 05 §7).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
@pytest.mark.parametrize("integrator", ["frequency", "time"])
def test_kinematics_amplitudes(integrator: str) -> None:
    """v and x amplitudes match A/omega and A/omega^2 within 2 % (both methods)."""
    f0, amp = 200.0, 9.80665
    a = _sine_accel(f0, amp)
    integrate = integrate_frequency if integrator == "frequency" else integrate_time
    velocity, displacement = integrate(a, FS, 1.0)
    omega = 2.0 * math.pi * f0
    assert np.std(velocity) * math.sqrt(2.0) == pytest.approx(amp / omega, rel=2e-2)
    assert np.std(displacement) * math.sqrt(2.0) == pytest.approx(amp / omega**2, rel=2e-2)


@pytest.mark.golden
def test_kinematics_phase_lag_pi() -> None:
    """Displacement lags acceleration by pi: corr(x, -sin) ~ +1 (doc 05 §7)."""
    f0, amp = 200.0, 9.80665
    a = _sine_accel(f0, amp)
    _, displacement = integrate_frequency(a, FS, 1.0)
    t = np.arange(a.size) / FS
    reference = -np.sin(2.0 * math.pi * f0 * t)
    corr = float(np.corrcoef(displacement, reference)[0, 1])
    assert corr > 0.999


def test_integrators_agree_in_band() -> None:
    """The frequency and time integrators agree on an in-band tone within 3 %."""
    a = _sine_accel(200.0, 9.80665)
    vf, xf = integrate_frequency(a, FS, 1.0)
    vt, xt = integrate_time(a, FS, 1.0)
    assert np.std(vt) == pytest.approx(np.std(vf), rel=3e-2)
    assert np.std(xt) == pytest.approx(np.std(xf), rel=3e-2)


def test_frequency_mask_removes_dc_and_subband() -> None:
    """The spectral high-pass removes a DC pedestal and bin-aligned sub-band sway."""
    a = _sine_accel(200.0, 9.80665)
    t = np.arange(a.size) / FS
    drift = 0.3 * 9.80665 + 3.0 * np.sin(2.0 * math.pi * 0.5 * t)  # DC + 0.5 Hz (a resolved bin)
    _, x_clean = integrate_frequency(a, FS, 1.0)
    _, x_drift = integrate_frequency(a + drift, FS, 1.0)
    assert np.std(x_drift) == pytest.approx(np.std(x_clean), rel=1e-3)


def test_time_detrend_removes_polynomial_drift() -> None:
    """The time integrator's detrend removes a polynomial drift (DC + ramp + parabola)."""
    a = _sine_accel(200.0, 9.80665)
    t = np.arange(a.size) / FS
    drift = 0.3 * 9.80665 + 0.4 * t + 0.2 * t**2
    _, x_clean = integrate_time(a, FS, 1.0)
    _, x_drift = integrate_time(a + drift, FS, 1.0)
    assert np.std(x_drift) == pytest.approx(np.std(x_clean), rel=2e-2)


# --------------------------------------------------------------------------- #
# Spectra (amplitude scaling, Parseval, dominant resolution).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_amplitude_spectrum_scaling() -> None:
    """A unit-amplitude tone peaks at its amplitude in the 2/N spectrum."""
    amp = 9.80665
    spectrum = amplitude_spectrum(_sine_accel(200.0, amp), FS)
    assert float(np.max(spectrum.values)) == pytest.approx(amp, rel=1e-6)
    assert spectrum.kind == "amplitude"


@pytest.mark.golden
def test_welch_parseval() -> None:
    """The integral of the Welch PSD equals the signal variance (Parseval)."""
    a = _sine_accel(200.0, 9.80665)
    psd = welch_psd(a, FS)
    integral = float(np.trapezoid(psd.values, psd.freq))
    assert integral == pytest.approx(float(np.var(a)), rel=1e-2)
    assert psd.kind == "psd"


def test_dominant_resolves_within_one_bin() -> None:
    """The dominant frequency is within one FFT bin of the true tone."""
    n = 10000
    bin_hz = FS / n
    spectrum = amplitude_spectrum(_sine_accel(173.0, 1.0, n), FS)
    dominant = dominant_frequencies(spectrum)
    assert abs(dominant[0] - 173.0) <= bin_hz


def test_dominant_multitone() -> None:
    """A two-tone signal yields both lines, strongest first."""
    n = 10000
    t = np.arange(n) / FS
    signal = 9.8 * np.sin(2 * math.pi * 150 * t) + 5.0 * np.sin(2 * math.pi * 420 * t)
    dominant = dominant_frequencies(amplitude_spectrum(np.asarray(signal), FS))
    assert dominant[0] == pytest.approx(150.0, abs=1.0)
    assert any(abs(f - 420.0) <= 1.0 for f in dominant)


# --------------------------------------------------------------------------- #
# Golden: NEA referred to the input (doc 05 §7; O-SW-08).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_nea_plateau_in_documented_window(variant_b: VariantConfig, constants: Constants) -> None:
    """The plateau NEA of variant B sits in the documented ~25 ug/sqrt(Hz) window."""
    detector = _forward(_drive(200.0, 1.0), variant_b, "photodiode")
    summary = nea_from_detector(detector, variant_b, constants)
    assert summary is not None
    nea_ug = summary.nea_plateau / G0 * 1e6
    assert 10.0 <= nea_ug <= 50.0


@pytest.mark.golden
def test_nea_analytic_cross_check(variant_b: VariantConfig, constants: Constants) -> None:
    """The simulated noise PSD matches the analytic re-assembly within 15 % (doc 11 §7)."""
    detector = _forward(_drive(200.0, 1.0), variant_b, "photodiode")
    psd_sim = float(detector.noise["psd_total_a2_hz"])  # type: ignore[arg-type]
    psd_analytic = analytic_noise_psd(detector, variant_b, constants)
    assert psd_analytic == pytest.approx(psd_sim, rel=0.15)


def test_nea_none_for_stub(variant_b: VariantConfig, constants: Constants) -> None:
    """A noiseless stub detector yields no NEA (nothing to refer)."""
    detector = _forward(_drive(200.0, 1.0), variant_b, "stub")
    assert nea_from_detector(detector, variant_b, constants) is None


def test_nea_spectrum_matches_plateau(variant_b: VariantConfig, constants: Constants) -> None:
    """NEA(f) equals the plateau figure well below f1 (flat region)."""
    detector = _forward(_drive(200.0, 1.0), variant_b, "photodiode")
    summary = nea_from_detector(detector, variant_b, constants)
    assert summary is not None
    freq = np.array([10.0, 50.0, 200.0])
    nea = nea_spectrum(detector, variant_b, freq, constants)
    assert np.allclose(nea, summary.nea_plateau, rtol=1e-3)


# --------------------------------------------------------------------------- #
# End-to-end "true vs recovered" on the full forward chain.
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_true_vs_recovered_photodiode(variant_b: VariantConfig, dsp: StandardDsp) -> None:
    """Full chain, SNR >> 1: recovered a tracks the applied a and the dominant is exact."""
    exc = _drive(200.0, 1.0)
    result = dsp.run(_forward(exc, variant_b, "photodiode"), variant_b, DspOptions())
    a_true = np.asarray(exc.a_x, dtype=np.float64)
    assert np.std(result.a) == pytest.approx(np.std(a_true), rel=1e-2)
    assert result.dominant_freqs_hz[0] == pytest.approx(200.0, abs=1.0)
    assert "nea" in (result.iso or {})


def test_cross_axis_residual_small(variant_b: VariantConfig, dsp: StandardDsp) -> None:
    """An off-axis (y) drive leaks weakly into the target-axis reconstruction."""
    rx = dsp.run(_forward(_drive(120.0, 1.0, "x"), variant_b, "stub"), variant_b, DspOptions())
    ry = dsp.run(_forward(_drive(120.0, 1.0, "y"), variant_b, "stub"), variant_b, DspOptions())
    suppression = cross_axis_suppression(float(np.std(ry.a)), float(np.std(rx.a)))
    assert suppression < 1e-4


def test_iso_assessment_present(variant_b: VariantConfig, dsp: StandardDsp) -> None:
    """The ISO assessment carries a zone and the standard reference."""
    result = dsp.run(_forward(_drive(50.0, 0.15), variant_b, "photodiode"), variant_b, DspOptions())
    assert result.iso is not None
    assert result.iso["zone"] in {"A", "B", "C", "D"}
    assert "ISO" in str(result.iso["standard"])


# --------------------------------------------------------------------------- #
# Property tests (hypothesis): linearity, determinism, round-trip.
# --------------------------------------------------------------------------- #
@settings(max_examples=15, deadline=None)
@given(scale=st.floats(min_value=0.1, max_value=5.0))
def test_calibration_linearity(
    variant_b: VariantConfig, constants: Constants, scale: float
) -> None:
    """Scaling the detector AC content scales the recovered acceleration linearly."""
    base = _forward(_drive(200.0, 0.2), variant_b, "stub")
    ac = np.asarray(base.samples, dtype=np.float64) - base.dc_level
    scaled = DetectorOutput(
        samples=np.ascontiguousarray(base.dc_level + scale * ac),
        fs=base.fs,
        dc_level=base.dc_level,
        units=base.units,
        noise=base.noise,
    )
    a0, _ = calibrate_acceleration(base, variant_b, constants)
    a1, _ = calibrate_acceleration(scaled, variant_b, constants)
    assert np.std(a1) == pytest.approx(scale * np.std(a0), rel=1e-6)


def test_seed_determinism(variant_b: VariantConfig, dsp: StandardDsp) -> None:
    """The same scenario seed reproduces the recovered signal bit-for-bit."""
    exc = _drive(200.0, 0.05)
    r1 = dsp.run(_forward(exc, variant_b, "photodiode"), variant_b, DspOptions())
    r2 = dsp.run(_forward(exc, variant_b, "photodiode"), variant_b, DspOptions())
    assert np.array_equal(r1.a, r2.a)


@settings(max_examples=10, deadline=None)
@given(f0=st.floats(min_value=50.0, max_value=500.0))
def test_round_trip_a_to_x_to_a(f0: float) -> None:
    """Integrating a -> x then double-differentiating recovers a in band (round-trip)."""
    a = _sine_accel(f0, 9.80665)
    _, x = integrate_frequency(a, FS, 1.0)
    spec = np.fft.rfft(x)
    omega = 2.0 * math.pi * np.fft.rfftfreq(x.size, 1.0 / FS)
    a_rt = np.fft.irfft(spec * (-(omega**2)), n=x.size)
    interior = slice(500, -500)
    assert np.std(a_rt[interior] - a[interior]) / np.std(a[interior]) < 2e-2


# --------------------------------------------------------------------------- #
# Regression: registry, default key, prior scenario dominants (SW-S5-01).
# --------------------------------------------------------------------------- #
def test_both_dsp_keys_registered() -> None:
    """Both inverse implementations resolve from the registry."""
    assert set(DSP_REGISTRY.keys()) == {"stub", "standard"}
    assert isinstance(DSP_REGISTRY.create("standard"), StandardDsp)
    integrator_keys = set(INTEGRATOR_REGISTRY.keys())
    assert "frequency" in integrator_keys
    assert "time" in integrator_keys


def test_default_stage_is_stub() -> None:
    """The default DSP stage stays the stub (decision SW-S5-01)."""
    assert StageSelection().dsp == "stub"


@pytest.mark.parametrize(
    ("scenario", "dominant_hz"),
    [
        ("hello.yaml", 120.0),
        ("cross_axis.yaml", 240.0),
        ("resonance_sweep.yaml", 5005.0),
        ("linearity_ramp.yaml", 25001.5),
        ("sine_with_noise.yaml", 200.0),
    ],
)
def test_prior_dominants_unchanged_under_standard(
    examples_dir: Path, scenario: str, dominant_hz: float
) -> None:
    """The calibrated chain reproduces the prior S1-S4 dominants (scalar calibration)."""
    base = load_scenario(examples_dir / scenario)
    stages = base.stages.model_copy(update={"dsp": "standard"})
    scenario_std = base.model_copy(update={"stages": stages})
    variant = load_variant(scenario_std.variant)
    result = Pipeline(scenario_std, variant).run().result
    assert result.dominant_freqs_hz
    assert result.dominant_freqs_hz[0] == pytest.approx(dominant_hz, abs=1.5)


# --------------------------------------------------------------------------- #
# Visualization smoke (SW-09): pure Figures build head-less.
# --------------------------------------------------------------------------- #
def test_viz_dsp_figures_build_headless(variant_b: VariantConfig, dsp: StandardDsp) -> None:
    """The inverse-chain figures build without Qt/pyplot (SW-09)."""
    from optivibe.viz.dsp import (
        plot_kinematics,
        plot_nea,
        plot_spectrogram,
        plot_spectrum,
        plot_true_vs_recovered,
    )

    exc = _drive(200.0, 1.0)
    detector = _forward(exc, variant_b, "photodiode")
    result = dsp.run(detector, variant_b, DspOptions())
    a_true = np.asarray(exc.a_x, dtype=np.float64)
    figures = (
        plot_true_vs_recovered(a_true, result.a, result.fs),
        plot_spectrum(result.spectrum, result.dominant_freqs_hz),
        plot_spectrogram(result.a, result.fs),
        plot_nea(detector, variant_b),
        plot_kinematics(result.v, result.x, result.fs),
    )
    for figure in figures:
        assert figure.get_axes()
