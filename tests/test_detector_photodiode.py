"""Tests for the S4 photodiode detector: noise budget, ADC, reproducibility.

Golden checks against the knowledge base (doc 07 noise budget, doc 05 §5.3
normalization), limit/edge behaviour, hypothesis property tests (seed
determinism, noise additivity) and the regression that the stub default and the
prior scenario dominants are unchanged (SW-27). Tolerances follow doc 11 §7
(detector <= 10 %; the end-to-end NEA self-check <= 15 %).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.signal import welch

from optivibe.core.config.loader import load_constants, load_variant
from optivibe.core.config.models import Constants, StageSelection, VariantConfig
from optivibe.core.types import OpticalResponse
from optivibe.detector import StubDetector
from optivibe.detector.photodiode import (
    PhotodiodeDetector,
    detector_seed_sequence,
    noise_psd,
    rin_psd,
    signal_multiplier,
)
from optivibe.optics.cylinder import CylinderOpticsModel

G0 = 9.80665


@pytest.fixture(scope="module")
def constants(config_dir: Path) -> Constants:
    """Physical constants bundle."""
    return load_constants(config_dir / "constants.yaml")


@pytest.fixture(scope="module")
def variant_b(config_dir: Path) -> VariantConfig:
    """Variant B preset (general-purpose wideband)."""
    return load_variant("B", config_dir)


@pytest.fixture(scope="module")
def eta0_b(variant_b: VariantConfig) -> float:
    """Computed optical working point eta0 of variant B."""
    return CylinderOpticsModel.from_config(variant_b).eta_working_point()


def _optical(eta0: float, fs: float, signal: np.ndarray) -> OpticalResponse:
    """Build an OpticalResponse with a given bias and AC signal around it."""
    eta = np.asarray(eta0 + signal, dtype=np.float64)
    return OpticalResponse(eta=eta, bias=eta0, fs=fs)


def _s_target(variant: VariantConfig, constants: Constants, eta0: float) -> float:
    """Effective target-axis current sensitivity s_target, A/(m/s^2) (docs 04/05)."""
    model = CylinderOpticsModel.from_config(variant)
    tcl = constants.tilt_displacement_coupling_per_l
    slope_eff = model.effective_slope_dx(variant.length_m, tcl)
    f = constants.fiber
    h_qs = (
        f.density_kg_m3
        * f.area_m2
        * variant.length_m**4
        / (8.0 * f.youngs_modulus_pa * f.inertia_m4)
    )
    gain = variant.responsivity_a_w * variant.source.power_w
    return abs(gain * variant.reflector.reflectivity * slope_eff * h_qs)


# --------------------------------------------------------------------------- #
# Golden: read-out and normalization (docs 04 §4, 05 §5.3).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_dc_level_matches_route2(variant_b: VariantConfig, eta0_b: float) -> None:
    """dc_level reproduces I_DC = R P (R1 + rho eta0) exactly (doc 04 §4)."""
    fs = 5000.0
    det = PhotodiodeDetector(scenario_seed=1)
    out = det.run(_optical(eta0_b, fs, np.zeros(256)), variant_b)
    gain = variant_b.responsivity_a_w * variant_b.source.power_w
    expected = gain * (variant_b.endface_reflectivity + variant_b.reflector.reflectivity * eta0_b)
    assert out.dc_level == pytest.approx(expected, rel=1e-9)
    assert out.units == "A"


@pytest.mark.golden
def test_signal_multiplier_references() -> None:
    """1 + R1/(rho eta0) = 5.0 (bare) and ~1.15 (metal) at eta0=0.25 (doc 05 §5.3)."""
    assert signal_multiplier(0.036, 0.036, 0.25) == pytest.approx(5.0, rel=1e-9)
    assert signal_multiplier(0.036, 0.98, 0.25) == pytest.approx(1.15, rel=2e-2)


# --------------------------------------------------------------------------- #
# Golden: noise budget (doc 07 §1).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_noise_psd_components_analytic(
    variant_b: VariantConfig, constants: Constants, eta0_b: float
) -> None:
    """Shot/RIN/Johnson PSDs match the closed forms of doc 07 §1."""
    gain = variant_b.responsivity_a_w * variant_b.source.power_w
    i_dc = gain * (variant_b.endface_reflectivity + variant_b.reflector.reflectivity * eta0_b)
    dc = constants.detector
    psd = noise_psd(i_dc, variant_b, constants, balanced=True)

    # Shot doubles under the balanced two-arm model (doc 07 §1.2).
    assert psd["shot"] == pytest.approx(2.0 * 2.0 * dc.elementary_charge_c * i_dc, rel=1e-12)
    rin0 = 10.0 ** (variant_b.source.rin_db_hz / 10.0)
    cmrr = variant_b.detector.cmrr_db
    assert psd["rin"] == pytest.approx(i_dc**2 * rin0 * 10.0 ** (-cmrr / 10.0), rel=1e-12)
    rf = variant_b.detector.transimpedance_ohm
    johnson_expected = 4.0 * dc.boltzmann_j_k * dc.temperature_k / rf
    assert psd["johnson"] == pytest.approx(johnson_expected, rel=1e-12)
    assert psd["total"] == pytest.approx(psd["shot"] + psd["rin"] + psd["johnson"], rel=1e-12)


@pytest.mark.golden
@pytest.mark.parametrize(
    ("balanced", "reference_arm", "expected_factor"),
    [(True, "matched", 2.0), (True, "bright", 1.0), (False, "matched", 1.0)],
)
def test_reference_arm_shot_conventions(
    variant_b: VariantConfig,
    constants: Constants,
    eta0_b: float,
    balanced: bool,
    reference_arm: str,
    expected_factor: float,
) -> None:
    """Both NEA shot conventions are first-class and switchable (O-SW-08).

    'matched' doubles the shot PSD (conservative two-arm floor); 'bright' keeps
    the bare 2 e I_DC (datasheet/doc-08 shot limit); a single-ended channel also
    uses the bare value.
    """
    from optivibe.detector import shot_arm_factor

    assert shot_arm_factor(balanced=balanced, reference_arm=reference_arm) == expected_factor
    gain = variant_b.responsivity_a_w * variant_b.source.power_w
    i_dc = gain * (variant_b.endface_reflectivity + variant_b.reflector.reflectivity * eta0_b)
    psd = noise_psd(i_dc, variant_b, constants, balanced=balanced, reference_arm=reference_arm)
    e = constants.detector.elementary_charge_c
    assert psd["shot"] == pytest.approx(expected_factor * 2.0 * e * i_dc, rel=1e-12)


@pytest.mark.golden
def test_rin_suppression_equals_cmrr(variant_b: VariantConfig) -> None:
    """Balanced RIN suppression equals the CMRR within 1 dB (doc 07 §1.2)."""
    i_dc = 3.0e-3
    rin = variant_b.source.rin_db_hz
    cmrr = variant_b.detector.cmrr_db
    suppressed = rin_psd(i_dc, rin, cmrr, balanced=True)
    unsuppressed = rin_psd(i_dc, rin, cmrr, balanced=False)
    suppression_db = -10.0 * math.log10(suppressed / unsuppressed)
    assert suppression_db == pytest.approx(cmrr, abs=1.0)


@pytest.mark.golden
def test_shot_psd_welch(variant_b: VariantConfig, constants: Constants, eta0_b: float) -> None:
    """A noise-only realization's Welch PSD matches the analytic total <= 10 %."""
    fs = 20000.0
    det = PhotodiodeDetector(scenario_seed=42)
    out = det.run(_optical(eta0_b, fs, np.zeros(200_000)), variant_b)
    expected = float(out.noise["psd_total_a2_hz"])
    ac = np.asarray(out.samples) - out.dc_level
    _, p_welch = welch(ac, fs=fs, nperseg=4096)
    measured = float(np.median(p_welch[1:]))
    assert measured == pytest.approx(expected, rel=0.10)


# --------------------------------------------------------------------------- #
# Golden: NEA self-check (docs 07/08; doc 11 §7).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_nea_self_check(variant_b: VariantConfig, constants: Constants, eta0_b: float) -> None:
    """Simulated noise floor matches the analytic NEA <= 15 %, right order vs doc 08."""
    fs = 20000.0
    s_target = _s_target(variant_b, constants, eta0_b)
    det = PhotodiodeDetector(scenario_seed=7)
    out = det.run(_optical(eta0_b, fs, np.zeros(200_000)), variant_b)

    nea_analytic = math.sqrt(float(out.noise["psd_total_a2_hz"])) / s_target  # (m/s^2)/rtHz
    ac = np.asarray(out.samples) - out.dc_level
    _, p_welch = welch(ac, fs=fs, nperseg=4096)
    nea_sim = math.sqrt(float(np.median(p_welch[1:]))) / s_target

    assert nea_sim == pytest.approx(nea_analytic, rel=0.15)

    # Order-of-magnitude agreement with the doc-08 shelf (~10.6 ug/rtHz); the
    # exact model is higher by the documented SW-26 R_c factor (x1.57, not x2)
    # and the balanced two-arm shot penalty (xsqrt(2)).
    nea_ug = nea_analytic / G0 * 1e6
    assert 5.0 < nea_ug < 60.0


# --------------------------------------------------------------------------- #
# Golden: ADC quantization and clipping (doc 07 §1.4).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_quantization_rms_is_lsb_over_sqrt12(variant_b: VariantConfig, eta0_b: float) -> None:
    """Quantization error RMS ~ LSB/sqrt(12) on a smooth signal (<= 10 %)."""
    fs = 5000.0
    n = 20_000
    full_scale = variant_b.detector.adc_full_scale
    bits = 10  # modest resolution so quantization dominates a smooth signal
    # Smooth signal spanning many levels, noise-free: a slow sine at 80 % FS.
    t = np.arange(n) / fs
    smooth = 0.8 * full_scale * np.sin(2.0 * math.pi * 11.0 * t)
    from optivibe.detector.photodiode import _quantize

    quantized, lsb, n_clipped = _quantize(smooth, full_scale, bits)
    err_rms = float(np.std(quantized - smooth))
    assert n_clipped == 0
    assert err_rms == pytest.approx(lsb / math.sqrt(12.0), rel=0.10)


@pytest.mark.golden
def test_adc_clip_detection(variant_b: VariantConfig, eta0_b: float) -> None:
    """Modulation beyond the full scale is clipped and counted (doc 10 §7)."""
    fs = 5000.0
    full_scale = variant_b.detector.adc_full_scale
    # AC current well beyond +/- full scale (in coupling units: eta swing).
    gain = variant_b.responsivity_a_w * variant_b.source.power_w
    rho = variant_b.reflector.reflectivity
    big_eta = (5.0 * full_scale / (gain * rho)) * np.sin(np.linspace(0.0, 20.0, 4096))
    det = PhotodiodeDetector(scenario_seed=3)
    out = det.run(_optical(eta0_b, fs, big_eta), variant_b)
    assert int(out.noise["n_clipped"]) > 0
    # Clipped samples never exceed the pedestal + full scale (one LSB tolerance).
    assert float(np.max(out.samples - out.dc_level)) <= full_scale + float(out.noise["adc_lsb"])


# --------------------------------------------------------------------------- #
# Limits (doc 07 §1).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_limit_power_to_zero(variant_b: VariantConfig, constants: Constants, eta0_b: float) -> None:
    """P -> 0 drives I_DC -> 0 so the signal-derived shot and RIN vanish.

    The Johnson term is thermal (power-independent) and remains as the
    electronics floor (doc 07 §1.3), so only shot and RIN are checked.
    """
    tiny = variant_b.model_copy(
        update={"source": variant_b.source.model_copy(update={"power_w": 1e-12})}
    )
    gain = tiny.responsivity_a_w * tiny.source.power_w
    i_dc = gain * (tiny.endface_reflectivity + tiny.reflector.reflectivity * eta0_b)
    psd = noise_psd(i_dc, tiny, constants, balanced=True)
    full = noise_psd(3.0e-3, variant_b, constants, balanced=True)
    assert psd["shot"] < full["shot"] * 1e-6
    assert psd["rin"] < full["rin"] * 1e-6


@pytest.mark.golden
def test_limit_rin_zero_rf_inf(
    variant_b: VariantConfig, constants: Constants, eta0_b: float
) -> None:
    """RIN -> 0 and Rf -> inf leave only the shot term."""
    quiet = variant_b.model_copy(
        update={
            "source": variant_b.source.model_copy(update={"rin_db_hz": -400.0}),
            "detector": variant_b.detector.model_copy(update={"transimpedance_ohm": None}),
        }
    )
    gain = quiet.responsivity_a_w * quiet.source.power_w
    i_dc = gain * (quiet.endface_reflectivity + quiet.reflector.reflectivity * eta0_b)
    psd = noise_psd(i_dc, quiet, constants, balanced=True)
    assert psd["rin"] < psd["shot"] * 1e-9
    assert psd["johnson"] == 0.0
    assert psd["total"] == pytest.approx(psd["shot"], rel=1e-9)


@pytest.mark.golden
def test_balanced_false_exposes_rin(
    variant_b: VariantConfig, constants: Constants, eta0_b: float
) -> None:
    """balanced=False removes the CMRR suppression -> RIN grows by 10^(CMRR/10)."""
    gain = variant_b.responsivity_a_w * variant_b.source.power_w
    i_dc = gain * (variant_b.endface_reflectivity + variant_b.reflector.reflectivity * eta0_b)
    on = noise_psd(i_dc, variant_b, constants, balanced=True)
    off = noise_psd(i_dc, variant_b, constants, balanced=False)
    ratio = off["rin"] / on["rin"]
    assert ratio == pytest.approx(10.0 ** (variant_b.detector.cmrr_db / 10.0), rel=1e-9)
    assert off["total"] > on["total"]


# --------------------------------------------------------------------------- #
# Optional voltage output and decimation.
# --------------------------------------------------------------------------- #
def test_voltage_output_units(variant_b: VariantConfig, eta0_b: float) -> None:
    """output='voltage' returns volts scaled by Rf and the right dc_level."""
    rf = 1.0e5
    volt = variant_b.model_copy(
        update={
            "detector": variant_b.detector.model_copy(
                update={"output": "voltage", "transimpedance_ohm": rf, "adc_full_scale": 1.0}
            )
        }
    )
    det = PhotodiodeDetector(scenario_seed=1)
    out = det.run(_optical(eta0_b, 5000.0, np.zeros(128)), volt)
    gain = volt.responsivity_a_w * volt.source.power_w
    i_dc = gain * (volt.endface_reflectivity + volt.reflector.reflectivity * eta0_b)
    assert out.units == "V"
    assert out.dc_level == pytest.approx(i_dc * rf, rel=1e-9)


def test_decimation_lowers_fs(variant_b: VariantConfig, eta0_b: float) -> None:
    """adc_fs_hz below the optical fs reduces the output sampling rate."""
    deci = variant_b.model_copy(
        update={"detector": variant_b.detector.model_copy(update={"adc_fs_hz": 1000.0})}
    )
    det = PhotodiodeDetector(scenario_seed=1)
    out = det.run(_optical(eta0_b, 5000.0, np.zeros(5000)), deci)
    assert out.fs < 5000.0
    assert out.n_samples < 5000


# --------------------------------------------------------------------------- #
# Property tests (hypothesis): determinism and additivity (doc 10 §8).
# --------------------------------------------------------------------------- #
@settings(max_examples=25, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_seed_determinism(variant_b: VariantConfig, eta0_b: float, seed: int) -> None:
    """Same seed -> bit-identical noise; the seed sequence is reproducible."""
    optical = _optical(eta0_b, 5000.0, np.zeros(2048))
    a = PhotodiodeDetector(scenario_seed=seed).run(optical, variant_b)
    b = PhotodiodeDetector(scenario_seed=seed).run(optical, variant_b)
    assert np.array_equal(a.samples, b.samples)
    # The derived sequences compare equal entropy/state for the same seed.
    assert detector_seed_sequence(seed).entropy == detector_seed_sequence(seed).entropy


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    amp=st.floats(min_value=1e-5, max_value=1e-3),
)
def test_noise_additivity(variant_b: VariantConfig, eta0_b: float, seed: int, amp: float) -> None:
    """With a shared seed the noise cancels under subtraction (noise is additive).

    The coupling swing is kept small enough that the AC current stays inside the
    ADC full scale (no clipping), so the only residual after subtraction is the
    (signal-dependent) quantization, bounded by a couple of LSB.
    """
    fs = 5000.0
    t = np.arange(4096) / fs
    sig_a = amp * np.sin(2.0 * math.pi * 137.0 * t)
    sig_b = 0.5 * amp * np.sin(2.0 * math.pi * 137.0 * t)
    out_a = PhotodiodeDetector(scenario_seed=seed).run(_optical(eta0_b, fs, sig_a), variant_b)
    out_b = PhotodiodeDetector(scenario_seed=seed).run(_optical(eta0_b, fs, sig_b), variant_b)
    stub_a = StubDetector().run(_optical(eta0_b, fs, sig_a), variant_b)
    stub_b = StubDetector().run(_optical(eta0_b, fs, sig_b), variant_b)
    diff_noisy = (out_a.samples - out_b.samples) - (stub_a.samples - stub_b.samples)
    lsb = float(out_a.noise["adc_lsb"])
    # Noise cancels exactly; only quantization remains (<= a couple of LSB).
    assert float(np.max(np.abs(diff_noisy))) <= 3.0 * lsb


# --------------------------------------------------------------------------- #
# Regression: stub default and prior dominants unchanged (SW-27).
# --------------------------------------------------------------------------- #
def test_default_detector_is_stub() -> None:
    """The default detector stays 'stub' (empirical decision SW-27)."""
    assert StageSelection().detector == "stub"


def test_stub_read_out_unchanged(variant_b: VariantConfig, eta0_b: float) -> None:
    """The stub read-out is the exact affine map, untouched by S4."""
    fs = 5000.0
    signal = 0.01 * np.sin(np.linspace(0.0, 30.0, 4096))
    out = StubDetector().run(_optical(eta0_b, fs, signal), variant_b)
    gain = variant_b.responsivity_a_w * variant_b.source.power_w
    r1 = variant_b.endface_reflectivity
    rho = variant_b.reflector.reflectivity
    expected = gain * (r1 + rho * (eta0_b + signal))
    assert np.allclose(out.samples, expected, rtol=1e-12, atol=0.0)
    assert out.dc_level == pytest.approx(gain * (r1 + rho * eta0_b), rel=1e-12)


@pytest.mark.parametrize(
    ("scenario_name", "dominant_hz"),
    [
        ("hello.yaml", 120.0),
        ("cross_axis.yaml", 240.0),
        ("resonance_sweep.yaml", 5005.0),
    ],
)
def test_prior_scenarios_dominants(
    examples_dir: Path, scenario_name: str, dominant_hz: float
) -> None:
    """S1-S3 acceptance scenarios keep their dominant frequency (stub detector)."""
    from optivibe.pipeline.orchestrator import run_scenario

    result = run_scenario(examples_dir / scenario_name)
    assert result.result.dominant_freqs_hz[0] == pytest.approx(dominant_hz, abs=1.0)


@pytest.mark.parametrize("scenario_name", ["noise_floor.yaml", "sine_with_noise.yaml"])
def test_photodiode_scenarios_run(examples_dir: Path, scenario_name: str) -> None:
    """The new photodiode scenarios run end-to-end (exit 0 equivalent)."""
    from optivibe.pipeline.orchestrator import run_scenario

    result = run_scenario(examples_dir / scenario_name)
    assert result.forward.detector.noise["model"] == "photodiode"
    assert result.result.n_samples == 10_000


def test_sine_with_noise_recovers_tone(examples_dir: Path) -> None:
    """The 10 mg tone is still the dominant above the noise floor."""
    from optivibe.pipeline.orchestrator import run_scenario

    result = run_scenario(examples_dir / "sine_with_noise.yaml")
    assert result.result.dominant_freqs_hz[0] == pytest.approx(200.0, abs=1.0)


# --------------------------------------------------------------------------- #
# Visualization smoke (SW-09): pure Figures build head-less.
# --------------------------------------------------------------------------- #
def test_viz_detector_figures_build_headless(
    variant_b: VariantConfig, constants: Constants, eta0_b: float
) -> None:
    """The detector figures build without Qt/pyplot (SW-09)."""
    from optivibe.viz.detector import plot_detector_timeseries, plot_noise_psd

    fs = 5000.0
    det = PhotodiodeDetector(scenario_seed=1)
    out = det.run(_optical(eta0_b, fs, np.zeros(4096)), variant_b)
    fig_psd = plot_noise_psd(variant_b, eta0_b, fs, realization=out, constants=constants)
    fig_psd_off = plot_noise_psd(variant_b, eta0_b, fs, balanced=False, constants=constants)
    fig_ts = plot_detector_timeseries(out, n_max=512)
    for fig in (fig_psd, fig_psd_off, fig_ts):
        assert fig.get_axes()
