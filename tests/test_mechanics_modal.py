"""S2 mechanics tests: golden references (docs 02/05/08), limits (11 §7),
integration on the S1 excitation family, and property invariants (10 §10)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.signal import welch

from optivibe.core.config import load_constants, load_variant
from optivibe.core.config.models import Constants, MultitoneSpec, RandomSpec, SweepSpec
from optivibe.core.types import Excitation
from optivibe.excitation import EXCITATION_REGISTRY
from optivibe.mechanics import (
    MECHANICS_REGISTRY,
    CantileverModel,
    ModalFrequencyMechanics,
    ModalTimeMechanics,
    axial_qs_compliance,
    first_mode_hz,
    lateral_qs_compliance,
    second_mode_hz,
)

VARIANTS = ("A", "B", "C", "D")


@pytest.fixture(scope="module")
def constants(config_dir: Path) -> Constants:
    return load_constants(config_dir / "constants.yaml")


def _g0(constants: Constants) -> float:
    return constants.universal.g0_m_s2


def _sine_excitation(
    freq_hz: float, amplitude_g: float, fs: float, duration_s: float, g0: float, axis: str = "x"
) -> Excitation:
    t = np.arange(round(duration_s * fs)) / fs
    signal = amplitude_g * g0 * np.sin(2.0 * np.pi * freq_hz * t)
    zeros = np.zeros_like(signal)
    channels = {"x": zeros.copy(), "y": zeros.copy(), "z": zeros.copy()}
    channels[axis] = signal
    return Excitation(a_x=channels["x"], a_y=channels["y"], a_z=channels["z"], fs=fs)


# --------------------------------------------------------------------------- #
# Golden: derived quantities against the documented references (11 §7).
# --------------------------------------------------------------------------- #
pytestmark_golden = pytest.mark.golden


@pytest.mark.golden
@pytest.mark.parametrize("name", VARIANTS)
def test_golden_f1_scaling_per_variant(name: str, constants: Constants, config_dir: Path) -> None:
    """f1(L) matches the documented 100/L[mm]^2 kHz law within 2 % (08 R-31)."""
    variant = load_variant(name, config_dir=config_dir)
    length_mm = variant.length_m * 1.0e3
    expected_hz = 100.0e3 / length_mm**2
    assert first_mode_hz(constants, variant.length_m) == pytest.approx(expected_hz, rel=2.0e-2)


@pytest.mark.golden
@pytest.mark.parametrize("name", VARIANTS)
def test_golden_h_qs_scaling_per_variant(name: str, constants: Constants, config_dir: Path) -> None:
    """H_lat^QS matches 0.0384 * L[mm]^4 nm/g within 2 % (docs 02 §6, 08 R-31)."""
    variant = load_variant(name, config_dir=config_dir)
    length_mm = variant.length_m * 1.0e3
    expected_m_per_g = 0.0384 * length_mm**4 * 1.0e-9
    h_qs_per_g = lateral_qs_compliance(constants, variant.length_m) * _g0(constants)
    assert h_qs_per_g == pytest.approx(expected_m_per_g, rel=2.0e-2)


@pytest.mark.golden
def test_golden_second_mode_ratio(constants: Constants) -> None:
    """f2 / f1 = (beta2/beta1)^2 ~ 6.27 (doc 02 §2/§8)."""
    ratio = second_mode_hz(constants, 3.0e-3) / first_mode_hz(constants, 3.0e-3)
    assert ratio == pytest.approx(6.27, rel=2.0e-3)


@pytest.mark.golden
def test_golden_resonant_peak_is_q_times_qs(constants: Constants, config_dir: Path) -> None:
    """|H(f1)| = Q * H_QS within 5 % (docs 02 §7.1, 11 §7)."""
    variant = load_variant("D", config_dir=config_dir)
    model = CantileverModel.from_config(constants, variant)
    peak = float(np.abs(model.h_lat(model.f1_hz))[0])
    assert peak == pytest.approx(model.q_total * model.h_lat_qs, rel=5.0e-2)


@pytest.mark.golden
def test_golden_axial_compliance_reference(constants: Constants) -> None:
    """dz/a_z = rho L^2 / (2E) ~ 1.4 pm/g at L = 3 mm within 5 % (doc 02 §7.2)."""
    per_g = axial_qs_compliance(constants, 3.0e-3) * _g0(constants)
    assert per_g == pytest.approx(1.4e-12, rel=5.0e-2)


@pytest.mark.golden
def test_golden_tilt_coupling_exact(constants: Constants, config_dir: Path) -> None:
    """theta_y / dx == 1.377 / L exactly, sample by sample (docs 02 §7.1, 05 §3.3)."""
    variant = load_variant("B", config_dir=config_dir)
    stage = ModalFrequencyMechanics(constants=constants)
    excitation = _sine_excitation(120.0, 1.0, 8000.0, 0.25, _g0(constants))
    tip = stage.run(excitation, variant)
    coupling = constants.tilt_displacement_coupling_per_l / variant.length_m
    np.testing.assert_array_equal(tip.theta_y, coupling * tip.dx)
    np.testing.assert_array_equal(tip.theta_x, coupling * tip.dy)


# --------------------------------------------------------------------------- #
# Limits and dimensions (11 §7).
# --------------------------------------------------------------------------- #
def test_limit_dynamic_factor_dc_is_unity(constants: Constants, config_dir: Path) -> None:
    """f -> 0 => |D| -> 1 exactly (doc 05 §2)."""
    model = CantileverModel.from_config(constants, load_variant("B", config_dir=config_dir))
    d0 = model.dynamic_factor(0.0)[0]
    assert d0 == pytest.approx(1.0 + 0.0j, abs=0.0)


def test_limit_dynamic_factor_at_resonance(constants: Constants, config_dir: Path) -> None:
    """|D(f1)| = Q and phase(D(f1)) = -90 deg exactly (docs 05 §2, 11 §7)."""
    model = CantileverModel.from_config(constants, load_variant("D", config_dir=config_dir))
    d1 = model.dynamic_factor(model.f1_hz)[0]
    assert abs(d1) == pytest.approx(model.q_total, rel=1.0e-12)
    assert math.degrees(np.angle(d1)) == pytest.approx(-90.0, abs=1.0e-9)


def test_limit_dynamic_factor_shape(constants: Constants, config_dir: Path) -> None:
    """|D| is monotone-rising below f1 and falls past it; phase passes -180 deg."""
    model = CantileverModel.from_config(constants, load_variant("B", config_dir=config_dir))
    freq = np.linspace(0.0, 2.0 * model.f1_hz, 4001)
    mag = np.abs(model.dynamic_factor(freq))
    below = mag[freq < 0.95 * model.f1_hz]
    assert np.all(np.diff(below) > 0.0)
    assert abs(model.dynamic_factor(2.0 * model.f1_hz)[0]) < 1.0
    phase_past = np.angle(model.dynamic_factor(5.0 * model.f1_hz)[0])
    assert math.degrees(phase_past) == pytest.approx(-180.0, abs=1.0)


def test_dimensions_plateau_sine_amplitude(constants: Constants, config_dir: Path) -> None:
    """1 g sine deep on the plateau -> dx amplitude == H_QS * g0 (m in, m out)."""
    variant = load_variant("A", config_dir=config_dir)  # f1 ~ 4 kHz, drive at 5 Hz
    stage = ModalFrequencyMechanics(constants=constants)
    excitation = _sine_excitation(5.0, 1.0, 2000.0, 2.0, _g0(constants))
    tip = stage.run(excitation, variant)
    expected = lateral_qs_compliance(constants, variant.length_m) * _g0(constants)
    assert float(np.max(np.abs(tip.dx))) == pytest.approx(expected, rel=1.0e-3)
    # Untouched axes stay identically zero.
    assert not np.any(tip.dy)
    assert not np.any(tip.dz)


def test_axial_channel_is_quasistatic(constants: Constants, config_dir: Path) -> None:
    """a_z maps through the frequency-independent compliance rho L^2/(2E)."""
    variant = load_variant("B", config_dir=config_dir)
    stage = ModalFrequencyMechanics(constants=constants)
    excitation = _sine_excitation(440.0, 2.0, 8000.0, 0.5, _g0(constants), axis="z")
    tip = stage.run(excitation, variant)
    compliance = axial_qs_compliance(constants, variant.length_m)
    np.testing.assert_allclose(tip.dz, compliance * excitation.a_z, rtol=0.0, atol=0.0)
    assert not np.any(tip.dx)
    assert not np.any(tip.theta_y)


def test_y_axis_response_equals_x_axis_response(constants: Constants, config_dir: Path) -> None:
    """Axisymmetric cantilever: H_y == H_x (doc 02 §1/§7.3)."""
    variant = load_variant("B", config_dir=config_dir)
    stage = ModalFrequencyMechanics(constants=constants)
    on_x = stage.run(_sine_excitation(300.0, 1.0, 8000.0, 0.5, _g0(constants), axis="x"), variant)
    on_y = stage.run(_sine_excitation(300.0, 1.0, 8000.0, 0.5, _g0(constants), axis="y"), variant)
    np.testing.assert_array_equal(on_y.dy, on_x.dx)
    np.testing.assert_array_equal(on_y.theta_x, on_x.theta_y)


def test_q_override_changes_resonant_response(constants: Constants, config_dir: Path) -> None:
    """The scenario-level q_total override reaches the model (docs 07/08)."""
    variant = load_variant("D", config_dir=config_dir)
    default_model = ModalFrequencyMechanics(constants=constants).model(variant)
    overridden = ModalFrequencyMechanics(constants=constants, q_total=3000.0).model(variant)
    assert default_model.q_total == pytest.approx(variant.q_total)
    assert overridden.q_total == pytest.approx(3000.0)
    ratio = (
        np.abs(overridden.h_lat(overridden.f1_hz))[0]
        / np.abs(default_model.h_lat(default_model.f1_hz))[0]
    )
    assert ratio == pytest.approx(3000.0 / variant.q_total, rel=1.0e-9)


def test_registry_keys_present() -> None:
    """ "modal" and "modal_time" are registered; the stub stays for regression."""
    assert "modal" in MECHANICS_REGISTRY
    assert "modal_time" in MECHANICS_REGISTRY
    assert "stub" in MECHANICS_REGISTRY


# --------------------------------------------------------------------------- #
# Integration on the S1 excitation family (14 §8).
# --------------------------------------------------------------------------- #
def test_integration_sweep_recovers_frf(constants: Constants, config_dir: Path) -> None:
    """Sweep through f1 (variant D): the spectral ratio dx/a_x reproduces the
    analytic plateau + resonance curve (doc 05 §1)."""
    variant = load_variant("D", config_dir=config_dir)
    spec = SweepSpec(
        kind="sweep",
        axis="x",
        fs_hz=20000.0,
        duration_s=2.0,
        f_start_hz=500.0,
        f_end_hz=6500.0,
        amplitude_g=0.001,
        method="linear",
    )
    excitation = EXCITATION_REGISTRY.create("sweep").generate(spec, seed=1)
    tip = ModalFrequencyMechanics(constants=constants).run(excitation, variant)

    n = excitation.n_samples
    freq = np.fft.rfftfreq(n, d=1.0 / excitation.fs)
    spec_in = np.fft.rfft(excitation.a_x)
    spec_out = np.fft.rfft(tip.dx)
    # Bins with solid input energy, away from the sweep edges.
    band = (freq >= 800.0) & (freq <= 6000.0) & (np.abs(spec_in) > 0.05 * np.abs(spec_in).max())
    measured = np.abs(spec_out[band] / spec_in[band])

    # Independent analytic transcription of H_lat(f) (docs 02 §6, 05 §1).
    fiber = constants.fiber
    h_qs = (
        fiber.density_kg_m3
        * fiber.area_m2
        * variant.length_m**4
        / (8.0 * fiber.youngs_modulus_pa * fiber.inertia_m4)
    )
    f1 = first_mode_hz(constants, variant.length_m)
    r = freq[band] / f1
    analytic = h_qs / np.abs(1.0 - r**2 + 1j * r / variant.q_total)

    np.testing.assert_allclose(measured, analytic, rtol=2.0e-2)
    # The curve actually shows plateau + resonance: dynamic range sanity.
    assert measured.max() / measured.min() > 10.0


def test_integration_multitone_plateau_vs_near_resonance(
    constants: Constants, config_dir: Path
) -> None:
    """Two equal tones (plateau and near f1): response ratio = |D(f_res)|/|D(f_pl)|."""
    variant = load_variant("B", config_dir=config_dir)  # f1 ~ 25.0 kHz
    f_plateau, f_near = 100.0, 24000.0
    spec = MultitoneSpec(
        kind="multitone",
        axis="x",
        fs_hz=80000.0,
        duration_s=1.0,
        tones=(
            {"frequency_hz": f_plateau, "amplitude_g": 0.5},
            {"frequency_hz": f_near, "amplitude_g": 0.5},
        ),
    )
    excitation = EXCITATION_REGISTRY.create("multitone").generate(spec, seed=1)
    model = CantileverModel.from_config(constants, variant)
    tip = ModalFrequencyMechanics(constants=constants).run(excitation, variant)

    n = excitation.n_samples
    spectrum = np.abs(np.fft.rfft(tip.dx))
    freq = np.fft.rfftfreq(n, d=1.0 / excitation.fs)
    amp_plateau = spectrum[np.argmin(np.abs(freq - f_plateau))]
    amp_near = spectrum[np.argmin(np.abs(freq - f_near))]

    expected = float(
        np.abs(model.dynamic_factor(f_near))[0] / np.abs(model.dynamic_factor(f_plateau))[0]
    )
    assert amp_near / amp_plateau == pytest.approx(expected, rel=1.0e-2)
    assert expected > 5.0  # the near-resonance tone is genuinely amplified


def test_integration_random_psd_shaping(constants: Constants, config_dir: Path) -> None:
    """Random(PSD) input: output PSD = input PSD * |H|^2 (Welch, in tolerance)."""
    variant = load_variant("B", config_dir=config_dir)
    spec = RandomSpec(
        kind="random",
        axis="x",
        fs_hz=8000.0,
        duration_s=8.0,
        band_hz=(50.0, 3000.0),
        psd_g2_hz=1.0e-4,
    )
    excitation = EXCITATION_REGISTRY.create("random").generate(spec, seed=7)
    model = CantileverModel.from_config(constants, variant)
    tip = ModalFrequencyMechanics(constants=constants).run(excitation, variant)

    nperseg = 2048
    freq, pxx = welch(excitation.a_x, fs=excitation.fs, nperseg=nperseg)
    _, pyy = welch(tip.dx, fs=excitation.fs, nperseg=nperseg)
    band = (freq >= 100.0) & (freq <= 2800.0)
    transfer_sq = pyy[band] / pxx[band]
    expected = np.abs(model.h_lat(freq[band])) ** 2
    ratio = transfer_sq / expected
    # Welch leakage at the band edges averages out; the median must sit on 1.
    assert float(np.median(ratio)) == pytest.approx(1.0, rel=5.0e-2)
    np.testing.assert_allclose(ratio, 1.0, rtol=0.25)


def test_time_solver_matches_frequency_solver_on_sine(
    constants: Constants, config_dir: Path
) -> None:
    """Steady-state sine: "modal_time" agrees with "modal" (plateau and f1/2)."""
    variant = load_variant("D", config_dir=config_dir)  # f1 ~ 5.005 kHz
    freq_stage = ModalFrequencyMechanics(constants=constants)
    time_stage = ModalTimeMechanics(constants=constants)
    for drive_hz, rel_tol in ((500.0, 2.0e-2), (2500.0, 5.0e-2)):
        excitation = _sine_excitation(drive_hz, 0.001, 50000.0, 0.5, _g0(constants))
        steady = slice(excitation.n_samples // 2, None)  # discard the start-up transient
        rms_freq = float(np.sqrt(np.mean(freq_stage.run(excitation, variant).dx[steady] ** 2)))
        rms_time = float(np.sqrt(np.mean(time_stage.run(excitation, variant).dx[steady] ** 2)))
        assert rms_time == pytest.approx(rms_freq, rel=rel_tol)


def test_scenario_resonance_sweep_runs_and_peaks_at_f1(repo_root: Path) -> None:
    """examples/resonance_sweep.yaml runs through the orchestrator; the
    dominant response frequency lands on f1 of variant D (~5.005 kHz)."""
    from optivibe.pipeline import run_scenario

    artifacts = run_scenario(
        repo_root / "examples" / "resonance_sweep.yaml", config_dir=repo_root / "configs"
    )
    assert artifacts.result.dominant_freqs_hz[0] == pytest.approx(5005.0, abs=25.0)


# --------------------------------------------------------------------------- #
# Property invariants (hypothesis; 10 §10).
# --------------------------------------------------------------------------- #
@settings(max_examples=20, deadline=None)
@given(scale=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False))
def test_property_linearity_in_amplitude(scale: float) -> None:
    """Scaling the input by k scales every tip channel by k (linear model)."""
    constants = load_constants(_repo_configs() / "constants.yaml")
    variant = load_variant("B", config_dir=_repo_configs())
    stage = ModalFrequencyMechanics(constants=constants)
    base = _sine_excitation(120.0, 1.0, 8000.0, 0.25, constants.universal.g0_m_s2)
    scaled = Excitation(a_x=scale * base.a_x, a_y=base.a_y, a_z=base.a_z, fs=base.fs)
    tip_base = stage.run(base, variant)
    tip_scaled = stage.run(scaled, variant)
    np.testing.assert_allclose(tip_scaled.dx, scale * tip_base.dx, rtol=1.0e-9, atol=1.0e-30)
    np.testing.assert_allclose(
        tip_scaled.theta_y, scale * tip_base.theta_y, rtol=1.0e-9, atol=1.0e-30
    )


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_property_seed_invariance(seed: int) -> None:
    """One seed -> one TipState: the chain excitation -> mechanics is reproducible."""
    constants = load_constants(_repo_configs() / "constants.yaml")
    variant = load_variant("B", config_dir=_repo_configs())
    spec = RandomSpec(
        kind="random",
        axis="x",
        fs_hz=4000.0,
        duration_s=0.5,
        band_hz=(20.0, 1000.0),
        g_rms=1.0,
    )
    stage = ModalFrequencyMechanics(constants=constants)
    source = EXCITATION_REGISTRY.create("random")
    tip_a = stage.run(source.generate(spec, seed=seed), variant)
    tip_b = stage.run(source.generate(spec, seed=seed), variant)
    np.testing.assert_array_equal(tip_a.dx, tip_b.dx)
    np.testing.assert_array_equal(tip_a.theta_y, tip_b.theta_y)


def _repo_configs() -> Path:
    """Repository configs/ for the hypothesis tests (fixtures unavailable there)."""
    return Path(__file__).resolve().parents[1] / "configs"
