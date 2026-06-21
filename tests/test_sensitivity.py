"""Tests for the switchable sensitivity model (task S6 §A; decision SW-33).

Each axis is exercised *independently* (the 24-way cross product is never
enumerated, SW-33): axis A (``calibration``), axis B (``SENSITIVITY_REGISTRY``:
static / operating_point / nonlinear_curve), axis C (``sensitivity_freq``) and
the axis-D vector seam. The headline invariant is regression equivalence:
``static + ideal + plateau`` reproduces the v1 (S5) calibration *bit-for-bit*
(``s_target`` and the recovered acceleration), so the default DSP path is
unchanged. Invalid combinations fail loudly (doc 10 §7).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optivibe.core.config.loader import load_constants, load_variant
from optivibe.core.config.models import Constants, DspOptions, SineSpec, VariantConfig
from optivibe.core.types import DetectorOutput
from optivibe.detector import PhotodiodeDetector, StubDetector
from optivibe.dsp import (
    SENSITIVITY_REGISTRY,
    NonlinearCurveSensitivity,
    OperatingPointSensitivity,
    StandardDsp,
    StaticSensitivity,
    build_sensitivity_model,
    calibrate_acceleration,
    recover_acceleration_3d,
    second_harmonic_ratio,
    target_sensitivity,
)
from optivibe.dsp.sensitivity import WORKING_POINT, TipPoint
from optivibe.dsp.spectra import amplitude_spectrum
from optivibe.excitation import EXCITATION_REGISTRY
from optivibe.mechanics.modal import ModalFrequencyMechanics
from optivibe.optics.cylinder import CylinderOptics, CylinderOpticsModel

G0 = 9.80665
FS = 5000.0
DURATION = 2.0


@pytest.fixture(scope="module")
def constants(config_dir: Path) -> Constants:
    """Physical constants bundle."""
    return load_constants(config_dir / "constants.yaml")


@pytest.fixture(scope="module")
def variant_b(config_dir: Path) -> VariantConfig:
    """Variant B preset."""
    return load_variant("B", config_dir)


def _detector(variant: VariantConfig, f0: float, amp_g: float, axis: str = "x") -> DetectorOutput:
    """Forward chain modal -> cylinder -> photodiode for a single tone."""
    spec = SineSpec(
        kind="sine", axis=axis, fs_hz=FS, duration_s=DURATION, frequency_hz=f0, amplitude_g=amp_g
    )
    exc = EXCITATION_REGISTRY.create("sine").generate(spec, seed=7)
    optical = CylinderOptics().run(ModalFrequencyMechanics().run(exc, variant), variant)
    return PhotodiodeDetector(scenario_seed=7).run(optical, variant)


# --------------------------------------------------------------------------- #
# Headline invariant: static + ideal + plateau == v1 (bit-for-bit).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_static_default_is_bit_identical_to_v1(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """The default model reproduces the v1 calibration exactly (regression-equivalence)."""
    det = _detector(variant_b, 200.0, 1.0)
    a_v1, s_v1 = calibrate_acceleration(det, variant_b, constants)  # model=None -> v1
    model = build_sensitivity_model(variant_b, DspOptions(), constants)
    assert isinstance(model, StaticSensitivity)
    a_static, s_static = calibrate_acceleration(det, variant_b, constants, model=model)
    assert s_static == s_v1  # exact float identity
    assert np.array_equal(a_static, a_v1)  # bit-for-bit


@pytest.mark.golden
def test_static_plateau_matches_documented_s_target(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """The static plateau scalar is the signed s_target ~ -1.93e-7 (doc 13 SW-33)."""
    model = StaticSensitivity(variant_b, constants)
    assert model.plateau_value == target_sensitivity(variant_b, constants)
    assert model.plateau_value == pytest.approx(-1.93e-7, rel=0.05)
    assert model.plateau_value < 0.0


@pytest.mark.golden
def test_standard_dsp_default_unchanged(variant_b: VariantConfig, constants: Constants) -> None:
    """StandardDsp with default options gives the same recovered a as the v1 calibration."""
    det = _detector(variant_b, 200.0, 1.0)
    result = StandardDsp(constants=constants).run(det, variant_b, DspOptions())
    a_v1, _ = calibrate_acceleration(det, variant_b, constants)
    assert np.array_equal(result.a, a_v1)


# --------------------------------------------------------------------------- #
# Axis B registry: keys and construction.
# --------------------------------------------------------------------------- #
def test_registry_keys() -> None:
    """The sensitivity family registers exactly the three documented strategies."""
    assert set(SENSITIVITY_REGISTRY.keys()) == {"static", "operating_point", "nonlinear_curve"}


# --------------------------------------------------------------------------- #
# Axis B: operating_point (0.37 rule).
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_operating_point_hits_037_ratio(variant_b: VariantConfig, constants: Constants) -> None:
    """operating_point rebiases to eta0/eta_peak = 0.37 exactly (doc 08 R-40)."""
    model = OperatingPointSensitivity(variant_b, constants, eta_ratio=0.37)
    optics = CylinderOpticsModel.from_config(variant_b)
    rebiased = CylinderOpticsModel(
        beam=optics.beam,
        gap_m=optics.gap_m,
        radius_of_curvature_m=optics.radius_of_curvature_m,
        bias_m=model.bias_m,
    )
    assert rebiased.eta_working_point() / rebiased.eta_peak() == pytest.approx(0.37, rel=1e-9)


def test_operating_point_distinct_from_static(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """The 0.37 operating point differs from variant B's configured working point."""
    static = StaticSensitivity(variant_b, constants)
    op = OperatingPointSensitivity(variant_b, constants, eta_ratio=0.37)
    assert abs(op.plateau_value - static.plateau_value) > 1e-12
    assert op.plateau_value < 0.0  # sign preserved


# --------------------------------------------------------------------------- #
# Axis B: nonlinear_curve.
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_nonlinear_reduces_to_static_small_signal(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """For a small tone the curve inversion agrees with the linear scalar (sign + scale)."""
    det = _detector(variant_b, 200.0, 1.0)
    i_ac = np.asarray(det.samples, dtype=np.float64) - det.dc_level
    model = NonlinearCurveSensitivity(variant_b, constants)
    a_nl = model.recover_acceleration(i_ac, det.fs)
    a_lin, _ = calibrate_acceleration(det, variant_b, constants)
    assert np.std(a_nl) == pytest.approx(np.std(a_lin), rel=1e-3)
    assert float(np.sign(np.mean(a_nl * a_lin))) == 1.0
    assert model.plateau_value == target_sensitivity(variant_b, constants)


@pytest.mark.golden
def test_nonlinear_beats_static_at_large_excursion(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """At a large optical excursion the curve inversion recovers the truth while the
    scalar leaves harmonic distortion (the point of nonlinear_curve, doc 00 / SW-33).

    Built from a *synthetic* noiseless, ADC-free I_AC so the test isolates the
    optical non-linearity (eta(dx) curvature) from ADC saturation: a clean tone
    with a ~0.3 um tip excursion stays on the monotonic working-point branch.
    """
    from optivibe.mechanics.cantilever import CantileverModel

    optics = CylinderOpticsModel.from_config(variant_b)
    cantilever = CantileverModel.from_config(constants, variant_b)
    tcl = constants.tilt_displacement_coupling_per_l
    h_qs = cantilever.h_lat_qs
    n = 8000
    t = np.arange(n) / FS
    dx_peak = 0.30e-6  # ~0.3 um excursion -> a few % optical THD (doc 03 §5)
    a_true = (dx_peak / h_qs) * np.sin(2.0 * np.pi * 50.0 * t)
    dx = h_qs * a_true
    theta_y = (tcl / variant_b.length_m) * dx
    eta = optics.eta(dx=dx, theta_y=theta_y)  # full non-linear forward coupling
    gain = variant_b.responsivity_a_w * variant_b.source.power_w
    rho = variant_b.reflector.reflectivity
    i_ac = gain * rho * (eta - optics.eta_working_point())  # no ADC, no noise

    a_nl = NonlinearCurveSensitivity(variant_b, constants).recover_acceleration(i_ac, FS)
    a_lin = i_ac / target_sensitivity(variant_b, constants)

    err_nl = float(np.std(a_nl - a_true) / np.std(a_true))
    err_lin = float(np.std(a_lin - a_true) / np.std(a_true))
    assert err_nl < 1e-6  # curve inversion is exact for pure-x motion (z-channel negligible)
    assert err_lin > 10.0 * err_nl  # the scalar materially mis-tracks at this excursion
    thd_lin = second_harmonic_ratio(amplitude_spectrum(a_lin, FS), 50.0)
    thd_nl = second_harmonic_ratio(amplitude_spectrum(a_nl, FS), 50.0)
    assert thd_nl < thd_lin


# --------------------------------------------------------------------------- #
# Axis C: sensitivity_freq plateau vs dynamic.
# --------------------------------------------------------------------------- #
def test_sensitivity_freq_plateau_is_noop(variant_b: VariantConfig, constants: Constants) -> None:
    """In band, plateau (default) leaves the recovered a unchanged vs the bare calibration."""
    det = _detector(variant_b, 200.0, 1.0)
    result = StandardDsp(constants=constants).run(
        det, variant_b, DspOptions(sensitivity_freq="plateau")
    )
    a_v1, _ = calibrate_acceleration(det, variant_b, constants)
    assert np.array_equal(result.a, a_v1)


def test_sensitivity_freq_dynamic_changes_recovery(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """sensitivity_freq=dynamic applies the D(f) deconvolution (differs from plateau)."""
    det = _detector(variant_b, 200.0, 1.0)
    plateau = StandardDsp(constants=constants).run(det, variant_b, DspOptions())
    dynamic = StandardDsp(constants=constants).run(
        det, variant_b, DspOptions(sensitivity_freq="dynamic")
    )
    assert not np.array_equal(plateau.a, dynamic.a)
    # In band (200 Hz << f1=25 kHz) the correction is tiny: the std barely moves.
    assert np.std(dynamic.a) == pytest.approx(np.std(plateau.a), rel=5e-2)


# --------------------------------------------------------------------------- #
# at() seam (axis D type) and vector guard.
# --------------------------------------------------------------------------- #
def test_at_returns_scalar_and_dynamic(variant_b: VariantConfig, constants: Constants) -> None:
    """at(working_point) is a scalar; at(freq) is a per-frequency complex array."""
    model = StaticSensitivity(variant_b, constants)
    scalar = model.at(WORKING_POINT)
    assert not scalar.is_dynamic
    assert scalar.plateau_scalar() == model.plateau_value
    dyn = model.at(WORKING_POINT, np.array([0.0, 1000.0, 25000.0]))
    assert dyn.is_dynamic
    assert np.asarray(dyn.value).shape == (3,)


def test_plateau_scalar_rejects_dynamic(variant_b: VariantConfig, constants: Constants) -> None:
    """plateau_scalar is undefined for a dynamic sample (loud)."""
    model = StaticSensitivity(variant_b, constants)
    dyn = model.at(TipPoint(), np.array([1000.0]))
    with pytest.raises(ValueError, match="QS scalar"):
        dyn.plateau_scalar()


def test_vector_inverse_is_deferred(variant_b: VariantConfig, constants: Constants) -> None:
    """The 3-D vector inverse (axis D) is reserved and fails loudly (SW-33)."""
    with pytest.raises(NotImplementedError, match="vector inverse"):
        recover_acceleration_3d(np.zeros(8), variant_b, constants)


# --------------------------------------------------------------------------- #
# Axis A and invalid-combination guards (loud failures, doc 10 §7).
# --------------------------------------------------------------------------- #
def test_nonlinear_plus_bench_fails_loudly(variant_b: VariantConfig, constants: Constants) -> None:
    """nonlinear_curve needs a measured curve; combining with bench is rejected."""
    options = DspOptions(sensitivity_model="nonlinear_curve", calibration="bench")
    with pytest.raises(ValueError, match="measured"):
        build_sensitivity_model(variant_b, options, constants)


def test_live_bench_is_deferred(variant_b: VariantConfig, constants: Constants) -> None:
    """Live-pipeline bench needs a reference excitation and is deferred (14 §8)."""
    with pytest.raises(ValueError, match="deferred"):
        build_sensitivity_model(variant_b, DspOptions(calibration="bench"), constants)


def test_unknown_model_key_fails(variant_b: VariantConfig, constants: Constants) -> None:
    """An unknown sensitivity key is rejected by the registry."""
    with pytest.raises(KeyError):
        SENSITIVITY_REGISTRY.get("does_not_exist")


def test_stub_detector_calibration_still_works(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """The model path works with the noiseless stub detector too (no NEA needed)."""
    spec = SineSpec(
        kind="sine", axis="x", fs_hz=FS, duration_s=DURATION, frequency_hz=200.0, amplitude_g=1.0
    )
    exc = EXCITATION_REGISTRY.create("sine").generate(spec, seed=7)
    optical = CylinderOptics().run(ModalFrequencyMechanics().run(exc, variant_b), variant_b)
    det = StubDetector().run(optical, variant_b)
    model = build_sensitivity_model(variant_b, DspOptions(), constants)
    a_model, _ = calibrate_acceleration(det, variant_b, constants, model=model)
    a_v1, _ = calibrate_acceleration(det, variant_b, constants)
    assert np.array_equal(a_model, a_v1)
