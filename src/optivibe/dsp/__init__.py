"""Inverse/DSP stage: detector samples -> reconstructed vibration (S5).

Two implementations of the :class:`~optivibe.core.stages.DspStage` protocol are
registered:

``StubDsp`` (key ``"stub"``, the default)
    Minimal uncalibrated inverse kept for regression (the prior S1-S4 dominants
    are reproduced unchanged). See :mod:`optivibe.dsp.stub`.

``StandardDsp`` (key ``"standard"``, S5)
    The physically calibrated inverse chain of documents 05/07: calibration
    against ``s_target`` (A/V -> m/s^2), drift-suppressed ``a -> v -> x``
    integration, Welch/spectrogram spectra, dominant lines, ISO 10816/20816
    severity, the cross-axis residual and the noise-equivalent acceleration
    (NEA). Selected explicitly per scenario via ``stages.dsp``; it is *not* the
    default because it changes the scale/meaning of ``VibrationResult.a``
    (calibrated SI vs raw photocurrent) and its NEA/ISO metrics rely on the
    photodiode detector's noise metadata (decision SW-S5-01). See
    :mod:`optivibe.dsp.standard`.

The submodules expose reusable helpers (calibration, kinematics, spectra,
metrics, NEA, ISO) so ``viz`` and the tests can call them directly (14 §8:
spectra are computed in ``dsp/``, ``viz`` only draws).
"""

from __future__ import annotations

from optivibe.core.registry import Registry
from optivibe.core.stages import DspStage
from optivibe.dsp.calibration import (
    bench_sensitivity,
    calibrate_acceleration,
    detector_ac_current,
    dynamic_sensitivity,
    target_sensitivity,
    target_sensitivity_via_multiplier,
)
from optivibe.dsp.iso import classify_velocity_rms, iso_assessment
from optivibe.dsp.kinematics import (
    INTEGRATOR_REGISTRY,
    integrate_frequency,
    integrate_time,
)
from optivibe.dsp.metrics import (
    band_rms_velocity,
    cross_axis_suppression,
    rms,
    second_harmonic_ratio,
)
from optivibe.dsp.nea import (
    NeaResult,
    analytic_noise_psd,
    nea_from_detector,
    nea_spectrum,
)
from optivibe.dsp.sensitivity import (
    SENSITIVITY_REGISTRY,
    NonlinearCurveSensitivity,
    OperatingPointSensitivity,
    Sensitivity,
    SensitivityModel,
    StaticSensitivity,
    TipPoint,
    build_sensitivity_model,
    recover_acceleration_3d,
)
from optivibe.dsp.spectra import (
    amplitude_spectrum,
    dominant_frequencies,
    spectrogram,
    welch_psd,
)
from optivibe.dsp.standard import StandardDsp
from optivibe.dsp.stub import StubDsp

DSP_REGISTRY: Registry[DspStage] = Registry("dsp")

# Register the two stage implementations under their keys (default = "stub").
DSP_REGISTRY.register("stub")(StubDsp)
DSP_REGISTRY.register("standard")(StandardDsp)

__all__ = [
    "DSP_REGISTRY",
    "INTEGRATOR_REGISTRY",
    "SENSITIVITY_REGISTRY",
    "NeaResult",
    "NonlinearCurveSensitivity",
    "OperatingPointSensitivity",
    "Sensitivity",
    "SensitivityModel",
    "StandardDsp",
    "StaticSensitivity",
    "StubDsp",
    "TipPoint",
    "amplitude_spectrum",
    "analytic_noise_psd",
    "band_rms_velocity",
    "bench_sensitivity",
    "build_sensitivity_model",
    "calibrate_acceleration",
    "classify_velocity_rms",
    "cross_axis_suppression",
    "detector_ac_current",
    "dominant_frequencies",
    "dynamic_sensitivity",
    "integrate_frequency",
    "integrate_time",
    "iso_assessment",
    "nea_from_detector",
    "nea_spectrum",
    "recover_acceleration_3d",
    "rms",
    "second_harmonic_ratio",
    "spectrogram",
    "target_sensitivity",
    "target_sensitivity_via_multiplier",
    "welch_psd",
]
