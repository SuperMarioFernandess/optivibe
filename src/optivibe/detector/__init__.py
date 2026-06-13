"""Detector stage: optical response -> digitized photodetector signal.

Two implementations of the :class:`~optivibe.core.stages.DetectorStage`
protocol are registered:

``StubDetector`` (key ``"stub"``, the S0 default)
    Noiseless, un-quantized affine read-out ``S = R * P * (R1 + rho * eta)``
    (doc 04 §4, route 2). Kept for regression: the prior dominants of every
    S1-S3 scenario are reproduced exactly.

``PhotodiodeDetector`` (key ``"photodiode"``, S4)
    The physical read-out of document 07 -- shot / RIN / Johnson noise, the
    balanced reference channel (R-23) and an AC-coupled ADC. Selected explicitly
    per scenario via ``stages.detector``; it is *not* the default because the
    realistic noise floor would bury the ~1e-9 cross-axis residual of
    ``cross_axis`` and shift the regression dominants (see journal SW-27).

Selection: ``stages.detector`` in the scenario (SW-02). The scenario seed is
forwarded to ``PhotodiodeDetector`` by the orchestrator (the S2 options pattern),
since the ``run()`` protocol carries no seed.
"""

from __future__ import annotations

from optivibe.core.config.models import VariantConfig
from optivibe.core.registry import Registry
from optivibe.core.stages import DetectorStage
from optivibe.core.types import DetectorOutput, OpticalResponse
from optivibe.detector.photodiode import (
    PhotodiodeDetector,
    detector_seed_sequence,
    noise_psd,
    signal_multiplier,
)

DETECTOR_REGISTRY: Registry[DetectorStage] = Registry("detector")

__all__ = [
    "DETECTOR_REGISTRY",
    "PhotodiodeDetector",
    "StubDetector",
    "detector_seed_sequence",
    "noise_psd",
    "signal_multiplier",
]


@DETECTOR_REGISTRY.register("stub")
class StubDetector:
    """Noiseless intensity read-out ``S = R * P * (R1 + rho * eta)`` (doc 04 §4).

    Warnings
    --------
    No noise and no quantization (S0); the physical read-out is
    :class:`~optivibe.detector.photodiode.PhotodiodeDetector` (S4).
    """

    def run(self, optical: OpticalResponse, variant: VariantConfig) -> DetectorOutput:
        """Convert the coupling response to photocurrent samples (amperes)."""
        responsivity = variant.responsivity_a_w
        power = variant.source.power_w
        r1 = variant.endface_reflectivity
        rho = variant.reflector.reflectivity
        gain = responsivity * power
        samples = gain * (r1 + rho * optical.eta)
        dc_level = float(gain * (r1 + rho * optical.bias))
        return DetectorOutput(
            samples=samples,
            fs=optical.fs,
            dc_level=dc_level,
            units="A",
            noise={"model": "none", "stage": "S0-stub"},
        )


DETECTOR_REGISTRY.register("photodiode")(PhotodiodeDetector)
