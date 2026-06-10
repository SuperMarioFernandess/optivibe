"""Detector stage: optical response -> digitized photodetector signal.

S0 applies the documented intensity read-out ``S = R * P * (R1 + rho * eta)``
(doc 04 §4, route 2) with no noise and no quantization. The affine read-out is
real; only the noise model (shot / RIN / Johnson / electronics), the
transimpedance front-end and the ADC are stubbed and arrive in S4 (doc 07).
"""

from __future__ import annotations

from optivibe.core.config.models import VariantConfig
from optivibe.core.registry import Registry
from optivibe.core.stages import DetectorStage
from optivibe.core.types import DetectorOutput, OpticalResponse

DETECTOR_REGISTRY: Registry[DetectorStage] = Registry("detector")

__all__ = ["DETECTOR_REGISTRY", "StubDetector"]


@DETECTOR_REGISTRY.register("stub")
class StubDetector:
    """Noiseless intensity read-out ``S = R * P * (R1 + rho * eta)`` (doc 04 §4).

    Warnings
    --------
    No noise and no quantization in S0; the noise budget and ADC arrive in S4.
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
