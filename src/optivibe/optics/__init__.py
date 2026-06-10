"""Optics stage: tip state -> coupling efficiency eta(t).

S0 ships a bounded, *non-physical* carrier: a small normalized modulation of the
target displacement around the bias working point, so the waveform survives to
the detector while ``eta`` stays in a sane range. S3 replaces it with the
Gaussian-beam model ``eta(Δx)`` + bias, the cylinder geometry and the
sensitivity matrix ``dS/dq`` (documents 03/04).
"""

from __future__ import annotations

import numpy as np

from optivibe.core.config.models import VariantConfig
from optivibe.core.registry import Registry
from optivibe.core.stages import OpticsStage
from optivibe.core.types import OpticalResponse, TipState

MECHANICS_TINY = 1.0e-30  # guard against division by zero in normalization

OPTICS_REGISTRY: Registry[OpticsStage] = Registry("optics")

__all__ = ["OPTICS_REGISTRY", "StubOptics"]


@OPTICS_REGISTRY.register("stub")
class StubOptics:
    """Identity-like placeholder producing a bounded eta(t) around the bias.

    Warnings
    --------
    Not physical: ``eta = eta0 * (1 + 0.1 * dx_normalized)``. Used only to keep
    the signal flowing within ``[0.9, 1.1] * eta0`` in S0. The real ``eta(Δx)``
    arrives in S3.
    """

    _modulation_depth: float = 0.1

    def run(self, tip: TipState, variant: VariantConfig) -> OpticalResponse:
        """Carry the displacement waveform as a bounded modulation of the bias."""
        bias = variant.eta_bias
        peak = float(np.max(np.abs(tip.dx)))
        normalized = tip.dx / peak if peak > MECHANICS_TINY else np.zeros_like(tip.dx)
        eta = bias * (1.0 + self._modulation_depth * normalized)
        return OpticalResponse(eta=eta, bias=bias, fs=tip.fs)
