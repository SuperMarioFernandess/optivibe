"""S0 placeholder optics, kept registered for regression (key "stub")."""

from __future__ import annotations

import numpy as np

from optivibe.core.config.models import VariantConfig
from optivibe.core.types import OpticalResponse, TipState

__all__ = ["StubOptics"]

MECHANICS_TINY = 1.0e-30  # guard against division by zero in normalization


class StubOptics:
    """Identity-like placeholder producing a bounded eta(t) around the bias.

    Warnings
    --------
    Not physical: ``eta = eta0 * (1 + 0.1 * dx_normalized)``. Used only to keep
    the signal flowing within ``[0.9, 1.1] * eta0`` in S0. The physical
    ``eta(q_tip)`` lives in :mod:`optivibe.optics.cylinder` (S3).
    """

    _modulation_depth: float = 0.1

    def run(self, tip: TipState, variant: VariantConfig) -> OpticalResponse:
        """Carry the displacement waveform as a bounded modulation of the bias."""
        bias = variant.eta_bias
        peak = float(np.max(np.abs(tip.dx)))
        normalized = tip.dx / peak if peak > MECHANICS_TINY else np.zeros_like(tip.dx)
        eta = bias * (1.0 + self._modulation_depth * normalized)
        return OpticalResponse(eta=eta, bias=bias, fs=tip.fs)
