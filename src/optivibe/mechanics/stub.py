"""S0 structural stub of the mechanics stage (kept for regression)."""

from __future__ import annotations

import numpy as np

from optivibe.core.config.models import VariantConfig
from optivibe.core.types import Excitation, TipState

__all__ = ["StubMechanics"]


class StubMechanics:
    """Identity placeholder mapping acceleration channels to tip channels.

    Warnings
    --------
    Not physical: ``dx/dy/dz`` are set equal to the input acceleration and the
    tilts are zero. Used only to validate the contract wiring in S0; since S2
    the physical default is "modal" and this stub stays registered under
    "stub" for regression.
    """

    def run(self, excitation: Excitation, variant: VariantConfig) -> TipState:
        """Carry the signal through (structural identity)."""
        zeros = np.zeros(excitation.n_samples, dtype=np.float64)
        return TipState(
            dx=excitation.a_x,
            dy=excitation.a_y,
            dz=excitation.a_z,
            theta_x=zeros.copy(),
            theta_y=zeros.copy(),
            fs=excitation.fs,
        )
