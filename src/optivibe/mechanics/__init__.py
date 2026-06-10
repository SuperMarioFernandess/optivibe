"""Mechanics stage: base acceleration -> tip state q_tip(t).

S0 ships a *non-physical* structural identity: it carries the time series through
unchanged so the pipeline is exercised end to end. S2 replaces it with the modal
model and the lateral transfer ``H_lat(f)`` from documents 02/05, including the
rigid tilt-displacement coupling ``theta = 1.377 * delta / L`` (doc 04 §2).
"""

from __future__ import annotations

import numpy as np

from optivibe.core.config.models import VariantConfig
from optivibe.core.registry import Registry
from optivibe.core.stages import MechanicsStage
from optivibe.core.types import Excitation, TipState

MECHANICS_REGISTRY: Registry[MechanicsStage] = Registry("mechanics")

__all__ = ["MECHANICS_REGISTRY", "StubMechanics"]


@MECHANICS_REGISTRY.register("stub")
class StubMechanics:
    """Identity placeholder mapping acceleration channels to tip channels.

    Warnings
    --------
    Not physical: ``dx/dy/dz`` are set equal to the input acceleration and the
    tilts are zero. Used only to validate the contract wiring in S0. The
    quantitative model arrives in S2.
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
