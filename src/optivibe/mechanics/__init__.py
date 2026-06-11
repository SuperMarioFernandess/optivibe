"""Mechanics stage: base acceleration -> tip state q_tip(t).

S2 replaces the S0 stub by the modal cantilever model of docs 02/05: the
frequency-domain solver "modal" (default since S2) and the time-domain
state-space solver "modal_time", both built on
:class:`~optivibe.mechanics.cantilever.CantileverModel`. The S0 stub remains
registered under "stub" for regression. Selection: ``stages.mechanics`` in the
scenario (SW-02); the quality factor comes from the variant preset
(``q_total``, docs 07/08) and may be overridden per scenario via
``mechanics.q_total``.
"""

from __future__ import annotations

from optivibe.core.registry import Registry
from optivibe.core.stages import MechanicsStage
from optivibe.mechanics.cantilever import (
    CantileverModel,
    axial_qs_compliance,
    first_mode_hz,
    lateral_qs_compliance,
    second_mode_hz,
    tilt_coupling_per_m,
)
from optivibe.mechanics.modal import ModalFrequencyMechanics, ModalTimeMechanics
from optivibe.mechanics.stub import StubMechanics

MECHANICS_REGISTRY: Registry[MechanicsStage] = Registry("mechanics")

MECHANICS_REGISTRY.register("stub")(StubMechanics)
MECHANICS_REGISTRY.register("modal")(ModalFrequencyMechanics)
MECHANICS_REGISTRY.register("modal_time")(ModalTimeMechanics)

__all__ = [
    "MECHANICS_REGISTRY",
    "CantileverModel",
    "ModalFrequencyMechanics",
    "ModalTimeMechanics",
    "StubMechanics",
    "axial_qs_compliance",
    "first_mode_hz",
    "lateral_qs_compliance",
    "second_mode_hz",
    "tilt_coupling_per_m",
]
