"""Mechanics stage: base acceleration -> tip state q_tip(t).

S2 replaces the S0 stub by the modal cantilever model of docs 02/05: the
frequency-domain solver "modal" (default since S2) and the time-domain
state-space solver "modal_time", both built on
:class:`~optivibe.mechanics.cantilever.CantileverModel`. The S0 stub remains
registered under "stub" for regression. Selection: ``stages.mechanics`` in the
scenario (SW-02); the quality factor comes from the variant preset
(``q_total``, docs 07/08) and may be overridden per scenario via
``mechanics.q_total``. Since M-02 a composition may omit ``q_total`` to use
the computable damping model Q(L) (:mod:`optivibe.mechanics.damping`: air +
anchor + internal losses, docs 02 §5 / 07 §2.3). Since M-12 the mechanics also
exposes the Brownian thermal floor of the mode (:mod:`optivibe.mechanics.thermal`:
``NEA_th = sqrt(4 kB T omega_1 / (Q M_a))``, ``M_a ~ 0.58 rho S L``, doc 07 §2),
consumed by the NEA budget as the fourth branch (doc 17 §2, ``(+)NEA_th``).
"""

from __future__ import annotations

from optivibe.core.registry import Registry
from optivibe.core.stages import MechanicsStage
from optivibe.mechanics.cantilever import (
    CantileverModel,
    axial_qs_compliance,
    first_mode_hz,
    first_mode_shape,
    lateral_qs_compliance,
    second_mode_hz,
    tilt_coupling_per_m,
)
from optivibe.mechanics.damping import (
    damping_budget,
    hydrodynamic_function,
    knudsen_number,
    q_air,
    q_anchor,
    q_total_model,
    reynolds_number,
)
from optivibe.mechanics.modal import ModalFrequencyMechanics, ModalTimeMechanics
from optivibe.mechanics.stub import StubMechanics
from optivibe.mechanics.thermal import (
    acceleration_effective_mass,
    kinetic_effective_mass,
    nea_thermal,
    thermal_force_psd,
)

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
    "acceleration_effective_mass",
    "axial_qs_compliance",
    "damping_budget",
    "first_mode_hz",
    "first_mode_shape",
    "hydrodynamic_function",
    "kinetic_effective_mass",
    "knudsen_number",
    "lateral_qs_compliance",
    "nea_thermal",
    "q_air",
    "q_anchor",
    "q_total_model",
    "reynolds_number",
    "second_mode_hz",
    "thermal_force_psd",
    "tilt_coupling_per_m",
]
