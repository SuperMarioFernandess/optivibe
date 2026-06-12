"""Optics stage: tip state -> coupling efficiency eta(t).

S3 ships the physical Gaussian-beam model of the cylindrical reflector
(docs 03/04): :mod:`optivibe.optics.gaussian` holds the beam/ABCD/overlap
formulas, :mod:`optivibe.optics.cylinder` the frozen
:class:`~optivibe.optics.cylinder.CylinderOpticsModel` and its pipeline stage
(registry key "cylinder", the default since S3). The S0 stub remains
registered under "stub" for regression. Selection: ``stages.optics`` in the
scenario (SW-02).
"""

from __future__ import annotations

from optivibe.core.registry import Registry
from optivibe.core.stages import OpticsStage
from optivibe.optics.cylinder import CylinderOptics, CylinderOpticsModel
from optivibe.optics.gaussian import (
    GaussianBeam,
    eta_parallel_curved,
    eta_parallel_flat,
    misalignment_factor,
    round_trip_q,
)
from optivibe.optics.stub import StubOptics

OPTICS_REGISTRY: Registry[OpticsStage] = Registry("optics")

OPTICS_REGISTRY.register("stub")(StubOptics)
OPTICS_REGISTRY.register("cylinder")(CylinderOptics)

__all__ = [
    "OPTICS_REGISTRY",
    "CylinderOptics",
    "CylinderOpticsModel",
    "GaussianBeam",
    "StubOptics",
    "eta_parallel_curved",
    "eta_parallel_flat",
    "misalignment_factor",
    "round_trip_q",
]
