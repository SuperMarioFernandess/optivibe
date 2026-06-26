"""Optics stage: tip state -> coupling efficiency eta(t).

S3 shipped the physical Gaussian-beam model of the convex *cylinder*
(docs 03/04). S9-B generalises it to a **reflector family** behind a shape
layer (:mod:`optivibe.optics.reflector`):

* :mod:`optivibe.optics.gaussian` holds the beam / ABCD / overlap formulas;
* :mod:`optivibe.optics.reflector` defines the ``ReflectorModel`` protocol, the
  ``REFLECTOR_MODEL_REGISTRY`` (shape -> frozen model) and the shape-dispatching
  :class:`~optivibe.optics.reflector.ReflectorOptics` stage;
* :mod:`optivibe.optics.cylinder` / :mod:`~optivibe.optics.sphere` /
  :mod:`~optivibe.optics.plane` / :mod:`~optivibe.optics.wedge` provide the four
  per-shape models (cylinder kept byte-for-byte; sphere isotropic; plane the
  ``R_c -> inf`` reference; wedge a tilted plane).

The optics *stage* registry ``OPTICS_REGISTRY`` keeps the S0 ``"stub"`` and the
physical reflector optics under ``"cylinder"`` (the S3 default key, kept for
back-compat) and the self-documenting alias ``"reflector"``; both dispatch by
``reflector.shape``. Selection: ``stages.optics`` in the scenario (SW-02);
the shape itself comes from the variant's reflector preset (SW-43).
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
from optivibe.optics.plane import PlaneOpticsModel
from optivibe.optics.reflector import (
    REFLECTOR_MODEL_REGISTRY,
    GaussianOverlapModel,
    ReflectorModel,
    ReflectorOptics,
    build_reflector_model,
)
from optivibe.optics.sphere import SphereOpticsModel
from optivibe.optics.stub import StubOptics
from optivibe.optics.wedge import WedgeOpticsModel

# Reflector-shape models keyed by ``reflector.shape`` (doc 03 §c; S9-B). Each
# factory is the model's ``from_config`` classmethod; the cylinder is registered
# here (rather than self-registering) so its S3 module stays byte-for-byte.
REFLECTOR_MODEL_REGISTRY.register("cylinder")(CylinderOpticsModel.from_config)
REFLECTOR_MODEL_REGISTRY.register("sphere")(SphereOpticsModel.from_config)
REFLECTOR_MODEL_REGISTRY.register("plane")(PlaneOpticsModel.from_config)
REFLECTOR_MODEL_REGISTRY.register("wedge")(WedgeOpticsModel.from_config)

OPTICS_REGISTRY: Registry[OpticsStage] = Registry("optics")

OPTICS_REGISTRY.register("stub")(StubOptics)
OPTICS_REGISTRY.register("cylinder")(ReflectorOptics)  # S3 default key (back-compat)
OPTICS_REGISTRY.register("reflector")(ReflectorOptics)  # self-documenting alias (S9-B)

__all__ = [
    "OPTICS_REGISTRY",
    "REFLECTOR_MODEL_REGISTRY",
    "CylinderOptics",
    "CylinderOpticsModel",
    "GaussianBeam",
    "GaussianOverlapModel",
    "PlaneOpticsModel",
    "ReflectorModel",
    "ReflectorOptics",
    "SphereOpticsModel",
    "StubOptics",
    "WedgeOpticsModel",
    "build_reflector_model",
    "eta_parallel_curved",
    "eta_parallel_flat",
    "misalignment_factor",
    "round_trip_q",
]
