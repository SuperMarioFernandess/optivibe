"""Reflector-shape model layer: protocol, registry and dispatching stage (S9-B).

The ``ReflectorModel`` protocol, a Gaussian overlap mixin, the shape
sub-registry and the shape-dispatching optics stage.

Why a layer (doc 09 §3; task S9-B)
----------------------------------
S3 shipped a single convex *cylinder* coupling model (:mod:`optivibe.optics.
cylinder`). The reflector family (sphere / plane / wedge) shares the very same
Gaussian-overlap framework (doc 03 §4): ``eta = eta_x * eta_y`` with per-plane
parallel factors and a transverse/angular misalignment factor. Each shape only
differs in (i) which parallel factor each plane uses (curved ABCD vs flat
defocus) and (ii) how the geometric maps ``(d, alpha)`` depend on the tip
state. This module factors that commonality:

* :class:`ReflectorModel` -- the structural contract every shape model meets
  (``eta_components`` + ``eta`` + ``eta_working_point``); identical to the
  surface the S3 :class:`~optivibe.optics.cylinder.CylinderOpticsModel` already
  exposes, so the cylinder satisfies it unchanged (bit-identity, task S9-B);
* :class:`GaussianOverlapModel` -- a mixin giving ``eta`` and
  ``eta_working_point`` in terms of a shape's ``eta_components`` so each new
  shape only implements the maps;
* :data:`REFLECTOR_MODEL_REGISTRY` -- maps ``reflector.shape`` to a model
  factory (the doc-string example of :class:`~optivibe.core.registry.Registry`);
* :class:`ReflectorOptics` -- the pipeline stage that builds the right model
  from the variant's ``reflector.shape`` and returns the same
  :class:`~optivibe.core.types.OpticalResponse` the S3 stage returned.

Physics references live with each shape model (cylinder: docs 03 §4-§5 / 04
§3-§4; sphere / plane / wedge: doc 03 §c-§e and the S9-B addendum). Numbers come
only from the variant configuration (SW-03, 10 §13); nothing is hard-coded here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from optivibe.core.registry import Registry
from optivibe.core.types import FloatArray, OpticalResponse, TipState

if TYPE_CHECKING:  # avoid importing the heavy resolved model at import time
    from optivibe.core.config.models import VariantConfig

__all__ = [
    "REFLECTOR_MODEL_REGISTRY",
    "GaussianOverlapModel",
    "ReflectorModel",
    "ReflectorOptics",
    "build_reflector_model",
]


@runtime_checkable
class ReflectorModel(Protocol):
    """Structural contract of a frozen per-shape coupling model (doc 03 §4).

    A reflector model maps a tip-state trajectory to the per-plane coupling
    factors ``(eta_x, eta_y)`` whose product is the coupling efficiency
    ``eta(q_tip)``. The convex cylinder (S3), sphere, plane and wedge all
    satisfy this protocol. ``eta_working_point`` returns the static coupling
    ``eta0`` at the configured working point (zero perturbation) and feeds
    :attr:`~optivibe.core.types.OpticalResponse.bias` (doc 04 §4).
    """

    def eta_components(
        self,
        dx: FloatArray | float,
        dy: FloatArray | float,
        dz: FloatArray | float,
        theta_x: FloatArray | float,
        theta_y: FloatArray | float,
    ) -> tuple[FloatArray, FloatArray]:
        """Return the per-plane coupling factors ``(eta_x, eta_y)``."""
        ...

    def eta(
        self,
        dx: FloatArray | float = 0.0,
        dy: FloatArray | float = 0.0,
        dz: FloatArray | float = 0.0,
        theta_x: FloatArray | float = 0.0,
        theta_y: FloatArray | float = 0.0,
    ) -> FloatArray:
        """Return the total coupling ``eta = eta_x * eta_y``."""
        ...

    def eta_working_point(self) -> float:
        """Return the static working-point coupling ``eta0`` (doc 04 §4)."""
        ...


class GaussianOverlapModel:
    """Mixin deriving ``eta`` and ``eta_working_point`` from ``eta_components``.

    Shape models that subclass this only implement :meth:`eta_components` (the
    per-plane geometric maps) and :meth:`from_config`; the total coupling and
    the working point follow from ``eta = eta_x * eta_y`` (doc 03 §4). The
    convex cylinder keeps its own copies of these methods (it predates the
    layer and must stay bit-identical), so it does *not* use this mixin.
    """

    def eta_components(
        self,
        dx: FloatArray | float,
        dy: FloatArray | float,
        dz: FloatArray | float,
        theta_x: FloatArray | float,
        theta_y: FloatArray | float,
    ) -> tuple[FloatArray, FloatArray]:
        """Per-plane factors ``(eta_x, eta_y)`` -- implemented by each shape."""
        raise NotImplementedError  # pragma: no cover - overridden by every shape

    def eta(
        self,
        dx: FloatArray | float = 0.0,
        dy: FloatArray | float = 0.0,
        dz: FloatArray | float = 0.0,
        theta_x: FloatArray | float = 0.0,
        theta_y: FloatArray | float = 0.0,
    ) -> FloatArray:
        """Total coupling ``eta = eta_x * eta_y`` (doc 03 §4).

        Parameters
        ----------
        dx, dy, dz : numpy.ndarray or float, optional
            Tip displacements (``dz`` changes the gap ``g = A + dz``), m.
        theta_x, theta_y : numpy.ndarray or float, optional
            Tip tilts, rad.

        Returns
        -------
        numpy.ndarray
            Coupling efficiency, dimensionless, broadcast over the inputs.
        """
        eta_x, eta_y = self.eta_components(dx, dy, dz, theta_x, theta_y)
        out: FloatArray = eta_x * eta_y
        return out

    def eta_working_point(self) -> float:
        """Return the static working point ``eta0 = eta(0, 0, 0, 0, 0)`` (doc 04 §4)."""
        return float(self.eta().item())


# Maps ``reflector.shape`` -> a model factory ``VariantConfig -> ReflectorModel``
# (each shape registers its ``from_config`` classmethod in
# :mod:`optivibe.optics`). The doc-string example of ``Registry`` is exactly a
# reflector family; the optics *stage* registry (``OPTICS_REGISTRY``) is a
# separate, coarser selection (stub vs physical reflector optics).
REFLECTOR_MODEL_REGISTRY: Registry[ReflectorModel] = Registry("reflector-model")


def build_reflector_model(variant: VariantConfig) -> ReflectorModel:
    """Build the coupling model for a variant's ``reflector.shape`` (S9-B).

    Parameters
    ----------
    variant : VariantConfig
        Resolved sensor variant (reflector, source, optics blocks).

    Returns
    -------
    ReflectorModel
        The frozen per-shape model, validated by its ``from_config``.

    Raises
    ------
    optivibe.core.registry.RegistryError
        If ``reflector.shape`` has no registered model.
    ValueError
        If the per-shape geometry guards fail (raised by ``from_config``).
    """
    factory = REFLECTOR_MODEL_REGISTRY.get(variant.reflector.shape)
    return factory(variant)


class ReflectorOptics:
    """Pipeline stage: ``TipState -> OpticalResponse`` dispatched by shape (S9-B).

    Registered under the optics key ``"cylinder"`` (the S3 default, kept for
    back-compat) and the alias ``"reflector"``. The shape model is built from
    the variant at each call (pure construction, no I/O) by
    :func:`build_reflector_model`; ``OpticalResponse.bias`` carries the
    *computed* working point ``eta0``. For a cylinder variant this is
    byte-for-byte the S3 :class:`~optivibe.optics.cylinder.CylinderOptics`
    behaviour (same model, same arithmetic).
    """

    def run(self, tip: TipState, variant: VariantConfig) -> OpticalResponse:
        """Compute the coupling response of a tip trajectory for any shape.

        Parameters
        ----------
        tip : TipState
            Tip-state time series (m, rad).
        variant : VariantConfig
            Sensor variant; ``reflector.shape`` selects the model.

        Returns
        -------
        OpticalResponse
            ``eta(t)`` with per-plane factors and the computed working point.
        """
        model = build_reflector_model(variant)
        eta_x, eta_y = model.eta_components(tip.dx, tip.dy, tip.dz, tip.theta_x, tip.theta_y)
        eta: FloatArray = eta_x * eta_y
        return OpticalResponse(
            eta=eta,
            bias=model.eta_working_point(),
            fs=tip.fs,
            eta_x=eta_x,
            eta_y=eta_y,
        )
