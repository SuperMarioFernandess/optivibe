"""Convex-sphere reflector coupling model eta(q_tip) (S9-B).

Physics (doc 03 §c-§e; S9-B addendum to 03/04)
----------------------------------------------
A convex *sphere* has the same curvature radius ``R_c`` in **both** transverse
planes (doc 03 §c table), so the focal length ``f = -R_c/2`` applies to the
x-z and the y-z plane alike. The Gaussian overlap still factorises,
``eta = eta_x * eta_y`` (doc 03 §4), but now *both* factors take the **curved**
form the cylinder used only in its power plane:

* x-z plane:  ``eta_x = eta_par_curved(g, R_c) * exp[-(d_x/w0)^2 - (alpha_x/theta0)^2]``
  with ``d_x = (2 g / R_c) * dx_eff``, ``alpha_x = (2 / R_c) * dx_eff`` and
  ``dx_eff = Delta x0 + dx + (R_c + g) * theta_y`` (doc 04 §3 lever arm);
* y-z plane:  ``eta_y = eta_par_curved(g, R_c) * exp[-(d_y/w0)^2 - (alpha_y/theta0)^2]``
  with ``d_y = (2 g / R_c) * dy_eff``, ``alpha_y = (2 / R_c) * dy_eff`` and
  ``dy_eff = Delta y0 + dy + (R_c + g) * theta_x``.

Both planes share the *same* parallel factor ``eta_par_curved`` because the
curvature is identical, so the on-axis coupling is its square. The response is
therefore **isotropic**: the two axes carry equal displacement sensitivity, the
target-to-cross anisotropy collapses to 1 and ``|d eta / d Delta y|`` equals
``|d eta / d Delta x|`` (doc 03 §c: a sphere gives the same sensitivity on both
axes). Unlike the cylinder, the cross axis is **not** suppressed -- which is
exactly why v1 picks the cylinder for single-axis recovery (doc 00 §3).

Working point. The static de-centering is applied **radially**: the model uses
``Delta y0 = Delta x0 = bias_offset_m`` so the symmetric working point sits on
equal slopes in x and y and the isotropy is exact (doc 03 §5). The gap is
``g = A + dz``; the Delta z channel enters only through ``eta_par_curved(g)``
and the ``d``-maps.

The validity guards are the cylinder's (doc 03 §6): ``R_c >= 5 w0`` and the
spot must fit the mirror, ``w(A) <= R_c/3``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optivibe.core.config.models import VariantConfig
from optivibe.core.types import FloatArray
from optivibe.optics.gaussian import GaussianBeam, eta_parallel_curved, misalignment_factor
from optivibe.optics.reflector import GaussianOverlapModel

__all__ = ["SphereOpticsModel"]

# Paraxial validity guards (doc 03 §6) -- identical to the cylinder: the sphere
# is curved in both planes, so it must be wide relative to the mode and the spot
# must fit the mirror.
_MIN_RADIUS_PER_WAIST = 5.0
_MAX_SPOT_PER_RADIUS = 1.0 / 3.0


@dataclass(frozen=True)
class SphereOpticsModel(GaussianOverlapModel):
    """Frozen coupling model of a convex *spherical* reflector (doc 03 §c).

    Parameters
    ----------
    beam : GaussianBeam
        Fiber mode (lambda from the source, w0 from the optics config).
    gap_m : float
        Nominal one-way air gap A, m.
    radius_of_curvature_m : float
        Convex-sphere curvature radius R_c, m (same in both planes).
    bias_m : float
        Intentional static radial de-centering Delta x0 = Delta y0, m
        (applied to *both* planes, so the response stays isotropic).
    """

    beam: GaussianBeam
    gap_m: float
    radius_of_curvature_m: float
    bias_m: float

    def __post_init__(self) -> None:
        if self.gap_m <= 0.0:
            msg = f"gap_m must be positive, got {self.gap_m!r}"
            raise ValueError(msg)
        if self.radius_of_curvature_m <= 0.0:
            msg = f"radius_of_curvature_m must be positive, got {self.radius_of_curvature_m!r}"
            raise ValueError(msg)
        if self.bias_m < 0.0:
            msg = f"bias_m must be non-negative, got {self.bias_m!r}"
            raise ValueError(msg)

    @classmethod
    def from_config(cls, variant: VariantConfig) -> SphereOpticsModel:
        """Build the model from a variant, validating its geometry (doc 03 §6).

        Parameters
        ----------
        variant : VariantConfig
            Sensor variant providing the reflector, source and optics blocks.

        Returns
        -------
        SphereOpticsModel
            Frozen coupling model.

        Raises
        ------
        ValueError
            If ``reflector.shape`` is not ``"sphere"``, if ``R_c`` is missing,
            if ``R_c < 5 w0`` or if the spot on the mirror exceeds ``R_c/3``.
        """
        shape = variant.reflector.shape
        if shape != "sphere":
            msg = f"SphereOptics requires reflector.shape == 'sphere', got {shape!r}"
            raise ValueError(msg)
        radius = variant.reflector.radius_of_curvature_m
        if radius is None:
            msg = "sphere reflector requires a finite radius_of_curvature_m (got None)"
            raise ValueError(msg)
        beam = GaussianBeam(
            wavelength_m=variant.source.wavelength_m,
            waist_radius_m=variant.optics.mode_field_radius_m,
        )
        gap = variant.optics.gap_m
        if radius < _MIN_RADIUS_PER_WAIST * beam.waist_radius_m:
            msg = (
                f"R_c = {radius:.3e} m violates the paraxial guard "
                f"R_c >= {_MIN_RADIUS_PER_WAIST:g} w0 = "
                f"{_MIN_RADIUS_PER_WAIST * beam.waist_radius_m:.3e} m (doc 03 §6)"
            )
            raise ValueError(msg)
        spot = float(beam.spot_radius_m(gap))
        if spot > _MAX_SPOT_PER_RADIUS * radius:
            msg = (
                f"spot w(A) = {spot:.3e} m exceeds R_c/3 = "
                f"{_MAX_SPOT_PER_RADIUS * radius:.3e} m (doc 03 §6)"
            )
            raise ValueError(msg)
        return cls(
            beam=beam,
            gap_m=gap,
            radius_of_curvature_m=radius,
            bias_m=variant.optics.bias_offset_m,
        )

    def eta_components(
        self,
        dx: FloatArray | float,
        dy: FloatArray | float,
        dz: FloatArray | float,
        theta_x: FloatArray | float,
        theta_y: FloatArray | float,
    ) -> tuple[FloatArray, FloatArray]:
        """Per-plane factors ``(eta_x, eta_y)`` -- both curved (doc 03 §c).

        Parameters
        ----------
        dx, dy, dz : numpy.ndarray or float
            Tip displacements (``dz`` changes the gap ``g = A + dz``), m.
        theta_x, theta_y : numpy.ndarray or float
            Tip tilts, rad.

        Returns
        -------
        tuple of numpy.ndarray
            ``(eta_x, eta_y)``, broadcast over the inputs. Both planes use the
            curved ABCD parallel factor and the curved geometric maps, so the
            response is isotropic (``eta_x`` and ``eta_y`` are the same function
            of their plane's effective de-centering).
        """
        radius = self.radius_of_curvature_m
        gap = self.gap_m + np.asarray(dz, dtype=np.float64)
        dx_eff = (
            self.bias_m
            + np.asarray(dx, dtype=np.float64)
            + (radius + gap) * np.asarray(theta_y, dtype=np.float64)
        )
        dy_eff = (
            self.bias_m
            + np.asarray(dy, dtype=np.float64)
            + (radius + gap) * np.asarray(theta_x, dtype=np.float64)
        )
        eta_par = eta_parallel_curved(self.beam, gap, radius)
        offset_x = (2.0 * gap / radius) * dx_eff
        angle_x = (2.0 / radius) * dx_eff
        eta_x = eta_par * misalignment_factor(self.beam, offset_x, angle_x)
        offset_y = (2.0 * gap / radius) * dy_eff
        angle_y = (2.0 / radius) * dy_eff
        eta_y = eta_par * misalignment_factor(self.beam, offset_y, angle_y)
        return np.atleast_1d(eta_x), np.atleast_1d(eta_y)
