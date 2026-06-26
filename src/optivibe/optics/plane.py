"""Plane reflector coupling model eta(q_tip) -- the R_c -> inf reference (S9-B).

Physics (doc 03 §c-§e, §g; S9-B addendum to 03/04)
--------------------------------------------------
A flat mirror is the ``R_c -> inf`` limit: there is **no focusing**, so the
curvature maps vanish (``2/R_c -> 0``) and a pure lateral shift of the mirror
leaves the overlap unchanged to first order. The plane therefore has **zero
displacement sensitivity** in both axes, ``d eta / d Delta x = d eta / d Delta y
= 0`` (doc 03 §c: a plane gives zero sensitivity on both axes; doc 03 §g: the
flat-mirror insensitivity is a known result). It is the curvature that creates
sensitivity to displacement -- which is why the cylinder and sphere do, and the
plane does not.

What remains are the two flat-mirror channels (doc 03 §d, flat plane):

* tilt: a tip tilt ``theta`` sends the return beam off by ``2 theta``, giving
  ``d = 2 g * theta``, ``alpha = 2 * theta`` in each plane;
* gap (defocus): the aligned overlap is ``eta_par_flat(g) = 1/sqrt(1+(g/zR)^2)``
  in **both** planes, so the on-axis ceiling is ``1/(1+(g/zR)^2)`` (doc 03 §c).

Hence ``eta = eta_x * eta_y`` with, for each plane,
``eta_i = eta_par_flat(g) * exp[-(d_i/w0)^2 - (alpha_i/theta0)^2]`` and the
x-z plane driven by ``theta_y``, the y-z plane by ``theta_x``. The gap is
``g = A + dz``. The displacement bias ``Delta x0`` is irrelevant for a flat
mirror (no displacement coupling) and is ignored.

Validity: only the gap guard (``A > 0``) applies -- the finite-aperture guard
``w(A) <= R_c/3`` is vacuous for an (idealised) infinite plane (doc 03 §6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optivibe.core.config.models import VariantConfig
from optivibe.core.types import FloatArray
from optivibe.optics.gaussian import GaussianBeam, eta_parallel_flat, misalignment_factor
from optivibe.optics.reflector import GaussianOverlapModel

__all__ = ["PlaneOpticsModel"]


@dataclass(frozen=True)
class PlaneOpticsModel(GaussianOverlapModel):
    """Frozen coupling model of a *plane* reflector (R_c -> inf; doc 03 §c).

    Parameters
    ----------
    beam : GaussianBeam
        Fiber mode (lambda from the source, w0 from the optics config).
    gap_m : float
        Nominal one-way air gap A, m.
    """

    beam: GaussianBeam
    gap_m: float

    def __post_init__(self) -> None:
        if self.gap_m <= 0.0:
            msg = f"gap_m must be positive, got {self.gap_m!r}"
            raise ValueError(msg)

    @classmethod
    def from_config(cls, variant: VariantConfig) -> PlaneOpticsModel:
        """Build the model from a variant (gap guard only; doc 03 §6).

        Parameters
        ----------
        variant : VariantConfig
            Sensor variant providing the reflector, source and optics blocks.
            ``reflector.radius_of_curvature_m`` may be ``None`` (the plane is
            the ``R_c -> inf`` limit and does not use it).

        Returns
        -------
        PlaneOpticsModel
            Frozen coupling model.

        Raises
        ------
        ValueError
            If ``reflector.shape`` is not ``"plane"`` (the gap positivity guard
            is enforced in ``__post_init__``).
        """
        shape = variant.reflector.shape
        if shape != "plane":
            msg = f"PlaneOptics requires reflector.shape == 'plane', got {shape!r}"
            raise ValueError(msg)
        beam = GaussianBeam(
            wavelength_m=variant.source.wavelength_m,
            waist_radius_m=variant.optics.mode_field_radius_m,
        )
        return cls(beam=beam, gap_m=variant.optics.gap_m)

    def eta_components(
        self,
        dx: FloatArray | float,
        dy: FloatArray | float,
        dz: FloatArray | float,
        theta_x: FloatArray | float,
        theta_y: FloatArray | float,
    ) -> tuple[FloatArray, FloatArray]:
        """Per-plane factors ``(eta_x, eta_y)`` -- both flat (doc 03 §c-§d).

        Parameters
        ----------
        dx, dy, dz : numpy.ndarray or float
            Tip displacements (``dz`` changes the gap ``g = A + dz``), m.
            ``dx`` and ``dy`` do **not** enter: a flat mirror has no
            displacement coupling (doc 03 §c).
        theta_x, theta_y : numpy.ndarray or float
            Tip tilts, rad -- the only transverse drivers (through ``2 theta``).

        Returns
        -------
        tuple of numpy.ndarray
            ``(eta_x, eta_y)``, broadcast over the inputs.
        """
        del dx, dy  # flat mirror: no displacement coupling (doc 03 §c)
        gap = self.gap_m + np.asarray(dz, dtype=np.float64)
        theta_y_arr = np.asarray(theta_y, dtype=np.float64)
        theta_x_arr = np.asarray(theta_x, dtype=np.float64)
        eta_par = eta_parallel_flat(self.beam, gap)
        offset_x = 2.0 * gap * theta_y_arr
        angle_x = 2.0 * theta_y_arr
        eta_x = eta_par * misalignment_factor(self.beam, offset_x, angle_x)
        offset_y = 2.0 * gap * theta_x_arr
        angle_y = 2.0 * theta_x_arr
        eta_y = eta_par * misalignment_factor(self.beam, offset_y, angle_y)
        return np.atleast_1d(eta_x), np.atleast_1d(eta_y)
