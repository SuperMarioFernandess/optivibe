"""Wedge reflector coupling model eta(q_tip) -- a tilted plane (S9-B).

Physics (doc 03 §c; S9-B addendum to 03/04)
-------------------------------------------
A *wedge* is a flat mirror carrying a fixed built-in tilt ``alpha_w`` of its
face (doc 03 §c table: a wedge is a face-tilted plane). Reflection doubles the
surface tilt, so the return beam picks up a **constant angular offset
``2 alpha_w``** on top of the tip-tilt term -- the wedge biases the working
point in the *angular* coordinate, not by a decentering. It is otherwise a
plane: there is no curvature and hence no displacement sensitivity
(``d eta / d Delta x = d eta / d Delta y = 0``); the bias is purely angular.

Taking the wedge face tilted about ``y`` (deflecting in the power x-z plane),
the flat-mirror maps (doc 03 §d) become, with ``theta_eff = theta_y + alpha_w``:

* x-z plane:  ``d_x = 2 g * (theta_y + alpha_w)``, ``alpha_x = 2 * (theta_y + alpha_w)``;
* y-z plane:  ``d_y = 2 g * theta_x``, ``alpha_y = 2 * theta_x`` (untouched plane).

Each plane keeps the flat parallel factor ``eta_par_flat(g) = 1/sqrt(1+(g/zR)^2)``,
so ``eta = eta_x * eta_y`` and ``g = A + dz``. At ``alpha_w = 0`` every map
reduces to :class:`~optivibe.optics.plane.PlaneOpticsModel` exactly. With
``alpha_w != 0`` the aligned coupling drops as ``exp[-(2 g alpha_w/w0)^2
-(2 alpha_w/theta0)^2]`` (doc 03 §c: the connection falls off with the wedge
angle), shifting the operating point. The displacement bias ``Delta x0`` is
irrelevant for a flat mirror and is ignored.

Validity: the gap guard (``A > 0``) and the paraxial wedge-angle range
``|alpha_w| <= 0.15 rad`` (keeps the doubled return angle ``2 alpha_w`` within
the paraxial regime, doc 03 §6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optivibe.core.config.models import VariantConfig
from optivibe.core.types import FloatArray
from optivibe.optics.gaussian import GaussianBeam, eta_parallel_flat, misalignment_factor
from optivibe.optics.reflector import GaussianOverlapModel

__all__ = ["MAX_WEDGE_ANGLE_RAD", "WedgeOpticsModel"]

# Paraxial range of the built-in wedge angle (doc 03 §6): the return beam is
# deflected by 2 alpha_w, so cap |alpha_w| so that 2 alpha_w stays paraxial.
MAX_WEDGE_ANGLE_RAD = 0.15


@dataclass(frozen=True)
class WedgeOpticsModel(GaussianOverlapModel):
    """Frozen coupling model of a *wedge* reflector (tilted plane; doc 03 §c).

    Parameters
    ----------
    beam : GaussianBeam
        Fiber mode (lambda from the source, w0 from the optics config).
    gap_m : float
        Nominal one-way air gap A, m.
    wedge_angle_rad : float
        Built-in wedge (face-tilt) angle alpha_w, rad. The return beam is
        deflected by ``2 * alpha_w``; ``alpha_w = 0`` is a plane.
    """

    beam: GaussianBeam
    gap_m: float
    wedge_angle_rad: float

    def __post_init__(self) -> None:
        if self.gap_m <= 0.0:
            msg = f"gap_m must be positive, got {self.gap_m!r}"
            raise ValueError(msg)
        if abs(self.wedge_angle_rad) > MAX_WEDGE_ANGLE_RAD:
            msg = (
                f"|wedge_angle_rad| = {abs(self.wedge_angle_rad):.3e} rad exceeds the paraxial "
                f"range {MAX_WEDGE_ANGLE_RAD:g} rad (doc 03 §6)"
            )
            raise ValueError(msg)

    @classmethod
    def from_config(cls, variant: VariantConfig) -> WedgeOpticsModel:
        """Build the model from a variant (gap + angle-range guards; doc 03 §6).

        Parameters
        ----------
        variant : VariantConfig
            Sensor variant providing the reflector, source and optics blocks.
            ``optics.wedge_angle_rad`` carries the built-in wedge angle (it
            "flows" through ``VariantConfig.optics``, task S9-B §3).

        Returns
        -------
        WedgeOpticsModel
            Frozen coupling model.

        Raises
        ------
        ValueError
            If ``reflector.shape`` is not ``"wedge"`` or ``optics.wedge_angle_rad``
            is ``None`` (the gap and paraxial-angle guards are enforced in
            ``__post_init__``).
        """
        shape = variant.reflector.shape
        if shape != "wedge":
            msg = f"WedgeOptics requires reflector.shape == 'wedge', got {shape!r}"
            raise ValueError(msg)
        angle = variant.optics.wedge_angle_rad
        if angle is None:
            msg = "wedge reflector requires optics.wedge_angle_rad (got None)"
            raise ValueError(msg)
        beam = GaussianBeam(
            wavelength_m=variant.source.wavelength_m,
            waist_radius_m=variant.optics.mode_field_radius_m,
        )
        return cls(beam=beam, gap_m=variant.optics.gap_m, wedge_angle_rad=angle)

    def eta_components(
        self,
        dx: FloatArray | float,
        dy: FloatArray | float,
        dz: FloatArray | float,
        theta_x: FloatArray | float,
        theta_y: FloatArray | float,
    ) -> tuple[FloatArray, FloatArray]:
        """Per-plane factors ``(eta_x, eta_y)`` -- flat with an angular bias.

        Parameters
        ----------
        dx, dy, dz : numpy.ndarray or float
            Tip displacements (``dz`` changes the gap ``g = A + dz``), m.
            ``dx`` and ``dy`` do **not** enter (flat mirror; doc 03 §c).
        theta_x, theta_y : numpy.ndarray or float
            Tip tilts, rad. The x-z plane is driven by ``theta_y + alpha_w``
            (the wedge adds a constant ``alpha_w``); the y-z plane by ``theta_x``.

        Returns
        -------
        tuple of numpy.ndarray
            ``(eta_x, eta_y)``, broadcast over the inputs.
        """
        del dx, dy  # flat mirror: no displacement coupling (doc 03 §c)
        gap = self.gap_m + np.asarray(dz, dtype=np.float64)
        eff_y = np.asarray(theta_y, dtype=np.float64) + self.wedge_angle_rad
        theta_x_arr = np.asarray(theta_x, dtype=np.float64)
        eta_par = eta_parallel_flat(self.beam, gap)
        offset_x = 2.0 * gap * eff_y
        angle_x = 2.0 * eff_y
        eta_x = eta_par * misalignment_factor(self.beam, offset_x, angle_x)
        offset_y = 2.0 * gap * theta_x_arr
        angle_y = 2.0 * theta_x_arr
        eta_y = eta_par * misalignment_factor(self.beam, offset_y, angle_y)
        return np.atleast_1d(eta_x), np.atleast_1d(eta_y)
