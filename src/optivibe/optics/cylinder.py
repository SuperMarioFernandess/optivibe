"""Cylindrical-reflector coupling model eta(q_tip) and its pipeline stage (S3).

Physics (docs 03 §d-§e, 04 §3-§4; R-59)
---------------------------------------
The convex *cylindrical* mirror (curved in the x-z plane, flat along y) makes
the coupling factorize per plane, ``eta = eta_x * eta_y``. Each factor is the
**full** overlap of the fiber mode with the returned beam (doc 03 §d, R-59):
``eta_i = eta_par^i(g) * exp[-(C_dd d_i^2 + C_da d_i alpha_i + C_aa alpha_i^2)]``,
coefficients of the returned-beam parameter ``q_out`` (:func:`~optivibe.optics.
gaussian.overlap_coefficients`).

* curved plane: ``q_out`` of the round trip with ``f = -R_c/2``; maps
  ``d_x = (2 g / R_c) * dx_eff``, ``alpha_x = (2 / R_c) * dx_eff`` with the
  *effective* decentering ``dx_eff = Delta x0 + dx + (R_c + g) * theta_y``
  (R-43: tilt acts as a displacement through the lever arm ``R_c + g``, the
  distance from the endface to the centre of curvature);
* flat plane: ``q_out = i zR + 2g``; maps ``d_y = 2 g * theta_x``,
  ``alpha_y = 2 * theta_x`` -- *independent of dy* (translation along the
  cylinder axis is a symmetry of the mirror).

The gap is ``g = A + dz``. Closed forms (doc 03 §d, R-59; identities pinned by
golden tests): ``eta_x = eta_par^x(g) * exp(-dx_eff^2 / sigma_eff^2)`` with
``sigma_eff = (w0/2) sqrt(1 + ((R_c + g)/zR)^2) = w(R_c + g)/2`` -- half the
beam radius in the plane of the centre of curvature, ``= sigma_base /
eta_par^x`` -- and ``eta_y = eta_par^y(g) * exp[-(2 theta_x/theta0)^2]`` (angular
width ``theta0/2``, independent of the gap). Because ``eta_x`` is an exact
Gaussian in ``dx_eff``, the working point, slopes and bias rule keep closed
forms in ``sigma_eff``.

Reference values (doc 03 §d'', variant A: w0 = 5.2 um, A = 31 um,
R_c = 62.5 um, Delta x0 = 2 um): ``sigma_eff = 5.1415 um``, ``eta_peak = 0.4369``,
``eta0 = 0.3755``, bare slope ``-5.683e4 1/m``. The S3 calibration of A = 31 um
(journal 2026-06-12) anchored the *pre*-R-59 multiplier to the doc 04/05
references; after R-59 those references moved (doc 03 §d''). This is a model
statement: L3 is not passed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from optivibe.core.config.models import VariantConfig
from optivibe.core.types import FloatArray
from optivibe.optics.gaussian import (
    GaussianBeam,
    eta_parallel_curved,
    eta_parallel_flat,
    flat_round_trip_q,
    misalignment_factor,
    overlap_coefficients,
    round_trip_q,
)
from optivibe.optics.reflector import ReflectorOptics

__all__ = ["CylinderOptics", "CylinderOpticsModel"]

# Validity guards of the Gaussian/paraxial model (doc 03 §6):
# the mirror must be wide relative to the mode (R_c >= ~5 w0, i.e. >= 26 um at
# w0 = 5.2 um) and the spot on the mirror must fit the cylinder (w(A) <= R_c/3).
_MIN_RADIUS_PER_WAIST = 5.0
_MAX_SPOT_PER_RADIUS = 1.0 / 3.0


@dataclass(frozen=True)
class CylinderOpticsModel:
    """Frozen coupling model of one variant's cylindrical reflector (doc 03).

    Parameters
    ----------
    beam : GaussianBeam
        Fiber mode (lambda from the source, w0 from the optics config).
    gap_m : float
        Nominal one-way air gap A, m.
    radius_of_curvature_m : float
        Convex-cylinder curvature radius R_c, m.
    bias_m : float
        Intentional static de-centering Delta x0 (working point), m.
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
    def from_config(cls, variant: VariantConfig) -> CylinderOpticsModel:
        """Build the model from a variant preset, validating its geometry.

        Parameters
        ----------
        variant : VariantConfig
            Sensor variant providing the reflector, source and optics blocks
            (mirrors of docs 03/08).

        Returns
        -------
        CylinderOpticsModel
            Frozen coupling model.

        Raises
        ------
        ValueError
            If the reflector shape is not "cylinder", if R_c < 5 w0, or if the
            spot on the mirror exceeds R_c/3 (doc 03 §6 validity guards).
        """
        shape = variant.reflector.shape
        if shape != "cylinder":
            msg = f"CylinderOptics requires reflector.shape == 'cylinder', got {shape!r}"
            raise ValueError(msg)
        beam = GaussianBeam(
            wavelength_m=variant.source.wavelength_m,
            waist_radius_m=variant.optics.mode_field_radius_m,
        )
        radius = variant.reflector.radius_of_curvature_m
        if radius is None:
            msg = "cylinder reflector requires a finite radius_of_curvature_m (got None)"
            raise ValueError(msg)
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

    # ------------------------------------------------------------------ #
    # Vectorized response.
    # ------------------------------------------------------------------ #
    def eta_components(
        self,
        dx: FloatArray | float,
        dy: FloatArray | float,
        dz: FloatArray | float,
        theta_x: FloatArray | float,
        theta_y: FloatArray | float,
    ) -> tuple[FloatArray, FloatArray]:
        """Per-plane coupling factors (eta_x, eta_y) for a tip trajectory.

        Parameters
        ----------
        dx, dy, dz : numpy.ndarray or float
            Tip displacements (dz changes the gap: ``g = A + dz``), m.
        theta_x, theta_y : numpy.ndarray or float
            Tip tilts, rad.

        Returns
        -------
        tuple of numpy.ndarray
            ``(eta_x, eta_y)``, each broadcast over the inputs.

        Notes
        -----
        ``dy`` is accepted for interface symmetry but does not enter the maps:
        translation along the cylinder axis is a mirror symmetry (doc 03 §4).
        Cross-axis sensitivity arises only through the mechanically coupled
        tilt ``theta_x = 1.377 dy / L`` and is quadratic at the working point.
        """
        del dy  # symmetry of the cylinder: no dependence on dy (doc 03 §4)
        radius = self.radius_of_curvature_m
        gap = self.gap_m + np.asarray(dz, dtype=np.float64)
        dx_eff = (
            self.bias_m
            + np.asarray(dx, dtype=np.float64)
            + (radius + gap) * np.asarray(theta_y, dtype=np.float64)
        )
        offset_x = (2.0 * gap / radius) * dx_eff
        angle_x = (2.0 / radius) * dx_eff
        eta_x = eta_parallel_curved(self.beam, gap, radius) * misalignment_factor(
            self.beam, offset_x, angle_x, q_out=round_trip_q(self.beam, gap, -radius / 2.0)
        )
        theta_x_arr = np.asarray(theta_x, dtype=np.float64)
        offset_y = 2.0 * gap * theta_x_arr
        angle_y = 2.0 * theta_x_arr
        eta_y = eta_parallel_flat(self.beam, gap) * misalignment_factor(
            self.beam, offset_y, angle_y, q_out=flat_round_trip_q(self.beam, gap)
        )
        return np.atleast_1d(eta_x), np.atleast_1d(eta_y)

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
            Tip displacements, m.
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

    # ------------------------------------------------------------------ #
    # Working-point scalars (closed forms, docs 03 §e / 04 §4; R-59).
    # ------------------------------------------------------------------ #
    @property
    def sigma_m(self) -> float:
        """Gaussian width ``sigma_eff`` of eta(dx_eff) at the nominal gap, m (R-59).

        Evaluated from the same full overlap that drives :meth:`eta`: on the
        curved-plane maps the exponent per unit ``dx_eff^2`` is
        ``1/sigma^2 = (2/R_c)^2 (C_dd A^2 + C_da A + C_aa)``. It equals the closed
        form ``sigma_eff = (w0/2) sqrt(1 + ((R_c + A)/zR)^2) = sigma_base / eta_par^x``
        of doc 03 §d (golden identity; 5.1415 um for variant A, 3.9257 um for
        B/C/D, doc 03 §d''). The pre-R-59 width
        ``sigma_base = (w0/2) R_c / sqrt(A^2 + zR^2)`` (2.5808 / 1.2801 um) belonged
        to the matched-beam multiplier and is revoked.
        """
        radius = self.radius_of_curvature_m
        gap = self.gap_m
        c_dd, c_da, c_aa = overlap_coefficients(
            self.beam, round_trip_q(self.beam, gap, -radius / 2.0)
        )
        inv_sigma_sq = float((2.0 / radius) ** 2 * (c_dd * gap**2 + c_da * gap + c_aa))
        return 1.0 / math.sqrt(inv_sigma_sq)

    def eta_peak(self) -> float:
        """Aligned-peak coupling ``eta_par^x(A) * eta_par^y(A)`` (dx_eff = 0)."""
        return float(
            eta_parallel_curved(self.beam, self.gap_m, self.radius_of_curvature_m)
            * eta_parallel_flat(self.beam, self.gap_m)
        )

    def eta_working_point(self) -> float:
        """Return the static working point ``eta0 = eta_peak * exp(-(Dx0/sigma_eff)^2)``."""
        return float(self.eta().item())

    def slope_dx(self) -> float:
        """Analytical bare slope ``d eta / d dx = -2 eta0 Delta x0 / sigma_eff^2``, 1/m.

        Exact derivative of the Gaussian ``eta(dx_eff)`` at the working point
        (docs 03 §e / 04 §4, R-59; reference -5.683e4 1/m for variant A, doc 03
        §d''; the pre-R-59 -1.44e5 is revoked). Zero when the bias is zero — at
        the peak the response is purely quadratic.
        """
        return -2.0 * self.eta_working_point() * self.bias_m / self.sigma_m**2

    def tilt_multiplier(self, length_m: float, tilt_coupling_per_l: float) -> float:
        """Tilt-enhancement factor ``1 + 1.377 (R_c + A) / L`` (doc 05 §1).

        Parameters
        ----------
        length_m : float
            Cantilever length L, m.
        tilt_coupling_per_l : float
            Dimensionless coupling ``theta L / delta`` (1.377, doc 01).

        Returns
        -------
        float
            Multiplier applied to the bare slope (> 1).
        """
        if length_m <= 0.0:
            msg = f"length_m must be positive, got {length_m!r}"
            raise ValueError(msg)
        return 1.0 + tilt_coupling_per_l * (self.radius_of_curvature_m + self.gap_m) / length_m

    def effective_slope_dx(self, length_m: float, tilt_coupling_per_l: float) -> float:
        """Effective slope including the mechanical tilt, 1/m (docs 04 §4, 05 §1).

        ``(d eta / d dx)_eff = slope_dx * [1 + 1.377 (R_c + A) / L]`` — the
        tip tilt theta_y = 1.377 dx / L adds a displacement through the lever
        arm R_c + A (R-43). References (doc 03 §d'', R-59): -5.829e4 1/m for
        variant A (L = 5 mm), -6.205e4 1/m at L = 1.4 mm on the same platform.

        Parameters
        ----------
        length_m : float
            Cantilever length L, m.
        tilt_coupling_per_l : float
            Dimensionless coupling ``theta L / delta`` (1.377, doc 01).

        Returns
        -------
        float
            Effective slope, 1/m.
        """
        return self.slope_dx() * self.tilt_multiplier(length_m, tilt_coupling_per_l)

    def anisotropy(self, length_m: float, tilt_coupling_per_l: float) -> float:
        """Target-to-cross angular-map anisotropy ``L / (1.377 R_c)`` (doc 04 §6).

        Ratio of the angular misalignment produced per unit dx (through the
        curved-plane map) to that per unit dy (through the tilt-only flat-plane
        map): ~35x at L = 3 mm, R_c = 62.5 um.

        After R-59 this is the ratio of the angular **maps**, not of the
        sensitivities: the curved-plane angular term is almost cancelled by the
        overlap cross term, and the width ratio is
        ``M L / (1.377 sqrt((R_c + A)^2 + zR^2))`` (doc 03 §e). Which ratio is the
        tolerance metric is an open decision (backlog M-25); the value returned
        here is unchanged.

        Parameters
        ----------
        length_m : float
            Cantilever length L, m.
        tilt_coupling_per_l : float
            Dimensionless coupling ``theta L / delta`` (1.377, doc 01).

        Returns
        -------
        float
            Anisotropy factor, dimensionless.
        """
        if length_m <= 0.0:
            msg = f"length_m must be positive, got {length_m!r}"
            raise ValueError(msg)
        return length_m / (tilt_coupling_per_l * self.radius_of_curvature_m)

    def bias_for_eta_ratio(self, ratio: float) -> float:
        """Bias Delta x0 placing the working point at ``eta0 = ratio * eta_peak``.

        Closed form of the SNR-optimum rule eta0 ~ 0.37 eta_peak (doc 08,
        R-40/O-05): since eta(dx_eff) is exactly Gaussian,
        ``Delta x0 = sigma_eff * sqrt(-ln ratio)`` -- 5.1267 um (R_c = 62.5 um)
        and 3.9144 um (R_c = 31 um) at the 0.37 rule (doc 03 §d'', R-59).
        Exposed as a design helper (S4/S6); in v1 the bias is a static config
        value (the configured biases are not changed here -- backlog M-25).

        Parameters
        ----------
        ratio : float
            Target ``eta0 / eta_peak`` in (0, 1].

        Returns
        -------
        float
            Bias offset, m.

        Raises
        ------
        ValueError
            If ``ratio`` is outside (0, 1].
        """
        if not 0.0 < ratio <= 1.0:
            msg = f"ratio must be in (0, 1], got {ratio!r}"
            raise ValueError(msg)
        return self.sigma_m * math.sqrt(-math.log(ratio))


class CylinderOptics(ReflectorOptics):
    """Back-compat alias of the shape-dispatching optics stage (S3 -> S9-B).

    Historically (S3) this *was* the cylinder-only stage. Since S9-B the optics
    stage dispatches by ``reflector.shape`` through
    :class:`~optivibe.optics.reflector.ReflectorOptics`; ``CylinderOptics`` is
    kept as a thin subclass so existing imports
    (``from optivibe.optics.cylinder import CylinderOptics``) and the registry
    key ``"cylinder"`` keep working unchanged. For a cylinder variant the
    inherited :meth:`~optivibe.optics.reflector.ReflectorOptics.run` builds the
    very same :class:`CylinderOpticsModel`, so the output is byte-for-byte the
    S3 behaviour. The cylinder *model* (:class:`CylinderOpticsModel`) still
    refuses non-cylinder shapes in :meth:`CylinderOpticsModel.from_config`.
    """
