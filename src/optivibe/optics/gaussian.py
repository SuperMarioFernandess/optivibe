"""Gaussian-beam formulas of the optical subsystem (doc 03 §a-§d).

This module holds the *formulas* shared by reflector geometries: the beam
descriptor (w0, zR, theta0), the round-trip ABCD propagation fiber -> gap ->
mirror -> gap -> fiber, the mode-overlap (parallel) coupling factors and the
transverse/angular misalignment factor of the returned beam. Numbers (lambda,
w0, A, R_c) come exclusively from the variant configuration (mirror of docs
03/08) — nothing is hard-coded here (SW-03, 10 §13).

Conventions (doc 03 §b-§d)
--------------------------
* The fiber mode is a Gaussian beam with waist w0 at the endface: q_in = i zR.
* A convex mirror of curvature radius R_c acts as a diverging mirror with focal
  length f = -R_c/2 (doc 03 §c); a flat mirror is the f -> inf limit, so its
  round trip is free propagation over 2g: q_out = q_in + 2g.
* Round trip over a gap ``g``: M = [[1 - g/f, 2g - g^2/f], [-1/f, 1 - g/f]].
* Mode overlap of the aligned returned beam with the fiber mode (doc 03 §d):
  ``eta_par = 2 sqrt(zR Im q_out) / | -i zR - q_out |``.
* Misalignment factor (doc 03 §d, R-59): the **full** overlap of the mode with
  the returned beam of parameter ``q_out``, offset ``d`` and tilt ``alpha``,
  ``exp[-(C_dd d^2 + C_da d alpha + C_aa alpha^2)]`` with the coefficients of
  :func:`overlap_coefficients`. The returned beam of a convex mirror is
  *mismatched* to the mode (``q_out != i zR``), and the cross term ``C_da``
  (negative for a diverging return) almost cancels the angular one; the former
  form ``exp[-(d/w0)^2 - (alpha/theta0)^2]`` is only the matched limit
  ``q_out -> q_in`` (revoked by R-59 for every reflector shape).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from optivibe.core.types import FloatArray

ComplexArray = npt.NDArray[np.complex128]

__all__ = [
    "GaussianBeam",
    "eta_parallel_curved",
    "eta_parallel_flat",
    "flat_round_trip_q",
    "misalignment_factor",
    "overlap_coefficients",
    "round_trip_q",
]


@dataclass(frozen=True)
class GaussianBeam:
    """Fundamental fiber mode as a Gaussian beam (doc 03 §1).

    Parameters
    ----------
    wavelength_m : float
        Vacuum wavelength lambda, m (air index taken as 1, doc 03 §2).
    waist_radius_m : float
        Mode-field radius w0 at the endface, m (5.2 um at 1550 nm, SMF-28).
    """

    wavelength_m: float
    waist_radius_m: float

    def __post_init__(self) -> None:
        if self.wavelength_m <= 0.0:
            msg = f"wavelength_m must be positive, got {self.wavelength_m!r}"
            raise ValueError(msg)
        if self.waist_radius_m <= 0.0:
            msg = f"waist_radius_m must be positive, got {self.waist_radius_m!r}"
            raise ValueError(msg)

    @property
    def rayleigh_range_m(self) -> float:
        """Rayleigh range ``zR = pi w0^2 / lambda``, m (54.8 um at 1550 nm)."""
        return float(np.pi * self.waist_radius_m**2 / self.wavelength_m)

    @property
    def divergence_rad(self) -> float:
        """Far-field half-divergence ``theta0 = lambda / (pi w0)``, rad (94.9 mrad)."""
        return float(self.wavelength_m / (np.pi * self.waist_radius_m))

    def spot_radius_m(self, distance_m: FloatArray | float) -> FloatArray:
        """Beam radius ``w(z) = w0 sqrt(1 + (z/zR)^2)`` at distance z, m.

        Parameters
        ----------
        distance_m : numpy.ndarray or float
            Propagation distance from the waist, m.

        Returns
        -------
        numpy.ndarray
            Spot radius, m (broadcast over the input).
        """
        z = np.asarray(distance_m, dtype=np.float64)
        out: FloatArray = self.waist_radius_m * np.sqrt(1.0 + (z / self.rayleigh_range_m) ** 2)
        return out


def round_trip_q(beam: GaussianBeam, gap_m: FloatArray | float, focal_m: float) -> ComplexArray:
    """Round-trip complex beam parameter q_out at the fiber plane (doc 03 §3).

    ABCD matrix of gap -> mirror(f) -> gap applied to ``q_in = i zR``:
    ``M = [[1 - g/f, 2g - g^2/f], [-1/f, 1 - g/f]]``, ``q_out = (A q + B)/(C q + D)``.

    Parameters
    ----------
    beam : GaussianBeam
        Fiber mode.
    gap_m : numpy.ndarray or float
        One-way gap g (possibly time-varying ``A + dz``), m.
    focal_m : float
        Mirror focal length f, m (``-R_c/2`` for a convex mirror).

    Returns
    -------
    numpy.ndarray of complex
        q_out (broadcast over ``gap_m``).
    """
    g = np.asarray(gap_m, dtype=np.float64)
    m11 = 1.0 - g / focal_m
    m12 = 2.0 * g - g**2 / focal_m
    m21 = -1.0 / focal_m
    m22 = m11
    q_in = 1j * beam.rayleigh_range_m
    out: ComplexArray = (m11 * q_in + m12) / (m21 * q_in + m22)
    return out


def flat_round_trip_q(beam: GaussianBeam, gap_m: FloatArray | float) -> ComplexArray:
    """Round-trip beam parameter of a *flat* mirror, ``q_out = i zR + 2g`` (doc 03 §b).

    The ``f -> inf`` limit of :func:`round_trip_q`: the return is free
    propagation of the mode over the full path ``2g``. Used by the flat plane
    of the cylinder and by the plane and wedge reflectors (doc 03 §d, R-59).

    Parameters
    ----------
    beam : GaussianBeam
        Fiber mode.
    gap_m : numpy.ndarray or float
        One-way gap g (possibly time-varying ``A + dz``), m.

    Returns
    -------
    numpy.ndarray of complex
        q_out (broadcast over ``gap_m``).
    """
    g = np.asarray(gap_m, dtype=np.float64)
    out: ComplexArray = np.asarray(2.0 * g + 1j * beam.rayleigh_range_m, dtype=np.complex128)
    return out


def eta_parallel_curved(
    beam: GaussianBeam, gap_m: FloatArray | float, radius_of_curvature_m: float
) -> FloatArray:
    """Aligned mode-overlap in the *curved* plane of a convex mirror (doc 03 §4).

    ``eta_par^x = 2 sqrt(zR Im q_out) / | -i zR - q_out |`` with
    ``q_out = round_trip_q(beam, g, f = -R_c/2)``.

    Closed form (doc 03 §d): ``eta_par^x = R_c zR / sqrt((g^2 + zR^2)((R_c + g)^2 + zR^2))``.
    Limits: ``R_c -> inf`` reproduces :func:`eta_parallel_flat`; ``g -> 0``
    gives ``R_c / sqrt(R_c^2 + zR^2)`` -- **not** 1: the convex mirror imprints
    its curvature on the returned wavefront even at zero gap (0.752 at
    ``R_c = 62.5 um``, 1550 nm; doc 03 §g).

    Parameters
    ----------
    beam : GaussianBeam
        Fiber mode.
    gap_m : numpy.ndarray or float
        One-way gap, m.
    radius_of_curvature_m : float
        Convex-mirror curvature radius R_c > 0, m.

    Returns
    -------
    numpy.ndarray
        Overlap factor in (0, 1] (broadcast over ``gap_m``).
    """
    if radius_of_curvature_m <= 0.0:
        msg = f"radius_of_curvature_m must be positive, got {radius_of_curvature_m!r}"
        raise ValueError(msg)
    z_r = beam.rayleigh_range_m
    q_out = round_trip_q(beam, gap_m, -radius_of_curvature_m / 2.0)
    out: FloatArray = 2.0 * np.sqrt(z_r * q_out.imag) / np.abs(-1j * z_r - q_out)
    return out


def eta_parallel_flat(beam: GaussianBeam, gap_m: FloatArray | float) -> FloatArray:
    """Aligned mode-overlap in the *flat* plane (cylinder axis; doc 03 §4).

    ``eta_par^y = 1 / sqrt(1 + (g / zR)^2)`` — the flat-mirror image-waist
    overlap; equals the ``R_c -> inf`` limit of :func:`eta_parallel_curved`.

    Parameters
    ----------
    beam : GaussianBeam
        Fiber mode.
    gap_m : numpy.ndarray or float
        One-way gap, m.

    Returns
    -------
    numpy.ndarray
        Overlap factor in (0, 1] (broadcast over ``gap_m``).
    """
    g = np.asarray(gap_m, dtype=np.float64)
    out: FloatArray = 1.0 / np.sqrt(1.0 + (g / beam.rayleigh_range_m) ** 2)
    return out


def overlap_coefficients(
    beam: GaussianBeam, q_out: ComplexArray | complex
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Coefficients ``(C_dd, C_da, C_aa)`` of the full mode overlap (doc 03 §d, R-59).

    The per-plane overlap of the fiber mode (``q_in = i zR``) with a returned
    Gaussian beam of parameter ``q_out``, offset ``d`` and tilt ``alpha`` is
    ``eta_par * exp[-(C_dd d^2 + C_da d alpha + C_aa alpha^2)]`` with
    ``a = i k / (2 q_out) = 1/w^2 + i k / (2 R_wave)``, ``b = 1/w0^2`` and

    * ``C_dd = 2 b (|a|^2 + b Re a) / |a + b|^2``,
    * ``C_da = -2 k b Im a / |a + b|^2``,
    * ``C_aa = k^2 (Re a + b) / (2 |a + b|^2)``.

    Matched limit ``q_out -> q_in`` (``a -> b``): ``C_dd -> 1/w0^2``,
    ``C_da -> 0``, ``C_aa -> 1/theta0^2``. A diverging return (convex mirror,
    ``Im a > 0``) gives ``C_da < 0``: for ``d alpha > 0`` the cross term lowers
    the loss (the wavefront curvature partly compensates the angular walk-off;
    doc 03 §d').

    Parameters
    ----------
    beam : GaussianBeam
        Fiber mode.
    q_out : numpy.ndarray or complex
        Complex beam parameter of the returned beam at the fiber plane, m.

    Returns
    -------
    tuple of numpy.ndarray
        ``(C_dd [1/m^2], C_da [1/(m rad)], C_aa [1/rad^2])`` broadcast over
        ``q_out``.
    """
    q = np.asarray(q_out, dtype=np.complex128)
    k = 2.0 * np.pi / beam.wavelength_m
    a = 1j * k / (2.0 * q)
    b = 1.0 / beam.waist_radius_m**2
    norm = np.abs(a + b) ** 2
    c_dd: FloatArray = 2.0 * b * (np.abs(a) ** 2 + b * a.real) / norm
    c_da: FloatArray = -2.0 * k * b * a.imag / norm
    c_aa: FloatArray = k**2 * (a.real + b) / (2.0 * norm)
    return c_dd, c_da, c_aa


def misalignment_factor(
    beam: GaussianBeam,
    offset_m: FloatArray | float,
    angle_rad: FloatArray | float,
    *,
    q_out: ComplexArray | complex,
) -> FloatArray:
    """Full-overlap misalignment factor of the returned beam (doc 03 §d, R-59).

    ``exp[-(C_dd d^2 + C_da d alpha + C_aa alpha^2)]`` with the coefficients
    of :func:`overlap_coefficients` evaluated at the returned beam parameter
    ``q_out``. ``d`` is the transverse offset and ``alpha`` the angular tilt of
    the returned beam relative to the fiber mode at the endface (field
    convention ``exp(-ikz)``).

    ``q_out`` is **required** (keyword-only): the former three-argument form
    ``exp[-(d/w0)^2 - (alpha/theta0)^2]`` is the matched limit
    ``q_out = i zR`` and is wrong for the mismatched return of a convex or
    flat mirror (R-59, S-30). Passing ``q_out = 1j * beam.rayleigh_range_m``
    reproduces it identically.

    Closed forms (doc 03 §d, special cases verified by golden tests): on the
    maps of a curved plane (``d = (2g/R_c) D``, ``alpha = (2/R_c) D``) the factor
    is ``exp(-D^2/sigma_eff^2)`` with ``sigma_eff = (w0/2) sqrt(1 + ((R_c + g)/zR)^2)``;
    on a flat plane (``d = 2 g theta``, ``alpha = 2 theta``) it is
    ``exp[-(2 theta/theta0)^2]``, independent of the gap.

    Parameters
    ----------
    beam : GaussianBeam
        Fiber mode.
    offset_m : numpy.ndarray or float
        Transverse offset d, m.
    angle_rad : numpy.ndarray or float
        Angular misalignment alpha, rad.
    q_out : numpy.ndarray or complex
        Returned-beam parameter at the fiber plane (:func:`round_trip_q` for a
        curved mirror, :func:`flat_round_trip_q` for a flat one), m.

    Returns
    -------
    numpy.ndarray
        Misalignment factor in (0, 1] (broadcast over the inputs).
    """
    d = np.asarray(offset_m, dtype=np.float64)
    alpha = np.asarray(angle_rad, dtype=np.float64)
    c_dd, c_da, c_aa = overlap_coefficients(beam, q_out)
    out: FloatArray = np.exp(-(c_dd * d**2 + c_da * d * alpha + c_aa * alpha**2))
    return out
