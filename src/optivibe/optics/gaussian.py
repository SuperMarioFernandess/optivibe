"""Gaussian-beam formulas of the optical subsystem (doc 03 §1-§4).

This module holds the *formulas* shared by reflector geometries: the beam
descriptor (w0, zR, theta0), the round-trip ABCD propagation fiber -> gap ->
convex mirror -> gap -> fiber, the mode-overlap (parallel) coupling factors and
the Gaussian misalignment factor. Numbers (lambda, w0, A, R_c) come exclusively
from the variant configuration (mirror of docs 03/08) — nothing is hard-coded
here (SW-03, 10 §13).

Conventions (doc 03 §2)
-----------------------
* The fiber mode is a Gaussian beam with waist w0 at the endface: q_in = i zR.
* A convex mirror of curvature radius R_c acts as a diverging mirror with focal
  length f = -R_c/2 (doc 03 §3).
* Round trip over a gap ``g``: M = [[1 - g/f, 2g - g^2/f], [-1/f, 1 - g/f]].
* Mode overlap of the returned beam with the fiber mode (doc 03 §4):
  ``eta_par = 2 sqrt(zR Im q_out) / | -i zR - q_out |``.
* Transverse/angular misalignment factor: ``exp[-(d/w0)^2 - (alpha/theta0)^2]``.
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
    "misalignment_factor",
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


def eta_parallel_curved(
    beam: GaussianBeam, gap_m: FloatArray | float, radius_of_curvature_m: float
) -> FloatArray:
    """Aligned mode-overlap in the *curved* plane of a convex mirror (doc 03 §4).

    ``eta_par^x = 2 sqrt(zR Im q_out) / | -i zR - q_out |`` with
    ``q_out = round_trip_q(beam, g, f = -R_c/2)``.

    Limits: ``R_c -> inf`` reproduces :func:`eta_parallel_flat`; ``g -> 0``
    gives 1 (perfect re-coupling at zero gap).

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


def misalignment_factor(
    beam: GaussianBeam,
    offset_m: FloatArray | float,
    angle_rad: FloatArray | float,
) -> FloatArray:
    """Gaussian misalignment factor ``exp[-(d/w0)^2 - (alpha/theta0)^2]`` (doc 03 §4).

    ``d`` is the transverse offset and ``alpha`` the angular tilt of the
    returned beam relative to the fiber mode.

    Parameters
    ----------
    beam : GaussianBeam
        Fiber mode.
    offset_m : numpy.ndarray or float
        Transverse offset d, m.
    angle_rad : numpy.ndarray or float
        Angular misalignment alpha, rad.

    Returns
    -------
    numpy.ndarray
        Misalignment factor in (0, 1] (broadcast over the inputs).
    """
    d = np.asarray(offset_m, dtype=np.float64)
    alpha = np.asarray(angle_rad, dtype=np.float64)
    out: FloatArray = np.exp(-((d / beam.waist_radius_m) ** 2) - (alpha / beam.divergence_rad) ** 2)
    return out
