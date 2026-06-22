"""Analytical cantilever model: derived quantities and the lateral FRF.

This module holds the *formulas* of the mechanics subsystem (docs 02/05) as
plain functions plus a frozen :class:`CantileverModel` that bundles the derived
quantities of one variant. Numbers (E, I, rho, S, beta_n L, 1.377, Q) come
exclusively from ``configs/constants.yaml`` / ``configs/variants/*.yaml``
(mirrors of docs 01/08) — nothing is hard-coded here (SW-03, 10 §13).

The mode-1 frequency formula previously lived only in the S0 golden test
(``tests/test_constants_golden.py``); per the S2 task it now lives here and the
test cross-checks this function instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from optivibe.core.config.models import Constants, VariantConfig
from optivibe.core.types import FloatArray

ComplexArray = npt.NDArray[np.complex128]

__all__ = [
    "CantileverModel",
    "axial_qs_compliance",
    "first_mode_hz",
    "first_mode_shape",
    "lateral_qs_compliance",
    "second_mode_hz",
    "tilt_coupling_per_m",
]


def _mode_hz(constants: Constants, length_m: float, beta_l: float) -> float:
    """Natural frequency of one bending mode of the clamped-free fiber.

    ``f_n = (beta_n L)^2 / (2 pi) * sqrt(E I / (rho S)) / L^2``
    (Euler-Bernoulli, doc 02 §2).

    Parameters
    ----------
    constants : Constants
        Physical constants (doc 01 mirror).
    length_m : float
        Cantilever length L, m.
    beta_l : float
        Dimensionless eigenvalue ``beta_n * L`` of the mode.

    Returns
    -------
    float
        Natural frequency, Hz.
    """
    if length_m <= 0.0:
        msg = f"length_m must be positive, got {length_m!r}"
        raise ValueError(msg)
    fiber = constants.fiber
    stiffness = math.sqrt(
        fiber.youngs_modulus_pa * fiber.inertia_m4 / (fiber.density_kg_m3 * fiber.area_m2)
    )
    return beta_l**2 / (2.0 * math.pi) * stiffness / length_m**2


def first_mode_hz(constants: Constants, length_m: float) -> float:
    """First (working) bending-mode frequency ``f1``, Hz (doc 02 §2).

    Scaling reference: ``f1 ~ 100 / L[mm]^2`` kHz (doc 08, R-31).

    Parameters
    ----------
    constants : Constants
        Physical constants (doc 01 mirror).
    length_m : float
        Cantilever length L, m.

    Returns
    -------
    float
        First-mode frequency, Hz.
    """
    return _mode_hz(constants, length_m, constants.universal.beta1_l)


def second_mode_hz(constants: Constants, length_m: float) -> float:
    """Second bending-mode frequency ``f2 ~ 6.27 * f1``, Hz (doc 02 §2).

    Exposed as a derived/reporting quantity; the S2 response synthesis is
    single-mode (mode-2 contribution < 0.5 % in band for the documented
    variants, doc 05 §1). Multi-mode synthesis is a recorded loop (14 §8).

    Parameters
    ----------
    constants : Constants
        Physical constants (doc 01 mirror).
    length_m : float
        Cantilever length L, m.

    Returns
    -------
    float
        Second-mode frequency, Hz.
    """
    return _mode_hz(constants, length_m, constants.universal.beta2_l)


def lateral_qs_compliance(constants: Constants, length_m: float) -> float:
    """Quasi-static lateral tip compliance ``H_lat^QS = rho S L^4 / (8 E I)``.

    Units: m per m/s^2 (i.e. s^2). Reference: ``H_lat^QS ~ 0.0384 * L[mm]^4``
    nm/g (docs 02 §6, 08 R-31). Sign convention follows the ICD-facing form
    ``dx = H_lat(f) * a_x`` of doc 05 §1 (the relative-frame minus sign of doc
    02 §6 is absorbed into that interface convention).

    Parameters
    ----------
    constants : Constants
        Physical constants (doc 01 mirror).
    length_m : float
        Cantilever length L, m.

    Returns
    -------
    float
        Quasi-static compliance, m/(m/s^2).
    """
    if length_m <= 0.0:
        msg = f"length_m must be positive, got {length_m!r}"
        raise ValueError(msg)
    fiber = constants.fiber
    return (
        fiber.density_kg_m3
        * fiber.area_m2
        * length_m**4
        / (8.0 * fiber.youngs_modulus_pa * fiber.inertia_m4)
    )


def axial_qs_compliance(constants: Constants, length_m: float) -> float:
    """Quasi-static axial tip compliance ``dz/a_z = rho L^2 / (2 E)``.

    Units: m per m/s^2. Reference: ~1.4 pm/g at L = 3 mm — negligible but
    modelled (doc 02 §7.2). The axial resonance (~0.5 MHz at 3 mm) is far above
    the 20 kHz band, so the response is treated as frequency-independent. The
    second-order *geometric* gap change from transverse motion (doc 02 §7.2) is
    not modelled in S2 (recorded loop, 14 §8).

    Parameters
    ----------
    constants : Constants
        Physical constants (doc 01 mirror).
    length_m : float
        Cantilever length L, m.

    Returns
    -------
    float
        Quasi-static axial compliance, m/(m/s^2).
    """
    if length_m <= 0.0:
        msg = f"length_m must be positive, got {length_m!r}"
        raise ValueError(msg)
    fiber = constants.fiber
    return fiber.density_kg_m3 * length_m**2 / (2.0 * fiber.youngs_modulus_pa)


def first_mode_shape(xi: npt.ArrayLike, beta1_l: float) -> FloatArray:
    """Return the normalized first cantilever mode ``phi_1(xi) / phi_1(1)``.

    The Euler-Bernoulli clamped-free eigenfunction (doc 02 §2)
    ``phi(s) = cosh(b s) - cos(b s) - sigma (sinh(b s) - sin(b s))`` with
    ``b = beta1_l`` and ``sigma = (cosh b + cos b) / (sinh b + sin b)``, rescaled
    so the tip value is 1. By construction the tip slope (in units of ``1/L``)
    equals the rigid coupling ``phi_1'(1) / phi_1(1) = 1.377`` (doc 04 §2): a tip
    deflection ``dx`` drawn as ``dx * phi(z / L)`` therefore carries the matching
    tip tilt ``theta_y = 1.377 dx / L`` for free. This is pure geometry -- the
    GUI scales this unit shape by the tip displacement of a
    :class:`~optivibe.core.types.TipState` to animate the bend, so no physics is
    computed in the view (task S7; SW-09).

    Parameters
    ----------
    xi : array_like
        Normalized arc-length ``z / L`` in ``[0, 1]`` (clamp at 0, tip at 1).
    beta1_l : float
        First eigenvalue ``beta_1 * L`` (doc 01 §4.3, ``1.8751``); pass
        ``constants.universal.beta1_l`` rather than hard-coding it.

    Returns
    -------
    numpy.ndarray
        Mode shape with ``phi(0) = 0`` and ``phi(1) = 1``, dimensionless.

    Raises
    ------
    ValueError
        If ``beta1_l`` is not positive.
    """
    if beta1_l <= 0.0:
        msg = f"beta1_l must be positive, got {beta1_l!r}"
        raise ValueError(msg)
    s = np.ascontiguousarray(xi, dtype=np.float64)
    b = beta1_l
    sigma = (math.cosh(b) + math.cos(b)) / (math.sinh(b) + math.sin(b))
    phi = np.cosh(b * s) - np.cos(b * s) - sigma * (np.sinh(b * s) - np.sin(b * s))
    tip = math.cosh(b) - math.cos(b) - sigma * (math.sinh(b) - math.sin(b))
    return np.ascontiguousarray(phi / tip, dtype=np.float64)


def tilt_coupling_per_m(constants: Constants, length_m: float) -> float:
    """Rigid tilt-displacement coupling ``theta / delta = 1.377 / L``, rad/m.

    ``theta_y = 1.377 * dx / L`` and ``theta_x = 1.377 * dy / L`` — a
    frequency-independent modal relation (docs 02 §7.1, 05 §3.3, 04 §2).

    Parameters
    ----------
    constants : Constants
        Physical constants (doc 01 mirror).
    length_m : float
        Cantilever length L, m.

    Returns
    -------
    float
        Tilt per unit tip displacement, rad/m.
    """
    if length_m <= 0.0:
        msg = f"length_m must be positive, got {length_m!r}"
        raise ValueError(msg)
    return constants.tilt_displacement_coupling_per_l / length_m


@dataclass(frozen=True)
class CantileverModel:
    """Derived mechanical quantities of one sensor variant (docs 02/05).

    Parameters
    ----------
    length_m : float
        Cantilever length L, m.
    q_total : float
        Total quality factor of mode 1 (docs 07/08; variant value or scenario
        override).
    f1_hz, f2_hz : float
        First/second bending-mode frequencies, Hz (doc 02 §2).
    h_lat_qs : float
        Quasi-static lateral compliance, m/(m/s^2) (doc 02 §6).
    axial_compliance : float
        Quasi-static axial compliance dz/a_z, m/(m/s^2) (doc 02 §7.2).
    tilt_per_m : float
        Tilt-displacement coupling 1.377/L, rad/m (doc 02 §7.1).
    """

    length_m: float
    q_total: float
    f1_hz: float
    f2_hz: float
    h_lat_qs: float
    axial_compliance: float
    tilt_per_m: float

    @classmethod
    def from_config(
        cls,
        constants: Constants,
        variant: VariantConfig,
        *,
        q_total: float | None = None,
    ) -> CantileverModel:
        """Build the model from the constants and a variant preset.

        Parameters
        ----------
        constants : Constants
            Physical constants (doc 01 mirror).
        variant : VariantConfig
            Sensor variant providing L and the default ``q_total`` (doc 08).
        q_total : float or None, optional
            Scenario-level override of the variant's quality factor.

        Returns
        -------
        CantileverModel
            Frozen bundle of derived quantities.

        Raises
        ------
        ValueError
            If the resolved quality factor is not positive.
        """
        q = variant.q_total if q_total is None else q_total
        if q <= 0.0:
            msg = f"q_total must be positive, got {q!r}"
            raise ValueError(msg)
        length = variant.length_m
        return cls(
            length_m=length,
            q_total=q,
            f1_hz=first_mode_hz(constants, length),
            f2_hz=second_mode_hz(constants, length),
            h_lat_qs=lateral_qs_compliance(constants, length),
            axial_compliance=axial_qs_compliance(constants, length),
            tilt_per_m=tilt_coupling_per_m(constants, length),
        )

    def dynamic_factor(self, freq_hz: FloatArray | float) -> ComplexArray:
        """Single-mode dynamic amplification ``D(f)`` (doc 05 §1).

        ``D(f) = 1 / (1 - (f/f1)^2 + i (f/f1)/Q)``. Limits: ``|D| -> 1`` for
        ``f -> 0``; ``|D(f1)| = Q`` with phase -90 deg (docs 05 §2, 11 §7).

        Parameters
        ----------
        freq_hz : numpy.ndarray or float
            Frequencies, Hz (non-negative).

        Returns
        -------
        numpy.ndarray
            Complex dynamic factor, dimensionless, same shape as the input.
        """
        ratio = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64)) / self.f1_hz
        denom = 1.0 - ratio**2 + 1j * ratio / self.q_total
        return np.asarray(1.0 / denom, dtype=np.complex128)

    def h_lat(self, freq_hz: FloatArray | float) -> ComplexArray:
        """Lateral FRF ``H_lat(f) = H_lat^QS * D(f)``, m/(m/s^2) (doc 05 §1).

        Identical for the x and y axes: the bare cantilever is axially
        symmetric; axis separation is performed by the optics, not the
        mechanics (docs 02 §1/§7.3, 03/04).

        Parameters
        ----------
        freq_hz : numpy.ndarray or float
            Frequencies, Hz (non-negative).

        Returns
        -------
        numpy.ndarray
            Complex lateral transfer function, m/(m/s^2).
        """
        return self.h_lat_qs * self.dynamic_factor(freq_hz)
