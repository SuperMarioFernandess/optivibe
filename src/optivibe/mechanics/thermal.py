"""Brownian (thermomechanical) noise floor of the cantilever: ``NEA_th`` (M-12).

Backlog item M-12 (doc 16; model gap MG-1 of plan 19 §5) adds the fundamental
thermal noise floor of doc 07 §2 to the model. The fluctuation-dissipation
theorem (Callen-Welton; Saulson, Phys. Rev. D 42, 2437 (1990)) attaches a white
Langevin force to the dissipation of mode 1 (viscous damping ``b = m_eff
omega_1 / Q``):

``S_F = 4 kB T b = 4 kB T m_eff omega_1 / Q``  [N^2/Hz],

and referring it to the *input acceleration* divides by the equivalent tip
force per unit acceleration of the distributed beam, ``F_a / a = (3/8) m_tot``
(doc 07 §2.2); expressing the result through the "acceleration" effective mass
``M_a = (0.375 m_tot)^2 / m_eff`` folds it into the closed form

``NEA_th = sqrt(S_F) / (0.375 m_tot) = sqrt(4 kB T omega_1 / (Q M_a))``
(Gabrielson, IEEE Trans. Electron Devices 40, 903 (1993)).

Two effective masses (doc 07 §2.2, R-25) -- do not mix them up:

* the **kinetic** modal mass sets the frequency,
  ``m_eff = k_eff / omega_1^2 = 3 rho S L / (beta_1 L)^4 = 0.2427 rho S L``;
* the **acceleration** mass converts force to input acceleration. The
  equivalent tip force reproducing the exact quasi-static response
  ``H_lat^QS = rho S L^4 / (8 E I)`` is ``F_a = k_eff H_lat^QS a =
  (3/8) m_tot a``, hence
  ``M_a = (0.375 m_tot)^2 / m_eff = 3 (beta_1 L)^4 / 64 * rho S L
  ~ 0.58 rho S L`` -- about 2.4x the kinetic mass. Substituting the kinetic
  mass instead *overestimates* the floor by ``sqrt(0.58/0.243) = 1.55x``
  (R-25). At ``L = 1.4 mm``: ``M_a = 2.19e-8 kg ~ 22 ug`` (micrograms, not
  nanograms -- R-45).

Dimensional check: ``[4 kB T omega / (Q M)] = J s^-1 / kg = m^2/s^3``, so the
square root is ``(m/s^2)/sqrt(Hz)``. Limits: ``T -> 0`` and ``Q -> inf`` both
give ``NEA_th -> 0``; larger ``M_a`` lowers the floor. Fixed-Q length scaling:
``omega_1 ~ L^-2`` and ``M_a ~ L`` give ``NEA_th ~ L^-3/2`` exactly (with the
computable Q(L) of :mod:`optivibe.mechanics.damping` the effective exponent
over the design window is ~ -1.4, doc 07 §4.3 -- *not* the naive ``L^-2``).

Frequency shape (doc 07 §2.4, R-53). The thermal force and the input
acceleration drive the *same* mode, so the resonant denominator ``|D(f)|^2``
cancels in the input-referred ratio: the resonance is invisible in
``NEA_th(f)`` (exact, independent of the damping model). Whiteness across the
band additionally assumes the viscous-equivalent damping frozen at the mode,
``b(omega) ~ b(omega_1)``. Per loss channel this is (a) an in-band
*conservative upper bound* for the air share -- the exact hydrodynamic
dissipation ``b_air(omega) ~ omega Gamma_i(Re(omega))`` falls monotonically
toward DC (<~0.1x at the documented band edges); (b) exact for anchor loss
modelled as half-space radiation (frequency-independent radiation resistance,
Lysmer-type dashpot); (c) an *underestimate* toward DC if a loss channel is
hysteretic (structural loss angle: ``S_F ~ 1/omega``, Saulson 1990) -- for the
documented variants this corner matters only in the bottom decade of variant A
(see R-53 and backlog M-14). The flat model below is the canonical doc 07 §2
form; the deviation analysis lives in the physics log, not in code.
"""

from __future__ import annotations

import math

from optivibe.core.config.models import Constants
from optivibe.mechanics.cantilever import first_mode_hz

__all__ = [
    "acceleration_effective_mass",
    "kinetic_effective_mass",
    "nea_thermal",
    "thermal_force_psd",
]


def _total_mass(constants: Constants, length_m: float) -> float:
    """Total beam mass ``m_tot = rho S L``, kg (guarded by the callers)."""
    fiber = constants.fiber
    return fiber.density_kg_m3 * fiber.area_m2 * length_m


def kinetic_effective_mass(constants: Constants, length_m: float) -> float:
    """Kinetic modal mass ``m_eff = 3 rho S L / (beta_1 L)^4``, kg (doc 07 §2.2a).

    The mass that, with the tip stiffness ``k_eff = 3 E I / L^3``, reproduces
    the mode-1 frequency: ``omega_1 = sqrt(k_eff / m_eff)`` exactly matches the
    Euler-Bernoulli ``omega_1 = (beta_1 L)^2 sqrt(E I / (rho S L^4))``. The
    coefficient is ``3 / (beta_1 L)^4 = 0.2427``. This mass sets the frequency,
    **not** the thermal floor (use :func:`acceleration_effective_mass` there).

    Parameters
    ----------
    constants : Constants
        Physical constants (doc 01 mirror).
    length_m : float
        Cantilever length L, m (> 0).

    Returns
    -------
    float
        Kinetic effective mass, kg.

    Raises
    ------
    ValueError
        If ``length_m`` is not positive.
    """
    if length_m <= 0.0:
        msg = f"length_m must be positive, got {length_m!r}"
        raise ValueError(msg)
    beta1_l = constants.universal.beta1_l
    return 3.0 / beta1_l**4 * _total_mass(constants, length_m)


def acceleration_effective_mass(constants: Constants, length_m: float) -> float:
    """Acceleration effective mass ``M_a = 3 (beta_1 L)^4 / 64 * rho S L`` (07 §2.2b).

    Derivation (doc 07 §2.2, R-25): the distributed inertial load couples to
    the tip coordinate as the equivalent force ``F_a = k_eff H_lat^QS a =
    (3/8) m_tot a`` (using the exact all-mode quasi-static compliance of doc
    02), and the FDT referral gives ``M_a = (0.375 m_tot)^2 / m_eff``.
    Substituting ``m_eff`` yields the closed form ``3 (beta_1 L)^4 / 64 *
    m_tot = 0.5795 m_tot ~ 0.58 rho S L`` -- the coefficient is *derived*
    from ``beta_1 L``, not hard-coded (SW-03). Reference: ``M_a(1.4 mm) =
    2.19e-8 kg ~ 22 ug`` (R-45: micrograms).

    Parameters
    ----------
    constants : Constants
        Physical constants (doc 01 mirror).
    length_m : float
        Cantilever length L, m (> 0).

    Returns
    -------
    float
        Acceleration effective mass, kg.

    Raises
    ------
    ValueError
        If ``length_m`` is not positive.
    """
    if length_m <= 0.0:
        msg = f"length_m must be positive, got {length_m!r}"
        raise ValueError(msg)
    beta1_l = constants.universal.beta1_l
    return 3.0 * beta1_l**4 / 64.0 * _total_mass(constants, length_m)


def thermal_force_psd(constants: Constants, length_m: float, q_total: float) -> float:
    """One-sided Langevin force PSD ``S_F = 4 kB T m_eff omega_1 / Q`` (07 §2.1).

    The fluctuation-dissipation theorem for the mode-1 oscillator with viscous
    damping ``b = m_eff omega_1 / Q`` (Callen-Welton; Saulson 1990). White in
    the viscous-equivalent model (see the module note on the frequency shape).

    Parameters
    ----------
    constants : Constants
        Physical constants (``kB``, ``T`` from the detector block, doc 07 §2.5).
    length_m : float
        Cantilever length L, m (> 0).
    q_total : float
        Total mode-1 quality factor (> 0; variant value or
        :func:`optivibe.mechanics.damping.q_total_model`).

    Returns
    -------
    float
        Thermal force PSD, N^2/Hz.

    Raises
    ------
    ValueError
        If ``length_m`` or ``q_total`` is not positive.
    """
    if q_total <= 0.0:
        msg = f"q_total must be positive, got {q_total!r}"
        raise ValueError(msg)
    det = constants.detector
    omega1 = 2.0 * math.pi * first_mode_hz(constants, length_m)
    m_eff = kinetic_effective_mass(constants, length_m)
    return 4.0 * det.boltzmann_j_k * det.temperature_k * m_eff * omega1 / q_total


def nea_thermal(constants: Constants, length_m: float, q_total: float) -> float:
    """Thermal noise floor ``NEA_th = sqrt(4 kB T omega_1 / (Q M_a))`` (doc 07 §2).

    The Brownian floor referred to the input acceleration, flat across the
    band in the canonical viscous-equivalent model (doc 07 §2.4: the resonance
    cancels between the thermal displacement and the signal transfer -- exact;
    whiteness holds for ``b(omega) ~ b(omega_1)``, see the module note).
    References (doc 07 §2.5 / §4.3): 1.12 ug/sqrt(Hz) at ``L = 1.4 mm, Q =
    1950``; 0.67 at ``Q = 5490``; scaling column 2.0 mm -> 0.57, 3.0 -> 0.34,
    4.0 -> 0.25 ug/sqrt(Hz) with the computable Q(L).

    Parameters
    ----------
    constants : Constants
        Physical constants (doc 01 mirror; ``kB``/``T`` from the detector block).
    length_m : float
        Cantilever length L, m (> 0).
    q_total : float
        Total mode-1 quality factor (> 0).

    Force-referral identity (doc 07 §2.2): the Langevin force divides by the
    force-per-unit-acceleration ``F_a / a = (3/8) m_tot = sqrt(M_a m_eff)``,
    and ``M_a = (0.375 m_tot)^2 / m_eff`` folds this into the closed form:
    ``sqrt(S_F) / (0.375 m_tot) == sqrt(4 kB T omega_1 / (Q M_a))`` exactly
    (pinned by the golden identity test). Note ``sqrt(S_F) / M_a`` is *wrong*.

    Returns
    -------
    float
        Thermal noise-equivalent acceleration density, (m/s^2)/sqrt(Hz).

    Raises
    ------
    ValueError
        If ``length_m`` or ``q_total`` is not positive.
    """
    if q_total <= 0.0:
        msg = f"q_total must be positive, got {q_total!r}"
        raise ValueError(msg)
    det = constants.detector
    omega1 = 2.0 * math.pi * first_mode_hz(constants, length_m)
    m_a = acceleration_effective_mass(constants, length_m)
    return math.sqrt(4.0 * det.boltzmann_j_k * det.temperature_k * omega1 / (q_total * m_a))
