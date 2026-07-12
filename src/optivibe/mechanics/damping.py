"""Computable quality-factor model Q(L): air + anchor + internal losses.

Backlog item M-02 (doc 16 §1) replaces the per-variant ``q_total`` constant by
a *computable* model of the mode-1 quality factor of the fiber cantilever
(docs 02 §5, 07 §2.3, R-26/R-31):

* **Viscous air damping** via the hydrodynamic function of a circular-section
  beam oscillating in a viscous fluid (Stokes; Sader, J. Appl. Phys. 84, 64
  (1998)): ``Q_air = (rho + rho_f Gamma_r) / (rho_f Gamma_i)`` evaluated at the
  in-vacuo mode-1 frequency ``omega_1(L)``. The added-mass frequency shift
  ``rho_f Gamma_r / (2 rho) ~ 4e-4`` is neglected (doc 07 §2.3).
* **Anchor (support) loss** ``Q_anchor = c (L/D)^3`` with ``c = 2.17``
  (doc 02 §5; Hosaka, Itao & Kuroda, Sens. Actuators A 49, 87 (1995)) --
  strongly mounting-dependent, taken as-per-formula.
* **Internal losses**: structural ``Q_struct ~ 3.3e5`` and thermoelastic
  ``Q_TED ~ 1e7`` (doc 02 §5; both negligible in air).

The channels are independent, so ``1/Q_total = sum(1/Q_i)`` (doc 02 §5).
``Q_total(L)`` is *non-monotonic*: anchor loss ``~(L/D)^3`` dominates at short
``L``, air ``~1/L`` at long ``L``; the peak ``Q ~ 2610`` sits near
``L ~ 2.0-2.1 mm`` (doc 07 §4.3, R-31; reproduced by the golden tests).

Regime of validity (doc 07 §2.3)
--------------------------------
* Continuum viscous flow: Knudsen number ``Kn = lambda_mfp / R ~ 1.1e-3 << 1``
  for the 125-um fiber (independent of ``L``); no slip correction needed.
* The hydrodynamic function uses the documented 3-term large-argument
  expansion ``K_1/K_0 ~ 1 + 1/(2z) - 1/(8 z^2)`` (doc 07 §2.3), accurate to
  <~1 % for ``Re >~ 5`` -- satisfied over the design window ``L = 1.0-5.0 mm``
  (``Re = 163 ... 6.5``). Below ``Re ~ 5`` the truncation error grows; the
  function warns (it does not fail) because such lengths are outside the
  documented window.

Precedence (backward compatibility)
-----------------------------------
An explicit ``q_total`` in a composition/variant remains the priority override
(O-SW-06 note of SW-43); this model is used only when the composition omits
``q_total`` (see :meth:`optivibe.core.config.subsystems.SystemConfig.resolve`).
The built-in A-D presets keep their explicit numbers, so the resolved-variant
golden contract is untouched.
"""

from __future__ import annotations

import math

from optivibe.core.config.models import Constants
from optivibe.core.logging import get_logger
from optivibe.mechanics.cantilever import first_mode_hz

logger = get_logger(__name__)

__all__ = [
    "damping_budget",
    "hydrodynamic_function",
    "knudsen_number",
    "q_air",
    "q_anchor",
    "q_total_model",
    "reynolds_number",
]

# Below this oscillatory Reynolds number the 3-term K_1/K_0 expansion of the
# hydrodynamic function degrades past the ~1 % level (doc 07 §2.3 uses it for
# Re = 18-83; the design window L = 1-5 mm keeps Re >= ~6.5).
_MIN_RELIABLE_REYNOLDS = 5.0


def reynolds_number(constants: Constants, omega_rad_s: float) -> float:
    """Oscillatory Reynolds number ``Re = rho_f omega R^2 / mu_f`` (doc 02 §5).

    Parameters
    ----------
    constants : Constants
        Physical constants (doc 01 mirror; provides ``rho_f``, ``mu_f``, ``R``).
    omega_rad_s : float
        Angular oscillation frequency, rad/s (> 0).

    Returns
    -------
    float
        Reynolds number, dimensionless.

    Raises
    ------
    ValueError
        If ``omega_rad_s`` is not positive.
    """
    if omega_rad_s <= 0.0:
        msg = f"omega_rad_s must be positive, got {omega_rad_s!r}"
        raise ValueError(msg)
    air = constants.air
    radius = constants.fiber.radius_m
    return air.density_kg_m3 * omega_rad_s * radius**2 / air.dynamic_viscosity_pa_s


def knudsen_number(constants: Constants) -> float:
    """Knudsen number ``Kn = lambda_mfp / R`` of the fiber in air (doc 07 §2.3).

    Continuum viscous flow (the regime of :func:`hydrodynamic_function`)
    requires ``Kn << 1``; for the 125-um fiber ``Kn ~ 1.1e-3`` independently of
    the cantilever length.

    Parameters
    ----------
    constants : Constants
        Physical constants (mean free path from ``constants.damping``).

    Returns
    -------
    float
        Knudsen number, dimensionless.
    """
    return constants.damping.air_mean_free_path_m / constants.fiber.radius_m


def hydrodynamic_function(reynolds: float) -> complex:
    """Hydrodynamic function ``Gamma(Re)`` of a circular cylinder (doc 07 §2.3).

    ``Gamma = 1 + (4 i / w) K_1(z) / K_0(z)`` with ``w = sqrt(i Re)`` and
    ``z = -i w = sqrt(Re/2) (1 - i)`` (Stokes; Sader 1998, circular section),
    using the documented large-argument expansion
    ``K_1/K_0 ~ 1 + 1/(2 z) - 1/(8 z^2)``. The real part ``Gamma_r`` is the
    added-mass coefficient, the imaginary part ``Gamma_i`` the dissipation.

    Dimensional check: ``Gamma`` is dimensionless (``Re`` dimensionless).
    Limits: ``Re -> inf`` gives ``Gamma -> 1`` (inviscid added mass of the
    cylinder, ``Gamma_i -> 0``); decreasing ``Re`` raises ``Gamma_i``
    (relatively stronger viscous drag).

    Parameters
    ----------
    reynolds : float
        Oscillatory Reynolds number (> 0). Below ``Re ~ 5`` the truncated
        expansion degrades past ~1 % and a warning is logged (doc 07 §2.3).

    Returns
    -------
    complex
        ``Gamma_r + i Gamma_i``, dimensionless.

    Raises
    ------
    ValueError
        If ``reynolds`` is not positive.
    """
    if reynolds <= 0.0:
        msg = f"reynolds must be positive, got {reynolds!r}"
        raise ValueError(msg)
    if reynolds < _MIN_RELIABLE_REYNOLDS:
        logger.warning(
            "hydrodynamic_function: Re = %.3g < %.3g -- truncated K_1/K_0 expansion "
            "degrades past ~1 %% (doc 07 §2.3)",
            reynolds,
            _MIN_RELIABLE_REYNOLDS,
        )
    half = math.sqrt(reynolds / 2.0)
    w = complex(half, half)  # sqrt(i Re)
    z = complex(half, -half)  # -i w
    k_ratio = 1.0 + 1.0 / (2.0 * z) - 1.0 / (8.0 * z * z)
    return 1.0 + (4.0j / w) * k_ratio


def q_air(constants: Constants, length_m: float) -> float:
    """Air-damping quality factor ``Q_air(L)`` at the mode-1 frequency (doc 02 §5).

    ``Q_air = (rho + rho_f Gamma_r) / (rho_f Gamma_i)`` with ``Gamma`` evaluated
    at ``Re(omega_1(L))``. References (doc 07 §2.3): ``Q_air ~ 2377`` at
    ``L = 3 mm`` (``Re = 18.1``) and ``~5487`` at ``L = 1.4 mm``
    (``Re = 83.3``); scaling ``Q_air ~ 1/L`` over the window.

    Parameters
    ----------
    constants : Constants
        Physical constants (doc 01 mirror).
    length_m : float
        Cantilever length L, m (> 0; guarded by :func:`first_mode_hz`).

    Returns
    -------
    float
        Air-limited quality factor, dimensionless.
    """
    omega1 = 2.0 * math.pi * first_mode_hz(constants, length_m)
    gamma = hydrodynamic_function(reynolds_number(constants, omega1))
    rho = constants.fiber.density_kg_m3
    rho_f = constants.air.density_kg_m3
    return (rho + rho_f * gamma.real) / (rho_f * gamma.imag)


def q_anchor(constants: Constants, length_m: float) -> float:
    """Anchor (support) loss quality factor ``Q_anchor = c (L/D)^3`` (doc 02 §5).

    Reference: ``Q_anchor ~ 3050`` at ``L = 1.4 mm`` (``L/D = 11.2``; doc 07
    §2.3). Strongly mounting-dependent (docs 02/07): the coefficient
    ``c = 2.17`` (Hosaka 1995) is the as-per-formula value; an ideal clamp
    raises it, which is why the thermal floor of doc 07 is quoted as a range.

    Parameters
    ----------
    constants : Constants
        Physical constants (``c`` from ``constants.damping``, ``D`` from
        ``constants.fiber``).
    length_m : float
        Cantilever length L, m (> 0).

    Returns
    -------
    float
        Anchor-limited quality factor, dimensionless.

    Raises
    ------
    ValueError
        If ``length_m`` is not positive.
    """
    if length_m <= 0.0:
        msg = f"length_m must be positive, got {length_m!r}"
        raise ValueError(msg)
    aspect = length_m / constants.fiber.diameter_m
    return constants.damping.anchor_coefficient * aspect**3


def damping_budget(
    constants: Constants, length_m: float, *, vacuum: bool = False
) -> dict[str, float]:
    """Per-channel quality factors and their parallel total (docs 02 §5, 07 §2.3).

    Independent loss channels add reciprocally:
    ``1/Q_total = 1/Q_air + 1/Q_anchor + 1/Q_struct + 1/Q_TED``; under vacuum
    the air channel is removed (variant A/D option, doc 08).

    Parameters
    ----------
    constants : Constants
        Physical constants (doc 01 mirror).
    length_m : float
        Cantilever length L, m (> 0).
    vacuum : bool, optional
        Exclude the air channel (``vacuum: true`` variants). Default ``False``.

    Returns
    -------
    dict of str to float
        Keys ``"air"`` (``inf`` under vacuum), ``"anchor"``, ``"structural"``,
        ``"ted"`` and ``"total"``; each a quality factor, dimensionless.
    """
    damping = constants.damping
    budget = {
        "air": math.inf if vacuum else q_air(constants, length_m),
        "anchor": q_anchor(constants, length_m),
        "structural": damping.q_structural,
        "ted": damping.q_ted,
    }
    budget["total"] = 1.0 / sum(1.0 / q for q in budget.values())
    return budget


def q_total_model(constants: Constants, length_m: float, *, vacuum: bool = False) -> float:
    """Total mode-1 quality factor ``Q_total(L)`` of the damping model (M-02).

    Convenience wrapper over :func:`damping_budget`. References (doc 07 §4.3,
    R-31 scaling table): ``Q_total ~ 970`` (anchor-limited) at ``L = 1 mm``,
    ``~1950`` at 1.4 mm, non-monotonic peak ``~2610`` near ``L ~ 2.0-2.1 mm``,
    ``~2188`` at 3 mm, ``~1660`` (air-limited) at 4 mm.

    Parameters
    ----------
    constants : Constants
        Physical constants (doc 01 mirror).
    length_m : float
        Cantilever length L, m (> 0).
    vacuum : bool, optional
        Exclude the air channel. Default ``False``.

    Returns
    -------
    float
        Total quality factor, dimensionless.
    """
    return damping_budget(constants, length_m, vacuum=vacuum)["total"]
