"""Source linewidth physics: ASE intensity noise and coherence wash-out (M-01).

Backlog item M-01 (doc 16 §1) makes the source linewidth an explicit, physical
input. This module holds the closed-form links between the linewidth and the
two quantities the model consumes:

RIN of a thermal / ASE source (doc 07 §1.2)
-------------------------------------------
Broadband SLD light has thermal (ASE) statistics: the Siegert relation
``g2(tau) = 1 + |g1(tau)|^2`` makes the relative-intensity PSD at ``f << dnu``

    ``RIN(0) = 2 tau_c``,  ``tau_c = integral |g1(tau)|^2 dtau``,

the spontaneous-spontaneous beat-noise floor (Goodman, *Statistical Optics*,
ch. 6; Mandel & Wolf ch. 9; Derickson, *Fiber Optic Test and Measurement*).
The lineshape enters through ``tau_c``: for a rectangular spectrum of width
``dnu`` (FWHM) ``tau_c = 1/dnu`` and ``RIN = 2/dnu`` -- the knowledge-base
convention (doc 07 §1.2); a Gaussian line gives ``1.33/dnu`` (-1.8 dB), a
Lorentzian ``0.64/dnu`` (-5 dB), and fully unpolarized light halves the level
(-3 dB). The project adopts the **rectangular-equivalent, polarized**
convention ``RIN = 2/dnu`` -- the conservative upper edge, matching the doc-07
anchor (``dlam = 60 nm`` at 1550 nm -> ``-125.7 dB/Hz`` vs. the SLD reference
``-126``). Lineshape refinement is deferred (backlog M-10; O-SW-10).

**Applicability boundary:** the relation holds for thermal/ASE light only.
A single-frequency laser (DFB) is *not* thermal -- its RIN is set by
relaxation-oscillation/pump dynamics (datasheet ``~-155 dB/Hz``), while
``2/dnu`` at ``dnu ~ 1 MHz`` would give ``-57 dB/Hz``, off by ~100 dB. The
config layer therefore derives RIN from the linewidth for ``SLD`` sources only.

Coherence length and fringe visibility (doc 03 §f', R-13)
---------------------------------------------------------
For a Gaussian lineshape the endface-fringe visibility over the round-trip
path difference ``OPD = 2A`` is

    ``V(A) = 2^{-(2A/L_c)^2}``,  ``L_c = (2 ln 2 / pi) lambda^2 / dlam``,

(doc 03 §f'; ``2 ln 2 / pi = 0.4413`` is the ``0.44`` of the base rounded).
Route 2 (coherent wash-out, R-13) requires ``V < 0.03``, i.e.
``A >= 1.1246 L_c`` (the documented ``A >~ 1.12 L_c``). The alignment margin
(worst case ``A - 10 um``, doc 03 §f') is a *design* margin on top of this
nominal-gap criterion; :func:`min_gap_for_washout` supports both checks.

Dimensions and limits are asserted in ``tests/test_optics_source.py`` against
the doc 03 §f' table and the doc 07 §1.2 anchors.
"""

from __future__ import annotations

import math

from optivibe.core.units import SPEED_OF_LIGHT_M_S

__all__ = [
    "WASHOUT_VISIBILITY_MAX",
    "coherence_length_m",
    "fringe_visibility",
    "linewidth_nu_hz",
    "min_gap_for_washout_m",
    "rin_ase",
    "rin_ase_db_hz",
]

# Full-wash-out visibility criterion of route 2 (doc 03 §f'; R-13): the
# residual endface fringe must satisfy V < 0.03, i.e. A >= 1.1246 L_c.
WASHOUT_VISIBILITY_MAX: float = 0.03

# Gaussian-lineshape coherence-length coefficient 2 ln 2 / pi = 0.4413...
# (doc 03 §f' quotes the 2-digit rounding 0.44). Kept exact so the identity
# V = 2^{-(2A/L_c)^2} holds to machine precision.
_COHERENCE_COEFFICIENT: float = 2.0 * math.log(2.0) / math.pi

# Rectangular-equivalent beat-noise coefficient of the RIN convention
# RIN = kappa / dnu (doc 07 §1.2 anchor; see the module docstring).
_RIN_LINESHAPE_COEFFICIENT: float = 2.0


def linewidth_nu_hz(wavelength_m: float, linewidth_fwhm_m: float) -> float:
    """Convert a wavelength linewidth to a frequency linewidth (doc 07 §1.2).

    ``dnu = c dlam / lambda^2`` -- the first-order differential of
    ``nu = c / lambda``, valid for ``dlam << lambda`` (relative error
    ``~(dlam/lambda)^2 ~ 0.2 %`` at the widest documented SLD, 100 nm @ 1550).

    Parameters
    ----------
    wavelength_m : float
        Centre wavelength lambda, m (> 0).
    linewidth_fwhm_m : float
        Spectral FWHM dlam, m (> 0; must stay well below ``lambda``).

    Returns
    -------
    float
        Frequency FWHM dnu, Hz.

    Raises
    ------
    ValueError
        If an argument is not positive or ``dlam`` is not small next to
        ``lambda`` (first-order validity, ``dlam < 0.2 lambda``).
    """
    if wavelength_m <= 0.0 or linewidth_fwhm_m <= 0.0:
        msg = (
            "wavelength_m and linewidth_fwhm_m must be positive, got "
            f"{wavelength_m!r}, {linewidth_fwhm_m!r}"
        )
        raise ValueError(msg)
    if linewidth_fwhm_m >= 0.2 * wavelength_m:
        msg = (
            f"linewidth_fwhm_m = {linewidth_fwhm_m:.3e} m is not small next to "
            f"wavelength_m = {wavelength_m:.3e} m; the first-order conversion "
            "dnu = c dlam / lambda^2 no longer applies (doc 07 §1.2)"
        )
        raise ValueError(msg)
    return SPEED_OF_LIGHT_M_S * linewidth_fwhm_m / wavelength_m**2


def rin_ase(delta_nu_hz: float) -> float:
    """Spontaneous-beat RIN floor of thermal/ASE light, linear 1/Hz (doc 07 §1.2).

    ``RIN = 2 / dnu`` -- the rectangular-equivalent, polarized convention (see
    the module docstring for the lineshape/polarization spread of -5...0 dB).
    Thermal/ASE sources (SLD) only; not applicable to coherent lasers (DFB).

    Dimensional check: ``[2/dnu] = 1/Hz``. Limits: ``dnu -> inf`` gives
    ``RIN -> 0`` (more independent spectral modes average the beats out).

    Parameters
    ----------
    delta_nu_hz : float
        Frequency FWHM dnu of the source spectrum, Hz (> 0).

    Returns
    -------
    float
        Relative intensity noise PSD, 1/Hz (one-sided, ``f << dnu``).

    Raises
    ------
    ValueError
        If ``delta_nu_hz`` is not positive.
    """
    if delta_nu_hz <= 0.0:
        msg = f"delta_nu_hz must be positive, got {delta_nu_hz!r}"
        raise ValueError(msg)
    return _RIN_LINESHAPE_COEFFICIENT / delta_nu_hz


def rin_ase_db_hz(delta_nu_hz: float) -> float:
    """Spontaneous-beat RIN floor in dB/Hz (doc 07 §1.2).

    ``10 log10(2/dnu)``; anchor: ``dlam = 60 nm`` @ 1550 nm
    (``dnu = 7.49e12 Hz``) gives ``-125.7 dB/Hz`` (doc 07 §1.2).

    Parameters
    ----------
    delta_nu_hz : float
        Frequency FWHM dnu of the source spectrum, Hz (> 0).

    Returns
    -------
    float
        Relative intensity noise, dB/Hz.
    """
    return 10.0 * math.log10(rin_ase(delta_nu_hz))


def coherence_length_m(wavelength_m: float, linewidth_fwhm_m: float) -> float:
    """Coherence length ``L_c = (2 ln 2 / pi) lambda^2 / dlam`` (doc 03 §f').

    Gaussian-lineshape convention; the base quotes the coefficient as 0.44.
    Anchors (doc 03 §f' table, 1550 nm): 20 nm -> 52.9 um, 40 nm -> 26.4 um,
    60 nm -> 17.6 um.

    Dimensional check: ``[lambda^2 / dlam] = m``. Limits: ``dlam -> 0`` gives
    ``L_c -> inf`` (monochromatic light never washes out).

    Parameters
    ----------
    wavelength_m : float
        Centre wavelength lambda, m (> 0).
    linewidth_fwhm_m : float
        Spectral FWHM dlam, m (> 0).

    Returns
    -------
    float
        Coherence length, m.

    Raises
    ------
    ValueError
        If an argument is not positive.
    """
    if wavelength_m <= 0.0 or linewidth_fwhm_m <= 0.0:
        msg = (
            "wavelength_m and linewidth_fwhm_m must be positive, got "
            f"{wavelength_m!r}, {linewidth_fwhm_m!r}"
        )
        raise ValueError(msg)
    return _COHERENCE_COEFFICIENT * wavelength_m**2 / linewidth_fwhm_m


def fringe_visibility(gap_m: float, coherence_length_m: float) -> float:
    """Endface-fringe visibility ``V = 2^{-(2A/L_c)^2}`` (doc 03 §f').

    The round-trip path difference across the air gap is ``OPD = 2A``; a
    Gaussian lineshape gives ``V = exp(-pi^2 A^2 dlam^2 / (lambda^4 ln 2))``,
    identical to ``2^{-(2A/L_c)^2}`` with the exact ``L_c`` of
    :func:`coherence_length_m`. Anchors (doc 03 §f'): ``V(30 um)`` = 0.41 for
    ``dlam = 20 nm``, 0.029 for 40 nm, ~0 for >= 60 nm.

    Limits: ``A -> 0`` gives ``V -> 1`` (full interference); ``L_c -> 0`` or
    ``A -> inf`` gives ``V -> 0`` (complete wash-out).

    Parameters
    ----------
    gap_m : float
        One-way air gap A, m (>= 0).
    coherence_length_m : float
        Coherence length L_c, m (> 0).

    Returns
    -------
    float
        Fringe visibility, dimensionless in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``gap_m`` is negative or ``coherence_length_m`` is not positive.
    """
    if gap_m < 0.0 or coherence_length_m <= 0.0:
        msg = (
            "gap_m must be non-negative and coherence_length_m positive, got "
            f"{gap_m!r}, {coherence_length_m!r}"
        )
        raise ValueError(msg)
    return float(2.0 ** (-((2.0 * gap_m / coherence_length_m) ** 2)))


def min_gap_for_washout_m(
    coherence_length_m: float, *, visibility_max: float = WASHOUT_VISIBILITY_MAX
) -> float:
    """Minimum gap for the route-2 wash-out criterion (doc 03 §f'; R-13).

    Inverts ``V(A) < V_max``: ``A_min = (L_c / 2) sqrt(log2(1 / V_max))``. For
    the documented ``V_max = 0.03`` this is ``A_min = 1.1246 L_c`` (the base's
    ``A >~ 1.12 L_c``; the popular ``A > L_c / 2`` rule gives only ``V ~ 0.5``).
    The +-10 um alignment tolerance of doc 03 §f' is a design margin on top:
    check the worst-case gap against this value.

    Parameters
    ----------
    coherence_length_m : float
        Coherence length L_c, m (> 0).
    visibility_max : float, optional
        Wash-out criterion on the residual visibility, in ``(0, 1)``. Default
        :data:`WASHOUT_VISIBILITY_MAX` = 0.03 (R-13).

    Returns
    -------
    float
        Minimum one-way gap A, m.

    Raises
    ------
    ValueError
        If ``coherence_length_m`` is not positive or ``visibility_max`` is
        outside ``(0, 1)``.
    """
    if coherence_length_m <= 0.0:
        msg = f"coherence_length_m must be positive, got {coherence_length_m!r}"
        raise ValueError(msg)
    if not 0.0 < visibility_max < 1.0:
        msg = f"visibility_max must lie in (0, 1), got {visibility_max!r}"
        raise ValueError(msg)
    return 0.5 * coherence_length_m * math.sqrt(math.log2(1.0 / visibility_max))
