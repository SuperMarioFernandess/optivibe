"""Source spectrum physics: lineshape, coherence wash-out, ASE noise (M-01/M-10).

Backlog item M-01 (doc 16 §1) made the source linewidth an explicit, physical
input; backlog item M-10 promotes the *shape* of the line from an effective
scalar to a spectral model. This module holds the closed-form links between
the spectrum and the two quantities the model consumes:

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

Coherence length and fringe visibility (doc 03 §f', R-13; M-10)
----------------------------------------------------------------
By the Wiener-Khinchin theorem the complex degree of coherence is the Fourier
transform of the *normalized* source spectrum ``s(nu)`` (Mandel & Wolf ch. 4;
Goodman, *Statistical Optics* ch. 5):

    ``gamma(tau) = integral s(nu) exp(-i 2 pi nu tau) dnu``,
    ``V(A) = |gamma(2A/c)|``  (round-trip path difference ``OPD = 2A``).

M-10 derives the visibility from the lineshape instead of postulating it.
For a **Gaussian** line of FWHM ``dnu`` the transform gives
``|gamma| = exp(-(pi dnu tau)^2 / (4 ln 2))``, which with ``tau = 2A/c``
is exactly the knowledge-base law

    ``V_G(A) = 2^{-(2A/L_c)^2}``,  ``L_c = (2 ln 2 / pi) lambda^2 / dlam``

(doc 03 §f'; ``2 ln 2 / pi = 0.4413`` is the ``0.44`` of the base rounded).
For a **Lorentzian** line of the *same* FWHM the transform gives
``|gamma| = exp(-pi dnu |tau|)``, i.e. in units of the *same* (Gaussian-
convention) ``L_c``

    ``V_L(A) = 2^{-4A/L_c}``.

The two laws cross exactly at ``A = L_c`` where ``V_G = V_L = 1/16``: for
``A < L_c`` the Lorentzian (sharper peak) decays *faster*, for ``A > L_c``
its heavy tails decay *slower*. Route 2 (coherent wash-out, R-13) requires
``V < 0.03 < 1/16`` -- past the crossing -- so Lorentzian tails *tighten* the
wash-out criterion:

    Gaussian:    ``A >= 1.1246 L_c``  (R-46),
    Lorentzian:  ``A >= ln(1/0.03)/(4 ln 2) L_c = 1.2647 L_c``  (+12.5 %),

with the shape-independent identity ``A_min^L L_c = (A_min^G)^2`` at any
threshold. The alignment margin (worst case ``A - 10 um``, doc 03 §f') is a
*design* margin on top of this nominal-gap criterion;
:func:`min_gap_for_washout_m` / :func:`min_gap_for_washout_lorentzian_m`
support both checks.

Measured spectra (M-10; feeds D-03 / E1-P6)
-------------------------------------------
A tabulated spectrum ``S_lambda(lambda)`` (an OSA trace, arbitrary scale) is
converted to the frequency density ``S_nu = S_lambda lambda^2 / c``,
normalized, and evaluated numerically: ``V(A)`` by direct quadrature of the
Wiener-Khinchin integral, and the noise floor through the coherence time,
which by Parseval needs no transform at all:

    ``tau_c = integral |gamma|^2 dtau = integral s(nu)^2 dnu``,
    ``dnu_eff = 1/tau_c``  (noise-equivalent linewidth, Derickson ch. 5),
    ``RIN = 2 tau_c = 2 / dnu_eff``.

The numeric path also captures what no single-lobe shape can: spectral
ripple/multi-lobe SLD spectra produce *coherence revivals* (two lines split
by ``delta nu`` give ``gamma = cos(pi delta-nu tau)`` times the single-line
envelope -- full revival at ``OPD = c/delta-nu``), which is precisely the
route-2 risk that the phase-0 spectrum measurement (D-03, prediction E1-P6)
must rule out.

The lineshape/RIN form factors (``kappa = RIN dnu``: rectangular 2, Gaussian
1.3286, Lorentzian 0.6366 -- doc 07 §1.2 table) are hereby *derived*; the
project default remains the rectangular-equivalent ``kappa = 2`` (R-46,
conservative upper edge) unless a lineshape is stated explicitly.

Dimensions and limits are asserted in ``tests/test_optics_source.py`` against
the doc 03 §f' table and the doc 07 §1.2 anchors.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from optivibe.core.units import SPEED_OF_LIGHT_M_S

__all__ = [
    "RIN_KAPPA_BY_LINESHAPE",
    "WASHOUT_VISIBILITY_MAX",
    "Lineshape",
    "coherence_length_m",
    "coherence_time_measured_s",
    "effective_linewidth_measured_hz",
    "fringe_visibility",
    "fringe_visibility_lorentzian",
    "fringe_visibility_measured",
    "linewidth_nu_hz",
    "min_gap_for_washout_lorentzian_m",
    "min_gap_for_washout_m",
    "rin_ase",
    "rin_ase_db_hz",
    "rin_ase_measured",
    "rin_ase_measured_db_hz",
]

#: Analytic lineshape family of the source spectrum (M-10). ``None`` at the
#: config level keeps the pre-M-10 effective-scalar behaviour (R-46).
Lineshape = Literal["gaussian", "lorentzian", "measured"]

# Full-wash-out visibility criterion of route 2 (doc 03 §f'; R-13): the
# residual endface fringe must satisfy V < 0.03, i.e. A >= 1.1246 L_c.
WASHOUT_VISIBILITY_MAX: float = 0.03

# Gaussian-lineshape coherence-length coefficient 2 ln 2 / pi = 0.4413...
# (doc 03 §f' quotes the 2-digit rounding 0.44). Kept exact so the identity
# V = 2^{-(2A/L_c)^2} holds to machine precision.
_COHERENCE_COEFFICIENT: float = 2.0 * math.log(2.0) / math.pi

# Rectangular-equivalent beat-noise coefficient of the RIN convention
# RIN = kappa / dnu (doc 07 §1.2 anchor; see the module docstring). This is
# the project default (R-46, conservative upper edge of the lineshape family).
_RIN_LINESHAPE_COEFFICIENT: float = 2.0

#: Beat-noise form factors ``kappa = RIN(0) dnu_FWHM = 2 tau_c dnu_FWHM``
#: derived from the Wiener-Khinchin/Siegert chain per lineshape (doc 07 §1.2
#: table, now derived -- M-10): Gaussian ``2 sqrt(2 ln 2 / pi) = 1.3286``
#: (-1.78 dB vs. the convention), Lorentzian ``2/pi = 0.6366`` (-4.97 dB).
#: The rectangular-equivalent ``2`` remains the default convention (R-46).
RIN_KAPPA_BY_LINESHAPE: dict[str, float] = {
    "rectangular": _RIN_LINESHAPE_COEFFICIENT,
    "gaussian": 2.0 * math.sqrt(2.0 * math.log(2.0) / math.pi),
    "lorentzian": 2.0 / math.pi,
}


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


def _rin_kappa(lineshape: str | None) -> float:
    """Beat-noise form factor ``kappa = RIN dnu`` for a named lineshape (M-10).

    ``None`` keeps the R-46 rectangular-equivalent convention ``kappa = 2``.

    Parameters
    ----------
    lineshape : str or None
        ``"gaussian"``, ``"lorentzian"``, ``"rectangular"`` or ``None``.

    Returns
    -------
    float
        Form factor kappa, dimensionless.

    Raises
    ------
    ValueError
        If the lineshape is not in the closed-form family (``"measured"``
        spectra go through :func:`rin_ase_measured` instead).
    """
    if lineshape is None:
        return _RIN_LINESHAPE_COEFFICIENT
    try:
        return RIN_KAPPA_BY_LINESHAPE[lineshape]
    except KeyError:
        msg = (
            f"no closed-form RIN factor for lineshape {lineshape!r}; expected one of "
            f"{sorted(RIN_KAPPA_BY_LINESHAPE)} (a measured spectrum goes through "
            "rin_ase_measured, doc 07 §1.2; M-10)"
        )
        raise ValueError(msg) from None


def rin_ase(delta_nu_hz: float, *, lineshape: str | None = None) -> float:
    """Spontaneous-beat RIN floor of thermal/ASE light, linear 1/Hz (doc 07 §1.2).

    ``RIN = kappa / dnu`` with ``kappa = 2`` by default -- the rectangular-
    equivalent, polarized convention (R-46, conservative upper edge). An
    explicit ``lineshape`` (M-10) selects the derived form factor instead:
    Gaussian ``2 sqrt(2 ln 2/pi) = 1.3286`` (-1.78 dB), Lorentzian
    ``2/pi = 0.6366`` (-4.97 dB) -- the Siegert chain ``RIN(0) = 2 tau_c``
    with ``tau_c = integral |g1|^2 dtau`` of that shape.
    Thermal/ASE sources (SLD) only; not applicable to coherent lasers (DFB).

    Dimensional check: ``[kappa/dnu] = 1/Hz``. Limits: ``dnu -> inf`` gives
    ``RIN -> 0`` (more independent spectral modes average the beats out).

    Parameters
    ----------
    delta_nu_hz : float
        Frequency FWHM dnu of the source spectrum, Hz (> 0).
    lineshape : str or None, optional
        ``None`` (default, ``kappa = 2``), ``"gaussian"``, ``"lorentzian"``
        or ``"rectangular"``.

    Returns
    -------
    float
        Relative intensity noise PSD, 1/Hz (one-sided, ``f << dnu``).

    Raises
    ------
    ValueError
        If ``delta_nu_hz`` is not positive or the lineshape is unknown.
    """
    if delta_nu_hz <= 0.0:
        msg = f"delta_nu_hz must be positive, got {delta_nu_hz!r}"
        raise ValueError(msg)
    return _rin_kappa(lineshape) / delta_nu_hz


def rin_ase_db_hz(delta_nu_hz: float, *, lineshape: str | None = None) -> float:
    """Spontaneous-beat RIN floor in dB/Hz (doc 07 §1.2).

    ``10 log10(kappa/dnu)``; anchor: ``dlam = 60 nm`` @ 1550 nm
    (``dnu = 7.49e12 Hz``) gives ``-125.7 dB/Hz`` with the default
    ``kappa = 2`` (doc 07 §1.2); an explicit Gaussian lineshape sits 1.78 dB
    below, a Lorentzian 4.97 dB below (M-10).

    Parameters
    ----------
    delta_nu_hz : float
        Frequency FWHM dnu of the source spectrum, Hz (> 0).
    lineshape : str or None, optional
        See :func:`rin_ase`.

    Returns
    -------
    float
        Relative intensity noise, dB/Hz.
    """
    return 10.0 * math.log10(rin_ase(delta_nu_hz, lineshape=lineshape))


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


def fringe_visibility_lorentzian(gap_m: float, coherence_length_m: float) -> float:
    """Endface-fringe visibility of a Lorentzian line, ``V = 2^{-4A/L_c}`` (M-10).

    Wiener-Khinchin for a Lorentzian spectrum of FWHM ``dnu`` gives
    ``|gamma(tau)| = exp(-pi dnu |tau|)``; with ``tau = 2A/c`` and the
    *project* (Gaussian-convention) coherence length
    ``L_c = (2 ln 2/pi) lambda^2 / dlam`` of :func:`coherence_length_m` at
    the same FWHM, this is exactly ``V = 2^{-4A/L_c}``
    (equivalently ``exp(-2 pi A dlam / lambda^2)``).

    Crossing identity: ``V_L(L_c) = V_G(L_c) = 1/16`` exactly; below ``L_c``
    the Lorentzian decays faster, above -- slower (heavy tails). The route-2
    threshold 0.03 < 1/16 lies past the crossing, hence the tightened wash-out
    criterion of :func:`min_gap_for_washout_lorentzian_m`.

    Limits: ``A -> 0`` gives ``V -> 1``; ``A -> inf`` gives ``V -> 0``.

    Parameters
    ----------
    gap_m : float
        One-way air gap A, m (>= 0).
    coherence_length_m : float
        Project coherence length L_c (Gaussian convention, same FWHM), m (> 0).

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
    return float(2.0 ** (-4.0 * gap_m / coherence_length_m))


def min_gap_for_washout_lorentzian_m(
    coherence_length_m: float, *, visibility_max: float = WASHOUT_VISIBILITY_MAX
) -> float:
    """Minimum gap for wash-out under a Lorentzian lineshape (M-10).

    Inverts ``V_L(A) < V_max``: ``A_min = L_c ln(1/V_max) / (4 ln 2)``. For
    the documented ``V_max = 0.03`` this is ``A_min = 1.2647 L_c`` -- 12.5 %
    tighter than the Gaussian ``1.1246 L_c`` (R-46), because the route-2
    threshold lies in the heavy-tail region past the ``A = L_c`` crossing.
    Shape-independent identity at any threshold:
    ``A_min^L L_c = (A_min^G)^2``.

    Parameters
    ----------
    coherence_length_m : float
        Project coherence length L_c (Gaussian convention, same FWHM), m (> 0).
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
    return coherence_length_m * math.log(1.0 / visibility_max) / (4.0 * math.log(2.0))


# --------------------------------------------------------------------------- #
# Measured (tabulated) spectra -- the D-03 / E1-P6 bridge (M-10).
# --------------------------------------------------------------------------- #
def _normalized_spectrum_nu(
    spectrum_wavelength_m: NDArray[np.float64] | list[float],
    spectrum_psd: NDArray[np.float64] | list[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert a tabulated ``S_lambda(lambda)`` trace to a unit-area ``s(nu)``.

    An OSA trace samples the *wavelength* density ``S_lambda`` (arbitrary
    scale). Conservation ``S_nu dnu = S_lambda dlam`` with ``nu = c/lambda``
    gives the Jacobian ``S_nu = S_lambda lambda^2 / c``; the frequency grid
    (descending in lambda order) is flipped ascending and the density is
    normalized to unit area for the Wiener-Khinchin integral.

    Parameters
    ----------
    spectrum_wavelength_m : array-like of float
        Strictly increasing wavelength samples, m (> 0). At least 4 points.
    spectrum_psd : array-like of float
        Spectral density samples ``S_lambda`` at those wavelengths, arbitrary
        scale (finite, >= 0, not all zero).

    Returns
    -------
    tuple of ndarray
        ``(nu_hz, s_nu)`` -- ascending frequency grid, Hz, and the unit-area
        normalized spectral density, 1/Hz.

    Raises
    ------
    ValueError
        On a malformed table (wrong lengths, non-monotonic wavelengths,
        negative/non-finite values, zero total power).
    """
    lam = np.asarray(spectrum_wavelength_m, dtype=np.float64)
    psd = np.asarray(spectrum_psd, dtype=np.float64)
    if lam.ndim != 1 or psd.ndim != 1 or lam.shape != psd.shape:
        msg = (
            "spectrum_wavelength_m and spectrum_psd must be 1-D arrays of equal "
            f"length, got shapes {lam.shape} and {psd.shape}"
        )
        raise ValueError(msg)
    if lam.size < 4:
        msg = f"a measured spectrum needs at least 4 samples, got {lam.size}"
        raise ValueError(msg)
    if not np.all(np.isfinite(lam)) or not np.all(np.isfinite(psd)):
        msg = "spectrum samples must be finite"
        raise ValueError(msg)
    if np.any(lam <= 0.0):
        msg = "spectrum_wavelength_m samples must be positive (metres)"
        raise ValueError(msg)
    if np.any(np.diff(lam) <= 0.0):
        msg = "spectrum_wavelength_m must be strictly increasing"
        raise ValueError(msg)
    if np.any(psd < 0.0):
        msg = "spectrum_psd samples must be non-negative"
        raise ValueError(msg)
    # Jacobian to the frequency density, then flip to an ascending nu grid.
    nu = SPEED_OF_LIGHT_M_S / lam[::-1]
    s_nu = (psd * lam**2 / SPEED_OF_LIGHT_M_S)[::-1]
    area = float(np.trapezoid(s_nu, nu))
    if area <= 0.0:
        msg = "spectrum_psd must carry power (the integral over the table is zero)"
        raise ValueError(msg)
    return nu, s_nu / area


def fringe_visibility_measured(
    gap_m: float,
    spectrum_wavelength_m: NDArray[np.float64] | list[float],
    spectrum_psd: NDArray[np.float64] | list[float],
) -> float:
    """Endface-fringe visibility of a measured spectrum (M-10; D-03/E1-P6).

    Direct quadrature of the Wiener-Khinchin integral at the round-trip delay
    ``tau = 2A/c``:

        ``V(A) = | integral s(nu) exp(-i 2 pi nu tau) dnu |``.

    Unlike the closed-form family this captures spectral ripple / multi-lobe
    SLD spectra, whose *coherence revivals* (full revival at
    ``OPD = c / delta-nu`` for two lines split by ``delta-nu``) are the
    route-2 risk that prediction E1-P6 (doc 19 §3.2) must rule out. Check the
    whole alignment band ``A +- 10 um``, not only the nominal gap, when the
    spectrum is structured.

    Limits: ``A = 0`` gives ``V = 1`` (unit-area normalization); a truncated
    tail biases ``V`` by at most the clipped spectral fraction.

    Parameters
    ----------
    gap_m : float
        One-way air gap A, m (>= 0).
    spectrum_wavelength_m : array-like of float
        Strictly increasing wavelength samples, m.
    spectrum_psd : array-like of float
        Spectral density samples (arbitrary scale) at those wavelengths.

    Returns
    -------
    float
        Fringe visibility, dimensionless in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``gap_m`` is negative or the table is malformed.
    """
    if gap_m < 0.0:
        msg = f"gap_m must be non-negative, got {gap_m!r}"
        raise ValueError(msg)
    nu, s_nu = _normalized_spectrum_nu(spectrum_wavelength_m, spectrum_psd)
    tau = 2.0 * gap_m / SPEED_OF_LIGHT_M_S
    gamma = np.trapezoid(s_nu * np.exp(-2j * np.pi * nu * tau), nu)
    return float(min(abs(gamma), 1.0))


def coherence_time_measured_s(
    spectrum_wavelength_m: NDArray[np.float64] | list[float],
    spectrum_psd: NDArray[np.float64] | list[float],
) -> float:
    """Coherence time ``tau_c`` of a measured spectrum (doc 07 §1.2; M-10).

    ``tau_c = integral |g1(tau)|^2 dtau``, which by Parseval equals
    ``integral s(nu)^2 dnu`` of the unit-area normalized spectrum -- no
    Fourier transform needed. This is the quantity the Siegert chain feeds
    into the ASE beat-noise floor ``RIN(0) = 2 tau_c``.

    Dimensional check: ``[s] = 1/Hz``, so ``[integral s^2 dnu] = 1/Hz = s``.
    Anchors: a Gaussian of FWHM ``dnu`` gives
    ``tau_c = sqrt(2 ln 2 / pi)/dnu = 0.6643/dnu``; a Lorentzian
    ``1/(pi dnu)`` (doc 07 §1.2 table).

    Parameters
    ----------
    spectrum_wavelength_m : array-like of float
        Strictly increasing wavelength samples, m.
    spectrum_psd : array-like of float
        Spectral density samples (arbitrary scale) at those wavelengths.

    Returns
    -------
    float
        Coherence time tau_c, s.

    Raises
    ------
    ValueError
        If the table is malformed.
    """
    nu, s_nu = _normalized_spectrum_nu(spectrum_wavelength_m, spectrum_psd)
    return float(np.trapezoid(s_nu**2, nu))


def effective_linewidth_measured_hz(
    spectrum_wavelength_m: NDArray[np.float64] | list[float],
    spectrum_psd: NDArray[np.float64] | list[float],
) -> float:
    """Noise-equivalent linewidth ``dnu_eff = 1/tau_c`` (Derickson ch. 5; M-10).

    ``dnu_eff = (integral S dnu)^2 / integral S^2 dnu`` -- the width in which
    the universal law ``RIN = 2/dnu_eff`` is *shape-independent*; expressed
    against the FWHM it recovers the form factors of
    :data:`RIN_KAPPA_BY_LINESHAPE` (Gaussian ``dnu_eff = 1.5053 dnu``,
    Lorentzian ``pi dnu``).

    Parameters
    ----------
    spectrum_wavelength_m : array-like of float
        Strictly increasing wavelength samples, m.
    spectrum_psd : array-like of float
        Spectral density samples (arbitrary scale) at those wavelengths.

    Returns
    -------
    float
        Noise-equivalent linewidth, Hz.
    """
    return 1.0 / coherence_time_measured_s(spectrum_wavelength_m, spectrum_psd)


def rin_ase_measured(
    spectrum_wavelength_m: NDArray[np.float64] | list[float],
    spectrum_psd: NDArray[np.float64] | list[float],
) -> float:
    """ASE beat-noise RIN floor of a measured spectrum, linear 1/Hz (M-10).

    ``RIN(0) = 2 tau_c`` -- the Siegert chain evaluated on the actual
    spectrum; no lineshape convention enters (doc 07 §1.2). Polarized light
    assumed (unpolarized halves the level, documented spread).
    Thermal/ASE sources (SLD) only.

    Parameters
    ----------
    spectrum_wavelength_m : array-like of float
        Strictly increasing wavelength samples, m.
    spectrum_psd : array-like of float
        Spectral density samples (arbitrary scale) at those wavelengths.

    Returns
    -------
    float
        Relative intensity noise PSD, 1/Hz (one-sided, ``f << dnu_eff``).
    """
    return 2.0 * coherence_time_measured_s(spectrum_wavelength_m, spectrum_psd)


def rin_ase_measured_db_hz(
    spectrum_wavelength_m: NDArray[np.float64] | list[float],
    spectrum_psd: NDArray[np.float64] | list[float],
) -> float:
    """ASE beat-noise RIN floor of a measured spectrum, dB/Hz (M-10).

    Parameters
    ----------
    spectrum_wavelength_m : array-like of float
        Strictly increasing wavelength samples, m.
    spectrum_psd : array-like of float
        Spectral density samples (arbitrary scale) at those wavelengths.

    Returns
    -------
    float
        Relative intensity noise, dB/Hz.
    """
    return 10.0 * math.log10(rin_ase_measured(spectrum_wavelength_m, spectrum_psd))
