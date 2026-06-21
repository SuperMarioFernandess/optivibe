"""NEA budget: spectrum, contribution split and displacement floor (task S6 §B6).

Refers the detector noise to the input as the noise-equivalent acceleration
``NEA(f) = sqrt(S_i(f)) / |s_target(f)|`` (doc 07 §1, 05 §7) and decomposes the
plateau floor into the shot / RIN / Johnson contributions read from the detector
metadata, with an independent analytic cross-check (the convention is propagated
from ``reference_arm``, never re-picked -- O-SW-08/SW-32). It also derives the
**displacement noise floor**: double integration amplifies the low-frequency
noise as ``PSD_x = PSD_a / omega^4`` (and ``PSD_v = PSD_a / omega^2``), so the
RMS displacement floor is dominated by the lower band edge and is *much higher*
than the acceleration error -- the two must never be conflated in a report
(task §8; the elevated ``rms x`` of ``recover_sine`` is exactly this effect, not
a calibration error).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from optivibe.core.config.loader import load_constants
from optivibe.core.config.models import Constants, VariantConfig
from optivibe.core.types import DetectorOutput, FloatArray
from optivibe.dsp.nea import analytic_noise_psd, nea_from_detector, nea_spectrum

__all__ = ["NeaBudget", "nea_budget"]


@dataclass(frozen=True)
class NeaBudget:
    """Full NEA budget referred to the input (doc 07; task S6 §B6).

    Attributes
    ----------
    freq_hz : numpy.ndarray
        Frequency grid over the assessment band, Hz.
    nea_density : numpy.ndarray
        NEA spectral density across the band, (m/s^2)/sqrt(Hz) (dips toward f1).
    nea_plateau : float
        Plateau NEA density, (m/s^2)/sqrt(Hz).
    nea_full_band : float
        Full-band NEA, m/s^2 (RMS) -- ``nea_plateau * sqrt(B)``.
    bandwidth_hz : float
        Noise bandwidth ``B`` used for the full-band figure, Hz.
    contributions : Mapping[str, float]
        Per-contribution plateau NEA density (keys ``"shot"``, ``"rin"``,
        ``"johnson"``, ``"total"``), (m/s^2)/sqrt(Hz).
    psd_components : Mapping[str, float]
        The current-noise PSD components from the metadata, A^2/Hz.
    psd_total_analytic : float
        Analytic re-assembly of the total PSD (same convention), A^2/Hz.
    psd_rel_error : float
        Relative gap ``|sim - analytic| / analytic`` (target <= 0.15, SW-29).
    reference_arm : str
        The shot-model convention propagated from the detector (O-SW-08).
    s_target : float
        Signed plateau sensitivity, A/(m/s^2).
    velocity_floor_rms : float
        RMS velocity floor over the band from ``PSD_v = PSD_a / omega^2``, m/s.
    displacement_floor_rms : float
        RMS displacement floor over the band from ``PSD_x = PSD_a / omega^4``, m
        -- the low-frequency-amplified figure that must be separated from the
        (small) acceleration error.
    """

    freq_hz: FloatArray
    nea_density: FloatArray
    nea_plateau: float
    nea_full_band: float
    bandwidth_hz: float
    contributions: dict[str, float]
    psd_components: dict[str, float]
    psd_total_analytic: float
    psd_rel_error: float
    reference_arm: str
    s_target: float
    velocity_floor_rms: float
    displacement_floor_rms: float


def _band_grid(variant: VariantConfig, n_points: int = 256) -> FloatArray:
    """Log-spaced frequency grid over the variant's assessment band, Hz."""
    f_min = max(variant.band.f_min_hz, 1.0e-3)
    f_max = variant.band.f_max_hz
    return np.geomspace(f_min, f_max, n_points).astype(np.float64)


def _floor_rms(nea_density: FloatArray, freq_hz: FloatArray, power: int) -> float:
    """RMS of ``x`` or ``v`` from ``PSD = NEA^2 / omega^(2*power)`` over the band.

    ``power = 1`` gives the velocity floor (``/omega^2``); ``power = 2`` the
    displacement floor (``/omega^4``). The integral is over the band grid.
    """
    omega = 2.0 * math.pi * freq_hz
    psd = nea_density**2 / omega ** (2 * power)
    variance = float(np.trapezoid(psd, freq_hz))
    return math.sqrt(max(variance, 0.0))


def nea_budget(
    detector: DetectorOutput,
    variant: VariantConfig,
    constants: Constants | None = None,
    *,
    n_points: int = 256,
) -> NeaBudget | None:
    """Assemble the NEA budget from a photodiode detector output (task S6 §B6).

    Returns ``None`` for a noiseless (stub) detector -- there is no noise to refer.

    Parameters
    ----------
    detector : DetectorOutput
        Digitized photodiode signal with noise metadata.
    variant : VariantConfig
        Sensor variant.
    constants : Constants or None, optional
        Physical constants (default loaded when ``None``).
    n_points : int, optional
        Number of points in the band frequency grid (default 256).

    Returns
    -------
    NeaBudget or None
        The full budget, or ``None`` for a noiseless detector.
    """
    consts = load_constants() if constants is None else constants
    summary = nea_from_detector(detector, variant, consts)
    if summary is None:
        return None
    noise = detector.noise
    freq = _band_grid(variant, n_points)
    nea_density = nea_spectrum(detector, variant, freq, consts)

    s_target = summary.s_target
    abs_s = abs(s_target)
    psd_components = {
        "shot": float(noise["psd_shot_a2_hz"]),  # type: ignore[arg-type]
        "rin": float(noise["psd_rin_a2_hz"]),  # type: ignore[arg-type]
        "johnson": float(noise["psd_johnson_a2_hz"]),  # type: ignore[arg-type]
        "total": float(noise["psd_total_a2_hz"]),  # type: ignore[arg-type]
    }
    contributions = {key: math.sqrt(value) / abs_s for key, value in psd_components.items()}

    psd_total_analytic = analytic_noise_psd(detector, variant, consts)
    psd_rel_error = abs(psd_components["total"] - psd_total_analytic) / psd_total_analytic

    return NeaBudget(
        freq_hz=freq,
        nea_density=nea_density,
        nea_plateau=summary.nea_plateau,
        nea_full_band=summary.nea_full_band,
        bandwidth_hz=summary.bandwidth_hz,
        contributions=contributions,
        psd_components=psd_components,
        psd_total_analytic=psd_total_analytic,
        psd_rel_error=psd_rel_error,
        reference_arm=summary.reference_arm,
        s_target=s_target,
        velocity_floor_rms=_floor_rms(nea_density, freq, power=1),
        displacement_floor_rms=_floor_rms(nea_density, freq, power=2),
    )
