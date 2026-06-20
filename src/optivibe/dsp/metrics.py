"""Metrics: RMS, band RMS velocity (ISO) and cross-axis residual (task S5 §4).

* :func:`rms` -- root-mean-square of a signal;
* :func:`band_rms_velocity` -- broadband RMS velocity inside the assessment band
  ``[f_lo, f_hi]`` (the quantity ISO 10816-3 grades), integrated from the
  velocity PSD;
* :func:`second_harmonic_ratio` -- amplitude at ``2 f0`` over the amplitude at
  the fundamental ``f0``: a self-contained proxy for nonlinear / cross-axis
  contamination (the cylinder cross channel is quadratic and appears at ``2 f``,
  doc 04 §5);
* :func:`cross_axis_suppression` -- rigorous cross metric when the applied
  off-axis acceleration is known (tests / S6 orchestration): ratio of recovered
  target-axis RMS to the applied off-axis RMS.
"""

from __future__ import annotations

import numpy as np

from optivibe.core.types import FloatArray, Spectrum

__all__ = [
    "band_rms_velocity",
    "cross_axis_suppression",
    "rms",
    "second_harmonic_ratio",
]


def rms(values: FloatArray) -> float:
    """Return the root-mean-square of ``values`` (same units as the input)."""
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)))))


def band_rms_velocity(psd: Spectrum, band_hz: tuple[float, float]) -> float:
    """Broadband RMS velocity in ``[f_lo, f_hi]`` from a velocity PSD, m/s (S5 §4).

    Integrates the one-sided velocity PSD over the assessment band (trapezoid)
    and takes the square root -- the broadband velocity ISO 10816-3 grades.

    Parameters
    ----------
    psd : Spectrum
        Velocity power spectral density (``kind="psd"``), units (m/s)^2/Hz.
    band_hz : tuple of float
        Assessment band ``(f_lo, f_hi)``, Hz.

    Returns
    -------
    float
        RMS velocity in the band, m/s.

    Raises
    ------
    ValueError
        If ``psd`` is not a PSD spectrum.
    """
    if psd.kind != "psd":
        msg = f"band_rms_velocity expects a PSD spectrum, got kind={psd.kind!r}"
        raise ValueError(msg)
    f_lo, f_hi = band_hz
    freq = psd.freq
    in_band = (freq >= f_lo) & (freq <= f_hi)
    if np.count_nonzero(in_band) < 2:
        return 0.0
    power = float(np.trapezoid(psd.values[in_band], freq[in_band]))
    return float(np.sqrt(max(power, 0.0)))


def second_harmonic_ratio(spectrum: Spectrum, fundamental_hz: float) -> float:
    """Amplitude ratio ``|X(2 f0)| / |X(f0)|`` at the fundamental ``f0`` (S5 §4).

    A proxy for the quadratic cross-axis / nonlinear contamination that the
    cylinder coupling routes to ``2 f`` (doc 04 §5). Returns 0 when the
    fundamental bin has no amplitude or the second harmonic falls outside the
    spectrum.

    Parameters
    ----------
    spectrum : Spectrum
        Amplitude spectrum of the recovered target-axis signal.
    fundamental_hz : float
        Fundamental frequency ``f0``, Hz.

    Returns
    -------
    float
        Second-harmonic amplitude ratio (dimensionless, >= 0).
    """
    freq = spectrum.freq
    mag = spectrum.values
    if freq.size < 2 or fundamental_hz <= 0.0:
        return 0.0
    f0_bin = int(np.argmin(np.abs(freq - fundamental_hz)))
    f2_target = 2.0 * fundamental_hz
    if f2_target > freq[-1]:
        return 0.0
    f2_bin = int(np.argmin(np.abs(freq - f2_target)))
    a_f0 = float(mag[f0_bin])
    if a_f0 <= 0.0:
        return 0.0
    return float(mag[f2_bin] / a_f0)


def cross_axis_suppression(recovered_target_rms: float, applied_offaxis_rms: float) -> float:
    """Ratio of recovered target-axis RMS to applied off-axis RMS (S5 §4).

    Rigorous cross-sensitivity used when the applied off-axis excitation is known
    (tests, S6). A small value means the off-axis input leaks weakly into the
    target-axis reconstruction.

    Parameters
    ----------
    recovered_target_rms : float
        RMS of the recovered target-axis acceleration, m/s^2.
    applied_offaxis_rms : float
        RMS of the applied off-axis acceleration, m/s^2.

    Returns
    -------
    float
        Dimensionless cross ratio (0 when the off-axis input is zero).
    """
    if applied_offaxis_rms <= 0.0:
        return 0.0
    return recovered_target_rms / applied_offaxis_rms
