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

Degenerate inputs (task S-25, `SW-77`; doc 17 §1a)
--------------------------------------------------
A metric that **cannot be computed** on the given input returns ``None`` --
"undefined here" -- and never a silent ``0.0``. An honest zero (a signal that
really carries no second harmonic, a band that really holds no energy) and a
degeneracy (a band that holds fewer than two bins, a spectrum without a
fundamental) are different statements about the world, and a measuring
instrument that renders them identically lies quietly; coding convention 10 §7
forbids silent failures. The convention is the project's existing one, not a new
one: ``dropped_samples`` is reported as ``None`` rather than a reassuring zero
(`SW-72`), unknown expected-peak amplitudes are ``None`` rather than a guess
(`SW-70`), and both :class:`~optivibe.analysis.instrument.InstrumentAnalysis`
and :class:`~optivibe.analysis.compare.ChainMetrics` already type the second
harmonic ratio as ``float | None``.

Two shapes of the same rule, by the container that has to hold it:

* a **scalar** metric that is undefined is ``None`` (this module);
* an **element of a sample array** that is undefined is ``nan`` -- ``None`` is
  not expressible in a float ``ndarray``, and the analysis layer already filters
  non-finite samples (``analysis/monte_carlo._percentiles``).

Every degenerate return is logged with its reason (10 §7); the machine-readable
reason vocabulary is :data:`DEGENERATE_REASONS`, which doc 17 §1a mirrors.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from optivibe.core.logging import get_logger
from optivibe.core.types import FloatArray, Spectrum

logger = get_logger(__name__)

__all__ = [
    "DEGENERATE_REASONS",
    "band_rms_velocity",
    "cross_axis_suppression",
    "rms",
    "second_harmonic_ratio",
]

#: Closed vocabulary of reasons a metric of this module can be undefined
#: (doc 17 §1a). Reported in logs and, where a verdict is rendered for the user,
#: in the assessment bag (see :func:`~optivibe.dsp.iso.iso_assessment`).
DEGENERATE_REASONS: Final[tuple[str, ...]] = (
    "band_has_fewer_than_two_bins",
    "spectrum_too_short",
    "fundamental_not_positive",
    "second_harmonic_outside_spectrum",
    "fundamental_amplitude_not_positive",
    "offaxis_input_is_zero",
    "spectrum_not_ready",
)


def rms(values: FloatArray) -> float:
    """Return the root-mean-square of ``values`` (same units as the input)."""
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)))))


def band_rms_velocity(psd: Spectrum, band_hz: tuple[float, float]) -> float | None:
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
    float or None
        RMS velocity in the band, m/s, or ``None`` when the band holds fewer
        than two bins: the trapezoid rule needs an interval, so there is no
        integral to take (reason ``"band_has_fewer_than_two_bins"``). A short
        record or a narrow ISO band against a coarse resolution ``fs / N`` is
        exactly this case, and it must not read as a quiet ``0`` m/s -- which
        ISO 10816-3 would grade as zone A, the best possible verdict.

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
        logger.info(
            "band_rms_velocity undefined (%s): band [%g, %g] Hz holds %d bin(s) of the spectrum",
            "band_has_fewer_than_two_bins",
            f_lo,
            f_hi,
            int(np.count_nonzero(in_band)),
        )
        return None
    power = float(np.trapezoid(psd.values[in_band], freq[in_band]))
    return float(np.sqrt(max(power, 0.0)))


def second_harmonic_ratio(spectrum: Spectrum, fundamental_hz: float) -> float | None:
    """Amplitude ratio ``|X(2 f0)| / |X(f0)|`` at the fundamental ``f0`` (S5 §4).

    A proxy for the quadratic cross-axis / nonlinear contamination that the
    cylinder coupling routes to ``2 f`` (doc 04 §5).

    Parameters
    ----------
    spectrum : Spectrum
        Amplitude spectrum of the recovered target-axis signal.
    fundamental_hz : float
        Fundamental frequency ``f0``, Hz.

    Returns
    -------
    float or None
        Second-harmonic amplitude ratio (dimensionless, >= 0), or ``None`` when
        the ratio is not defined on this input: a spectrum of fewer than two
        bins (``"spectrum_too_short"``), a non-positive fundamental
        (``"fundamental_not_positive"``), a second harmonic beyond the last bin
        (``"second_harmonic_outside_spectrum"``) or a fundamental bin without
        amplitude (``"fundamental_amplitude_not_positive"``). The third case is
        the sharpest: ``2 f0`` above Nyquist means the harmonic was never
        observed, which is not the same statement as "there is no harmonic".
    """
    freq = spectrum.freq
    mag = spectrum.values
    if freq.size < 2:
        logger.info(
            "second_harmonic_ratio undefined (%s): spectrum has %d bin(s)",
            "spectrum_too_short",
            int(freq.size),
        )
        return None
    if fundamental_hz <= 0.0:
        logger.info(
            "second_harmonic_ratio undefined (%s): f0 = %g Hz",
            "fundamental_not_positive",
            fundamental_hz,
        )
        return None
    f0_bin = int(np.argmin(np.abs(freq - fundamental_hz)))
    f2_target = 2.0 * fundamental_hz
    if f2_target > freq[-1]:
        logger.info(
            "second_harmonic_ratio undefined (%s): 2 f0 = %g Hz is above the last bin %g Hz",
            "second_harmonic_outside_spectrum",
            f2_target,
            float(freq[-1]),
        )
        return None
    f2_bin = int(np.argmin(np.abs(freq - f2_target)))
    a_f0 = float(mag[f0_bin])
    if a_f0 <= 0.0:
        logger.info(
            "second_harmonic_ratio undefined (%s): |X(f0)| = %g at %g Hz",
            "fundamental_amplitude_not_positive",
            a_f0,
            float(freq[f0_bin]),
        )
        return None
    return float(mag[f2_bin] / a_f0)


def cross_axis_suppression(recovered_target_rms: float, applied_offaxis_rms: float) -> float | None:
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
    float or None
        Dimensionless cross ratio, or ``None`` when no off-axis excitation was
        applied (``"offaxis_input_is_zero"``): the ratio has no denominator, so
        nothing about cross-sensitivity was measured. Reporting ``0`` there
        would claim perfect suppression -- the most flattering number the metric
        can produce -- on the strength of no measurement at all.
    """
    if applied_offaxis_rms <= 0.0:
        logger.info(
            "cross_axis_suppression undefined (%s): applied off-axis RMS = %g m/s^2",
            "offaxis_input_is_zero",
            applied_offaxis_rms,
        )
        return None
    return recovered_target_rms / applied_offaxis_rms
