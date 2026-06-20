"""Spectra and dominant-frequency extraction (task S5 §3; computed in dsp/).

All spectral products are computed here (``viz`` only draws them, 14 §8):

* :func:`amplitude_spectrum` -- one-sided rFFT amplitude (same units as the
  signal), correctly scaled (``2/N``, DC and Nyquist not doubled), so a pure
  tone of amplitude ``A`` reads ``A`` at its bin;
* :func:`welch_psd` -- one-sided power spectral density via
  :func:`scipy.signal.welch` (window / segment / overlap from
  :class:`~optivibe.core.config.models.DspOptions`);
* :func:`spectrogram` -- a time-frequency map via
  :func:`scipy.signal.spectrogram` (for ``viz``);
* :func:`dominant_frequencies` -- the most prominent spectral peaks with
  parabolic (quadratic) sub-bin interpolation, ordered by prominence.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, welch
from scipy.signal import spectrogram as _scipy_spectrogram

from optivibe.core.types import FloatArray, Spectrum

__all__ = [
    "amplitude_spectrum",
    "dominant_frequencies",
    "spectrogram",
    "welch_psd",
]


def amplitude_spectrum(signal: FloatArray, fs: float, *, window: str = "boxcar") -> Spectrum:
    """One-sided amplitude spectrum (rFFT), same units as ``signal`` (S5 §3).

    Scaled by ``2/N`` with the DC (and Nyquist, for even ``N``) bins halved, so a
    sinusoid of amplitude ``A`` reads ``A`` at its frequency. An optional window
    is applied with amplitude (coherent-gain) correction.

    Parameters
    ----------
    signal : numpy.ndarray
        Real time series.
    fs : float
        Sampling frequency, Hz.
    window : str, optional
        Window name (``"boxcar"`` = rectangular, the default).

    Returns
    -------
    Spectrum
        Amplitude spectrum (``kind="amplitude"``, ``method="fft"``).
    """
    from scipy.signal import get_window

    n = signal.size
    win = get_window(window, n).astype(np.float64) if window != "boxcar" else np.ones(n)
    coherent_gain = float(np.sum(win) / n)
    spectrum = np.fft.rfft(signal * win)
    freq = np.fft.rfftfreq(n, d=1.0 / fs).astype(np.float64)
    amplitude = np.abs(spectrum).astype(np.float64) * (2.0 / n) / coherent_gain
    if amplitude.size:
        amplitude[0] *= 0.5  # DC is one-sided already
        if n % 2 == 0:
            amplitude[-1] *= 0.5  # Nyquist is one-sided already
    return Spectrum(freq=freq, values=amplitude, kind="amplitude", window=window, method="fft")


def welch_psd(
    signal: FloatArray,
    fs: float,
    *,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> Spectrum:
    """One-sided power spectral density via Welch's method (S5 §3).

    Parameters
    ----------
    signal : numpy.ndarray
        Real time series, units ``u``.
    fs : float
        Sampling frequency, Hz.
    window : str, optional
        Window name (default ``"hann"``).
    nperseg : int or None, optional
        Segment length; defaults to ``min(len(signal), 256-rounded)`` capped at
        the signal length so short records still produce a PSD.
    noverlap : int or None, optional
        Segment overlap; defaults to ``nperseg // 2``.

    Returns
    -------
    Spectrum
        Power spectral density (``kind="psd"``, ``method="welch"``), units
        ``u^2/Hz``.
    """
    n = signal.size
    seg = n if nperseg is None else min(nperseg, n)
    seg = max(seg, 1)
    ov = noverlap if noverlap is not None else seg // 2
    ov = int(np.clip(ov, 0, seg - 1)) if seg > 1 else 0
    freq, psd = welch(signal, fs=fs, window=window, nperseg=seg, noverlap=ov)
    return Spectrum(
        freq=np.asarray(freq, dtype=np.float64),
        values=np.asarray(psd, dtype=np.float64),
        kind="psd",
        window=window,
        method="welch",
    )


def spectrogram(
    signal: FloatArray,
    fs: float,
    *,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Time-frequency power map via :func:`scipy.signal.spectrogram` (for viz).

    Parameters
    ----------
    signal : numpy.ndarray
        Real time series.
    fs : float
        Sampling frequency, Hz.
    window : str, optional
        Window name (default ``"hann"``).
    nperseg : int or None, optional
        Segment length; defaults to ``min(len(signal)//8, 256)`` (at least 8).
    noverlap : int or None, optional
        Overlap; defaults to ``nperseg // 2``.

    Returns
    -------
    freq : numpy.ndarray
        Frequency axis, Hz.
    time : numpy.ndarray
        Time axis, s.
    power : numpy.ndarray, shape (len(freq), len(time))
        Power spectral density per segment, units^2/Hz.
    """
    n = signal.size
    seg = nperseg if nperseg is not None else max(8, min(n // 8, 256))
    seg = max(1, min(seg, n))
    ov = noverlap if noverlap is not None else seg // 2
    ov = int(np.clip(ov, 0, seg - 1)) if seg > 1 else 0
    freq, time, power = _scipy_spectrogram(signal, fs=fs, window=window, nperseg=seg, noverlap=ov)
    return (
        np.asarray(freq, dtype=np.float64),
        np.asarray(time, dtype=np.float64),
        np.asarray(power, dtype=np.float64),
    )


def _parabolic_peak(freq: FloatArray, magnitude: FloatArray, index: int) -> float:
    """Sub-bin peak frequency by quadratic interpolation around ``index``.

    Fits a parabola to the three points ``(index-1, index, index+1)`` of the
    magnitude and returns the interpolated peak frequency. Falls back to the bin
    frequency at the array edges.
    """
    if index <= 0 or index >= magnitude.size - 1:
        return float(freq[index])
    y0, y1, y2 = magnitude[index - 1], magnitude[index], magnitude[index + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return float(freq[index])
    delta = 0.5 * (y0 - y2) / denom  # in bins, within [-0.5, 0.5]
    df = float(freq[1] - freq[0]) if freq.size > 1 else 0.0
    return float(freq[index] + delta * df)


def dominant_frequencies(
    spectrum: Spectrum, *, max_peaks: int = 3, min_prominence_ratio: float = 0.1
) -> tuple[float, ...]:
    """Most prominent spectral peaks, Hz, ordered by prominence (S5 §3).

    Peaks are found on the spectrum magnitude (excluding the DC bin) with
    :func:`scipy.signal.find_peaks`, ranked by prominence and refined by
    parabolic interpolation. A peak must clear ``min_prominence_ratio`` of the
    peak magnitude to count, which rejects noise ripple.

    Parameters
    ----------
    spectrum : Spectrum
        Amplitude or PSD spectrum.
    max_peaks : int, optional
        Maximum number of peaks to return (default 3).
    min_prominence_ratio : float, optional
        Minimum peak prominence as a fraction of the maximum magnitude
        (default 0.1).

    Returns
    -------
    tuple of float
        Dominant frequencies, Hz, ordered by descending prominence (empty if no
        peak clears the threshold).
    """
    freq = spectrum.freq
    mag = spectrum.values
    if mag.size < 3:
        if mag.size > 1:
            idx = int(np.argmax(mag[1:])) + 1
            return (float(freq[idx]),)
        return ()
    body = mag.copy()
    body[0] = 0.0  # ignore DC
    peak_max = float(np.max(body))
    if peak_max <= 0.0:
        return ()
    # The primary dominant is always the global maximum bin, so the reported
    # leading line matches a plain argmax (regression with the S1-S4 dominants);
    # any further lines are the next most prominent peaks.
    global_idx = int(np.argmax(body))
    dominant: list[float] = [_parabolic_peak(freq, mag, global_idx)]
    indices, props = find_peaks(body, prominence=min_prominence_ratio * peak_max)
    if indices.size:
        order = np.argsort(props["prominences"])[::-1]
        for i in order:
            idx = int(indices[i])
            if abs(idx - global_idx) <= 1:
                continue  # already counted as the primary
            dominant.append(_parabolic_peak(freq, mag, idx))
            if len(dominant) >= max_peaks:
                break
    return tuple(dominant)
