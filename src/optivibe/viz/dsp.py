"""Pure figure producers for the inverse/DSP stage (no Qt, no pyplot).

Per SW-09 / 09 §9 this module builds :class:`matplotlib.figure.Figure` objects
directly (Agg-compatible, headless-safe); the CLI saves them and the GUI embeds
them. Views: the "true vs recovered" acceleration overlay with its residual, the
representative spectrum with the dominant lines marked, a spectrogram, the
noise-equivalent acceleration ``NEA(f)`` against its analytic plateau, and the
``v``/``x`` kinematics. The numbers come from :mod:`optivibe.dsp`; this module
only draws them.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from optivibe.core.config.models import Constants, VariantConfig
from optivibe.core.types import DetectorOutput, FloatArray, Spectrum
from optivibe.dsp.nea import nea_from_detector, nea_spectrum
from optivibe.dsp.spectra import spectrogram

__all__ = [
    "plot_kinematics",
    "plot_nea",
    "plot_spectrogram",
    "plot_spectrum",
    "plot_true_vs_recovered",
]


def plot_true_vs_recovered(
    a_true: FloatArray, a_recovered: FloatArray, fs: float, *, n_max: int = 2000
) -> Figure:
    """Overlay the true and recovered acceleration with the residual (S5).

    Parameters
    ----------
    a_true : numpy.ndarray
        Applied target-axis acceleration, m/s^2.
    a_recovered : numpy.ndarray
        Reconstructed acceleration from the inverse chain, m/s^2.
    fs : float
        Sampling frequency, Hz.
    n_max : int, optional
        Maximum number of leading samples to plot.

    Returns
    -------
    matplotlib.figure.Figure
        Two subplots: the true/recovered overlay and their difference.
    """
    n = min(a_true.size, a_recovered.size, n_max)
    t = np.arange(n) / fs
    true = np.asarray(a_true, dtype=np.float64)[:n]
    rec = np.asarray(a_recovered, dtype=np.float64)[:n]

    fig = Figure(figsize=(8.0, 5.5), constrained_layout=True)
    ax_top, ax_bot = fig.subplots(2, 1, sharex=True)

    ax_top.plot(t, true, linewidth=1.2, label="true", color="tab:blue")
    ax_top.plot(t, rec, linewidth=0.9, label="recovered", color="tab:red", linestyle="--")
    ax_top.set_ylabel(r"$a$ [m/s$^2$]")
    ax_top.set_title("acceleration: true vs recovered")
    ax_top.legend(loc="best", fontsize=8)
    ax_top.grid(True, alpha=0.3)

    ax_bot.plot(t, rec - true, linewidth=0.8, color="tab:purple")
    ax_bot.set_xlabel("time [s]")
    ax_bot.set_ylabel(r"residual [m/s$^2$]")
    ax_bot.grid(True, alpha=0.3)
    return fig


def plot_spectrum(spectrum: Spectrum, dominant_freqs_hz: tuple[float, ...] = ()) -> Figure:
    """Plot a spectrum and mark the dominant lines (S5).

    Parameters
    ----------
    spectrum : Spectrum
        Amplitude or PSD spectrum from the inverse chain.
    dominant_freqs_hz : tuple of float, optional
        Dominant frequencies to annotate, Hz.

    Returns
    -------
    matplotlib.figure.Figure
        The spectrum (log-y for a PSD) with vertical markers at the dominants.
    """
    fig = Figure(figsize=(8.0, 4.5), constrained_layout=True)
    ax = fig.subplots()
    freq = spectrum.freq
    values = spectrum.values
    ax.plot(freq, values, linewidth=0.9, color="tab:blue")
    if spectrum.kind == "psd":
        ax.set_yscale("log")
        ax.set_ylabel("PSD")
    else:
        ax.set_ylabel("amplitude")
    for f_peak in dominant_freqs_hz:
        ax.axvline(f_peak, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.annotate(
            f"{f_peak:.3f} Hz",
            xy=(f_peak, float(np.max(values)) if values.size else 0.0),
            fontsize=8,
            color="tab:red",
        )
    ax.set_xlabel("frequency [Hz]")
    ax.set_title(f"spectrum ({spectrum.kind}, {spectrum.method}, window={spectrum.window})")
    ax.grid(True, alpha=0.3)
    return fig


def plot_spectrogram(
    accel: FloatArray, fs: float, *, window: str = "hann", nperseg: int | None = None
) -> Figure:
    """Plot a spectrogram of the recovered acceleration (S5).

    Parameters
    ----------
    accel : numpy.ndarray
        Acceleration time series, m/s^2.
    fs : float
        Sampling frequency, Hz.
    window : str, optional
        Window name (default ``"hann"``).
    nperseg : int or None, optional
        Segment length; chosen from the record when ``None``.

    Returns
    -------
    matplotlib.figure.Figure
        The time-frequency magnitude (dB) as a pcolormesh.
    """
    freqs, times, sxx = spectrogram(accel, fs, window=window, nperseg=nperseg)
    fig = Figure(figsize=(8.0, 4.5), constrained_layout=True)
    ax = fig.subplots()
    floor = float(np.max(sxx)) * 1e-12 + 1e-30
    sxx_db = 10.0 * np.log10(np.maximum(sxx, floor))
    mesh = ax.pcolormesh(times, freqs, sxx_db, shading="auto", cmap="magma")
    fig.colorbar(mesh, ax=ax, label="PSD [dB]")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("frequency [Hz]")
    ax.set_title("spectrogram")
    return fig


def plot_nea(
    detector: DetectorOutput,
    variant: VariantConfig,
    *,
    f_min_hz: float = 1.0,
    f_max_hz: float | None = None,
    n_points: int = 400,
    constants: Constants | None = None,
) -> Figure:
    """Plot ``NEA(f)`` against the analytic plateau (S5 §5; doc 05 §7).

    Parameters
    ----------
    detector : DetectorOutput
        Digitized detector with photodiode noise metadata.
    variant : VariantConfig
        Sensor variant.
    f_min_hz : float, optional
        Lower frequency of the curve, Hz.
    f_max_hz : float or None, optional
        Upper frequency, Hz; defaults to the detector Nyquist.
    n_points : int, optional
        Number of log-spaced frequency points.
    constants : Constants or None, optional
        Physical constants (default loaded).

    Returns
    -------
    matplotlib.figure.Figure
        ``NEA(f)`` with the plateau line and the full-band figure annotated.

    Raises
    ------
    ValueError
        If the detector carries no noise PSD.
    """
    summary = nea_from_detector(detector, variant, constants)
    if summary is None:
        msg = "plot_nea requires a photodiode detector with a noise PSD"
        raise ValueError(msg)
    upper = detector.fs / 2.0 if f_max_hz is None else f_max_hz
    freq = np.logspace(np.log10(f_min_hz), np.log10(upper), n_points).astype(np.float64)
    nea = nea_spectrum(detector, variant, freq, constants)

    fig = Figure(figsize=(8.0, 4.5), constrained_layout=True)
    ax = fig.subplots()
    g0 = 9.80665
    ax.plot(freq, nea / g0 * 1e6, linewidth=1.2, color="tab:blue", label=r"NEA$(f)$")
    ax.axhline(
        summary.nea_plateau / g0 * 1e6,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label=f"plateau {summary.nea_plateau / g0 * 1e6:.2f} ug/rtHz",
    )
    ax.set_xscale("log")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"NEA [$\mu g/\sqrt{\mathrm{Hz}}$]")
    ax.set_title(
        f"noise-equivalent acceleration ({summary.reference_arm}, "
        f"full-band {summary.nea_full_band / g0 * 1e3:.2f} mg)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_kinematics(
    velocity: FloatArray, displacement: FloatArray, fs: float, *, n_max: int = 2000
) -> Figure:
    """Plot the recovered velocity and displacement (S5).

    Parameters
    ----------
    velocity : numpy.ndarray
        Velocity, m/s.
    displacement : numpy.ndarray
        Displacement, m.
    fs : float
        Sampling frequency, Hz.
    n_max : int, optional
        Maximum number of leading samples to plot.

    Returns
    -------
    matplotlib.figure.Figure
        Two subplots: velocity and displacement against time.
    """
    n = min(velocity.size, displacement.size, n_max)
    t = np.arange(n) / fs
    fig = Figure(figsize=(8.0, 5.5), constrained_layout=True)
    ax_v, ax_x = fig.subplots(2, 1, sharex=True)

    ax_v.plot(t, np.asarray(velocity, dtype=np.float64)[:n], linewidth=0.9, color="tab:green")
    ax_v.set_ylabel(r"$v$ [m/s]")
    ax_v.set_title("recovered kinematics")
    ax_v.grid(True, alpha=0.3)

    ax_x.plot(t, np.asarray(displacement, dtype=np.float64)[:n], linewidth=0.9, color="tab:orange")
    ax_x.set_xlabel("time [s]")
    ax_x.set_ylabel(r"$x$ [m]")
    ax_x.grid(True, alpha=0.3)
    return fig
