"""Pure figure producers for excitation signals (no Qt, no pyplot state).

Per SW-09 / 09 §9 this module builds :class:`matplotlib.figure.Figure` objects
directly (Agg-compatible, headless-safe) and never imports Qt: the CLI saves
the figures to disk and the GUI may embed them later. Three views of an
:class:`~optivibe.core.types.Excitation` are provided — time series, spectrum
(amplitude or Welch PSD) and spectrogram (useful for sweep/random inputs).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from matplotlib.figure import Figure
from scipy import signal as sps

from optivibe.core.types import Excitation, FloatArray

__all__ = ["plot_spectrogram", "plot_spectrum", "plot_time_series"]

_AXES: tuple[str, str, str] = ("x", "y", "z")


def _channel(excitation: Excitation, axis: str) -> FloatArray:
    """Return the acceleration array of one axis."""
    arrays = {"x": excitation.a_x, "y": excitation.a_y, "z": excitation.a_z}
    try:
        return arrays[axis]
    except KeyError as exc:
        msg = f"axis must be one of {_AXES}, got {axis!r}"
        raise ValueError(msg) from exc


def plot_time_series(excitation: Excitation) -> Figure:
    """Plot a(t) of all three axes on a shared time axis.

    Parameters
    ----------
    excitation : Excitation
        Input signal (SI units, m/s^2).

    Returns
    -------
    matplotlib.figure.Figure
        Figure with one stacked subplot per axis.
    """
    n = excitation.n_samples
    t = np.arange(n, dtype=np.float64) / excitation.fs
    fig = Figure(figsize=(8.0, 6.0), constrained_layout=True)
    axes = fig.subplots(3, 1, sharex=True)
    for ax, name in zip(axes, _AXES, strict=True):
        ax.plot(t, _channel(excitation, name), linewidth=0.8)
        ax.set_ylabel(f"a_{name} [m/s$^2$]")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t [s]")
    fig.suptitle("Excitation time series")
    return fig


def plot_spectrum(
    excitation: Excitation,
    axis: Literal["x", "y", "z"] = "x",
    kind: Literal["amplitude", "psd"] = "psd",
) -> Figure:
    """Plot the spectrum of one axis: rFFT amplitude or Welch one-sided PSD.

    Parameters
    ----------
    excitation : Excitation
        Input signal.
    axis : {"x", "y", "z"}, optional
        Which axis to analyze.
    kind : {"amplitude", "psd"}, optional
        ``"amplitude"`` — single-sided rFFT amplitude spectrum [m/s^2];
        ``"psd"`` — Welch PSD [(m/s^2)^2/Hz] on a log-log grid.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with one subplot.
    """
    data = _channel(excitation, axis)
    fs = excitation.fs
    fig = Figure(figsize=(8.0, 4.5), constrained_layout=True)
    ax = fig.subplots()
    if kind == "amplitude":
        n = data.size
        freq = np.fft.rfftfreq(n, d=1.0 / fs)
        amp = np.abs(np.fft.rfft(data)) * 2.0 / n
        if n % 2 == 0 and amp.size > 1:
            amp[-1] /= 2.0  # Nyquist bin is not doubled
        amp[0] /= 2.0  # DC is not doubled
        ax.plot(freq, amp, linewidth=0.8)
        ax.set_ylabel(f"|A_{axis}(f)| [m/s$^2$]")
    else:
        nperseg = min(data.size, 4096)
        freq, psd = sps.welch(data, fs=fs, nperseg=nperseg)
        positive = freq > 0.0
        ax.loglog(freq[positive], psd[positive], linewidth=0.8)
        ax.set_ylabel(f"S_{axis}(f) [(m/s$^2$)$^2$/Hz]")
    ax.set_xlabel("f [Hz]")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(f"Excitation spectrum ({kind}, axis {axis})")
    return fig


def plot_spectrogram(
    excitation: Excitation,
    axis: Literal["x", "y", "z"] = "x",
) -> Figure:
    """Plot a spectrogram of one axis (time-frequency view for sweep/random).

    Parameters
    ----------
    excitation : Excitation
        Input signal.
    axis : {"x", "y", "z"}, optional
        Which axis to analyze.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with the spectrogram image and a colorbar (dB scale).
    """
    data = _channel(excitation, axis)
    fs = excitation.fs
    nperseg = max(16, min(data.size // 8, 1024))
    freq, t, sxx = sps.spectrogram(data, fs=fs, nperseg=nperseg)
    fig = Figure(figsize=(8.0, 4.5), constrained_layout=True)
    ax = fig.subplots()
    floor = np.finfo(np.float64).tiny
    mesh = ax.pcolormesh(t, freq, 10.0 * np.log10(sxx + floor), shading="gouraud")
    fig.colorbar(mesh, ax=ax, label="PSD [dB re 1 (m/s$^2$)$^2$/Hz]")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("f [Hz]")
    ax.set_title(f"Excitation spectrogram (axis {axis})")
    return fig
