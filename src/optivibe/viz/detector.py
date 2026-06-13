"""Pure figure producers for the detector stage (no Qt, no pyplot).

Per SW-09 / 09 §9 this module builds :class:`matplotlib.figure.Figure` objects
directly (Agg-compatible, headless-safe). Two views are provided: the analytic
one-sided current-noise budget (shot / RIN / Johnson / total) of a variant
(doc 07 §1), optionally overlaid with a Welch estimate of a noise realization,
and the digitized detector time series with its DC pedestal.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from scipy.signal import welch

from optivibe.core.config.loader import default_config_dir, load_constants
from optivibe.core.config.models import Constants, VariantConfig
from optivibe.core.types import DetectorOutput, FloatArray
from optivibe.detector.photodiode import noise_psd

__all__ = ["plot_detector_timeseries", "plot_noise_psd"]


def _dc_current(variant: VariantConfig, eta0: float) -> float:
    """DC photocurrent ``I_DC = R P (R1 + rho eta0)`` of a variant, A."""
    gain = variant.responsivity_a_w * variant.source.power_w
    return float(gain * (variant.endface_reflectivity + variant.reflector.reflectivity * eta0))


def plot_noise_psd(
    variant: VariantConfig,
    eta0: float,
    fs: float,
    *,
    balanced: bool | None = None,
    realization: DetectorOutput | None = None,
    constants: Constants | None = None,
) -> Figure:
    """Plot the analytic current-noise budget of a variant (doc 07 §1).

    Parameters
    ----------
    variant : VariantConfig
        Sensor variant (source RIN, detector electronics).
    eta0 : float
        Optical working point eta0 setting the DC photocurrent.
    fs : float
        Sampling frequency, Hz (defines the Nyquist band of the flat levels).
    balanced : bool or None, optional
        Whether the balanced reference channel is active; the variant value is
        used when None.
    realization : DetectorOutput or None, optional
        A noise realization to overlay as a Welch PSD estimate (its AC part,
        ``samples - dc_level``).
    constants : Constants or None, optional
        Physical constants; loaded from ``configs/constants.yaml`` when None.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with the shot / RIN / Johnson / total PSD levels (and the
        optional Welch overlay).
    """
    if constants is None:
        constants = load_constants(default_config_dir() / "constants.yaml")
    use_balanced = variant.detector.balanced if balanced is None else balanced
    i_dc = _dc_current(variant, eta0)
    psd = noise_psd(i_dc, variant, constants, balanced=use_balanced)

    fig = Figure(figsize=(8.0, 4.5), constrained_layout=True)
    ax = fig.subplots()
    freqs: FloatArray = np.array([1.0, fs / 2.0])
    styles = {
        "shot": ("tab:blue", "shot $2eI_{DC}$"),
        "rin": ("tab:orange", "RIN (effective)"),
        "johnson": ("tab:green", "Johnson $4k_BT/R_f$"),
        "total": ("black", "total"),
    }
    for key, (color, label) in styles.items():
        level = psd[key]
        if level <= 0.0:
            continue
        lw = 2.0 if key == "total" else 1.2
        ls = "-" if key == "total" else "--"
        ax.plot(freqs, [level, level], ls, color=color, linewidth=lw, label=label)

    if realization is not None:
        ac = np.asarray(realization.samples, dtype=np.float64) - realization.dc_level
        nperseg = min(ac.size, 1024)
        f_welch, p_welch = welch(ac, fs=realization.fs, nperseg=nperseg)
        ax.plot(f_welch[1:], p_welch[1:], color="tab:red", linewidth=0.8, alpha=0.6, label="Welch")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"current PSD [A$^2$/Hz]")
    ax.set_title(
        f"variant {variant.name} noise budget "
        f"($I_{{DC}}$={i_dc * 1e3:.2f} mA, balanced={use_balanced})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_detector_timeseries(detector: DetectorOutput, n_max: int = 2000) -> Figure:
    """Plot the digitized detector signal and its DC pedestal.

    Parameters
    ----------
    detector : DetectorOutput
        Digitized detector output.
    n_max : int, optional
        Maximum number of leading samples to plot.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with two subplots: the full signal with the DC level, and the
        AC-coupled modulation (samples - dc_level).
    """
    n = min(detector.n_samples, n_max)
    samples = np.asarray(detector.samples, dtype=np.float64)[:n]
    t = np.arange(n) / detector.fs
    unit = detector.units

    fig = Figure(figsize=(8.0, 5.5), constrained_layout=True)
    ax_top, ax_bot = fig.subplots(2, 1, sharex=True)

    ax_top.plot(t, samples, linewidth=0.8, label="samples")
    ax_top.axhline(detector.dc_level, color="tab:red", linestyle="--", linewidth=1.0, label="DC")
    ax_top.set_ylabel(f"signal [{unit}]")
    ax_top.set_title("digitized detector output")
    ax_top.legend(loc="best", fontsize=8)
    ax_top.grid(True, alpha=0.3)

    ax_bot.plot(t, samples - detector.dc_level, linewidth=0.8, color="tab:purple")
    ax_bot.set_xlabel("time [s]")
    ax_bot.set_ylabel(f"AC [{unit}]")
    ax_bot.grid(True, alpha=0.3)
    return fig
