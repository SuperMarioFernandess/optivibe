"""Pure figure producers for the analysis layer (no Qt, no pyplot; task S6 §B10).

Per SW-09 / 09 §9 this builds :class:`matplotlib.figure.Figure` objects directly
(Agg-compatible, headless). Views: the ``truth vs recovery`` a/v/x overlay; the
NEA budget ``NEA(f)`` with its analytic plateau and contribution split; the
design / response sweep maps (NEA-vs-parameter, response-vs-amplitude); and the
Monte-Carlo histograms / box plots. Spectra and metrics come from
:mod:`optivibe.analysis` / :mod:`optivibe.dsp`; this module only draws.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from optivibe.analysis.monte_carlo import MonteCarloResult
from optivibe.analysis.nea_budget import NeaBudget
from optivibe.analysis.sweep import SweepResult
from optivibe.core.types import FloatArray, VibrationResult

__all__ = [
    "plot_monte_carlo",
    "plot_nea_budget",
    "plot_sweep",
    "plot_truth_vs_recovery_avx",
]

G0 = 9.80665


def plot_truth_vs_recovery_avx(
    a_true: FloatArray, result: VibrationResult, *, n_max: int = 2000
) -> Figure:
    """Overlay true vs recovered a, plus the recovered v and x (task S6 §B10).

    Parameters
    ----------
    a_true : numpy.ndarray
        Applied target-axis acceleration, m/s^2.
    result : VibrationResult
        Reconstructed vibration (a/v/x).
    n_max : int, optional
        Maximum leading samples to plot.

    Returns
    -------
    matplotlib.figure.Figure
        Three stacked panels: a (true vs recovered), v, x.
    """
    n = min(a_true.size, result.a.size, n_max)
    t = np.arange(n) / result.fs
    fig = Figure(figsize=(8.0, 6.0))
    ax_a, ax_v, ax_x = fig.subplots(3, 1, sharex=True)
    ax_a.plot(t, np.asarray(a_true)[:n], lw=1.2, label="true a")
    ax_a.plot(t, np.asarray(result.a)[:n], lw=0.9, label="recovered a")
    ax_a.set_ylabel("a [m/s^2]")
    ax_a.legend(loc="upper right", fontsize=8)
    ax_a.set_title("truth vs recovery (target axis)")
    ax_v.plot(t, np.asarray(result.v)[:n], lw=0.9, color="tab:green")
    ax_v.set_ylabel("v [m/s]")
    ax_x.plot(t, np.asarray(result.x)[:n], lw=0.9, color="tab:red")
    ax_x.set_ylabel("x [m]")
    ax_x.set_xlabel("time [s]")
    for ax in (ax_a, ax_v, ax_x):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_nea_budget(budget: NeaBudget) -> Figure:
    """Plot NEA(f) with its plateau and the contribution split (task S6 §B10).

    Parameters
    ----------
    budget : NeaBudget
        The NEA budget.

    Returns
    -------
    matplotlib.figure.Figure
        Left: NEA(f) density and the plateau; right: contribution bar chart.
    """
    fig = Figure(figsize=(9.0, 4.0))
    ax_f, ax_bar = fig.subplots(1, 2)
    nea_ug = budget.nea_density / G0 * 1.0e6
    ax_f.loglog(budget.freq_hz, nea_ug, lw=1.2, label="NEA(f)")
    ax_f.axhline(budget.nea_plateau / G0 * 1.0e6, ls="--", color="grey", label="plateau (analytic)")
    ax_f.set_xlabel("frequency [Hz]")
    ax_f.set_ylabel("NEA [ug/sqrt(Hz)]")
    ax_f.set_title("noise-equivalent acceleration")
    ax_f.grid(True, which="both", alpha=0.3)
    ax_f.legend(fontsize=8)
    contribs = ["shot", "rin", "johnson"]
    values = [budget.contributions[c] / G0 * 1.0e6 for c in contribs]
    ax_bar.bar(contribs, values, color=["tab:blue", "tab:orange", "tab:green"])
    ax_bar.axhline(
        budget.contributions["total"] / G0 * 1.0e6, ls="--", color="black", label="total"
    )
    ax_bar.set_ylabel("NEA contribution [ug/sqrt(Hz)]")
    ax_bar.set_title(f"split (ref arm: {budget.reference_arm})")
    ax_bar.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_sweep(result: SweepResult, *, metric: str | None = None) -> Figure:
    """Plot a sweep map (design NEA-vs-parameter or response-vs-amplitude).

    Parameters
    ----------
    result : SweepResult
        The sweep result.
    metric : str or None, optional
        Metric to plot; defaults to ``nea_plateau_ug`` (design) or ``gain_ratio``
        (response) when present.

    Returns
    -------
    matplotlib.figure.Figure
        A single panel of the chosen metric vs the swept axis (log-x for the
        ``length_m`` / ``amplitude_g`` axes).
    """
    if metric is None:
        metric = "nea_plateau_ug" if result.mode == "design" else "gain_ratio"
    if metric not in result.metrics:
        metric = next(iter(result.metrics))
    fig = Figure(figsize=(8.0, 5.0))
    ax = fig.subplots()
    x = result.axis_values
    y = result.metrics[metric]
    log_x = result.parameter in {"length_m", "amplitude_g"}
    plotter = ax.semilogx if log_x else ax.plot
    plotter(x, y, marker="o", lw=1.2)
    if result.mode == "response" and result.parameter == "amplitude_g":
        ax.axvline(50.0, ls=":", color="red", label="50 g (spec limit)")
        ax.legend(fontsize=8)
    unit = result.meta.get("unit", "")
    ax.set_xlabel(f"{result.parameter} [{unit}]")
    ax.set_ylabel(metric)
    ax.set_title(f"sweep: {result.name} ({result.mode})")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_monte_carlo(result: MonteCarloResult, *, metric: str = "nea_full_band_ug") -> Figure:
    """Plot the Monte-Carlo distribution of a metric (histogram + box; §B10).

    Parameters
    ----------
    result : MonteCarloResult
        The Monte-Carlo result.
    metric : str, optional
        Sample key to plot (default ``"nea_full_band_ug"``).

    Returns
    -------
    matplotlib.figure.Figure
        Left: histogram with the median; right: box plot.
    """
    if metric not in result.samples:
        metric = next(iter(result.samples))
    values = result.samples[metric]
    finite = values[np.isfinite(values)]
    fig = Figure(figsize=(9.0, 4.0))
    ax_hist, ax_box = fig.subplots(1, 2)
    ax_hist.hist(finite, bins=min(30, max(5, finite.size // 4)), color="tab:blue", alpha=0.8)
    median = float(np.median(finite)) if finite.size else float("nan")
    ax_hist.axvline(median, ls="--", color="black", label=f"median {median:.3g}")
    ax_hist.set_xlabel(metric)
    ax_hist.set_ylabel("count")
    ax_hist.set_title(f"{result.name}: {metric} ({result.n_draws} draws)")
    ax_hist.legend(fontsize=8)
    ax_box.boxplot(finite, orientation="vertical", showfliers=True)
    ax_box.set_ylabel(metric)
    ax_box.set_title("distribution")
    ax_box.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
