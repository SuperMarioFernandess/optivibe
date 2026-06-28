"""Pure reference-curve figures for the parameter-choice tab (task S7-mod §5).

Per SW-09 / 09 §9 this module builds :class:`matplotlib.figure.Figure` objects
directly (Agg-compatible, headless-safe), so the GUI's *Physics / reference* tab
can show the key design dependencies for the **current composition** without any
Qt or DSP in the view. Three light, cheap curves are provided -- the first-mode
frequency versus length ``f1(L)``, the lateral transfer ``|H_lat(f)|`` and the
shape-agnostic coupling ``eta(dx)`` -- each annotated with the operating point of
the resolved variant. The physics comes from :mod:`optivibe.mechanics` and the
reflector model layer (:mod:`optivibe.optics`); this module only draws.

Heavier curves (measured ``NEA(f)`` with its shot/RIN/Johnson split, the sensor
family sweep) stay on the analysis/worker path and are drawn by
:mod:`optivibe.viz.dsp` / :mod:`optivibe.viz.analysis` (task S7-mod §5: light
curves recompute on edit, heavy curves go through the worker).
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from optivibe.core.config.models import Constants
from optivibe.core.types import FloatArray
from optivibe.mechanics.cantilever import CantileverModel, first_mode_hz
from optivibe.optics.reflector import ReflectorModel

__all__ = [
    "plot_first_mode_vs_length",
    "plot_lateral_transfer",
    "plot_reflector_eta_vs_dx",
]


def plot_first_mode_vs_length(
    constants: Constants,
    current_length_m: float,
    *,
    length_min_m: float = 0.5e-3,
    length_max_m: float = 6.0e-3,
    n: int = 200,
) -> Figure:
    """Plot the first bending-mode frequency ``f1(L)`` (doc 02 §2; 08 R-31).

    The working length of the resolved composition is marked, so the tab shows
    at a glance where the variant sits on the ``f1 ~ 1/L^2`` curve.

    Parameters
    ----------
    constants : Constants
        Physical constants (doc 01 mirror).
    current_length_m : float
        Length L of the current composition, m (marked on the curve).
    length_min_m, length_max_m : float, optional
        Length-axis span, m.
    n : int, optional
        Number of grid points.

    Returns
    -------
    matplotlib.figure.Figure
        One log-y subplot of ``f1`` (kHz) versus ``L`` (mm).
    """
    lengths: FloatArray = np.linspace(length_min_m, length_max_m, n)
    f1 = np.array([first_mode_hz(constants, length) for length in lengths], dtype=np.float64)
    fig = Figure(figsize=(7.5, 4.2), constrained_layout=True)
    ax = fig.subplots()
    ax.plot(lengths * 1e3, f1 * 1e-3, linewidth=1.4, label=r"$f_1(L)$")
    f1_here = first_mode_hz(constants, current_length_m)
    ax.plot(
        [current_length_m * 1e3],
        [f1_here * 1e-3],
        "o",
        color="tab:red",
        label=rf"L = {current_length_m * 1e3:.2f} mm $\to$ {f1_here * 1e-3:.2f} kHz",
    )
    ax.set_xlabel("cantilever length L [mm]")
    ax.set_ylabel(r"$f_1$ [kHz]")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("First bending mode vs length")
    return fig


def plot_lateral_transfer(
    model: CantileverModel,
    *,
    f_min_hz: float,
    f_max_hz: float,
    n: int = 400,
) -> Figure:
    """Plot the lateral FRF magnitude ``|H_lat(f)|`` for the current cantilever.

    ``H_lat(f) = H_lat^QS * D(f)`` with the single-mode amplification
    ``|D(f1)| = Q`` (docs 02 §6, 05 §1). The resonance ``f1`` is marked; this is
    the band-shape the inverse chain must equalise.

    Parameters
    ----------
    model : CantileverModel
        Derived mechanical model of the current composition.
    f_min_hz, f_max_hz : float
        Frequency-axis span, Hz (the composition's target band).
    n : int, optional
        Number of grid points.

    Returns
    -------
    matplotlib.figure.Figure
        One log-log subplot of ``|H_lat|`` (m per m/s^2) versus frequency.
    """
    freqs: FloatArray = np.logspace(np.log10(max(f_min_hz, 1e-3)), np.log10(f_max_hz), n)
    mag = np.abs(model.h_lat(freqs))
    fig = Figure(figsize=(7.5, 4.2), constrained_layout=True)
    ax = fig.subplots()
    ax.plot(freqs, mag, linewidth=1.4, label=r"$|H_{\mathrm{lat}}(f)|$")
    ax.axvline(
        model.f1_hz,
        linestyle="--",
        color="tab:red",
        linewidth=1.0,
        label=rf"$f_1$ = {model.f1_hz * 1e-3:.2f} kHz (Q = {model.q_total:.0f})",
    )
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"$|H_{\mathrm{lat}}|$ [m / (m/s$^2$)]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Lateral transfer function (current cantilever)")
    return fig


def plot_reflector_eta_vs_dx(
    model: ReflectorModel,
    *,
    span_m: float,
    eta0: float,
    bias_offset_m: float = 0.0,
    shape: str = "",
    n: int = 401,
) -> Figure:
    """Plot the shape-agnostic coupling ``eta(dx)`` with the working point.

    Works for any registered reflector model (cylinder/sphere/plane/wedge): it
    only calls :meth:`ReflectorModel.eta`. The plane is flat (``eta`` constant in
    ``dx``), the curved shapes show the Gaussian dip and the marked working point
    ``eta0`` set by the static de-centering (doc 03 §5).

    Parameters
    ----------
    model : ReflectorModel
        Coupling model of the current reflector composition.
    span_m : float
        Half-width of the symmetric ``dx`` grid, m.
    eta0 : float
        Working-point coupling ``eta0`` (marked at ``dx = 0``).
    bias_offset_m : float, optional
        Static de-centering Delta x0, m (annotated when non-zero).
    shape : str, optional
        Reflector shape name (for the title).
    n : int, optional
        Number of grid points.

    Returns
    -------
    matplotlib.figure.Figure
        One subplot of ``eta`` versus ``dx`` (um).
    """
    dx: FloatArray = np.linspace(-span_m, span_m, n)
    eta_raw = np.asarray(model.eta(dx=dx), dtype=np.float64)
    # Flat shapes (plane/wedge) have no displacement coupling and return a
    # scalar / size-1 array; broadcast it to the grid so the curve is drawable.
    eta = np.broadcast_to(eta_raw, dx.shape)
    fig = Figure(figsize=(7.5, 4.2), constrained_layout=True)
    ax = fig.subplots()
    ax.plot(dx * 1e6, eta, linewidth=1.4, label=r"$\eta(\Delta x)$")
    ax.plot(
        [0.0],
        [eta0],
        "o",
        color="tab:red",
        label=rf"working point $\eta_0$ = {eta0:.3f}",
    )
    if bias_offset_m > 0.0:
        ax.axvline(0.0, linestyle=":", color="tab:gray", linewidth=0.8)
    ax.set_xlabel(r"$\Delta x$ [$\mu$m]")
    ax.set_ylabel(r"$\eta$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    title = "Coupling vs target-axis displacement"
    if shape:
        title = f"{title} ({shape})"
    fig.suptitle(title)
    return fig
