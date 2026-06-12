"""Pure figure producers for the optical coupling model (no Qt, no pyplot).

Per SW-09 / 09 §9 this module builds :class:`matplotlib.figure.Figure` objects
directly (Agg-compatible, headless-safe). Three views of a
:class:`~optivibe.optics.cylinder.CylinderOpticsModel` are provided — the
coupling curve eta(dx) with the working point and its tangent, the cross-axis
slice eta(dy) through the mechanical tilt coupling, and an optional
(dx, dy) map.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from optivibe.core.types import FloatArray
from optivibe.optics.cylinder import CylinderOpticsModel

__all__ = ["plot_eta_map", "plot_eta_vs_dx", "plot_eta_vs_dy"]


def _dx_grid(model: CylinderOpticsModel, n: int = 401) -> FloatArray:
    """Symmetric dx grid covering the peak and the working point (+/- 3 sigma)."""
    span = 3.0 * model.sigma_m + model.bias_m
    out: FloatArray = np.linspace(-span, span, n)
    return out


def plot_eta_vs_dx(model: CylinderOpticsModel) -> Figure:
    """Plot eta(dx) with the working point eta0 and its tangent (doc 03 §5).

    Parameters
    ----------
    model : CylinderOpticsModel
        Coupling model of one variant.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with one subplot (dx in um on the abscissa).
    """
    dx = _dx_grid(model)
    eta = model.eta(dx=dx)
    eta0 = model.eta_working_point()
    slope = model.slope_dx()
    fig = Figure(figsize=(8.0, 4.5), constrained_layout=True)
    ax = fig.subplots()
    ax.plot(dx * 1e6, eta, linewidth=1.2, label=r"$\eta(\Delta x)$")
    ax.plot([0.0], [eta0], "o", color="tab:red", label=rf"working point $\eta_0$={eta0:.3f}")
    tangent_dx = np.linspace(-model.sigma_m, model.sigma_m, 2)
    ax.plot(
        tangent_dx * 1e6,
        eta0 + slope * tangent_dx,
        "--",
        color="tab:red",
        linewidth=1.0,
        label=rf"tangent $\partial\eta/\partial\Delta x$={slope:.3e} 1/m",
    )
    ax.set_xlabel(r"$\Delta x$ [$\mu$m]")
    ax.set_ylabel(r"$\eta$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Cylinder coupling vs target-axis displacement")
    return fig


def plot_eta_vs_dy(
    model: CylinderOpticsModel, length_m: float, tilt_coupling_per_l: float
) -> Figure:
    """Plot eta(dy) through the mechanical tilt coupling (cross-axis slice).

    Pure dy leaves eta exactly unchanged (cylinder symmetry, doc 03 §4); the
    quadratic residual shown here comes from ``theta_x = 1.377 dy / L``
    (doc 04 §5).

    Parameters
    ----------
    model : CylinderOpticsModel
        Coupling model of one variant.
    length_m : float
        Cantilever length L, m.
    tilt_coupling_per_l : float
        Dimensionless coupling ``theta L / delta`` (1.377, doc 01).

    Returns
    -------
    matplotlib.figure.Figure
        Figure with one subplot (dy in um on the abscissa).
    """
    dy = _dx_grid(model)
    theta_x = tilt_coupling_per_l / length_m * dy
    eta = model.eta(dy=dy, theta_x=theta_x)
    fig = Figure(figsize=(8.0, 4.5), constrained_layout=True)
    ax = fig.subplots()
    ax.plot(dy * 1e6, eta, linewidth=1.2)
    ax.axhline(model.eta_working_point(), linestyle="--", color="tab:red", linewidth=0.8)
    ax.set_xlabel(r"$\Delta y$ [$\mu$m]")
    ax.set_ylabel(r"$\eta$")
    ax.grid(True, alpha=0.3)
    fig.suptitle(rf"Cross-axis slice $\eta(\Delta y)$ via $\theta_x$ (L = {length_m * 1e3:.2f} mm)")
    return fig


def plot_eta_map(
    model: CylinderOpticsModel, length_m: float, tilt_coupling_per_l: float, n: int = 121
) -> Figure:
    """Plot the (dx, dy) coupling map with the mechanical tilt couplings applied.

    Parameters
    ----------
    model : CylinderOpticsModel
        Coupling model of one variant.
    length_m : float
        Cantilever length L, m.
    tilt_coupling_per_l : float
        Dimensionless coupling ``theta L / delta`` (1.377, doc 01).
    n : int, optional
        Grid points per axis.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with one pcolormesh subplot.
    """
    grid = _dx_grid(model, n)
    dx_2d, dy_2d = np.meshgrid(grid, grid)
    tilt = tilt_coupling_per_l / length_m
    eta = model.eta(
        dx=dx_2d.ravel(),
        dy=dy_2d.ravel(),
        theta_x=tilt * dy_2d.ravel(),
        theta_y=tilt * dx_2d.ravel(),
    ).reshape(dx_2d.shape)
    fig = Figure(figsize=(6.5, 5.5), constrained_layout=True)
    ax = fig.subplots()
    mesh = ax.pcolormesh(grid * 1e6, grid * 1e6, eta, shading="auto")
    ax.plot([0.0], [0.0], "o", color="tab:red", markersize=4)
    ax.set_xlabel(r"$\Delta x$ [$\mu$m]")
    ax.set_ylabel(r"$\Delta y$ [$\mu$m]")
    fig.colorbar(mesh, ax=ax, label=r"$\eta$")
    fig.suptitle("Coupling map with mechanical tilt couplings")
    return fig
