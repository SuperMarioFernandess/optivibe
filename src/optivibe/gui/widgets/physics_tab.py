"""Physics / reference tab: design curves for the current composition (S7-mod §5).

A read-only *help* surface that turns the resolved composition into the key
design dependencies, so a user can see where their edits land before running:

* **Light, auto-recomputed curves** (pure, cheap -- a few hundred point
  evaluations, safe to build inline): ``f1(L)`` with the working length, the
  lateral transfer ``|H_lat(f)|`` and the shape-agnostic coupling ``eta(dx)``.
  Built by :mod:`optivibe.viz.physics` from the resolved variant + models.
* **Heavy curve via the worker** (task S7-mod §5): the measured ``NEA(f)``
  budget with its shot/RIN/Johnson split is produced by the existing *Report*
  run (forward + analysis off the UI thread, SW-06) and pushed back here; the
  tab only emits :attr:`nea_requested` and embeds the returned figure.
* **Reference notes**: short descriptions of the mechanics, the reference-arm
  options, the inverse/DSP chain, the sensitivity models and the integrator,
  with knowledge-base pointers; the sensor *family* sweep lives on the Sweeps
  tab (an explicit, documented deferral rather than a duplicated heavy compute).

No physics here: the curves come from the core/viz layers; this tab embeds them.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from optivibe.core.config.loader import default_config_dir, load_constants
from optivibe.core.logging import get_logger
from optivibe.gui.controllers.system_builder import build_system_config, resolve_system_variant
from optivibe.gui.widgets.control_panel import ControlPanel
from optivibe.gui.widgets.mpl_canvas import MplFigureView
from optivibe.mechanics.cantilever import CantileverModel
from optivibe.optics.reflector import build_reflector_model
from optivibe.viz.physics import (
    plot_first_mode_vs_length,
    plot_lateral_transfer,
    plot_reflector_eta_vs_dx,
)

logger = get_logger(__name__)

__all__ = ["PhysicsTab"]

_REFERENCE_NOTES = """\
<h3>Reference models for the current composition</h3>
<p><b>Mechanics (docs 02 / 05).</b> The fiber cantilever is a clamped-free beam.
The first bending mode sets f1 ~ 1/L^2; the lateral transfer is
H_lat(f) = H_lat^QS * D(f) with single-mode amplification |D(f1)| = Q. Shorter L
raises f1 and widens the flat band but lowers the quasi-static compliance
(sensitivity) -- the core design trade shown by the f1(L) and |H_lat(f)| curves.</p>
<p><b>Reflector coupling (doc 03).</b> eta(dx) is the Gaussian overlap between the
returning beam and the fiber mode. The static de-centering Delta x0 sets the
working point eta0 on the slope; cylinder/sphere are curved (finite R_c), the
plane is flat (no displacement coupling) and the wedge adds an angular bias.</p>
<p><b>Detector reference arm (doc 07 §1.2).</b> "matched" balances the bright and
reference arms (common-mode RIN rejection limited by CMRR); "bright" leaves the
reference arm dark (no RIN cancellation, higher shot floor). This is the open
question O-SW-08.</p>
<p><b>Inverse / DSP (docs 05 / 11).</b> The standard inverse de-rotates D(f),
applies the calibrated optical sensitivity and integrates to v and x. The
sensitivity model -- "static" (plateau slope), "operating_point" (local slope at
eta0) or "nonlinear_curve" (full eta(dx) inversion) -- trades bias against
robustness. The integrator runs in "frequency" (omega-domain) or "time" form.</p>
<p><b>NEA(f) (docs 07 / 08).</b> The noise-equivalent acceleration density with
its shot / RIN / Johnson plateaus is a measured budget; press
<i>Compute NEA(f)</i> to run it through the worker (a Report run). The sensor
<b>family</b> sweep is on the <i>Sweeps</i> tab.</p>
"""


class PhysicsTab(QWidget):
    """Reference curves + notes for the current composition.

    Parameters
    ----------
    control_panel : ControlPanel
        Source of the current composition payload.
    config_dir : pathlib.Path or None, optional
        Configuration root (presets); defaults to the repository ``configs/``.
    parent : QWidget or None, optional
        Parent widget.
    """

    #: Emitted when the user asks for the measured NEA(f) (handled by the window
    #: by running a Report off the UI thread; the result returns via
    #: :meth:`set_nea_figure`).
    nea_requested = Signal()

    def __init__(
        self,
        control_panel: ControlPanel,
        config_dir: Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._panel = control_panel
        self._config_dir = config_dir or default_config_dir()
        self._constants = load_constants()

        self._f1 = MplFigureView("Press Refresh to build f1(L).")
        self._hlat = MplFigureView("Press Refresh to build |H_lat(f)|.")
        self._eta = MplFigureView("Press Refresh to build eta(dx).")
        self._nea = MplFigureView("Press Compute NEA(f) to run the budget.")

        self._refresh_button = QPushButton("Refresh from composition")
        self._nea_button = QPushButton("Compute NEA(f)")
        self._refresh_button.clicked.connect(self.refresh_light)
        self._nea_button.clicked.connect(self.nea_requested)

        self._notes = QTextEdit()
        self._notes.setReadOnly(True)
        self._notes.setHtml(_REFERENCE_NOTES)

        tabs = QTabWidget()
        tabs.addTab(self._curves_page(), "Design curves")
        tabs.addTab(self._notes, "Reference notes")

        actions = QHBoxLayout()
        actions.addWidget(self._refresh_button)
        actions.addWidget(self._nea_button)
        actions.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(actions)
        layout.addWidget(tabs, stretch=1)

    def _curves_page(self) -> QWidget:
        """Lay out the four figure slots in a 2x2 grid."""
        page = QWidget()
        grid = QGridLayout(page)
        grid.addWidget(self._f1, 0, 0)
        grid.addWidget(self._hlat, 0, 1)
        grid.addWidget(self._eta, 1, 0)
        grid.addWidget(self._nea, 1, 1)
        return page

    def refresh_light(self) -> bool:
        """Rebuild the light curves from the current composition.

        Returns
        -------
        bool
            ``True`` if the composition resolved and the curves were built;
            ``False`` if it was invalid (the caller may show the reason).
        """
        try:
            system = build_system_config(self._panel.system_payload())
            variant = resolve_system_variant(system, self._config_dir)
        except (ValueError, TypeError) as exc:
            logger.debug("physics refresh failed: %s", exc)
            return False

        self._f1.set_figure(plot_first_mode_vs_length(self._constants, variant.length_m))
        cantilever = CantileverModel.from_config(self._constants, variant)
        self._hlat.set_figure(
            plot_lateral_transfer(
                cantilever,
                f_min_hz=variant.band.f_min_hz,
                f_max_hz=variant.band.f_max_hz,
            )
        )
        model = build_reflector_model(variant)
        sigma = float(getattr(model, "sigma_m", 0.0))
        bias = float(getattr(model, "bias_m", variant.optics.bias_offset_m))
        if sigma > 0.0:
            span = 3.0 * sigma + bias
        else:
            span = max(6.0 * variant.optics.bias_offset_m, 8.0 * variant.optics.mode_field_radius_m)
        self._eta.set_figure(
            plot_reflector_eta_vs_dx(
                model,
                span_m=span,
                eta0=model.eta_working_point(),
                bias_offset_m=variant.optics.bias_offset_m,
                shape=variant.reflector.shape,
            )
        )
        return True

    def set_nea_figure(self, figure: Figure) -> None:
        """Embed a measured NEA(f) figure returned by the worker.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            A figure produced by :func:`optivibe.viz.analysis.plot_nea_budget`.
        """
        self._nea.set_figure(figure)
