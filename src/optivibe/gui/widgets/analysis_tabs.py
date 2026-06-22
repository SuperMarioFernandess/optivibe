"""Analysis tab widgets: Report, Sweep and Monte-Carlo (task S7 §4/§5).

The Report panel is display-only: it embeds the pure :mod:`optivibe.viz` figures
(truth-vs-recovery a/v/x, NEA budget, recovered-acceleration spectrogram) and the
error-budget text from a :class:`~optivibe.gui.workers.jobs.ReportBundle`. The
Sweep and Monte-Carlo panels add a few controls and a Run button; they only
*assemble a payload* (validated off-widget by the scenario builder) and *embed*
the result figure -- the heavy ``run_sweep`` / ``run_monte_carlo`` runs in the
worker, this code never computes (09 §9).
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from optivibe.analysis import MonteCarloResult, SweepResult
from optivibe.gui.widgets.mpl_canvas import MplFigureView
from optivibe.gui.workers.jobs import ReportBundle
from optivibe.viz.analysis import (
    plot_monte_carlo,
    plot_nea_budget,
    plot_sweep,
    plot_truth_vs_recovery_avx,
)
from optivibe.viz.dsp import plot_spectrogram

__all__ = ["MonteCarloPanel", "ReportPanel", "SweepPanel"]

_VARIANTS = ("A", "B", "C", "D")
_DESIGN_PARAMS = ("length_m", "radius_of_curvature_m", "power_w", "bias_offset_m", "full_scale_g")
_RESPONSE_PARAMS = ("amplitude_g", "frequency_hz")
_MC_DEFAULTS: dict[str, dict[str, Any]] = {
    "q_total": {"dist": "lognormal", "rel_sigma": 0.30},
    "radius_of_curvature_m": {"dist": "normal", "rel_sigma": 0.05},
    "gap_m": {"dist": "normal", "abs_sigma": 5.0e-6},
    "bias_offset_m": {"dist": "normal", "abs_sigma": 0.1e-6},
    "epsilon_x": {"dist": "normal", "abs_sigma": 0.1e-6},
}


class ReportPanel(QWidget):
    """Display the truth-vs-recovery + NEA budgets and figures (display-only)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumHeight(220)
        self._truth = MplFigureView("Run 'Report' to build the truth-vs-recovery figure.")
        self._nea = MplFigureView("NEA budget (needs the photodiode detector).")
        self._spectrogram = MplFigureView("Recovered-acceleration spectrogram.")
        tabs = QTabWidget()
        tabs.addTab(self._truth, "Truth vs recovery")
        tabs.addTab(self._nea, "NEA budget")
        tabs.addTab(self._spectrogram, "Spectrogram")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Error budget"))
        layout.addWidget(self._text)
        layout.addWidget(tabs, stretch=1)

    def show_bundle(self, bundle: ReportBundle) -> None:
        """Render a :class:`ReportBundle` (budget text + figures)."""
        self._text.setPlainText(bundle.budget.summary_text())
        a_true = bundle.artifacts.forward.excitation.a_x
        self._truth.set_figure(plot_truth_vs_recovery_avx(a_true, bundle.artifacts.result))
        if bundle.nea is not None:
            self._nea.set_figure(plot_nea_budget(bundle.nea))
        self._spectrogram.set_figure(
            plot_spectrogram(bundle.artifacts.result.a, bundle.artifacts.result.fs)
        )


class SweepPanel(QWidget):
    """Parameter-sweep controls + result figure (task S7 §5)."""

    #: Emitted when the Run button is pressed.
    run_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._variant = QComboBox()
        self._variant.addItems(_VARIANTS)
        self._variant.setCurrentText("B")
        self._mode = QComboBox()
        self._mode.addItems(("design", "response"))
        self._parameter = QComboBox()
        self._start = self._dspin(1.0e-3, 6)
        self._stop = self._dspin(4.0e-3, 6)
        self._num = QSpinBox()
        self._num.setRange(2, 200)
        self._num.setValue(16)
        self._log = QCheckBox("log spacing")
        self._run = QPushButton("Run sweep")
        self._run.clicked.connect(self.run_requested)
        self._canvas = MplFigureView("Run a sweep to see NEA / response vs the parameter.")

        controls = QGroupBox("Sweep")
        form = QFormLayout(controls)
        form.addRow("Variant", self._variant)
        form.addRow("Mode", self._mode)
        form.addRow("Parameter", self._parameter)
        grid_row = QHBoxLayout()
        for label, widget in (("start", self._start), ("stop", self._stop), ("num", self._num)):
            grid_row.addWidget(QLabel(label))
            grid_row.addWidget(widget)
        grid_row.addWidget(self._log)
        holder = QWidget()
        holder.setLayout(grid_row)
        form.addRow("Grid", holder)
        form.addRow(self._run)

        layout = QVBoxLayout(self)
        layout.addWidget(controls)
        layout.addWidget(self._canvas, stretch=1)

        self._mode.currentTextChanged.connect(self._on_mode_changed)
        self._on_mode_changed(self._mode.currentText())

    @staticmethod
    def _dspin(value: float, decimals: int) -> QDoubleSpinBox:
        """Build a numeric entry spin box (kept simple)."""
        box = QDoubleSpinBox()
        box.setDecimals(decimals)
        box.setRange(1.0e-9, 1.0e9)
        box.setValue(value)
        return box

    def _on_mode_changed(self, mode: str) -> None:
        """Repopulate the parameter list and default grid for the mode."""
        self._parameter.clear()
        if mode == "design":
            self._parameter.addItems(_DESIGN_PARAMS)
            self._start.setValue(1.0e-3)
            self._stop.setValue(4.0e-3)
            self._log.setChecked(False)
        else:
            self._parameter.addItems(_RESPONSE_PARAMS)
            self._start.setValue(0.1)
            self._stop.setValue(120.0)
            self._log.setChecked(True)

    def payload(self) -> dict[str, Any]:
        """Return the sweep spec payload for ``build_sweep_spec``."""
        parameter = self._parameter.currentText()
        return {
            "kind": "sweep",
            "name": f"sweep_{parameter}",
            "mode": self._mode.currentText(),
            "variant": self._variant.currentText(),
            "parameter": parameter,
            "grid": {
                "start": self._start.value(),
                "stop": self._stop.value(),
                "num": self._num.value(),
                "log": self._log.isChecked(),
            },
        }

    def show_result(self, result: SweepResult) -> None:
        """Embed the sweep figure for ``result``."""
        self._canvas.set_figure(plot_sweep(result))


class MonteCarloPanel(QWidget):
    """Tolerance Monte-Carlo controls + result figure (task S7 §5)."""

    #: Emitted when the Run button is pressed.
    run_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._variant = QComboBox()
        self._variant.addItems(_VARIANTS)
        self._variant.setCurrentText("B")
        self._n_draws = QSpinBox()
        self._n_draws.setRange(2, 5000)
        self._n_draws.setValue(64)
        self._cross_axis = QCheckBox("estimate cross-axis (slower)")
        self._tolerances = {key: QCheckBox(key) for key in _MC_DEFAULTS}
        for box in self._tolerances.values():
            box.setChecked(True)
        self._run = QPushButton("Run Monte-Carlo")
        self._run.clicked.connect(self.run_requested)
        self._canvas = MplFigureView("Run a Monte-Carlo to see the metric distribution.")

        controls = QGroupBox("Monte-Carlo")
        form = QFormLayout(controls)
        form.addRow("Variant", self._variant)
        form.addRow("Draws", self._n_draws)
        form.addRow(self._cross_axis)
        tol_box = QVBoxLayout()
        for box in self._tolerances.values():
            tol_box.addWidget(box)
        holder = QWidget()
        holder.setLayout(tol_box)
        form.addRow("Tolerances", holder)
        form.addRow(self._run)

        layout = QVBoxLayout(self)
        layout.addWidget(controls)
        layout.addWidget(self._canvas, stretch=1)

    def payload(self) -> dict[str, Any]:
        """Return the Monte-Carlo spec payload for ``build_monte_carlo_spec``."""
        tolerances = {
            key: dict(_MC_DEFAULTS[key]) for key, box in self._tolerances.items() if box.isChecked()
        }
        return {
            "kind": "montecarlo",
            "name": "montecarlo_gui",
            "variant": self._variant.currentText(),
            "n_draws": self._n_draws.value(),
            "cross_axis": self._cross_axis.isChecked(),
            "tolerances": tolerances,
        }

    def show_result(self, result: MonteCarloResult) -> None:
        """Embed the Monte-Carlo figure for ``result``."""
        self._canvas.set_figure(plot_monte_carlo(result))
