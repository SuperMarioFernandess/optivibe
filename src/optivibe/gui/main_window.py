"""The OptiVibe main window: pick a scenario, run it, see the result.

The window is deliberately thin (architecture 09 §9): it owns no physics and no
threads of its own. It delegates execution to a
:class:`~optivibe.gui.controllers.run_controller.RunController` (which runs the
pipeline off the UI thread) and renders the outcome in a
:class:`~optivibe.gui.widgets.live_plot.LivePlotWidget`. While a run is active the
Run button is disabled so the UI stays responsive and re-entry is impossible.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from optivibe.core.logging import get_logger
from optivibe.gui.controllers.run_controller import RunController
from optivibe.gui.widgets.live_plot import LivePlotWidget
from optivibe.pipeline import RunArtifacts

logger = get_logger(__name__)

__all__ = ["MainWindow"]

_DEFAULT_SCENARIO = "examples/hello.yaml"


class MainWindow(QMainWindow):
    """Top-level window with a scenario selector, a Run button and a plot.

    Parameters
    ----------
    default_scenario : pathlib.Path or str or None, optional
        Pre-filled scenario path; defaults to ``examples/hello.yaml`` (relative
        to the working directory).
    """

    def __init__(self, default_scenario: Path | str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("OptiVibe - sensor digital twin (S0)")
        self.resize(900, 600)

        self._controller = RunController(self)
        self._controller.finished.connect(self._on_finished)
        self._controller.failed.connect(self._on_failed)

        self._scenario_edit = QLineEdit(str(default_scenario or _DEFAULT_SCENARIO))
        self._run_button = QPushButton("Run")
        self._run_button.clicked.connect(self._on_run_clicked)
        self._plot = LivePlotWidget()

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Scenario:"))
        controls.addWidget(self._scenario_edit, stretch=1)
        controls.addWidget(self._run_button)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self._plot, stretch=1)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.statusBar().showMessage("Ready.")

    def _on_run_clicked(self) -> None:
        """Start a run for the scenario currently in the text field."""
        if self._controller.is_running():
            return
        scenario_path = Path(self._scenario_edit.text().strip())
        self._run_button.setEnabled(False)
        self.statusBar().showMessage(f"Running {scenario_path} ...")
        try:
            self._controller.start(scenario_path)
        except RuntimeError as exc:  # pragma: no cover - guarded by is_running
            self._run_button.setEnabled(True)
            self.statusBar().showMessage(f"Could not start: {exc}")

    def _on_finished(self, artifacts: object) -> None:
        """Render the result and re-enable the Run button."""
        self._run_button.setEnabled(True)
        if not isinstance(artifacts, RunArtifacts):  # pragma: no cover - defensive
            self.statusBar().showMessage("Run finished with unexpected payload.")
            return
        result = artifacts.result
        self._plot.show_result(result)
        dominant = ", ".join(f"{f:.2f}" for f in result.dominant_freqs_hz) or "-"
        self.statusBar().showMessage(
            f"Done: variant {artifacts.variant.name}, "
            f"{result.n_samples} samples, dominant {dominant} Hz."
        )

    def _on_failed(self, message: str) -> None:
        """Report a failed run in the status bar and re-enable the button."""
        self._run_button.setEnabled(True)
        self.statusBar().showMessage(f"Run failed: {message}")

    # Convenience accessors for tests (read-only views of the widgets).
    @property
    def run_button(self) -> QPushButton:
        """The Run button (exposed for tests)."""
        return self._run_button

    @property
    def plot(self) -> LivePlotWidget:
        """The live-plot widget (exposed for tests)."""
        return self._plot
