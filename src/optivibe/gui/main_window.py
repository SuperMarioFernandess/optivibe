"""The OptiVibe desktop window: build a scenario, run it, see it live (task S7).

A thin shell over the core (09 §9): the left control panel assembles a scenario
/ analysis *payload*, the action buttons hand a Qt-free
:class:`~optivibe.gui.workers.jobs.Job` to a
:class:`~optivibe.gui.controllers.job_controller.JobController` (which runs it off
the UI thread), and the tabs render the outcome. The window owns no physics and
no threads of its own; while a job runs the action buttons are disabled and a
busy indicator shows, so the UI stays responsive and re-entry is impossible. The
result type selects the tab (``RunArtifacts`` -> Live, ``ReportBundle`` ->
Report, ``SweepResult`` -> Sweeps, ``MonteCarloResult`` -> Monte-Carlo).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from optivibe.analysis import (
    MonteCarloResult,
    SweepResult,
    save_monte_carlo_npz,
    save_sweep_npz,
)
from optivibe.core.config.loader import load_constants
from optivibe.core.config.models import ScenarioConfig
from optivibe.core.config.subsystems import SystemConfig
from optivibe.core.logging import get_logger
from optivibe.gui.controllers.job_controller import JobController
from optivibe.gui.controllers.scenario_builder import (
    build_monte_carlo_spec,
    build_scenario_config,
    build_sweep_spec,
)
from optivibe.gui.controllers.system_builder import build_system_config
from optivibe.gui.widgets import (
    ControlPanel,
    LiveView,
    MonteCarloPanel,
    PhysicsTab,
    ReportPanel,
    SweepPanel,
)
from optivibe.gui.workers.jobs import (
    Job,
    MonteCarloJob,
    ReportBundle,
    ReportJob,
    ScenarioJob,
    SweepJob,
)
from optivibe.pipeline import RunArtifacts
from optivibe.viz.analysis import plot_nea_budget, plot_truth_vs_recovery_avx

logger = get_logger(__name__)

__all__ = ["MainWindow"]


class MainWindow(QMainWindow):
    """Top-level window: controls, live plots, report / sweep / Monte-Carlo tabs.

    Parameters
    ----------
    config_dir : pathlib.Path or None, optional
        Override for the ``configs/`` directory (variant presets).
    parent : QWidget or None, optional
        Parent widget.
    """

    def __init__(self, config_dir: Path | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("OptiVibe - fiber-optic vibration sensor digital twin")
        self.resize(1280, 820)
        self._config_dir = config_dir
        self._beta1_l = load_constants().universal.beta1_l
        self._last_result: object | None = None

        self._controller = JobController(self)
        self._controller.progress.connect(self._on_progress)
        self._controller.finished.connect(self._on_finished)
        self._controller.failed.connect(self._on_failed)
        self._controller.cancelled.connect(self._on_cancelled)

        self._panel = ControlPanel(config_dir=config_dir)
        self._live = LiveView()
        self._report = ReportPanel()
        self._sweep = SweepPanel()
        self._monte = MonteCarloPanel()
        self._physics = PhysicsTab(self._panel, config_dir=config_dir)
        self._sweep.run_requested.connect(self._on_sweep)
        self._monte.run_requested.connect(self._on_monte_carlo)
        self._physics.nea_requested.connect(self._on_report)

        self._run_button = QPushButton("Run")
        self._report_button = QPushButton("Report")
        self._cancel_button = QPushButton("Cancel")
        self._export_button = QPushButton("Export...")
        self._run_button.clicked.connect(self._on_run)
        self._report_button.clicked.connect(self._on_report)
        self._cancel_button.clicked.connect(self._controller.cancel)
        self._export_button.clicked.connect(self._on_export_clicked)
        self._cancel_button.setEnabled(False)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._live, "Live")
        self._tabs.addTab(self._report, "Report")
        self._tabs.addTab(self._sweep, "Sweeps")
        self._tabs.addTab(self._monte, "Monte-Carlo")
        self._tabs.addTab(self._physics, "Physics")
        self._tabs.currentChanged.connect(self._on_tab_changed)

        self.setCentralWidget(self._build_central())
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(160)
        self._progress.hide()
        self.statusBar().addPermanentWidget(self._progress)
        self.statusBar().showMessage("Ready. Pick a variant and excitation, then Run.")

    def _build_central(self) -> QWidget:
        """Assemble the control column, action bar and tab area."""
        actions = QHBoxLayout()
        for button in (
            self._run_button,
            self._report_button,
            self._cancel_button,
            self._export_button,
        ):
            actions.addWidget(button)
        actions.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._panel)
        scroll.setMinimumWidth(330)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addLayout(actions)
        right_layout.addWidget(self._tabs, stretch=1)

        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        return splitter

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #
    def _on_run(self) -> None:
        """Run the current scenario and show it on the Live tab."""
        scenario = self._build_scenario()
        if scenario is None:
            return
        system = self._build_system()
        if system is None:
            return
        self._start(
            ScenarioJob(scenario=scenario, config_dir=self._config_dir, system=system), "run"
        )

    def _on_report(self) -> None:
        """Run the scenario and build the analysis report."""
        scenario = self._build_scenario()
        if scenario is None:
            return
        system = self._build_system()
        if system is None:
            return
        self._start(
            ReportJob(scenario=scenario, config_dir=self._config_dir, system=system), "report"
        )

    def _on_sweep(self) -> None:
        """Run a parameter sweep from the Sweep panel."""
        try:
            spec = build_sweep_spec(self._sweep.payload())
        except (ValueError, TypeError) as exc:
            self.statusBar().showMessage(f"Invalid sweep: {exc}")
            return
        self._start(SweepJob(spec=spec), "sweep")

    def _on_monte_carlo(self) -> None:
        """Run a tolerance Monte-Carlo from the Monte-Carlo panel."""
        try:
            spec = build_monte_carlo_spec(self._monte.payload())
        except (ValueError, TypeError) as exc:
            self.statusBar().showMessage(f"Invalid Monte-Carlo: {exc}")
            return
        self._start(MonteCarloJob(spec=spec), "monte-carlo")

    def _build_scenario(self) -> ScenarioConfig | None:
        """Validate the control-panel payload into a scenario (or report error)."""
        try:
            return build_scenario_config(self._panel.scenario_payload())
        except (ValueError, TypeError) as exc:
            self.statusBar().showMessage(f"Invalid scenario: {str(exc).splitlines()[0]}")
            return None

    def _build_system(self) -> SystemConfig | None:
        """Validate the edited composition payload (or report the error).

        The resolved variant is produced on the worker thread (SW-06); here we
        only assemble and validate the frozen :class:`SystemConfig` so a bad
        edit is reported before a job starts.
        """
        try:
            return build_system_config(self._panel.system_payload())
        except (ValueError, TypeError) as exc:
            self.statusBar().showMessage(f"Invalid composition: {str(exc).splitlines()[0]}")
            return None

    def _start(self, job: Job, label: str) -> None:
        """Start a job off the UI thread and lock the action buttons."""
        if self._controller.is_running():
            return
        self._set_running(True)
        self.statusBar().showMessage(f"Running {label} ...")
        try:
            self._controller.start(job)
        except RuntimeError as exc:  # pragma: no cover - guarded by is_running
            self._set_running(False)
            self.statusBar().showMessage(f"Could not start: {exc}")

    # ------------------------------------------------------------------ #
    # Controller signals
    # ------------------------------------------------------------------ #
    def _on_progress(self, message: str) -> None:
        """Show a coarse progress message."""
        self.statusBar().showMessage(f"... {message}")

    def _on_finished(self, result: object) -> None:
        """Route a finished job result to the matching tab."""
        self._set_running(False)
        self._last_result = result
        if isinstance(result, RunArtifacts):
            self._live.show_artifacts(result, self._beta1_l)
            self._tabs.setCurrentWidget(self._live)
            self._announce_run(result)
        elif isinstance(result, ReportBundle):
            self._report.show_bundle(result)
            self._live.show_artifacts(result.artifacts, self._beta1_l)
            self._live.show_nea(result.nea)
            if result.nea is not None:
                self._physics.set_nea_figure(plot_nea_budget(result.nea))
            self._tabs.setCurrentWidget(self._report)
            self.statusBar().showMessage(
                f"Report ready: amplitude ratio {result.budget.amplitude_ratio:.4f}, "
                f"recovery rel err {result.budget.rms_error_rel:.2e}."
            )
        elif isinstance(result, SweepResult):
            self._sweep.show_result(result)
            self._tabs.setCurrentWidget(self._sweep)
            self.statusBar().showMessage(
                f"Sweep '{result.name}' ({result.mode}) over {result.parameter}: "
                f"{len(result.axis_labels)} points."
            )
        elif isinstance(result, MonteCarloResult):
            self._monte.show_result(result)
            self._tabs.setCurrentWidget(self._monte)
            self.statusBar().showMessage(f"Monte-Carlo '{result.name}': {result.n_draws} draws.")
        else:  # pragma: no cover - defensive
            self.statusBar().showMessage("Finished with an unrecognised result.")

    def _announce_run(self, artifacts: RunArtifacts) -> None:
        """Status line for a finished scenario run."""
        result = artifacts.result
        dominant = ", ".join(f"{f:.2f}" for f in result.dominant_freqs_hz) or "-"
        self.statusBar().showMessage(
            f"Done: variant {artifacts.variant.name}, {result.n_samples} samples, "
            f"dominant {dominant} Hz."
        )

    def _on_tab_changed(self, index: int) -> None:
        """Rebuild the light physics curves when the Physics tab is shown."""
        if self._tabs.widget(index) is self._physics:
            self._physics.refresh_light()

    def _on_failed(self, message: str) -> None:
        """Report a failed job."""
        self._set_running(False)
        self.statusBar().showMessage(f"Failed: {message}")

    def _on_cancelled(self) -> None:
        """Report a cancelled job (its result was dropped)."""
        self._set_running(False)
        self.statusBar().showMessage("Cancelled.")

    def _set_running(self, running: bool) -> None:
        """Lock/unlock the action buttons and the busy indicator."""
        self._run_button.setEnabled(not running)
        self._report_button.setEnabled(not running)
        self._export_button.setEnabled(not running)
        self._cancel_button.setEnabled(running)
        self._progress.setRange(0, 0 if running else 1)
        self._progress.setVisible(running)

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def _on_export_clicked(self) -> None:  # pragma: no cover - dialog
        """Pick a directory and export the latest result into it."""
        if self._last_result is None:
            self.statusBar().showMessage("Nothing to export yet.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Export to directory")
        if directory:
            saved = self.export_to(Path(directory))
            self.statusBar().showMessage(f"Exported {len(saved)} file(s) to {directory}.")

    def export_to(self, directory: Path) -> list[Path]:
        """Export the latest result (figures + ``.npz``) into ``directory``.

        Parameters
        ----------
        directory : pathlib.Path
            Target directory (created if missing).

        Returns
        -------
        list of pathlib.Path
            Paths written.
        """
        directory.mkdir(parents=True, exist_ok=True)
        result = self._last_result
        written: list[Path] = []
        if isinstance(result, RunArtifacts):
            written.append(self._save_run_npz(result, directory))
        elif isinstance(result, ReportBundle):
            truth = directory / "truth_vs_recovery.png"
            plot_truth_vs_recovery_avx(
                result.artifacts.forward.excitation.a_x, result.artifacts.result
            ).savefig(truth, dpi=120)
            written.append(truth)
            if result.nea is not None:
                nea = directory / "nea_budget.png"
                plot_nea_budget(result.nea).savefig(nea, dpi=120)
                written.append(nea)
            written.append(self._save_run_npz(result.artifacts, directory))
        elif isinstance(result, SweepResult):
            written.append(save_sweep_npz(result, directory / result.name))
            from optivibe.viz.analysis import plot_sweep

            fig = directory / f"{result.name}.png"
            plot_sweep(result).savefig(fig, dpi=120)
            written.append(fig)
        elif isinstance(result, MonteCarloResult):
            written.append(save_monte_carlo_npz(result, directory / result.name))
            from optivibe.viz.analysis import plot_monte_carlo

            fig = directory / f"{result.name}.png"
            plot_monte_carlo(result).savefig(fig, dpi=120)
            written.append(fig)
        return written

    @staticmethod
    def _save_run_npz(artifacts: RunArtifacts, directory: Path) -> Path:
        """Save a run's input + recovered signals as a ``.npz``."""
        path = directory / "run_result.npz"
        result = artifacts.result
        np.savez(
            path,
            a_input=artifacts.forward.excitation.a_x,
            a_recovered=result.a,
            v_recovered=result.v,
            x_recovered=result.x,
            fs=result.fs,
        )
        return path

    # ------------------------------------------------------------------ #
    # Lifecycle / test accessors
    # ------------------------------------------------------------------ #
    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop the animation and cancel any running job on close."""
        self._live.stop()
        self._controller.cancel()
        super().closeEvent(event)

    @property
    def controller(self) -> JobController:
        """The job controller (exposed for tests)."""
        return self._controller

    @property
    def control_panel(self) -> ControlPanel:
        """The control panel (exposed for tests)."""
        return self._panel

    @property
    def run_button(self) -> QPushButton:
        """The Run button (exposed for tests; S0 contract)."""
        return self._run_button

    @property
    def plot(self) -> LiveView:
        """The live view (exposed for tests; S0 ``plot`` accessor)."""
        return self._live
