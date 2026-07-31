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

import contextlib
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt, QUrl
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWhatsThis,
    QWidget,
)

from optivibe.analysis import (
    MonteCarloResult,
    SweepResult,
    predict_expected_peaks,
    save_monte_carlo_npz,
    save_sweep_npz,
)
from optivibe.analysis.instrument import load_analyze_spec
from optivibe.core.config.loader import default_config_dir, load_constants
from optivibe.core.config.models import ScenarioConfig
from optivibe.core.config.subsystems import SystemConfig
from optivibe.core.logging import get_logger
from optivibe.gui.controllers.job_controller import JobController
from optivibe.gui.controllers.scenario_builder import (
    build_monte_carlo_spec,
    build_scenario_config,
    build_sweep_spec,
)
from optivibe.gui.controllers.stream_controller import StreamController
from optivibe.gui.controllers.system_builder import build_system_config, resolve_system_variant
from optivibe.gui.i18n import (
    LANGUAGES,
    current_language,
    language_bus,
    set_language,
    t,
    tr,
)
from optivibe.gui.theme import THEMES, apply_theme
from optivibe.gui.widgets import (
    ControlPanel,
    LiveView,
    MonteCarloPanel,
    PhysicsTab,
    ReportPanel,
    SweepPanel,
)
from optivibe.gui.widgets.live_controls import SOURCE_RECORD
from optivibe.gui.widgets.preferences_dialog import PreferencesDialog
from optivibe.gui.workers.jobs import (
    Job,
    MonteCarloJob,
    ReportBundle,
    ReportJob,
    ScenarioJob,
    SweepJob,
)
from optivibe.gui.workers.stream import (
    RecordSource,
    ScenarioSource,
    StreamConfig,
    StreamFrame,
    StreamSetup,
    StreamSource,
)
from optivibe.mechanics.cantilever import first_mode_hz
from optivibe.pipeline import RunArtifacts
from optivibe.viz.analysis import plot_nea_budget, plot_truth_vs_recovery_avx

logger = get_logger(__name__)

__all__ = ["MainWindow"]

#: QSettings identity (organization / application).
_SETTINGS_ORG = "OptiVibe"
_SETTINGS_APP = "OptiVibe"

#: About-box text (English msgid; translated via ``t`` at display time).
_ABOUT_TEXT = (
    "OptiVibe -- digital twin of a fiber-optic vibration sensor.\n"
    "Desktop shell over a Qt-free core; all physics lives in the core."
)


class _QtLogHandler(logging.Handler):
    """A logging handler that appends records to the log-dock text view."""

    def __init__(self, view: QPlainTextEdit) -> None:
        super().__init__()
        self._view = view

    def emit(self, record: logging.LogRecord) -> None:
        """Append a formatted record (called on the logging thread's caller)."""
        with contextlib.suppress(RuntimeError):  # view already destroyed
            self._view.appendPlainText(self.format(record))


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
        self._settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        # Apply the saved language BEFORE any widget is built, so the whole tree
        # is constructed in the right language (SW-65).
        saved_language = str(self._settings.value("language", current_language()))
        if saved_language in LANGUAGES:
            set_language(saved_language)
        self._theme = str(self._settings.value("theme", "light"))
        self._restore_geometry = self._settings.value("restore_geometry", True, type=bool)
        app = QApplication.instance()
        if isinstance(app, QApplication):
            apply_theme(app, self._theme)

        self.setWindowTitle(t("OptiVibe - fiber-optic vibration sensor digital twin"))
        self.resize(1280, 820)
        self._config_dir = config_dir
        self._constants = load_constants()
        self._beta1_l = self._constants.universal.beta1_l
        self._last_result: object | None = None
        self._streaming = False

        self._controller = JobController(self)
        self._controller.progress.connect(self._on_progress)
        self._controller.finished.connect(self._on_finished)
        self._controller.failed.connect(self._on_failed)
        self._controller.cancelled.connect(self._on_cancelled)

        # The second worker family (task O-SW-03): a live stream instead of a
        # one-shot job. Its own controller, its own thread; the two are mutually
        # exclusive by policy (see _set_streaming), not by accident.
        self._stream = StreamController(self)
        self._stream.opened.connect(self._on_stream_opened)
        self._stream.frame.connect(self._on_stream_frame)
        self._stream.stopped.connect(self._on_stream_stopped)
        self._stream.failed.connect(self._on_stream_failed)

        self._panel = ControlPanel(config_dir=config_dir)
        self._live = LiveView()
        self._live.controls.start_requested.connect(self._on_live_start)
        self._live.controls.stop_requested.connect(self._stream.stop)
        self._report = ReportPanel()
        self._sweep = SweepPanel()
        self._monte = MonteCarloPanel()
        self._physics = PhysicsTab(self._panel, config_dir=config_dir)
        self._sweep.run_requested.connect(self._on_sweep)
        self._monte.run_requested.connect(self._on_monte_carlo)
        self._physics.nea_requested.connect(self._on_report)

        self._run_button = QPushButton(t("Run"))
        self._report_button = QPushButton(t("Report"))
        self._cancel_button = QPushButton(t("Cancel"))
        self._export_button = QPushButton(t("Export..."))
        self._check_button = QPushButton(t("Check composition"))
        self._run_button.clicked.connect(self._on_run)
        self._report_button.clicked.connect(self._on_report)
        self._cancel_button.clicked.connect(self._controller.cancel)
        self._export_button.clicked.connect(self._on_export_clicked)
        self._check_button.clicked.connect(self._on_check)
        self._cancel_button.setEnabled(False)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._live, t("Live"))
        self._tabs.addTab(self._report, t("Report"))
        self._tabs.addTab(self._sweep, t("Sweeps"))
        self._tabs.addTab(self._monte, t("Monte-Carlo"))
        self._tabs.addTab(self._physics, t("Physics"))
        self._tabs.currentChanged.connect(self._on_tab_changed)

        self.setCentralWidget(self._build_central())
        self._build_menu()
        self._build_log_dock()
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(160)
        self._progress.hide()
        self.statusBar().addPermanentWidget(self._progress)
        self.statusBar().showMessage(t("Ready. Pick a variant and excitation, then Run."))

        language_bus().changed.connect(self._relanguage)
        if self._restore_geometry:
            geometry = self._settings.value("geometry")
            if geometry is not None:
                self.restoreGeometry(geometry)

    def _build_central(self) -> QWidget:
        """Assemble the control column, action bar and tab area."""
        actions = QHBoxLayout()
        for button in (
            self._run_button,
            self._report_button,
            self._check_button,
            self._cancel_button,
            self._export_button,
        ):
            actions.addWidget(button)
        actions.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._panel)
        scroll.setMinimumWidth(330)
        self._scroll = scroll

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
    # Menu bar / shortcuts (Batch 1)
    # ------------------------------------------------------------------ #
    def _build_menu(self) -> None:
        """Build the File / Run / View / Help menu bar with shortcuts."""
        bar = self.menuBar()
        self._menu_actions: dict[str, QAction] = {}

        def _act(
            menu: QMenu,
            key: str,
            slot: Callable[[], object],
            shortcut: QKeySequence | QKeySequence.StandardKey | None = None,
        ) -> QAction:
            action = QAction(tr(key), self)
            if shortcut is not None:
                action.setShortcut(shortcut)
            action.triggered.connect(slot)
            menu.addAction(action)
            self._menu_actions[key] = action
            return action

        file_menu = bar.addMenu(tr("menu.file"))
        self._menus = {"menu.file": file_menu}
        _act(file_menu, "menu.export", self._on_export_clicked)
        file_menu.addSeparator()
        _act(file_menu, "menu.quit", self.close, QKeySequence.StandardKey.Quit)

        run_menu = bar.addMenu(tr("menu.run"))
        self._menus["menu.run"] = run_menu
        _act(run_menu, "menu.run_action", self._on_run, QKeySequence("Ctrl+R"))
        _act(run_menu, "menu.report_action", self._on_report, QKeySequence("Ctrl+Shift+R"))
        _act(run_menu, "menu.check_action", self._on_check, QKeySequence("Ctrl+K"))
        _act(run_menu, "menu.cancel_action", self._controller.cancel, QKeySequence("Esc"))

        view_menu = bar.addMenu(tr("menu.view"))
        self._menus["menu.view"] = view_menu
        _act(view_menu, "menu.preferences", self._on_preferences, QKeySequence("Ctrl+,"))
        self._log_action = _act(view_menu, "menu.toggle_log", self._toggle_log)
        self._log_action.setCheckable(True)

        help_menu = bar.addMenu(tr("menu.help"))
        self._menus["menu.help"] = help_menu
        _act(help_menu, "menu.whats_this", self._on_whats_this, QKeySequence("Shift+F1"))
        _act(help_menu, "menu.manual", self._open_manual)
        help_menu.addSeparator()
        _act(help_menu, "menu.about", self._on_about)

    def _retranslate_menu(self) -> None:
        """Re-label the menu titles and actions after a language change."""
        for key, menu in getattr(self, "_menus", {}).items():
            menu.setTitle(tr(key))
        for key, action in getattr(self, "_menu_actions", {}).items():
            action.setText(tr(key))

    # ------------------------------------------------------------------ #
    # Log dock (Batch 1)
    # ------------------------------------------------------------------ #
    def _build_log_dock(self) -> None:
        """Add a dockable log panel that mirrors the application logger."""
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(2000)
        clear = QPushButton(t("Clear"))
        clear.clicked.connect(self._log_view.clear)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._log_view, stretch=1)
        layout.addWidget(clear)
        self._log_dock = QDockWidget(t("Log"), self)
        self._log_dock.setObjectName("log_dock")
        self._log_dock.setWidget(body)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._log_dock)
        self._log_dock.hide()

        self._log_handler = _QtLogHandler(self._log_view)
        self._log_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logging.getLogger("optivibe").addHandler(self._log_handler)

    def _toggle_log(self) -> None:
        """Show or hide the log dock (keeps the menu check in sync)."""
        visible = not self._log_dock.isVisible()
        self._log_dock.setVisible(visible)
        self._log_action.setChecked(visible)

    # ------------------------------------------------------------------ #
    # Composition check (Batch 1)
    # ------------------------------------------------------------------ #
    def _on_check(self) -> None:
        """Dry-run resolve the composition and report the guard outcome.

        A lightweight, synchronous resolve (config read only, not the pipeline):
        it surfaces the geometry / wash-out guards (doc 03 §6, R-13) as a visible
        pass/fail instead of a run-time failure. On success it reports f1 and Q.
        """
        try:
            system = build_system_config(self._panel.system_payload())
        except (ValueError, TypeError) as exc:
            self._show_check(False, str(exc).splitlines()[0])
            return
        config_dir = self._config_dir or default_config_dir()
        try:
            variant = resolve_system_variant(system, config_dir)
        except (ValueError, TypeError) as exc:
            self._show_check(False, str(exc).splitlines()[0])
            return
        try:
            constants = load_constants(config_dir / "constants.yaml")
            f1 = f"{first_mode_hz(constants, variant.length_m):.1f}"
        except Exception:
            f1 = "-"
        q = f"{variant.q_total:.6g}" if variant.q_total is not None else "-"
        self._show_check(True, tr("check.resolved", name=variant.name, f1=f1, q=q))

    def _show_check(self, ok: bool, detail: str) -> None:
        """Show the composition-check result in a message box."""
        box = QMessageBox(self)
        box.setWindowTitle(tr("check.title"))
        if ok:
            box.setIcon(QMessageBox.Icon.Information)
            box.setText(tr("check.ok"))
            box.setInformativeText(detail)
        else:
            box.setIcon(QMessageBox.Icon.Warning)
            box.setText(tr("check.fail", reason=detail))
        box.exec()

    # ------------------------------------------------------------------ #
    # Preferences / theme / language (Batch 1)
    # ------------------------------------------------------------------ #
    def _on_preferences(self) -> None:
        """Open the Preferences dialog and apply / persist the selection."""
        dialog = PreferencesDialog(
            language=current_language(),
            theme=self._theme,
            restore_geometry=bool(self._restore_geometry),
            parent=self,
        )
        dialog.language_previewed.connect(set_language)
        dialog.theme_previewed.connect(self._apply_theme)
        if dialog.exec():
            choice = dialog.selection()
            set_language(str(choice["language"]))
            self._apply_theme(str(choice["theme"]))
            self._restore_geometry = bool(choice["restore_geometry"])
            self._settings.setValue("language", current_language())
            self._settings.setValue("theme", self._theme)
            self._settings.setValue("restore_geometry", self._restore_geometry)
        else:
            self._apply_theme(self._theme)

    def _apply_theme(self, name: str) -> None:
        """Apply a theme name to the application and remember it."""
        if name not in THEMES:
            return
        self._theme = name
        app = QApplication.instance()
        if isinstance(app, QApplication):
            apply_theme(app, name)

    def _relanguage(self, code: str) -> None:
        """Rebuild the control panel and re-label the chrome in the new language.

        Composition and scenario state survive through a payload round-trip
        (SW-65): everything the user edited is re-applied to the freshly-built
        panel, so the switch is transparent.
        """
        system_payload = self._panel.system_payload()
        scenario_payload = self._panel.scenario_payload()
        current_tab = self._tabs.currentIndex()

        new_panel = ControlPanel(config_dir=self._config_dir)
        try:
            new_panel.apply_system_payload(system_payload)
            new_panel.restore_scenario(scenario_payload)
        except (ValueError, TypeError):  # pragma: no cover - defensive
            logger.warning("could not fully restore panel state across a language change")

        old_panel = self._panel
        self._panel = new_panel
        # QScrollArea.setWidget deletes the previously set widget, so we must not
        # delete old_panel again afterwards (double-free).
        self._scroll.setWidget(new_panel)
        del old_panel

        new_physics = PhysicsTab(new_panel, config_dir=self._config_dir)
        new_physics.nea_requested.connect(self._on_report)
        self._tabs.removeTab(self._tabs.indexOf(self._physics))
        self._physics.deleteLater()
        self._physics = new_physics
        self._tabs.addTab(self._physics, t("Physics"))

        for panel in (self._live, self._report, self._sweep, self._monte):
            panel.retranslate()
        self._retranslate_chrome()
        self._tabs.setCurrentIndex(current_tab)

    def _retranslate_chrome(self) -> None:
        """Re-label the window title, tabs, buttons and menu after a switch."""
        self.setWindowTitle(t("OptiVibe - fiber-optic vibration sensor digital twin"))
        self._run_button.setText(t("Run"))
        self._report_button.setText(t("Report"))
        self._cancel_button.setText(t("Cancel"))
        self._export_button.setText(t("Export..."))
        self._check_button.setText(t("Check composition"))
        for index, key in enumerate(("Live", "Report", "Sweeps", "Monte-Carlo")):
            self._tabs.setTabText(index, t(key))
        self._log_dock.setWindowTitle(t("Log"))
        self._retranslate_menu()

    # ------------------------------------------------------------------ #
    # Help (Batch 1)
    # ------------------------------------------------------------------ #
    def _on_whats_this(self) -> None:
        """Enter Qt's What's-This mode (click a widget for its help)."""
        QWhatsThis.enterWhatsThisMode()

    def _open_manual(self) -> None:
        """Open the project documentation folder (``<repo>/docs``).

        Resolves ``docs`` relative to the source tree (editable install). If it
        is not found next to the package (e.g. a wheel install), the resolved
        path is shown so the user knows where to look.
        """
        docs = Path(__file__).resolve().parents[3] / "docs"
        if docs.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(docs)))
        else:
            QMessageBox.information(self, tr("menu.manual"), tr("manual.not_found", path=str(docs)))

    def _on_about(self) -> None:
        """Show the About box."""
        QMessageBox.about(self, tr("menu.about"), t(_ABOUT_TEXT))

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
            self.statusBar().showMessage(tr("status.invalid_sweep", exc=exc))
            return
        self._start(SweepJob(spec=spec), "sweep")

    def _on_monte_carlo(self) -> None:
        """Run a tolerance Monte-Carlo from the Monte-Carlo panel."""
        try:
            spec = build_monte_carlo_spec(self._monte.payload())
        except (ValueError, TypeError) as exc:
            self.statusBar().showMessage(tr("status.invalid_mc", exc=exc))
            return
        self._start(MonteCarloJob(spec=spec), "monte-carlo")

    def _build_scenario(self) -> ScenarioConfig | None:
        """Validate the control-panel payload into a scenario (or report error)."""
        try:
            return build_scenario_config(self._panel.scenario_payload())
        except (ValueError, TypeError) as exc:
            self.statusBar().showMessage(
                tr("status.invalid_scenario", exc=str(exc).splitlines()[0])
            )
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
            self.statusBar().showMessage(
                tr("status.invalid_composition", exc=str(exc).splitlines()[0])
            )
            return None

    def _start(self, job: Job, label: str) -> None:
        """Start a job off the UI thread and lock the action buttons."""
        if self._controller.is_running():
            return
        if self._stream.is_running():
            # Mutual exclusion with the live stream (the panel run buttons of
            # Sweeps / Monte-Carlo reach this method directly).
            self.statusBar().showMessage(tr("status.job_live_busy"))
            return
        self._set_running(True)
        self.statusBar().showMessage(tr("status.running", label=label))
        try:
            self._controller.start(job)
        except RuntimeError as exc:  # pragma: no cover - guarded by is_running
            self._set_running(False)
            self.statusBar().showMessage(tr("status.could_not_start", exc=exc))

    # ------------------------------------------------------------------ #
    # Controller signals
    # ------------------------------------------------------------------ #
    def _on_progress(self, message: str) -> None:
        """Show a coarse progress message."""
        self.statusBar().showMessage(tr("status.progress", message=message))

    def _on_finished(self, result: object) -> None:
        """Route a finished job result to the matching tab."""
        self._set_running(False)
        self._last_result = result
        if isinstance(result, RunArtifacts):
            self._live.show_artifacts(result, self._beta1_l)
            self._attach_expected(result)
            self._tabs.setCurrentWidget(self._live)
            self._announce_run(result)
        elif isinstance(result, ReportBundle):
            self._report.show_bundle(result)
            self._live.show_artifacts(result.artifacts, self._beta1_l)
            self._attach_expected(result.artifacts)
            self._live.show_nea(result.nea)
            if result.nea is not None:
                self._physics.set_nea_figure(plot_nea_budget(result.nea))
            self._tabs.setCurrentWidget(self._report)
            self.statusBar().showMessage(
                tr(
                    "status.report_ready",
                    ratio=result.budget.amplitude_ratio,
                    rel=result.budget.rms_error_rel,
                )
            )
        elif isinstance(result, SweepResult):
            self._sweep.show_result(result)
            self._tabs.setCurrentWidget(self._sweep)
            self.statusBar().showMessage(
                tr(
                    "status.sweep_done",
                    name=result.name,
                    mode=result.mode,
                    parameter=result.parameter,
                    n=len(result.axis_labels),
                )
            )
        elif isinstance(result, MonteCarloResult):
            self._monte.show_result(result)
            self._tabs.setCurrentWidget(self._monte)
            self.statusBar().showMessage(tr("status.mc_done", name=result.name, n=result.n_draws))
        else:  # pragma: no cover - defensive
            self.statusBar().showMessage(tr("status.unrecognised"))

    def _attach_expected(self, artifacts: RunArtifacts) -> None:
        """Hand the run's predicted peak set to the Live spectrum (S-16/S-17).

        Runs on the GUI thread, and that is deliberate. The S7 invariant
        (``SW-06``) forbids work **whose cost grows with the data** -- the
        pipeline, the DSP tract, an FFT, any walk over the samples; the
        precedent is the input FFT that was moved out to Report for exactly
        that reason. Building an ``ExpectedPeaks`` is O(1) arithmetic over the
        resolved configuration: no time series is read, and the physics lives
        in :mod:`optivibe.analysis.expected_peaks`, not here. The invariant is
        also guaranteed *structurally* rather than by convention --
        :func:`~optivibe.analysis.expected_peaks.predict_expected_peaks` accepts
        only a scenario and a variant, so it is not even able to receive a
        record. Criterion settled by coordination on 2026-07-30 (doc 13,
        ``SW-70``); do not re-litigate it per call site.

        The view still computes nothing: it only draws what the artifact
        carries (09 §9). A failure must never cost the user a finished run, so
        it degrades to "no overlay".
        """
        try:
            expected = predict_expected_peaks(
                artifacts.scenario, artifacts.variant, self._constants
            )
        except (ValueError, KeyError) as exc:  # pragma: no cover - defensive
            logger.debug("expected-peak prediction skipped: %s", exc)
            self._live.set_expected_peaks(None)
            return
        self._live.set_expected_peaks(expected)

    def _announce_run(self, artifacts: RunArtifacts) -> None:
        """Status line for a finished scenario run."""
        result = artifacts.result
        dominant = ", ".join(f"{f:.2f}" for f in result.dominant_freqs_hz) or "-"
        self.statusBar().showMessage(
            tr(
                "status.run_done",
                name=artifacts.variant.name,
                n=result.n_samples,
                dominant=dominant,
            )
        )

    def _on_tab_changed(self, index: int) -> None:
        """Rebuild the light physics curves when the Physics tab is shown."""
        if self._tabs.widget(index) is self._physics:
            self._physics.refresh_light()

    def _on_failed(self, message: str) -> None:
        """Report a failed job."""
        self._set_running(False)
        self.statusBar().showMessage(tr("status.failed", message=message))

    def _on_cancelled(self) -> None:
        """Report a cancelled job (its result was dropped)."""
        self._set_running(False)
        self.statusBar().showMessage(tr("status.cancelled"))

    def _set_running(self, running: bool) -> None:
        """Lock/unlock the action buttons and the busy indicator."""
        self._run_button.setEnabled(not running)
        self._report_button.setEnabled(not running)
        self._export_button.setEnabled(not running)
        self._cancel_button.setEnabled(running)
        self._live.controls.setEnabled(not running)
        self._progress.setRange(0, 0 if running else 1)
        self._progress.setVisible(running)

    # ------------------------------------------------------------------ #
    # Real-time mode (task O-SW-03)
    # ------------------------------------------------------------------ #
    def _on_live_start(self) -> None:
        """Assemble the selected source and start the live stream."""
        if self._stream.is_running():
            return
        if self._controller.is_running():
            self.statusBar().showMessage(tr("status.live_busy"))
            return
        source = self._build_stream_source()
        if source is None:
            return
        controls = self._live.controls
        config = StreamConfig(
            rate_hz=controls.rate_hz(),
            speed=controls.speed(),
            loop=controls.loop_enabled(),
        )
        self._set_streaming(True)
        self._live.begin_stream()
        try:
            self._stream.start(source, config)
        except RuntimeError as exc:  # pragma: no cover - guarded by is_running
            self._set_streaming(False)
            self._live.end_stream(str(exc))
            return
        self.statusBar().showMessage(tr("status.live_started", label=source.label))

    def _build_stream_source(self) -> StreamSource | None:
        """Build the source for the current selection, or report why not.

        Only *config* is read here (a scenario payload, or a small analyze-spec
        YAML). The record itself, the composition resolve and the forward run
        happen inside :meth:`StreamSource.open` on the worker thread (SW-06):
        this method must stay O(1) in the data, like the composition check.
        """
        controls = self._live.controls
        if controls.source_kind() == SOURCE_RECORD:
            path = controls.spec_path()
            if path is None:
                self.statusBar().showMessage(tr("status.live_no_spec"))
                return None
            try:
                spec = load_analyze_spec(path)
            except (ValueError, TypeError, OSError) as exc:
                self.statusBar().showMessage(
                    tr("status.live_bad_spec", exc=str(exc).splitlines()[0])
                )
                return None
            return RecordSource(spec=spec, config_dir=self._config_dir)
        scenario = self._build_scenario()
        if scenario is None:
            return None
        system = self._build_system()
        if system is None:
            return None
        return ScenarioSource(scenario=scenario, system=system, config_dir=self._config_dir)

    def _on_stream_opened(self, payload: object) -> None:
        """Attach the expected-peak overlay once the source has resolved.

        The synthetic source knows its scenario and (now) its resolved variant,
        so the prediction is available live. It is assembled here, on the UI
        thread, for the reason settled in ``SW-70``: it is O(1) arithmetic over
        the configuration and cannot read a time series. A file record carries
        no scenario, so it gets no overlay rather than a guessed one.
        """
        if not isinstance(payload, StreamSetup):  # pragma: no cover - defensive
            return
        if payload.scenario is None:
            self._live.set_expected_peaks(None)
            return
        try:
            expected = predict_expected_peaks(payload.scenario, payload.variant, self._constants)
        except (ValueError, KeyError) as exc:  # pragma: no cover - defensive
            logger.debug("expected-peak prediction skipped for the stream: %s", exc)
            self._live.set_expected_peaks(None)
            return
        self._live.set_expected_peaks(expected)

    def _on_stream_frame(self, payload: object) -> None:
        """Render one live frame (payload arrives as ``object`` over a signal)."""
        if isinstance(payload, StreamFrame):
            self._live.show_stream_frame(payload)

    def _on_stream_stopped(self) -> None:
        """Return to the idle layout after the stream ended."""
        self._set_streaming(False)
        self._live.end_stream()
        self.statusBar().showMessage(tr("status.live_stopped"))

    def _on_stream_failed(self, message: str) -> None:
        """Report a stream that could not start or died."""
        self._set_streaming(False)
        self._live.end_stream(tr("status.live_failed", message=message))
        self.statusBar().showMessage(tr("status.live_failed", message=message))

    def _set_streaming(self, streaming: bool) -> None:
        """Lock/unlock the one-shot actions for the duration of a stream.

        The two worker families are mutually exclusive **by policy**, not
        because a thread happens to be shared: a live stream and a finished run
        would otherwise put two unrelated sets of numbers on the same panels,
        and no provenance line could say which is which.
        """
        self._streaming = streaming
        for button in (
            self._run_button,
            self._report_button,
            self._check_button,
            self._export_button,
        ):
            button.setEnabled(not streaming)

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def _on_export_clicked(self) -> None:  # pragma: no cover - dialog
        """Pick a directory and export the latest result into it."""
        if self._last_result is None:
            self.statusBar().showMessage(tr("status.nothing_export"))
            return
        directory = QFileDialog.getExistingDirectory(self, t("Export to directory"))
        if directory:
            saved = self.export_to(Path(directory))
            self.statusBar().showMessage(tr("status.exported", n=len(saved), directory=directory))

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
        """Persist settings, stop the animation and cancel any running job."""
        self._settings.setValue("language", current_language())
        self._settings.setValue("theme", self._theme)
        self._settings.setValue("restore_geometry", self._restore_geometry)
        self._settings.setValue("geometry", self.saveGeometry())
        handler = getattr(self, "_log_handler", None)
        if handler is not None:
            logging.getLogger("optivibe").removeHandler(handler)
        self._live.stop()
        self._stream.stop()
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
