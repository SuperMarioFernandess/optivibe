"""GUI smoke tests for the S7 desktop app (pytest-qt, offscreen; task S7 §7).

Skipped automatically without the ``gui`` extra (PySide6 + pyqtgraph + pytest-qt)
or a Qt platform; runs head-less via ``QT_QPA_PLATFORM=offscreen`` (conftest).
Proves the mandatory threading invariant (compute runs *off* the GUI thread,
SW-06), that the control panel assembles a runnable scenario, that a real run
updates the window, and that cancellation and errors are handled without crashing.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("pytestqt")

pytestmark = pytest.mark.gui

from PySide6.QtCore import QThread  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from optivibe.gui.controllers.job_controller import JobController  # noqa: E402
from optivibe.gui.controllers.scenario_builder import (  # noqa: E402
    build_scenario_config,
)
from optivibe.gui.main_window import MainWindow  # noqa: E402
from optivibe.gui.workers.jobs import ScenarioJob  # noqa: E402
from optivibe.pipeline import RunArtifacts  # noqa: E402


class _ThreadProbe:
    """A job that records the QThread it runs on (for the off-thread proof)."""

    label = "thread-probe"

    def __init__(self) -> None:
        self.worker_thread: int | None = None

    def run(self, *, progress: object, is_cancelled: object) -> object:
        self.worker_thread = id(QThread.currentThread())
        return "ok"


class _BlockingJob:
    """A job that polls cancellation so cancel() can stop it cooperatively."""

    label = "blocking"

    def run(self, *, progress: object, is_cancelled: object) -> object:
        for _ in range(300):
            if is_cancelled():  # type: ignore[operator]
                return "cancelled-early"
            time.sleep(0.01)
        return "done"


def test_main_window_builds_tabs(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.windowTitle().startswith("OptiVibe")
    assert window.run_button is not None
    assert window.plot is not None
    assert window.controller is not None
    # Live / Report / Sweeps / Monte-Carlo.
    assert window._tabs.count() == 4


def test_control_panel_assembles_scenario(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    panel = window.control_panel
    # Default: variant B, photodiode -> detector overrides present.
    payload = panel.scenario_payload()
    scenario = build_scenario_config(payload)
    assert scenario.variant == "B"
    assert scenario.stages.detector == "photodiode"
    assert "detector" in payload
    # Switching to the stub omits detector overrides (the option-less stub takes
    # no constructor arguments; 14 §5).
    panel._detector.setCurrentText("stub")
    stub_payload = panel.scenario_payload()
    assert stub_payload["stages"]["detector"] == "stub"
    assert "detector" not in stub_payload


def test_computation_runs_off_the_gui_thread(qtbot) -> None:
    """The mandatory invariant: the worker executes on a *different* thread."""
    controller = JobController()
    probe = _ThreadProbe()
    with qtbot.waitSignal(controller.finished, timeout=10000):
        controller.start(probe)
    main_thread = id(QApplication.instance().thread())
    assert probe.worker_thread is not None
    assert probe.worker_thread != main_thread
    assert not controller.is_running()


def test_real_run_updates_window(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    results: list[object] = []
    window.controller.finished.connect(results.append)
    with qtbot.waitSignal(window.controller.finished, timeout=20000):
        window.run_button.click()
    assert len(results) == 1
    artifacts = results[0]
    assert isinstance(artifacts, RunArtifacts)
    assert artifacts.result.dominant_freqs_hz[0] == pytest.approx(200.0, abs=1.0)
    assert not window.controller.is_running()
    assert window.run_button.isEnabled()  # re-enabled after the run


def test_cancel_drops_result(qtbot) -> None:
    controller = JobController()
    with qtbot.waitSignal(controller.cancelled, timeout=10000):
        controller.start(_BlockingJob())
        controller.cancel()
    assert not controller.is_running()


def test_failed_job_is_reported(qtbot, tmp_path: Path) -> None:
    controller = JobController()
    errors: list[str] = []
    controller.failed.connect(errors.append)
    with qtbot.waitSignal(controller.failed, timeout=10000):
        controller.start(ScenarioJob(source=tmp_path / "missing.yaml"))
    assert errors
    assert not controller.is_running()


def test_report_action_fills_report_tab(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    with qtbot.waitSignal(window.controller.finished, timeout=20000):
        window._on_report()
    # The report panel text was populated from the error budget.
    assert window._report._text.toPlainText() != ""
