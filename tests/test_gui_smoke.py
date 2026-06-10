"""GUI smoke test: the window builds and a run completes off the UI thread.

Skipped automatically when the optional ``gui`` extra (PySide6 + pyqtgraph) or a
Qt platform is unavailable. Runs head-less via ``QT_QPA_PLATFORM=offscreen``
(set in ``conftest.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("pytestqt")

pytestmark = pytest.mark.gui

from optivibe.gui.controllers.run_controller import RunController  # noqa: E402
from optivibe.gui.main_window import MainWindow  # noqa: E402
from optivibe.pipeline import RunArtifacts  # noqa: E402


def test_main_window_constructs(qtbot) -> None:
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.run_button is not None
    assert window.plot is not None
    assert window.windowTitle().startswith("OptiVibe")


def test_run_controller_runs_scenario_off_thread(
    qtbot,
    hello_scenario: Path,
    config_dir: Path,
) -> None:
    controller = RunController()
    results: list[object] = []
    controller.finished.connect(results.append)

    with qtbot.waitSignal(controller.finished, timeout=15000):
        controller.start(hello_scenario, config_dir=config_dir)

    assert len(results) == 1
    artifacts = results[0]
    assert isinstance(artifacts, RunArtifacts)
    assert artifacts.result.dominant_freqs_hz[0] == pytest.approx(120.0, abs=1.0)
    assert not controller.is_running()
