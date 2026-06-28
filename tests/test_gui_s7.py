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
    # Live / Report / Sweeps / Monte-Carlo / Physics (task S7-mod §5).
    assert window._tabs.count() == 5


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


# --------------------------------------------------------------------------- #
# S7-mod: editable composition, dynamic multitone, live visibility, physics tab.
# --------------------------------------------------------------------------- #
def test_composition_panel_resolves_to_starting_variant(qtbot) -> None:
    """The unedited composition panel resolves to its starting variant B."""
    from optivibe.core.config.loader import default_config_dir, load_variant
    from optivibe.gui.controllers.system_builder import (
        build_system_config,
        resolve_system_variant,
    )

    window = MainWindow()
    qtbot.addWidget(window)
    system = build_system_config(window.control_panel.system_payload())
    resolved = resolve_system_variant(system, default_config_dir())
    assert resolved.model_dump() == load_variant("B").model_dump()


def test_reflector_shape_switch_updates_payload(qtbot) -> None:
    """Selecting the plane shape clears the curvature radius in the payload."""
    window = MainWindow()
    qtbot.addWidget(window)
    reflector = window.control_panel.system._reflector
    reflector._shape.setCurrentText("plane")
    overrides = window.control_panel.system_payload()["reflector"]["overrides"]
    assert overrides["shape"] == "plane"
    assert overrides["curvature_radius_m"] is None


def test_multitone_dynamic_components(qtbot) -> None:
    """Multitone defaults to two components and supports add/remove + phase."""
    window = MainWindow()
    qtbot.addWidget(window)
    excitation = window.control_panel._excitation
    excitation._kind.setCurrentText("multitone")
    multitone = excitation._multitone
    assert multitone.count() == 2
    multitone._add_row(360.0, 0.25, 0.0)
    assert multitone.count() == 3
    multitone._remove_row(multitone._rows[-1])
    assert multitone.count() == 2
    multitone._phase.setChecked(True)
    tones = excitation.excitation_payload()["tones"]
    assert len(tones) == 2
    assert all(len(tone) == 3 for tone in tones)


def test_live_visibility_toggle_reflows(qtbot) -> None:
    """Hiding a Live panel reflows the layout (frees the space)."""
    window = MainWindow()
    qtbot.addWidget(window)
    live = window.plot
    before = len(live._plots.ci.items)
    live._checks["det"].setChecked(False)
    after = len(live._plots.ci.items)
    assert after == before - 1
    live._checks["det"].setChecked(True)
    assert len(live._plots.ci.items) == before


def test_physics_tab_builds_curves(qtbot) -> None:
    """The Physics tab rebuilds its light reference curves from the composition."""
    window = MainWindow()
    qtbot.addWidget(window)
    assert window._physics.refresh_light()
    assert window._physics._f1.figure is not None
    assert window._physics._hlat.figure is not None
    assert window._physics._eta.figure is not None


def test_edited_composition_run_off_thread(qtbot) -> None:
    """An edited composition is injected and runs off-thread with the full DSP.

    Edits the cantilever length (a cylinder-compatible change the standard
    calibrated inverse supports) and checks the resolved variant carries the
    edit while the 200 Hz tone is still recovered. Shape changes to the
    sphere/plane/wedge family run with the stub inverse (the calibrated
    sensitivity is cylinder-only, doc 14 §8); that path is covered Qt-free in
    ``test_gui_system_builder``.
    """
    window = MainWindow()
    qtbot.addWidget(window)
    window.control_panel.system._cantilever._edits["length_m"].setText("1.8e-3")
    results: list[object] = []
    window.controller.finished.connect(results.append)
    with qtbot.waitSignal(window.controller.finished, timeout=20000):
        window.run_button.click()
    assert len(results) == 1
    artifacts = results[0]
    assert isinstance(artifacts, RunArtifacts)
    assert artifacts.variant.length_m == pytest.approx(1.8e-3)
    assert artifacts.result.dominant_freqs_hz[0] == pytest.approx(200.0, abs=1.0)
