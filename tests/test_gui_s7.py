"""GUI smoke tests for the S7 desktop app (pytest-qt, offscreen; task S7 §7).

Skipped automatically without the ``gui`` extra (PySide6 + pyqtgraph + pytest-qt)
or a Qt platform; runs head-less via ``QT_QPA_PLATFORM=offscreen`` (conftest).
Proves the mandatory threading invariant (compute runs *off* the GUI thread,
SW-06), that the control panel assembles a runnable scenario, that a real run
updates the window, and that cancellation and errors are handled without crashing.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("pytestqt")

pytestmark = pytest.mark.gui

from PySide6.QtWidgets import QApplication  # noqa: E402

from optivibe.gui.controllers.job_controller import JobController  # noqa: E402
from optivibe.gui.controllers.scenario_builder import (  # noqa: E402
    build_scenario_config,
)
from optivibe.gui.main_window import MainWindow  # noqa: E402
from optivibe.gui.workers.jobs import ScenarioJob  # noqa: E402
from optivibe.pipeline import RunArtifacts  # noqa: E402


class _ThreadProbe:
    """A job that records the thread it runs on (for the off-thread proof).

    The identity recorded is :func:`threading.get_ident`, **not**
    ``id(QThread.currentThread())``. The latter is the address of an ephemeral
    PySide *wrapper*: the worker's wrapper dies with the call, CPython reuses
    its address for the next wrapper, and the comparison then comes out equal
    for two different threads (observed as a ~1-in-10 flake once the heap churn
    changed, O-SW-03). The failure mode also runs the other way -- a wrapper
    cache miss yields two addresses for one thread -- so the proof could pass
    while the invariant was broken. ``get_ident`` is unique among *live*
    threads, and the GUI thread is alive throughout, so it cannot alias.
    """

    label = "thread-probe"

    def __init__(self) -> None:
        self.worker_thread: int | None = None

    def run(self, *, progress: object, is_cancelled: object) -> object:
        self.worker_thread = threading.get_ident()
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
    # Live / Report / Sweeps / Monte-Carlo / Compare / Physics
    # (task S7-mod §5; the Compare tab is task S-22 W-2).
    assert window._tabs.count() == 6


def test_control_panel_assembles_scenario(qtbot) -> None:
    from optivibe.core.config.loader import default_config_dir
    from optivibe.gui.controllers.system_builder import (
        build_system_config,
        resolve_system_variant,
    )

    window = MainWindow()
    qtbot.addWidget(window)
    panel = window.control_panel
    # Default: variant B, photodiode stage, physical optics (key "cylinder").
    payload = panel.scenario_payload()
    scenario = build_scenario_config(payload)
    assert scenario.variant == "B"
    assert scenario.stages.detector == "photodiode"
    assert scenario.stages.optics == "cylinder"
    # The scenario no longer emits any detector override (single source of
    # truth: balanced / reference_arm live in the composition; S7-mod cleanup).
    assert "detector" not in payload
    # Switching the detector stage to the stub still works and emits no override.
    panel._detector.setCurrentText("stub")
    stub_payload = panel.scenario_payload()
    assert stub_payload["stages"]["detector"] == "stub"
    assert "detector" not in stub_payload
    # balanced / reference_arm are governed by the composition's Detector form
    # and flow into the resolved variant.detector.
    panel.system._balanced.setChecked(False)
    panel.system._reference_arm.setCurrentText("bright")
    variant = resolve_system_variant(
        build_system_config(panel.system_payload()), default_config_dir()
    )
    assert variant.detector.balanced is False
    assert variant.detector.reference_arm == "bright"


def test_computation_runs_off_the_gui_thread(qtbot) -> None:
    """The mandatory invariant: the worker executes on a *different* thread."""
    controller = JobController()
    probe = _ThreadProbe()
    with qtbot.waitSignal(controller.finished, timeout=10000):
        controller.start(probe)
    assert probe.worker_thread is not None
    assert probe.worker_thread != threading.get_ident()
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


# --------------------------------------------------------------------------- #
# S-13 GUI: source lineshape/spectrum entry, computed Q(L), thermal NEA branch.
# --------------------------------------------------------------------------- #
def _spectrum_artifact(directory: Path) -> Path:
    """A Gaussian OSA characterization artifact (sidecar + CSV; doc 16 §2a)."""
    import math

    import numpy as np
    import yaml

    lam = np.linspace(1450.0, 1650.0, 401)
    sigma = 60.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    psd = np.exp(-0.5 * ((lam - 1550.0) / sigma) ** 2)
    np.savetxt(directory / "sp.csv", np.column_stack([lam, psd]), delimiter=",")
    sidecar = directory / "sp.yaml"
    sidecar.write_text(
        yaml.safe_dump(
            {
                "kind": "spectrum",
                "instrument": "OSA",
                "timestamp": "2026-07-10T09:00:00Z",
                "data_file": "sp.csv",
                "columns": {
                    "x": {"name": "l", "unit": "nm"},
                    "y": {"name": "S", "unit": "arb"},
                },
                "uncertainties": {"wavelength_m": 0.5e-9},
            }
        ),
        encoding="utf-8",
    )
    return sidecar


def test_source_form_defaults_add_no_lineshape(qtbot) -> None:
    """An untouched source form emits no lineshape/table keys (R-46 default)."""
    window = MainWindow()
    qtbot.addWidget(window)
    overrides = window.control_panel.system_payload()["source"]["overrides"]
    assert "lineshape" not in overrides
    assert "spectrum_wavelength_m" not in overrides


def test_source_form_loads_measured_spectrum(qtbot, tmp_path: Path) -> None:
    """Loading a spectrum artifact enables lineshape='measured' (M-10 rules).

    The overrides then carry the table and force ``linewidth_fwhm_m = None``
    (single source of truth, R-57(a)); switching back to the default drops the
    table from the payload again.
    """
    window = MainWindow()
    qtbot.addWidget(window)
    source = window.control_panel.system._source
    source.load_spectrum_artifact(_spectrum_artifact(tmp_path))
    overrides = source.overrides()
    assert overrides["lineshape"] == "measured"
    assert len(overrides["spectrum_wavelength_m"]) == 401
    assert overrides["linewidth_fwhm_m"] is None
    assert "sha" in source._spectrum_note.text()
    source._lineshape.setCurrentText("(default)")
    overrides = source.overrides()
    assert "lineshape" not in overrides
    assert "spectrum_wavelength_m" not in overrides


def test_source_form_rejects_non_spectrum_artifact(qtbot, tmp_path: Path) -> None:
    """A non-spectrum characterization artifact is refused by the loader."""
    import yaml

    sidecar = tmp_path / "L.yaml"
    sidecar.write_text(
        yaml.safe_dump(
            {
                "kind": "scalar",
                "instrument": "caliper",
                "timestamp": "t",
                "parameter": "length_m",
                "value": 4.0,
                "unit": "mm",
                "u": 0.02,
            }
        ),
        encoding="utf-8",
    )
    window = MainWindow()
    qtbot.addWidget(window)
    with pytest.raises(ValueError, match="not a spectrum"):
        window.control_panel.system._source.load_spectrum_artifact(sidecar)


def test_q_total_model_hint_is_shown(qtbot) -> None:
    """The computed Q(L) value is displayed next to the override field (M-02)."""
    window = MainWindow()
    qtbot.addWidget(window)
    panel = window.control_panel.system
    text = panel._q_model.text()
    assert "Q(L) model:" in text and "n/a" not in text
    # It follows the cantilever length: a shorter fiber -> different Q.
    panel._cantilever._edits["length_m"].setText("1.8e-3")
    panel.refresh_q_model()
    assert panel._q_model.text() != text


def test_live_nea_panel_draws_thermal_branch(qtbot) -> None:
    """show_nea draws all four plateau branches incl. thermal (M-12 visible)."""
    import numpy as np

    from optivibe.analysis.nea_budget import NeaBudget

    window = MainWindow()
    qtbot.addWidget(window)
    freq = np.linspace(1.0, 1000.0, 32)
    level = 1.0e-4
    nea = NeaBudget(
        freq_hz=freq,
        nea_density=np.full_like(freq, 2.0 * level),
        nea_plateau=2.0 * level,
        nea_full_band=2.0 * level * np.sqrt(999.0),
        bandwidth_hz=999.0,
        contributions={
            "shot": level,
            "rin": level,
            "johnson": level,
            "thermal": level,
            "total": 2.0 * level,
        },
        nea_thermal=level,
        psd_components={},
        psd_total_analytic=1.0,
        psd_rel_error=0.0,
        reference_arm="matched",
        s_target=1.0,
        velocity_floor_rms=0.0,
        displacement_floor_rms=0.0,
    )
    live = window.plot
    live.show_nea(nea)
    names = {item.name() for item in live._p_nea.listDataItems() if item.name() is not None}
    assert {"shot", "rin", "johnson", "thermal"} <= names


# --------------------------------------------------------------------------- #
# S-13b GUI polish: tabbed panel, reseed clears stale overrides, measured-data
# loaders on their tabs, inline help, and the mouse-wheel guard.
# --------------------------------------------------------------------------- #
def test_parameter_panel_is_one_flat_tab_set(qtbot) -> None:
    """The parameter area is a single tab widget with the agreed tabs.

    The tenth tab is the DSP experiment page (task S-22 W-1): additive, after
    the physics layers it belongs with, before reproducibility.
    """
    from PySide6.QtWidgets import QTabWidget

    window = MainWindow()
    qtbot.addWidget(window)
    panel = window.control_panel.system
    assert isinstance(panel, QTabWidget)
    labels = [panel.tabText(i) for i in range(panel.count())]
    assert labels == [
        "System",
        "Source",
        "Fiber line",
        "Cantilever",
        "Reflector",
        "Detector",
        "Excitation",
        "Physics layers",
        "DSP experiment",
        "Reproducibility",
    ]


def test_reseed_clears_fields_absent_from_the_new_preset(qtbot) -> None:
    """Switching presets/compositions clears stale values (no silent overrides).

    A linewidth typed for one SLD must not survive a switch to a preset that
    has none -- it would silently become an override of the new preset. The
    same applies to switching the starting composition.
    """
    window = MainWindow()
    qtbot.addWidget(window)
    source = window.control_panel.system._source
    # Starting composition B already uses the bare ``sld`` preset, so switch to
    # a genuinely different preset first: ``sld_dl60`` states the linewidth and
    # derives (omits) the RIN, so the field follows the preset and RIN clears.
    source._preset.setCurrentText("sld_dl60")
    assert source._edits["linewidth_fwhm_m"].text() != ""
    assert source._edits["rin_db_hz"].text() == ""
    # Now switch to ``sld``: it has no linewidth, so the stale value must clear
    # rather than silently become an override; its RIN is stated (-126).
    source._preset.setCurrentText("sld")
    assert source._edits["linewidth_fwhm_m"].text() == ""
    assert "linewidth_fwhm_m" not in source.overrides()
    assert float(source._edits["rin_db_hz"].text()) == pytest.approx(-126.0)
    # Switching the starting composition reseeds too: A also uses ``sld``.
    source._edits["linewidth_fwhm_m"].setText("9e-8")
    window.control_panel.system._starting.setCurrentText("A")
    assert source._edits["linewidth_fwhm_m"].text() == ""


def _artifact(directory: Path, kind: str) -> Path:
    """Write one synthetic characterization artifact; return the CSV path."""
    import math

    import numpy as np
    import yaml

    if kind == "rin_psd":
        freq = np.linspace(100.0, 20_000.0, 100)
        np.savetxt(
            directory / "a.csv",
            np.column_stack([freq, np.full_like(freq, -121.5)]),
            delimiter=",",
        )
        body = {
            "kind": "rin_psd",
            "instrument": "ESA",
            "timestamp": "t",
            "data_file": "a.csv",
            "columns": {"x": {"name": "f", "unit": "hz"}, "y": {"name": "r", "unit": "db/hz"}},
            "band": {"f_min_hz": 1_000.0, "f_max_hz": 10_000.0},
            "uncertainties": {"rin_db_hz": 0.5},
        }
    elif kind == "profile":
        radius = 150.0e-6
        theta = np.linspace(-0.5, 0.5, 40)
        table = np.column_stack(
            [radius * np.sin(theta) * 1e6, radius * (1.0 - np.cos(theta)) * 1e6]
        )
        np.savetxt(directory / "a.csv", table, delimiter=",")
        body = {
            "kind": "profile",
            "instrument": "microscope",
            "timestamp": "t",
            "data_file": "a.csv",
            "columns": {"x": {"name": "x", "unit": "um"}, "y": {"name": "z", "unit": "um"}},
        }
    else:  # ringdown
        fs, f1, q = 200_000.0, 6250.0, 1661.0
        t = np.arange(0.0, 0.2, 1.0 / fs)
        y = np.exp(-math.pi * f1 / q * t) * np.cos(2.0 * math.pi * f1 * t)
        np.savetxt(directory / "a.csv", np.column_stack([t, y]), delimiter=",")
        body = {
            "kind": "ringdown",
            "instrument": "scope",
            "timestamp": "t",
            "data_file": "a.csv",
            "columns": {"x": {"name": "t", "unit": "s"}, "y": {"name": "a", "unit": "arb"}},
        }
    (directory / "a.yaml").write_text(yaml.safe_dump(body), encoding="utf-8")
    return directory / "a.csv"


def test_rin_trace_loader_seeds_the_rin_field(qtbot, tmp_path: Path) -> None:
    """Loading a RIN artifact (by its CSV) seeds the explicit RIN value (R-57v)."""
    window = MainWindow()
    qtbot.addWidget(window)
    source = window.control_panel.system._source
    source.load_rin_artifact(_artifact(tmp_path, "rin_psd"))
    assert float(source._edits["rin_db_hz"].text()) == pytest.approx(-121.5)
    assert "replaces the derived floor" in source._rin_note.text()


def test_profile_loader_seeds_the_curvature_field(qtbot, tmp_path: Path) -> None:
    """Loading a tip-profile artifact seeds R_c from the circle fit (M-17)."""
    window = MainWindow()
    qtbot.addWidget(window)
    reflector = window.control_panel.system._reflector
    reflector.load_profile_artifact(_artifact(tmp_path, "profile"))
    assert float(reflector._rc.text()) == pytest.approx(150.0e-6, rel=0.01)
    assert "one azimuth" in reflector._profile_note.text()


def test_ringdown_loader_seeds_the_q_override(qtbot, tmp_path: Path) -> None:
    """Loading a ring-down artifact seeds the Q override (measured Q wins)."""
    window = MainWindow()
    qtbot.addWidget(window)
    panel = window.control_panel.system
    panel.load_ringdown_artifact(_artifact(tmp_path, "ringdown"))
    assert float(panel._q_total.text()) == pytest.approx(1661.0, rel=0.02)
    assert "overrides the Q(L) model" in panel._q_note.text()
    assert panel.system_payload()["q_total"] == pytest.approx(1661.0, rel=0.02)


def test_loader_rejects_wrong_artifact_kind(qtbot, tmp_path: Path) -> None:
    """A wrong-kind artifact is refused loudly by each loader."""
    window = MainWindow()
    qtbot.addWidget(window)
    ringdown_csv = _artifact(tmp_path, "ringdown")
    with pytest.raises(ValueError, match="not a RIN trace"):
        window.control_panel.system._source.load_rin_artifact(ringdown_csv)
    with pytest.raises(ValueError, match="not a tip profile"):
        window.control_panel.system._reflector.load_profile_artifact(ringdown_csv)


def test_every_tab_carries_inline_help(qtbot) -> None:
    """Each parameter tab exposes the faint ``?`` reference buttons."""
    from PySide6.QtWidgets import QToolButton

    window = MainWindow()
    qtbot.addWidget(window)
    panel = window.control_panel.system
    for index in range(panel.count()):
        page = panel.widget(index)
        marks = [b for b in page.findChildren(QToolButton) if b.text() == "?"]
        assert marks, f"tab {panel.tabText(index)!r} has no help buttons"


def test_wheel_cannot_edit_combos_or_spins(qtbot) -> None:
    """The mouse wheel never changes a combo/spin value (S-13b §4 guard)."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    window = MainWindow()
    qtbot.addWidget(window)
    combo = window.control_panel.system._starting
    spin = window.control_panel._seed
    wheel = QWheelEvent(
        QPointF(5.0, 5.0),
        QPointF(5.0, 5.0),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    combo_before, spin_before = combo.currentIndex(), spin.value()
    QApplication.sendEvent(combo, wheel)
    QApplication.sendEvent(spin, wheel)
    assert combo.currentIndex() == combo_before
    assert spin.value() == spin_before
