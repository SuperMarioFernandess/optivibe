"""GUI tests for the DSP comparison bench (task S-22; pytest-qt, offscreen).

Covers the four promises the bench makes on screen: the default chain is still
bit-identical to an ordinary run (the acceptance check of W-1), the experiment
panel and its two mirrors on *Physics layers* can never disagree (owner
decision R-1), the comparison itself runs off the UI thread (invariant S7,
proven on :func:`threading.get_ident` per the 2026-07-31 reinforcement of
18 §5), and the exported numbers carry their verdict (rule 1 / discipline S-13).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("pytestqt")

pytestmark = pytest.mark.gui

from optivibe.analysis.compare import (  # noqa: E402
    CHAIN_APPLICABILITY,
    DEFAULT_CHAIN,
    EXPERIMENT_FIELDS,
    ChainSpec,
    ComparisonResult,
)
from optivibe.core.config.models import DspOptions  # noqa: E402
from optivibe.gui.controllers.job_controller import JobController  # noqa: E402
from optivibe.gui.controllers.scenario_builder import build_scenario_config  # noqa: E402
from optivibe.gui.main_window import MainWindow  # noqa: E402
from optivibe.gui.widgets.dsp_controls import DspControls  # noqa: E402
from optivibe.gui.workers.jobs import CancelFn, CompareJob, ProgressFn  # noqa: E402


# --------------------------------------------------------------------------- #
# W-1: the panel shows the existing choices, and the default is untouched
# --------------------------------------------------------------------------- #
def test_panel_exposes_every_experiment_field_with_its_applicability(qtbot) -> None:
    """Each exposed option has a row, and each row states which path it reaches."""
    panel = DspControls()
    qtbot.addWidget(panel)
    assert set(panel._labels) == set(EXPERIMENT_FIELDS)
    for field, label in panel._labels.items():
        text = label.text()
        assert "[" in text and "]" in text, f"{field} has no applicability tag"


def test_untouched_panel_is_the_verified_default_chain(qtbot) -> None:
    """The panel starts on ``DspOptions()`` -- bit-identity of a default run."""
    panel = DspControls()
    qtbot.addWidget(panel)
    assert panel.dsp_options() == DEFAULT_CHAIN
    assert panel.status() == "verified"
    assert "verified" in panel.status_text()


def test_default_gui_scenario_carries_the_default_chain(qtbot) -> None:
    """A scenario assembled from an untouched panel resolves to ``DspOptions()``.

    This is the acceptance check of W-1: exposing the knobs must not shift the
    default run by a single field (the hard-coded ``spectrum_method`` / ``window``
    / ``sensitivity_freq`` of the old payload are gone, and the values they
    carried now come from the model, which has the same defaults).
    """
    window = MainWindow()
    qtbot.addWidget(window)
    scenario = build_scenario_config(window.control_panel.scenario_payload())
    assert scenario.dsp == DspOptions()


def test_deviation_flips_the_badge_and_reset_restores_it(qtbot) -> None:
    """Any deviation is experimental, spelled out, and one click away from undone."""
    panel = DspControls()
    qtbot.addWidget(panel)
    with qtbot.waitSignal(panel.changed):
        panel._integrator.setCurrentText("time")
    assert panel.status() == "experimental"
    text = panel.status_text()
    assert "integrator" in text
    assert "time" in text
    with qtbot.waitSignal(panel.changed):
        panel.reset_to_default()
    assert panel.dsp_options() == DEFAULT_CHAIN
    assert panel.status() == "verified"


def test_auto_sentinels_mean_none_not_zero(qtbot) -> None:
    """Empty numeric fields read as 'not set' and stay ``None`` in the model."""
    panel = DspControls()
    qtbot.addWidget(panel)
    options = panel.dsp_options()
    assert options.f_hp_hz is None
    assert options.f_c_stream is None
    assert options.welch_nperseg is None
    panel._f_hp_hz.setValue(5.0)
    assert panel.dsp_options().f_hp_hz == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# R-1: two places, one value
# --------------------------------------------------------------------------- #
def test_physics_layers_combos_mirror_the_experiment_panel(qtbot) -> None:
    """The mirrors track the experiment panel and are tracked by it, both ways."""
    window = MainWindow()
    qtbot.addWidget(window)
    panel = window.control_panel
    experiment = panel.experiment

    experiment._integrator.setCurrentText("time")
    experiment._sensitivity_model.setCurrentText("operating_point")
    assert panel._integrator.currentText() == "time"
    assert panel._sensitivity.currentText() == "operating_point"

    panel._integrator.setCurrentText("frequency")
    assert experiment.dsp_options().integrator == "frequency"
    panel._sensitivity.setCurrentText("static")
    assert experiment.dsp_options().sensitivity_model == "static"

    # And the payload agrees with both of them (one truth: the config).
    scenario = build_scenario_config(panel.scenario_payload())
    assert scenario.dsp.integrator == "frequency"
    assert scenario.dsp.sensitivity_model == "static"


def test_payload_no_longer_hard_codes_the_spectral_options(qtbot) -> None:
    """Options the old payload pinned to literals now come from the panel."""
    window = MainWindow()
    qtbot.addWidget(window)
    panel = window.control_panel
    panel.experiment._spectrum_method.setCurrentText("welch")
    panel.experiment._window.setCurrentText("flattop")
    scenario = build_scenario_config(panel.scenario_payload())
    assert scenario.dsp.spectrum_method == "welch"
    assert scenario.dsp.window == "flattop"


def test_scenario_payload_round_trips_through_restore(qtbot) -> None:
    """A language rebuild must not quietly reset the experiment (SW-65 path)."""
    window = MainWindow()
    qtbot.addWidget(window)
    panel = window.control_panel
    panel.experiment.set_dsp_options(DspOptions(integrator="time", window="nuttall"))
    payload = panel.scenario_payload()
    panel.experiment.reset_to_default()
    panel.restore_scenario(payload)
    assert panel.dsp_options().integrator == "time"
    assert panel.dsp_options().window == "nuttall"
    assert panel._integrator.currentText() == "time"


# --------------------------------------------------------------------------- #
# The mandatory invariant (S7) for the comparison job
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _ThreadProbeCompareJob:
    """A real :class:`CompareJob` that records the thread it ran on."""

    inner: CompareJob
    seen: list[int]
    label: str = "thread-probe compare"

    def run(self, *, progress: ProgressFn, is_cancelled: CancelFn) -> object:
        """Record the current thread, then run the real comparison."""
        self.seen.append(threading.get_ident())
        return self.inner.run(progress=progress, is_cancelled=is_cancelled)


def test_comparison_runs_off_the_gui_thread(qtbot) -> None:
    """The comparison walks the samples twice -- so it runs on the worker thread.

    The probe records :func:`threading.get_ident`, not the address of a Qt
    wrapper: an ephemeral wrapper's address is reused and would let a broken
    invariant pass (18 §5, reinforcement of 2026-07-31).
    """
    window = MainWindow()
    qtbot.addWidget(window)
    scenario = build_scenario_config(window.control_panel.scenario_payload())
    inner = CompareJob(
        chains=(
            ChainSpec(name="A"),
            ChainSpec(name="B", dsp=DspOptions(integrator="time")),
        ),
        scenario=scenario,
    )
    seen: list[int] = []
    results: list[object] = []
    controller = JobController()
    controller.finished.connect(results.append)
    with qtbot.waitSignal(controller.finished, timeout=60000):
        controller.start(_ThreadProbeCompareJob(inner=inner, seen=seen))
    assert seen and seen[0] != threading.get_ident()
    assert isinstance(results[0], ComparisonResult)
    assert len(results[0].chains) == 2


# --------------------------------------------------------------------------- #
# W-2 end to end, and the provenance of what leaves the app
# --------------------------------------------------------------------------- #
def test_compare_tab_shows_two_chains_and_their_verdicts(qtbot) -> None:
    """The button leads to an overlay, a filled table and both verdicts on screen."""
    window = MainWindow()
    qtbot.addWidget(window)
    window._compare.set_chain_b_options(DspOptions(integrator="time"))
    with qtbot.waitSignal(window._controller.finished, timeout=60000):
        window._compare.compare_requested.emit()
    result = window._compare.result()
    assert isinstance(result, ComparisonResult)
    assert [outcome.status for outcome in result.chains] == ["verified", "experimental"]
    assert window._compare._table.rowCount() == len(result.rows)
    assert window._compare._table.columnCount() >= 4
    status = window._compare.status_text()
    assert "verified" in status
    assert "experimental" in status
    assert window._tabs.currentWidget() is window._compare


def test_run_export_carries_the_chain_provenance(qtbot, tmp_path: Path) -> None:
    """An exported run says which chain produced it and whether it is verified."""
    window = MainWindow()
    qtbot.addWidget(window)
    window.control_panel.experiment.set_dsp_options(DspOptions(integrator="time"))
    with qtbot.waitSignal(window._controller.finished, timeout=60000):
        window.run_button.click()
    written = window.export_to(tmp_path)
    assert (tmp_path / "run_provenance.yaml").is_file()
    text = (tmp_path / "run_provenance.yaml").read_text()
    assert "status: experimental" in text
    assert "integrator" in text
    with np.load(next(p for p in written if p.suffix == ".npz")) as data:
        assert str(data["chain_status"]) == "experimental"
        provenance = json.loads(str(data["chain_provenance_json"]))
    assert provenance["deviations_from_default"][0]["field"] == "integrator"
    # The pre-existing keys are untouched: older readers keep working.
    with np.load(tmp_path / "run_result.npz") as data:
        assert {"a_input", "a_recovered", "v_recovered", "x_recovered", "fs"} <= set(data.files)


def test_comparison_export_writes_table_and_provenance(qtbot, tmp_path: Path) -> None:
    """An exported comparison carries every chain's verdict and the diff table."""
    window = MainWindow()
    qtbot.addWidget(window)
    window._compare.set_chain_b_options(DspOptions(spectrum_method="welch"))
    with qtbot.waitSignal(window._controller.finished, timeout=60000):
        window._compare.compare_requested.emit()
    written = window.export_to(tmp_path)
    assert (tmp_path / "compare_table.txt").is_file()
    provenance = (tmp_path / "compare_provenance.yaml").read_text()
    assert "dsp_chain_comparison" in provenance
    assert provenance.count("status:") >= 3  # the comparison plus one per chain
    assert any(path.suffix == ".npz" for path in written)


def test_streaming_locks_the_comparison_trigger(qtbot) -> None:
    """Live mode and the comparison are mutually exclusive, like the other jobs."""
    window = MainWindow()
    qtbot.addWidget(window)
    window._set_streaming(True)
    assert not window._compare._run.isEnabled()
    window._set_streaming(False)
    assert window._compare._run.isEnabled()


def test_applicability_labels_are_translated(qtbot) -> None:
    """The batch/stream tag is catalog text, so it localizes with the rest."""
    from optivibe.gui.i18n import set_language

    panel = DspControls()
    qtbot.addWidget(panel)
    english = panel._labels["integrator"].text()
    set_language("ru")
    try:
        panel.retranslate()
        russian = panel._labels["integrator"].text()
    finally:
        set_language("en")
        panel.retranslate()
    assert english != russian
    assert CHAIN_APPLICABILITY["integrator"] == "batch"
