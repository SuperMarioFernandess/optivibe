"""GUI tests for the real-time oscilloscope (task O-SW-03; pytest-qt, offscreen).

Skipped automatically without the ``gui`` extra or a Qt platform. Covers the
mandatory threading invariant for the *streaming* worker family, the two
provenance obligations that make the screen readable (warm-up state and the
drop counter), the render path of the live panels, and the mutual exclusion
between the live stream and the one-shot jobs.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("pytestqt")

pytestmark = pytest.mark.gui

from optivibe.core.config.loader import load_constants, load_variant  # noqa: E402
from optivibe.core.config.models import Constants, DspOptions, VariantConfig  # noqa: E402
from optivibe.core.types import DetectorOutput  # noqa: E402
from optivibe.dsp.sensitivity import target_sensitivity  # noqa: E402
from optivibe.gui.controllers.stream_controller import StreamController  # noqa: E402
from optivibe.gui.main_window import MainWindow  # noqa: E402
from optivibe.gui.widgets.live_controls import SOURCE_RECORD, LiveControls  # noqa: E402
from optivibe.gui.widgets.live_view import LiveView  # noqa: E402
from optivibe.gui.workers.stream import (  # noqa: E402
    StreamConfig,
    StreamFrame,
    StreamSetup,
    run_stream,
)

FS = 10_000.0


def _tone_setup(config_dir: Path, *, seconds: float = 1.0) -> StreamSetup:
    """A ready setup over a synthetic 200 Hz record (variant B)."""
    constants: Constants = load_constants(config_dir / "constants.yaml")
    variant: VariantConfig = load_variant("B", config_dir=config_dir)
    n = int(seconds * FS)
    t = np.arange(n, dtype=np.float64) / FS
    a = np.cos(2.0 * math.pi * 200.0 * t)
    samples = 1.0e-3 + target_sensitivity(variant, constants) * a
    detector = DetectorOutput(samples=samples, fs=FS, dc_level=1.0e-3, units="A")
    return StreamSetup(detector=detector, variant=variant, options=DspOptions(), label="probe tone")


def _one_frame(config_dir: Path) -> StreamFrame:
    """Run the Qt-free loop once and return its final frame."""
    frames: list[StreamFrame] = []
    run_stream(
        _tone_setup(config_dir),
        StreamConfig(rate_hz=10.0, speed="max", loop=False, window_s=0.2),
        on_frame=frames.append,
        is_stopped=lambda: False,
    )
    return frames[-1]


class _ThreadProbeSource:
    """A stream source that records the thread it is opened (and run) on."""

    def __init__(self, config_dir: Path) -> None:
        self._config_dir = config_dir
        self.worker_thread: int | None = None

    @property
    def label(self) -> str:
        """Short human-readable source name."""
        return "thread-probe stream"

    def open(self) -> StreamSetup:
        """Record the current thread, then hand over a tiny record."""
        self.worker_thread = threading.get_ident()
        return _tone_setup(self._config_dir, seconds=0.5)


class _FailingSource:
    """A stream source that cannot be opened."""

    @property
    def label(self) -> str:
        """Short human-readable source name."""
        return "broken source"

    def open(self) -> StreamSetup:
        """Fail the way a missing/invalid record would."""
        msg = "record is not readable"
        raise ValueError(msg)


# --------------------------------------------------------------------------- #
# The mandatory invariant (S7) for the streaming worker family.
# --------------------------------------------------------------------------- #
def test_stream_loop_runs_off_the_gui_thread(qtbot, config_dir: Path) -> None:
    """The streaming loop executes on a *different* thread than the GUI.

    The counterpart of ``test_computation_runs_off_the_gui_thread`` for the
    second worker family: the source is opened and the loop driven inside one
    call stack in :meth:`StreamWorker.run`, so recording the thread at open
    time identifies the loop's thread. The identity is
    :func:`threading.get_ident` -- ``id(QThread.currentThread())`` is the
    address of an ephemeral wrapper and aliases across threads (see the note in
    ``tests/test_gui_s7.py``).
    """
    controller = StreamController()
    probe = _ThreadProbeSource(config_dir)
    frames: list[object] = []
    controller.frame.connect(frames.append)
    with qtbot.waitSignal(controller.stopped, timeout=20000):
        controller.start(probe, StreamConfig(rate_hz=20.0, speed="max", loop=False))
    assert probe.worker_thread is not None
    assert probe.worker_thread != threading.get_ident()
    assert frames  # the loop really ran and delivered snapshots to the UI thread
    assert not controller.is_running()


def test_stream_controller_reports_a_failed_source(qtbot) -> None:
    """A source that cannot be opened surfaces as ``failed``, and the thread ends."""
    controller = StreamController()
    errors: list[str] = []
    controller.failed.connect(errors.append)
    with qtbot.waitSignal(controller.failed, timeout=20000):
        controller.start(_FailingSource(), StreamConfig())
    assert errors
    assert "not readable" in errors[0]
    assert not controller.is_running()


def test_stream_controller_refuses_a_second_stream(qtbot, config_dir: Path) -> None:
    """Only one stream at a time (re-entry raises, as for jobs)."""
    controller = StreamController()
    probe = _ThreadProbeSource(config_dir)
    with qtbot.waitSignal(controller.stopped, timeout=20000):
        controller.start(probe, StreamConfig(rate_hz=20.0, speed="max", loop=True))
        assert controller.is_running()
        with pytest.raises(RuntimeError, match="already running"):
            controller.start(probe, StreamConfig())
        controller.stop()


# --------------------------------------------------------------------------- #
# Provenance on screen (decision 13 §4.5): warm-up and dropped samples.
# --------------------------------------------------------------------------- #
def test_live_controls_show_warmup_and_drop_provenance(qtbot, config_dir: Path) -> None:
    """The strip states the warm-up condition and the drop count, always."""
    controls = LiveControls()
    qtbot.addWidget(controls)
    frame = _one_frame(config_dir)

    warming = StreamFrame(
        result=frame.result,
        warmed=False,
        dropped_samples=0,
        n_samples=frame.n_samples,
        elapsed_s=0.4,
        loops=0,
        seam=False,
        paced=True,
        source_label="probe tone",
    )
    controls.show_frame(warming)
    text = controls.provenance_text()
    assert "WARMING UP" in text  # numbers are not final and the screen says so
    assert "probe tone" in text
    assert "dropped: 0" in text

    dropped = StreamFrame(
        result=frame.result,
        warmed=True,
        dropped_samples=512,
        n_samples=frame.n_samples,
        elapsed_s=1.0,
        loops=2,
        seam=True,
        paced=True,
        source_label="probe tone",
    )
    controls.show_frame(dropped)
    text = controls.provenance_text()
    assert "dropped: 512" in text
    assert "warmed up" in text
    assert "loops: 2" in text
    assert "LOOP SEAM" in text  # the splice is announced, not smoothed over


def test_live_controls_mark_the_drop_counter_undefined_without_a_clock(
    qtbot, config_dir: Path
) -> None:
    """Accelerated replay shows "n/a", never a reassuring zero."""
    controls = LiveControls()
    qtbot.addWidget(controls)
    frame = _one_frame(config_dir)
    assert frame.dropped_samples is None  # the loop itself reports it as undefined
    controls.show_frame(frame)
    text = controls.provenance_text()
    assert "n/a" in text
    assert "dropped: 0" not in text


def test_live_controls_expose_the_settings_and_freeze_them_while_running(qtbot) -> None:
    """The minimal control set, frozen for the duration of a stream."""
    controls = LiveControls()
    qtbot.addWidget(controls)
    assert controls.rate_hz() == pytest.approx(10.0)  # reference profile N = 10/s
    assert controls.speed() == "realtime"
    assert controls.loop_enabled() is False
    controls.set_running(True)
    assert not controls._source.isEnabled()
    assert not controls._rate.isEnabled()
    controls.set_running(False)
    assert controls._source.isEnabled()


# --------------------------------------------------------------------------- #
# Render path.
# --------------------------------------------------------------------------- #
def test_live_view_renders_a_stream_frame(qtbot, config_dir: Path) -> None:
    """The panels draw the streamed traces; what does not apply stays blank."""
    view = LiveView()
    qtbot.addWidget(view)
    frame = _one_frame(config_dir)
    view.begin_stream()
    view.show_stream_frame(frame)

    recovered = view._accel_rec.getData()[0]
    assert recovered is not None and len(recovered) > 0
    # A live stream has no truth series aligned to its window, and the raw
    # capture is not part of the snapshot: both stay empty rather than stale.
    assert view._accel_true.getData()[0] is None or len(view._accel_true.getData()[0]) == 0
    assert view._det.getData()[0] is None or len(view._det.getData()[0]) == 0
    assert len(view._spec.getData()[0]) > 0
    assert "n/a" in view.controls.provenance_text()


def test_live_view_trace_stays_bounded_over_a_long_stream(qtbot, config_dir: Path) -> None:
    """What reaches the UI thread is the window, not the whole stream.

    This is the structural half of the S7 criterion: the drawing cost may not
    grow with how long the stream has been running (SW-70). The frame carries a
    trace of the configured window, so it cannot.
    """
    frames: list[StreamFrame] = []
    run_stream(
        _tone_setup(config_dir, seconds=1.0),
        StreamConfig(rate_hz=50.0, speed="max", loop=True, window_s=0.2),
        on_frame=frames.append,
        is_stopped=lambda: len(frames) >= 40,
    )
    window = int(FS * 0.2)
    assert frames[-1].n_samples > 5 * FS  # several records went through
    assert {frame.result.a.size for frame in frames[5:]} == {window}


# --------------------------------------------------------------------------- #
# Mutual exclusion with the one-shot jobs (owner decision, 2026-07-31).
# --------------------------------------------------------------------------- #
def test_live_stream_locks_the_one_shot_actions(qtbot) -> None:
    """While the stream runs, Run/Report are locked -- and released after."""
    window = MainWindow()
    qtbot.addWidget(window)
    controls = window._live.controls
    controls._speed.setCurrentIndex(1)  # as fast as possible
    controls._loop.setCurrentIndex(1)  # keep running until stopped
    with qtbot.waitSignal(window._stream.frame, timeout=30000):
        controls._start.click()
    assert window._stream.is_running()
    assert not window.run_button.isEnabled()
    assert not window._report_button.isEnabled()
    with qtbot.waitSignal(window._stream.stopped, timeout=30000):
        controls._start.click()
    assert not window._stream.is_running()
    assert window.run_button.isEnabled()
    assert window._report_button.isEnabled()


def test_synthetic_stream_attaches_the_expected_peak_overlay(qtbot) -> None:
    """A synthetic source knows its scenario, so the predicted peaks are live.

    The prediction is assembled on the UI thread on purpose: it is O(1) over the
    configuration and cannot read a time series (SW-70). A file record carries
    no scenario and therefore gets no overlay rather than a guessed one.
    """
    window = MainWindow()
    qtbot.addWidget(window)
    controls = window._live.controls
    controls._speed.setCurrentIndex(1)
    with qtbot.waitSignal(window._stream.stopped, timeout=30000):
        controls._start.click()
    assert window._live._expected is not None
    assert window._live._expected.peaks


def test_record_source_without_a_spec_is_reported(qtbot) -> None:
    """Choosing the record source without a spec explains itself and starts nothing."""
    window = MainWindow()
    qtbot.addWidget(window)
    controls = window._live.controls
    controls._source.setCurrentIndex(1)
    assert controls.source_kind() == SOURCE_RECORD
    controls._start.click()
    assert not window._stream.is_running()
    assert "analyze spec" in window.statusBar().currentMessage()
