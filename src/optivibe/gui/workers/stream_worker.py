"""A ``QObject`` worker that runs the live stream off the UI thread (O-SW-03).

The streaming counterpart of
:class:`~optivibe.gui.workers.job_worker.JobWorker`, and deliberately a
*separate* type: a job runs once and emits exactly one ``finished``/``failed``,
whereas this one emits ``frame`` for as long as the user leaves it running. The
job contract (09 §9) is therefore untouched.

Like the job worker it carries no widgets and never touches the GUI -- it only
emits signals, which Qt delivers to the UI thread through queued connections.
Stopping is cooperative: :meth:`request_stop` sets a flag the loop polls once
per block, so the thread always unwinds on its own (a forced ``QThread``
termination is never used).
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from optivibe.core.logging import get_logger
from optivibe.gui.workers.stream import StreamConfig, StreamFrame, StreamSource, run_stream

logger = get_logger(__name__)

__all__ = ["StreamWorker"]


class StreamWorker(QObject):
    """Open a :class:`~optivibe.gui.workers.stream.StreamSource` and stream it.

    Parameters
    ----------
    source : StreamSource
        The Qt-free source to open and replay.
    config : StreamConfig
        Loop settings (rate, pace, looping, window).

    Notes
    -----
    Move an instance to a :class:`~PySide6.QtCore.QThread` and connect the
    thread's ``started`` signal to :meth:`run`. ``opened`` fires once (the
    source resolved into a setup), ``frame`` repeatedly, and exactly one of
    ``stopped`` / ``failed`` ends the run.
    """

    #: Emitted once with the opened :class:`~optivibe.gui.workers.stream.StreamSetup`.
    opened = Signal(object)
    #: Emitted with each :class:`~optivibe.gui.workers.stream.StreamFrame`.
    frame = Signal(object)
    #: Emitted once the loop has left (record exhausted or stop honoured).
    stopped = Signal()
    #: Emitted with a human-readable message on failure.
    failed = Signal(str)

    def __init__(self, source: StreamSource, config: StreamConfig) -> None:
        super().__init__()
        self._source = source
        self._config = config
        self._stopped = False

    def request_stop(self) -> None:
        """Ask the loop to leave after the current block (safe from the UI)."""
        self._stopped = True

    def is_stopped(self) -> bool:
        """Return ``True`` once :meth:`request_stop` has been called."""
        return self._stopped

    def _emit_frame(self, frame: StreamFrame) -> None:
        """Hand one frame to the UI thread (queued)."""
        self.frame.emit(frame)

    @Slot()
    def run(self) -> None:
        """Open the source, run the loop, then emit ``stopped`` or ``failed``."""
        logger.debug("stream worker started: %s", self._source.label)
        try:
            setup = self._source.open()
            self.opened.emit(setup)
            run_stream(
                setup,
                self._config,
                on_frame=self._emit_frame,
                is_stopped=self.is_stopped,
            )
        except (FileNotFoundError, ValueError, KeyError, RuntimeError, OSError) as exc:
            logger.error("stream %r failed: %s", self._source.label, exc)
            self.failed.emit(str(exc))
            return
        except Exception as exc:
            logger.exception("stream %r raised an unexpected error", self._source.label)
            self.failed.emit(f"unexpected error: {exc}")
            return
        self.stopped.emit()
