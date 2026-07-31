"""Stream controller: owns the live-stream thread and its teardown (O-SW-03).

The streaming sibling of
:class:`~optivibe.gui.controllers.job_controller.JobController`. It follows the
same canonical Qt lifecycle (09 §9), for the same hard-won reasons: the worker
runs on its own thread; on leaving it asks the thread to quit and schedules its
own deletion; only once the thread has actually stopped does the controller
clear its references and re-emit the outcome. **No synchronous**
``QThread.wait()`` inside a slot connected to a worker signal -- ``quit`` is
queued behind that slot on the same thread and waiting there dead-locks the
event loop (the S0 defect, SW-13).

Why a second controller rather than a mode of the first: the two lifecycles
differ in kind. A job has a result and ends by itself; a stream has no result,
ends when the user says so, and emits all the way through. Modelling that as a
flag inside :class:`JobController` would put two lifetimes behind one set of
signals; keeping them apart costs a few lines of duplicated wiring and keeps the
frozen job path free of streaming concerns.

Mutual exclusion between the two families is a **policy of the window**, not an
accident of there being one thread: while the stream runs, the one-shot actions
are disabled and vice versa (owner decision, 2026-07-31) -- so which numbers are
on screen is never ambiguous.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal

from optivibe.core.logging import get_logger
from optivibe.gui.workers.stream import StreamConfig, StreamSource
from optivibe.gui.workers.stream_worker import StreamWorker

logger = get_logger(__name__)

__all__ = ["StreamController"]


class StreamController(QObject):
    """Drive a single :class:`StreamWorker` on its own thread.

    The controller re-emits the outcome on the UI thread once the background
    thread has fully stopped, so :meth:`is_running` is guaranteed to be
    ``False`` by the time ``stopped`` / ``failed`` fire. Only one stream may run
    at a time (:meth:`is_running` guards re-entry).
    """

    #: Forwarded opened :class:`~optivibe.gui.workers.stream.StreamSetup`.
    opened = Signal(object)
    #: Forwarded live frame (a :class:`~optivibe.gui.workers.stream.StreamFrame`).
    frame = Signal(object)
    #: Emitted after the thread has stopped following a normal end.
    stopped = Signal()
    #: Emitted after the thread has stopped following a failure.
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: StreamWorker | None = None
        self._error: str | None = None

    def is_running(self) -> bool:
        """Return ``True`` while a stream is live."""
        return self._thread is not None

    def start(self, source: StreamSource, config: StreamConfig) -> None:
        """Start streaming ``source`` on a background thread.

        Parameters
        ----------
        source : StreamSource
            The Qt-free source to open and replay (opening it -- file I/O, a
            forward run, composition resolution -- happens on that thread).
        config : StreamConfig
            Loop settings.

        Raises
        ------
        RuntimeError
            If a stream is already running.
        """
        if self.is_running():
            msg = "a stream is already running"
            raise RuntimeError(msg)

        thread = QThread()
        worker = StreamWorker(source, config)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.opened.connect(self.opened)
        worker.frame.connect(self.frame)
        worker.failed.connect(self._stash_error)
        # Canonical Qt teardown (see the module docstring).
        worker.stopped.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.stopped.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._finalize)

        self._thread = thread
        self._worker = worker
        self._error = None
        logger.debug("controller starting stream: %s", source.label)
        thread.start()

    def stop(self) -> None:
        """Ask the running stream to stop (no-op if idle).

        Cooperative: the loop polls the flag once per block and unwinds on its
        own. The outcome arrives as ``stopped`` when the thread has ended.
        """
        if not self.is_running() or self._worker is None:
            return
        logger.debug("controller stopping current stream")
        self._worker.request_stop()

    def _stash_error(self, message: str) -> None:
        """Store the worker's error message until the thread stops."""
        self._error = message

    def _finalize(self) -> None:
        """Clear references (the stream is over) and re-emit the outcome."""
        self._thread = None
        self._worker = None
        error = self._error
        self._error = None
        if error is not None:
            self.failed.emit(error)
        else:
            self.stopped.emit()
