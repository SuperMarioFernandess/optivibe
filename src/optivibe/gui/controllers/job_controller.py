"""Job controller: owns the worker thread and re-emits its outcome (task S7 §1).

This generalises the S0 ``RunController`` (09 §9, SW-13) from a single
path-based scenario run to *any* :class:`~optivibe.gui.workers.jobs.Job`
(scenario, report, sweep, Monte-Carlo). The controller isolates the
:class:`~PySide6.QtCore.QThread` plumbing so views deal only with four signals
(``progress`` / ``finished`` / ``failed`` / ``cancelled``).

Lifecycle is the canonical Qt pattern: the worker runs on its own thread; on
completion it asks the thread to quit and schedules its own deletion; once the
thread has actually stopped, the controller clears its references and *then*
re-emits the outcome. **No synchronous** ``QThread.wait()`` is used inside a slot
connected to a worker signal -- ``quit`` is queued behind that slot on the same
thread, so waiting there dead-locks the event loop (the S0 defect, SW-13).
Cancellation is cooperative + "abandon the result": :meth:`cancel` flags the
worker and, when the thread stops, the stashed result is dropped and
``cancelled`` is emitted instead of ``finished`` (the background work is never
force-terminated).
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal

from optivibe.core.logging import get_logger
from optivibe.gui.workers.job_worker import JobWorker
from optivibe.gui.workers.jobs import Job

logger = get_logger(__name__)

__all__ = ["JobController"]


class JobController(QObject):
    """Drive a single :class:`JobWorker` on its own thread.

    The controller re-emits the worker's outcome on the UI thread once the
    background thread has fully stopped, so :meth:`is_running` is guaranteed to
    be ``False`` by the time ``finished`` / ``failed`` / ``cancelled`` fire. Only
    one job may run at a time (:meth:`is_running` guards re-entry).
    """

    #: Coarse status string forwarded from the worker.
    progress = Signal(str)
    #: Re-emitted job result on success.
    finished = Signal(object)
    #: Re-emitted error message on failure.
    failed = Signal(str)
    #: Emitted when a run was cancelled (its result is dropped).
    cancelled = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: JobWorker | None = None
        self._result: object | None = None
        self._error: str | None = None
        self._cancel_requested = False

    def is_running(self) -> bool:
        """Return ``True`` while a job is in progress."""
        return self._thread is not None

    def start(self, job: Job) -> None:
        """Start running ``job`` on a background thread.

        Parameters
        ----------
        job : Job
            The Qt-free unit of work to execute.

        Raises
        ------
        RuntimeError
            If a job is already in progress.
        """
        if self.is_running():
            msg = "a job is already in progress"
            raise RuntimeError(msg)

        thread = QThread()
        worker = JobWorker(job)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._stash_result)
        worker.failed.connect(self._stash_error)
        # Canonical Qt teardown: stop the event loop and schedule deletion; the
        # outcome is re-emitted from _finalize once the thread has stopped.
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._finalize)

        self._thread = thread
        self._worker = worker
        self._result = None
        self._error = None
        self._cancel_requested = False
        logger.debug("controller starting job: %s", job.label)
        thread.start()

    def cancel(self) -> None:
        """Request cancellation of the running job (no-op if idle).

        Flags the worker (cooperative) and marks the run cancelled so its result
        is dropped when the thread stops. The work itself is never force-killed.
        """
        if not self.is_running() or self._worker is None:
            return
        logger.debug("controller cancelling current job")
        self._cancel_requested = True
        self._worker.request_cancel()
        self.progress.emit("cancelling...")

    def _on_progress(self, message: str) -> None:
        """Forward a worker progress message (suppressed once cancelling)."""
        if not self._cancel_requested:
            self.progress.emit(message)

    def _stash_result(self, result: object) -> None:
        """Store the worker's successful result until the thread stops."""
        self._result = result

    def _stash_error(self, message: str) -> None:
        """Store the worker's error message until the thread stops."""
        self._error = message

    def _finalize(self) -> None:
        """Clear references (run is over) and re-emit the stashed outcome."""
        self._thread = None
        self._worker = None
        result, error, cancelled = self._result, self._error, self._cancel_requested
        self._result = None
        self._error = None
        self._cancel_requested = False
        if cancelled:
            self.cancelled.emit()
        elif error is not None:
            self.failed.emit(error)
        elif result is not None:
            self.finished.emit(result)
