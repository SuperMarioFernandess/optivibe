"""A generic ``QObject`` worker that runs a :class:`Job` off the UI thread.

This is the mandatory threading boundary of architecture 09 §9 generalised from
the S0 ``PipelineWorker`` to *any* job (scenario, report, sweep, Monte-Carlo;
task S7 §1). The worker carries no widgets and never touches the GUI: it only
emits ``progress`` / ``finished`` / ``failed`` signals, which Qt delivers back
to the UI thread through the controller's queued connections. Cancellation is
cooperative -- :meth:`request_cancel` sets a flag the job may poll; a job that
cannot stop mid-call still runs to completion, and the controller simply drops a
late result (a forced ``QThread`` termination is never used).
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from optivibe.core.logging import get_logger
from optivibe.gui.workers.jobs import Job

logger = get_logger(__name__)

__all__ = ["JobWorker"]


class JobWorker(QObject):
    """Run a :class:`~optivibe.gui.workers.jobs.Job` in a background thread.

    Parameters
    ----------
    job : Job
        The unit of work to execute (Qt-free; see
        :mod:`optivibe.gui.workers.jobs`).

    Notes
    -----
    Move an instance to a :class:`~PySide6.QtCore.QThread` and connect the
    thread's ``started`` signal to :meth:`run`. Exactly one of ``finished`` or
    ``failed`` is emitted per run; ``progress`` may fire any number of times
    before it.
    """

    #: Emitted with a coarse status string while the job runs.
    progress = Signal(str)
    #: Emitted with the job result object on success.
    finished = Signal(object)
    #: Emitted with a human-readable message on failure.
    failed = Signal(str)

    def __init__(self, job: Job) -> None:
        super().__init__()
        self._job = job
        self._cancelled = False

    def request_cancel(self) -> None:
        """Ask the running job to stop (cooperative; safe from the UI thread)."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """Return ``True`` once :meth:`request_cancel` has been called."""
        return self._cancelled

    def _report(self, message: str) -> None:
        """Emit a progress message (queued to the UI thread)."""
        self.progress.emit(message)

    @Slot()
    def run(self) -> None:
        """Execute the job and emit ``finished`` or ``failed``."""
        logger.debug("worker started: %s", self._job.label)
        try:
            result = self._job.run(progress=self._report, is_cancelled=self.is_cancelled)
        except (FileNotFoundError, ValueError, KeyError, RuntimeError, OSError) as exc:
            logger.error("job %r failed: %s", self._job.label, exc)
            self.failed.emit(str(exc))
            return
        except Exception as exc:
            logger.exception("job %r raised an unexpected error", self._job.label)
            self.failed.emit(f"unexpected error: {exc}")
            return
        self.finished.emit(result)
