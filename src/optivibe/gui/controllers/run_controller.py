"""Run controller: owns the worker thread and re-emits its results.

The controller isolates the :class:`~PySide6.QtCore.QThread` plumbing (create,
move worker, connect, start, clean up) so the main window only deals with two
signals (``finished`` / ``failed``). This keeps the view thin and the threading
contract testable in isolation (09 §9).

Lifecycle follows the canonical Qt pattern: the worker runs on its own thread;
on completion it asks the thread to quit and schedules its own deletion; once the
thread has actually stopped, the controller clears its references and *then*
re-emits the outcome. No synchronous ``QThread.wait()`` is used (waiting inside a
slot that is queued ahead of ``quit`` would dead-lock the event loop).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from optivibe.core.logging import get_logger
from optivibe.gui.workers.pipeline_worker import PipelineWorker

logger = get_logger(__name__)

__all__ = ["RunController"]


class RunController(QObject):
    """Drive a single :class:`PipelineWorker` on its own thread.

    The controller re-emits the worker's outcome on the UI thread once the
    background thread has fully stopped, so :meth:`is_running` is guaranteed to
    be ``False`` by the time ``finished`` / ``failed`` fire. Only one run may be
    active at a time (:meth:`is_running` guards re-entry).
    """

    #: Re-emitted RunArtifacts on success.
    finished = Signal(object)
    #: Re-emitted error message on failure.
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._result: object | None = None
        self._error: str | None = None

    def is_running(self) -> bool:
        """Return ``True`` while a run is in progress."""
        return self._thread is not None

    def start(self, scenario_path: Path | str, config_dir: Path | None = None) -> None:
        """Start running ``scenario_path`` on a background thread.

        Parameters
        ----------
        scenario_path : pathlib.Path or str
            Scenario YAML to run.
        config_dir : pathlib.Path or None, optional
            Override for the ``configs/`` directory.

        Raises
        ------
        RuntimeError
            If a run is already in progress.
        """
        if self.is_running():
            msg = "a run is already in progress"
            raise RuntimeError(msg)

        thread = QThread()
        worker = PipelineWorker(scenario_path, config_dir)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._stash_result)
        worker.failed.connect(self._stash_error)
        # Stop the thread's event loop and schedule object deletion (canonical
        # Qt pattern); the outcome is re-emitted from _finalize once stopped.
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
        logger.debug("controller starting thread for %s", scenario_path)
        thread.start()

    def _stash_result(self, artifacts: object) -> None:
        """Store the worker's successful result until the thread stops."""
        self._result = artifacts

    def _stash_error(self, message: str) -> None:
        """Store the worker's error message until the thread stops."""
        self._error = message

    def _finalize(self) -> None:
        """Clear references (run is over) and re-emit the stashed outcome."""
        self._thread = None
        self._worker = None
        result, error = self._result, self._error
        self._result = None
        self._error = None
        if error is not None:
            self.failed.emit(error)
        elif result is not None:
            self.finished.emit(result)
