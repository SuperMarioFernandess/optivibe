"""Back-compatible scenario controller (thin adapter over :class:`JobController`).

``RunController`` is the S0 reference path (09 §9, SW-13): "run one scenario from
a path, off the UI thread, and re-emit ``finished`` / ``failed``". S7 generalises
the threading engine into :class:`~optivibe.gui.controllers.job_controller.JobController`
(any job, plus progress and cancellation); this class is kept as a thin adapter
so the published S0 contract -- ``start(scenario_path, config_dir)`` with
``finished(RunArtifacts)`` / ``failed(str)`` and ``is_running()`` -- and its
smoke test stay valid. It simply wraps the request in a
:class:`~optivibe.gui.workers.jobs.ScenarioJob` and forwards the inner signals.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from optivibe.core.logging import get_logger
from optivibe.gui.controllers.job_controller import JobController
from optivibe.gui.workers.jobs import ScenarioJob

logger = get_logger(__name__)

__all__ = ["RunController"]


class RunController(QObject):
    """Run a single scenario file off the UI thread (S0 contract).

    A thin facade over :class:`JobController`: it owns one inner controller, runs
    a :class:`ScenarioJob`, and re-emits the outcome. ``is_running()`` is
    guaranteed ``False`` by the time ``finished`` / ``failed`` fire (canonical
    teardown in the inner controller). Only one run may be active at a time.
    """

    #: Re-emitted RunArtifacts on success.
    finished = Signal(object)
    #: Re-emitted error message on failure.
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._inner = JobController(self)
        # Signal-to-signal forwarding (synchronous: same thread / direct).
        self._inner.finished.connect(self.finished)
        self._inner.failed.connect(self.failed)

    def is_running(self) -> bool:
        """Return ``True`` while a run is in progress."""
        return self._inner.is_running()

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
        self._inner.start(ScenarioJob(source=scenario_path, config_dir=config_dir))
