"""A ``QObject`` worker that runs a scenario off the UI thread.

The worker carries no Qt widgets and never touches the GUI directly: it only
emits signals (``finished`` / ``failed``), which the controller delivers back to
the UI thread via Qt's queued connections. This is the mandatory threading
boundary of architecture 09 §9 (the long-running pipeline must not block the
event loop).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from optivibe.core.logging import get_logger
from optivibe.pipeline import RunArtifacts, run_scenario

logger = get_logger(__name__)

__all__ = ["PipelineWorker"]


class PipelineWorker(QObject):
    """Run :func:`optivibe.pipeline.run_scenario` in a background thread.

    Parameters
    ----------
    scenario_path : pathlib.Path or str
        Scenario YAML to run.
    config_dir : pathlib.Path or None, optional
        Override for the ``configs/`` directory (variant presets).

    Notes
    -----
    Move an instance to a :class:`~PySide6.QtCore.QThread` and connect the
    thread's ``started`` signal to :meth:`run`. Exactly one of ``finished`` or
    ``failed`` is emitted per run.
    """

    #: Emitted with the :class:`~optivibe.pipeline.RunArtifacts` on success.
    finished = Signal(object)
    #: Emitted with a human-readable message on failure.
    failed = Signal(str)

    def __init__(self, scenario_path: Path | str, config_dir: Path | None = None) -> None:
        super().__init__()
        self._scenario_path = Path(scenario_path)
        self._config_dir = config_dir

    @Slot()
    def run(self) -> None:
        """Run the scenario and emit ``finished`` or ``failed``."""
        logger.debug("worker started for %s", self._scenario_path)
        try:
            artifacts: RunArtifacts = run_scenario(self._scenario_path, config_dir=self._config_dir)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("worker failed: %s", exc)
            self.failed.emit(str(exc))
            return
        self.finished.emit(artifacts)
