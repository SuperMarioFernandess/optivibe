"""Qt-free *jobs* that run the core off the UI thread (task S7 §1).

A :class:`Job` is a tiny, **Qt-free** unit of work: it calls
:mod:`optivibe.pipeline` / :mod:`optivibe.analysis` and returns a result object.
Keeping the jobs free of Qt means the heavy core calls are unit-testable without
a display (10 §10) and that *all* physics/DSP lives in the core -- the GUI only
selects a job and renders its result (architecture 09 §9; the GUI introduces no
new physical quantity).

The job runner (:class:`~optivibe.gui.workers.job_worker.JobWorker`) hands every
:meth:`Job.run` two callbacks: ``progress`` (a status string for the UI) and
``is_cancelled`` (a cooperative cancel poll). Long single-call analyses
(``run_sweep`` / ``run_monte_carlo``) cannot be interrupted mid-call without
changing the frozen core, so cancellation there is "abandon the result": the
controller drops a late result and frees the UI immediately (never a forced
``QThread`` termination, which is unsafe). The scenario jobs *do* poll between
the forward and inverse passes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from optivibe.analysis import (
    ErrorBudget,
    MonteCarloResult,
    MonteCarloSpec,
    NeaBudget,
    SweepResult,
    SweepSpec,
    nea_budget,
    run_monte_carlo,
    run_sweep,
    truth_vs_recovery,
)
from optivibe.core.config.loader import default_config_dir, load_scenario, load_variant
from optivibe.core.config.models import ScenarioConfig, VariantConfig
from optivibe.core.config.subsystems import SystemConfig
from optivibe.core.logging import get_logger
from optivibe.gui.controllers.system_builder import resolve_system_variant
from optivibe.pipeline import Pipeline, RunArtifacts

logger = get_logger(__name__)

#: A status reporter the worker wires to a Qt signal (queued to the UI thread).
ProgressFn = Callable[[str], None]
#: A cooperative cancel poll the worker exposes to the job.
CancelFn = Callable[[], bool]

__all__ = [
    "Job",
    "MonteCarloJob",
    "ReportBundle",
    "ReportJob",
    "ScenarioJob",
    "SweepJob",
    "build_run_artifacts",
]


@runtime_checkable
class Job(Protocol):
    """A unit of off-thread work returning a result object.

    Notes
    -----
    ``label`` is a read-only property so frozen-dataclass jobs (whose fields are
    immutable) satisfy the protocol.
    """

    @property
    def label(self) -> str:
        """Short human-readable name (shown while the job runs)."""
        ...

    def run(self, *, progress: ProgressFn, is_cancelled: CancelFn) -> object:
        """Execute the work and return its result.

        Parameters
        ----------
        progress : Callable[[str], None]
            Report a coarse progress message to the UI.
        is_cancelled : Callable[[], bool]
            Return ``True`` if the run has been asked to stop.

        Returns
        -------
        object
            The job's result (a ``RunArtifacts``, ``ReportBundle``,
            ``SweepResult`` or ``MonteCarloResult``).
        """
        ...


@dataclass(frozen=True)
class ReportBundle:
    """Everything the *Report* action produces (task S7 §5).

    Attributes
    ----------
    artifacts : RunArtifacts
        The forward + inverse run (its intermediates feed the live tab too).
    budget : ErrorBudget
        The end-to-end ``truth vs recovery`` error budget (analysis layer).
    nea : NeaBudget or None
        The NEA budget; ``None`` for a noiseless (stub) detector.
    """

    artifacts: RunArtifacts
    budget: ErrorBudget
    nea: NeaBudget | None


def build_run_artifacts(
    scenario: ScenarioConfig | None,
    source: Path | str | None,
    config_dir: Path | None,
    *,
    progress: ProgressFn,
    is_cancelled: CancelFn,
    system: SystemConfig | None = None,
) -> RunArtifacts:
    """Run forward + inverse for a config *or* a scenario file (shared helper).

    Exactly one of ``scenario`` / ``source`` is used: a GUI-built
    :class:`~optivibe.core.config.models.ScenarioConfig` runs directly (no YAML
    round-trip), or a path is loaded first. The forward and inverse passes are
    run separately so the worker can report progress and poll for cancellation
    between them.

    Variant selection has two routes (task S7-mod §1/§6):

    * **By name (default).** ``scenario.variant`` (one of ``A``..``D`` or a
      ``*_demo`` literal) is loaded from ``config_dir`` -- the S0..S9 path,
      unchanged.
    * **By edited composition.** When ``system`` is given (the GUI assembled an
      *editable* :class:`~optivibe.core.config.subsystems.SystemConfig`), it is
      **resolved on this worker thread** into the flat variant and used instead
      of loading by name. ``scenario.variant`` then merely labels the starting
      composition (its ``Literal`` type is a frozen ICD contract; the resolved
      variant carries the real, edited parameters). Resolution reads preset
      files off disk, so it must not run in the UI thread (SW-06).

    Parameters
    ----------
    scenario : ScenarioConfig or None
        A ready scenario (preferred when the GUI assembled one).
    source : pathlib.Path or str or None
        Scenario YAML path (used when ``scenario`` is ``None``).
    config_dir : pathlib.Path or None
        Override for the ``configs/`` directory (variant presets).
    progress : Callable[[str], None]
        Progress reporter.
    is_cancelled : Callable[[], bool]
        Cancel poll.
    system : SystemConfig or None, optional
        An edited composition to resolve and run instead of loading
        ``scenario.variant`` by name.

    Returns
    -------
    RunArtifacts
        Scenario, variant, forward intermediates and the recovered result.

    Raises
    ------
    ValueError
        If neither ``scenario`` nor ``source`` is given.
    """
    if scenario is None:
        if source is None:
            msg = "build_run_artifacts needs either a scenario or a source path"
            raise ValueError(msg)
        progress("loading scenario")
        scenario = load_scenario(Path(source))
    variant: VariantConfig
    if system is not None:
        progress("resolving composition")
        variant = resolve_system_variant(system, config_dir or default_config_dir())
    else:
        progress("loading variant")
        variant = load_variant(scenario.variant, config_dir=config_dir)
    pipeline = Pipeline(scenario, variant)
    progress("forward model")
    forward = pipeline.forward()
    if is_cancelled():
        logger.debug("scenario cancelled after forward pass")
    progress("inverse / DSP")
    result = pipeline.inverse(forward.detector)
    return RunArtifacts(scenario=scenario, variant=variant, forward=forward, result=result)


@dataclass(frozen=True)
class ScenarioJob:
    """Run one scenario (forward + inverse) to :class:`RunArtifacts`.

    Parameters
    ----------
    scenario : ScenarioConfig or None, optional
        A GUI-assembled scenario (preferred). Mutually exclusive with ``source``.
    source : pathlib.Path or str or None, optional
        A scenario YAML path (back-compatible with the S0 path-based run).
    config_dir : pathlib.Path or None, optional
        Override for the ``configs/`` directory.
    system : SystemConfig or None, optional
        An edited composition to resolve and run instead of the named variant
        (task S7-mod §1); ``None`` keeps the by-name path.
    """

    scenario: ScenarioConfig | None = None
    source: Path | str | None = None
    config_dir: Path | None = None
    system: SystemConfig | None = None
    label: str = "run scenario"

    def run(self, *, progress: ProgressFn, is_cancelled: CancelFn) -> object:
        """Run the scenario and return its :class:`RunArtifacts`."""
        return build_run_artifacts(
            self.scenario,
            self.source,
            self.config_dir,
            progress=progress,
            is_cancelled=is_cancelled,
            system=self.system,
        )


@dataclass(frozen=True)
class ReportJob:
    """Run a scenario and build the analysis budgets (task S7 §5).

    Parameters
    ----------
    scenario : ScenarioConfig
        The scenario to run.
    config_dir : pathlib.Path or None, optional
        Override for the ``configs/`` directory.
    band_hz : tuple of float or None, optional
        Assessment band for the spectral error (full spectrum when ``None``).
    system : SystemConfig or None, optional
        An edited composition to resolve and run instead of the named variant.
    """

    scenario: ScenarioConfig
    config_dir: Path | None = None
    band_hz: tuple[float, float] | None = None
    system: SystemConfig | None = None
    label: str = "report"

    def run(self, *, progress: ProgressFn, is_cancelled: CancelFn) -> object:
        """Run the scenario then assemble the ``ReportBundle``."""
        artifacts = build_run_artifacts(
            self.scenario,
            None,
            self.config_dir,
            progress=progress,
            is_cancelled=is_cancelled,
            system=self.system,
        )
        progress("error budget")
        budget = truth_vs_recovery(
            artifacts.forward.excitation,
            artifacts.result,
            artifacts.forward.detector,
            variant=artifacts.variant,
            band_hz=self.band_hz,
        )
        progress("NEA budget")
        nea = nea_budget(artifacts.forward.detector, artifacts.variant)
        return ReportBundle(artifacts=artifacts, budget=budget, nea=nea)


@dataclass(frozen=True)
class SweepJob:
    """Run a parameter sweep (design or response) to a :class:`SweepResult`.

    Parameters
    ----------
    spec : SweepSpec
        The sweep specification (analysis layer).
    """

    spec: SweepSpec
    label: str = "parameter sweep"

    def run(self, *, progress: ProgressFn, is_cancelled: CancelFn) -> object:
        """Run the sweep and return its :class:`SweepResult`."""
        progress(f"sweeping {self.spec.parameter}")
        return run_sweep(self.spec)


@dataclass(frozen=True)
class MonteCarloJob:
    """Run a tolerance Monte-Carlo to a :class:`MonteCarloResult`.

    Parameters
    ----------
    spec : MonteCarloSpec
        The Monte-Carlo specification (analysis layer).
    """

    spec: MonteCarloSpec
    label: str = "monte-carlo"

    def run(self, *, progress: ProgressFn, is_cancelled: CancelFn) -> object:
        """Run the Monte-Carlo and return its :class:`MonteCarloResult`."""
        progress(f"{self.spec.n_draws} draws")
        return run_monte_carlo(self.spec)


# Re-exported result types so consumers can ``isinstance``-dispatch on the
# worker payload without importing the analysis layer directly.
_RESULT_TYPES = (RunArtifacts, ReportBundle, SweepResult, MonteCarloResult)
