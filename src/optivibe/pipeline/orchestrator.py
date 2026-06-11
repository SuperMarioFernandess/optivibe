"""Pipeline orchestrator: wire the stages and run forward -> inverse.

The orchestrator is the only place that knows the *order* of the stages
(``excitation -> mechanics -> optics -> detector`` forward, then ``dsp`` inverse;
architecture 09 §8). It owns no physics: every stage is looked up in its domain
registry by the key given in the scenario, so swapping an implementation is a
config change, not a code change (SW-02). Because the orchestrator imports the
domain registries (and the domains import only ``core``), there is no import
cycle: dependencies still point inward (09 §3).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from optivibe.core.config.loader import load_scenario, load_variant
from optivibe.core.config.models import ScenarioConfig, VariantConfig
from optivibe.core.logging import get_logger
from optivibe.core.stages import (
    DetectorStage,
    DspStage,
    ExcitationSource,
    MechanicsStage,
    OpticsStage,
)
from optivibe.core.types import (
    DetectorOutput,
    Excitation,
    OpticalResponse,
    TipState,
    VibrationResult,
)
from optivibe.detector import DETECTOR_REGISTRY
from optivibe.dsp import DSP_REGISTRY
from optivibe.excitation import EXCITATION_REGISTRY
from optivibe.mechanics import MECHANICS_REGISTRY
from optivibe.optics import OPTICS_REGISTRY

logger = get_logger(__name__)

__all__ = ["ForwardArtifacts", "Pipeline", "RunArtifacts", "run_scenario"]


@dataclass(frozen=True)
class ForwardArtifacts:
    """Intermediate signals produced by the forward chain.

    Attributes
    ----------
    excitation : Excitation
        Generated base acceleration (system input).
    tip : TipState
        Fiber tip state q_tip(t) from the mechanics stage.
    optical : OpticalResponse
        Coupling efficiency eta(t) from the optics stage.
    detector : DetectorOutput
        Digitized photodetector signal from the detector stage.
    """

    excitation: Excitation
    tip: TipState
    optical: OpticalResponse
    detector: DetectorOutput


@dataclass(frozen=True)
class RunArtifacts:
    """Everything a single scenario run produces (forward intermediates + result).

    Attributes
    ----------
    scenario : ScenarioConfig
        The scenario that was run.
    variant : VariantConfig
        The sensor variant used.
    forward : ForwardArtifacts
        Forward-chain intermediates.
    result : VibrationResult
        Reconstructed vibration on the target axis (system output).
    """

    scenario: ScenarioConfig
    variant: VariantConfig
    forward: ForwardArtifacts
    result: VibrationResult


class Pipeline:
    """Forward + inverse orchestrator for one sensor variant.

    The stage implementations are resolved from the domain registries using the
    keys in ``scenario.stages`` at construction time, so an invalid key fails
    early with a clear :class:`~optivibe.core.registry.RegistryError`.

    Parameters
    ----------
    scenario : ScenarioConfig
        Run description (selects stages, excitation and DSP options).
    variant : VariantConfig
        Sensor-variant parameters passed to the physical stages.
    """

    def __init__(self, scenario: ScenarioConfig, variant: VariantConfig) -> None:
        self._scenario = scenario
        self._variant = variant
        stages = scenario.stages
        self._excitation: ExcitationSource = EXCITATION_REGISTRY.create(stages.excitation)
        # Scenario-level mechanics overrides (S2): only explicitly set options
        # are forwarded, so option-less implementations (the stub) still
        # construct; an unsupported option fails loudly (10 §7).
        mechanics_overrides = {
            key: value
            for key, value in scenario.mechanics.model_dump().items()
            if value is not None
        }
        self._mechanics: MechanicsStage = MECHANICS_REGISTRY.create(
            stages.mechanics, **mechanics_overrides
        )
        self._optics: OpticsStage = OPTICS_REGISTRY.create(stages.optics)
        self._detector: DetectorStage = DETECTOR_REGISTRY.create(stages.detector)
        self._dsp: DspStage = DSP_REGISTRY.create(stages.dsp)
        logger.debug(
            "pipeline built: variant=%s stages=%s",
            variant.name,
            stages.model_dump(),
        )

    @property
    def scenario(self) -> ScenarioConfig:
        """The scenario this pipeline runs."""
        return self._scenario

    @property
    def variant(self) -> VariantConfig:
        """The sensor variant this pipeline uses."""
        return self._variant

    def forward(self) -> ForwardArtifacts:
        """Run the forward chain: input -> tip -> optics -> detector.

        Returns
        -------
        ForwardArtifacts
            All forward-chain intermediates.
        """
        excitation = self._excitation.generate(self._scenario.excitation, seed=self._scenario.seed)
        tip = self._mechanics.run(excitation, self._variant)
        optical = self._optics.run(tip, self._variant)
        detector = self._detector.run(optical, self._variant)
        return ForwardArtifacts(excitation=excitation, tip=tip, optical=optical, detector=detector)

    def inverse(self, detector: DetectorOutput) -> VibrationResult:
        """Run the inverse/DSP chain on a detector signal.

        Parameters
        ----------
        detector : DetectorOutput
            Digitized detector signal.

        Returns
        -------
        VibrationResult
            Reconstructed vibration on the target axis.
        """
        return self._dsp.run(detector, self._variant, self._scenario.dsp)

    def run(self) -> RunArtifacts:
        """Run forward then inverse and bundle every artifact.

        Returns
        -------
        RunArtifacts
            Scenario, variant, forward intermediates and the result.
        """
        forward = self.forward()
        result = self.inverse(forward.detector)
        logger.info(
            "run complete: variant=%s n_samples=%d dominant_hz=%s",
            self._variant.name,
            result.n_samples,
            result.dominant_freqs_hz,
        )
        return RunArtifacts(
            scenario=self._scenario,
            variant=self._variant,
            forward=forward,
            result=result,
        )


def run_scenario(scenario_path: Path | str, config_dir: Path | None = None) -> RunArtifacts:
    """Load a scenario (and its variant) and run the whole pipeline.

    Parameters
    ----------
    scenario_path : pathlib.Path or str
        Path to a scenario YAML file.
    config_dir : pathlib.Path or None, optional
        Configuration directory for the variant preset; defaults to the resolved
        repository ``configs/`` (see
        :func:`optivibe.core.config.loader.default_config_dir`).

    Returns
    -------
    RunArtifacts
        The complete set of run artifacts.
    """
    scenario = load_scenario(Path(scenario_path))
    variant = load_variant(scenario.variant, config_dir=config_dir)
    return Pipeline(scenario, variant).run()
