"""Pipeline package: the stage orchestrator (forward + inverse).

The orchestrator wires the registered stages in order and runs a scenario; it
contains no physics of its own (architecture 09 §8). Stage *protocols* live in
:mod:`optivibe.core.stages`, not here, to keep dependencies pointing inward.
"""

from optivibe.pipeline.orchestrator import (
    ForwardArtifacts,
    Pipeline,
    RunArtifacts,
    run_scenario,
)

__all__ = ["ForwardArtifacts", "Pipeline", "RunArtifacts", "run_scenario"]
