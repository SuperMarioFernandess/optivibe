"""Pure core of OptiVibe: contracts, config, units, registry, stage protocols.

The core has no knowledge of the GUI, file formats or any concrete physical
implementation; dependencies point strictly inward (architecture 09 §3).
"""

from optivibe.core.registry import Registry, RegistryError
from optivibe.core.types import (
    DetectorOutput,
    Excitation,
    OpticalResponse,
    Spectrum,
    TipState,
    VibrationResult,
)

__all__ = [
    "DetectorOutput",
    "Excitation",
    "OpticalResponse",
    "Registry",
    "RegistryError",
    "Spectrum",
    "TipState",
    "VibrationResult",
]
