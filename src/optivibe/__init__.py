"""OptiVibe: a digital twin of a fiber-optic vibration sensor.

The package models the full forward chain (base acceleration -> fiber-cantilever
tip motion -> optical back-coupling -> photodetector signal) and the inverse
chain (reconstruction of the target-axis vibration), plus analytics, behind a
clean, contract-driven core with swappable implementations (architecture
document 09).

Layers
------
``core``
    Pure core: data contracts, config, units, registry, stage protocols.
``excitation`` / ``mechanics`` / ``optics`` / ``detector`` / ``dsp``
    Stage implementations registered for selection by config key.
``pipeline``
    Orchestrator wiring the stages forward then inverse.
``analysis`` / ``viz`` / ``io``
    Analytics, Qt-free plotting and persistence (filled in later stages).
``cli`` / ``gui``
    Head-less runner and the optional desktop shell.
"""

from __future__ import annotations

__version__ = "0.0.0"

__all__ = ["__version__"]
