"""Excitation stage: generators of 3-axis base acceleration a(t).

S1 implements the full generator family of doc 11 §2.1 behind the
:class:`~optivibe.core.stages.ExcitationSource` protocol — sine, multitone,
sweep/chirp, band-limited random noise with target PSD/RMS, half-sine shock —
plus replay of measured CSV/WAV records through the data-loader registry
(seam SW-08). Each implementation is registered under its spec ``kind``, so a
scenario selects it with ``stages.excitation: <kind>`` (SW-02). Input range of
the project spec (doc 00): 0.1g-50g anywhere in 0.1 Hz-20 kHz; behaviour above
50g is itself a study subject, so amplitudes are not clamped here.
"""

from __future__ import annotations

from optivibe.core.registry import Registry
from optivibe.core.stages import ExcitationSource
from optivibe.excitation.from_file import FileExcitationSource
from optivibe.excitation.random_noise import RandomExcitationSource
from optivibe.excitation.shock import ShockExcitationSource
from optivibe.excitation.sweep import SweepExcitationSource
from optivibe.excitation.tonal import MultitoneExcitationSource, SineExcitationSource

EXCITATION_REGISTRY: Registry[ExcitationSource] = Registry("excitation")

EXCITATION_REGISTRY.register("sine")(SineExcitationSource)
EXCITATION_REGISTRY.register("multitone")(MultitoneExcitationSource)
EXCITATION_REGISTRY.register("sweep")(SweepExcitationSource)
EXCITATION_REGISTRY.register("random")(RandomExcitationSource)
EXCITATION_REGISTRY.register("shock")(ShockExcitationSource)
EXCITATION_REGISTRY.register("csv")(FileExcitationSource)
EXCITATION_REGISTRY.register("wav")(FileExcitationSource)

__all__ = [
    "EXCITATION_REGISTRY",
    "FileExcitationSource",
    "MultitoneExcitationSource",
    "RandomExcitationSource",
    "ShockExcitationSource",
    "SineExcitationSource",
    "SweepExcitationSource",
]
