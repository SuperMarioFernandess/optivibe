"""Excitation stage: generators of 3-axis base acceleration a(t).

S0 ships only a deterministic sine source so the hello scenario can flow through
the whole pipeline. S1 adds multitone, sweep/chirp, random(PSD), shock and
CSV/WAV import, each registered under its own key (roadmap 12 S1).
"""

from __future__ import annotations

import numpy as np

from optivibe.core.config.models import ExcitationSpec
from optivibe.core.registry import Registry
from optivibe.core.stages import ExcitationSource
from optivibe.core.types import Excitation
from optivibe.core.units import G0_M_S2

EXCITATION_REGISTRY: Registry[ExcitationSource] = Registry("excitation")

__all__ = ["EXCITATION_REGISTRY", "SineExcitationSource"]


@EXCITATION_REGISTRY.register("sine")
class SineExcitationSource:
    """Single-tone sinusoidal acceleration on one axis.

    Generates ``a(t) = amplitude_g * g0 * sin(2*pi*f*t)`` on the chosen axis and
    zeros on the other two. Deterministic (the seed is recorded for traceability
    but the waveform does not use randomness).
    """

    def generate(self, spec: ExcitationSpec, *, seed: int | None = None) -> Excitation:
        """Generate a sine acceleration time series (see :class:`ExcitationSpec`)."""
        n_samples = round(spec.duration_s * spec.fs_hz)
        if n_samples < 1:
            msg = f"duration_s * fs_hz must yield >= 1 sample, got {n_samples}"
            raise ValueError(msg)
        t = np.arange(n_samples, dtype=np.float64) / spec.fs_hz
        amplitude_ms2 = spec.amplitude_g * G0_M_S2
        wave = amplitude_ms2 * np.sin(2.0 * np.pi * spec.frequency_hz * t)
        zeros = np.zeros(n_samples, dtype=np.float64)
        channels: dict[str, np.ndarray] = {"x": zeros.copy(), "y": zeros.copy(), "z": zeros.copy()}
        channels[spec.axis] = wave
        return Excitation(
            a_x=channels["x"],
            a_y=channels["y"],
            a_z=channels["z"],
            fs=spec.fs_hz,
            seed=seed,
            meta={
                "generator": "sine",
                "axis": spec.axis,
                "frequency_hz": spec.frequency_hz,
                "amplitude_g": spec.amplitude_g,
            },
        )
