"""Deterministic tonal generators: single sine and multitone.

The sine source reproduces the S0 acceptance signal exactly (hello scenario,
SW-11); multitone generalizes it to a sum of tones with individual phases
(doc 11 §2.1). Amplitudes are specified in g and converted to SI at this
boundary via :data:`optivibe.core.units.G0_M_S2` (10 §6, 01 §4.3).
"""

from __future__ import annotations

import numpy as np

from optivibe.core.config.models import ExcitationSpec, MultitoneSpec, SineSpec
from optivibe.core.types import Excitation
from optivibe.core.units import G0_M_S2
from optivibe.excitation._common import pack_on_axis, time_grid

__all__ = ["MultitoneExcitationSource", "SineExcitationSource"]


class SineExcitationSource:
    """Single-tone sinusoidal acceleration on one axis.

    Generates ``a(t) = amplitude_g * g0 * sin(2*pi*f*t)`` on the chosen axis and
    zeros on the other two. Deterministic (the seed is recorded for traceability
    but the waveform does not use randomness).
    """

    def generate(self, spec: ExcitationSpec, *, seed: int | None = None) -> Excitation:
        """Generate a sine acceleration time series (see :class:`SineSpec`)."""
        if not isinstance(spec, SineSpec):
            msg = f"'sine' source expects SineSpec, got kind={spec.kind!r}"
            raise TypeError(msg)
        t = time_grid(spec.fs_hz, spec.duration_s)
        wave = (spec.amplitude_g * G0_M_S2) * np.sin(2.0 * np.pi * spec.frequency_hz * t)
        meta: dict[str, object] = {
            "generator": "sine",
            "axis": spec.axis,
            "frequency_hz": spec.frequency_hz,
            "amplitude_g": spec.amplitude_g,
        }
        return pack_on_axis(wave, spec.axis, spec.fs_hz, seed, meta)


class MultitoneExcitationSource:
    """Sum of sine tones, each with its own frequency, amplitude and phase.

    ``a(t) = sum_i A_i * g0 * sin(2*pi*f_i*t + phi_i)`` on the chosen axis.
    Deterministic.
    """

    def generate(self, spec: ExcitationSpec, *, seed: int | None = None) -> Excitation:
        """Generate a multitone time series (see :class:`MultitoneSpec`)."""
        if not isinstance(spec, MultitoneSpec):
            msg = f"'multitone' source expects MultitoneSpec, got kind={spec.kind!r}"
            raise TypeError(msg)
        t = time_grid(spec.fs_hz, spec.duration_s)
        wave = np.zeros_like(t)
        for tone in spec.tones:
            wave += (tone.amplitude_g * G0_M_S2) * np.sin(
                2.0 * np.pi * tone.frequency_hz * t + tone.phase_rad
            )
        meta: dict[str, object] = {
            "generator": "multitone",
            "axis": spec.axis,
            "tones": [(tone.frequency_hz, tone.amplitude_g, tone.phase_rad) for tone in spec.tones],
        }
        return pack_on_axis(wave, spec.axis, spec.fs_hz, seed, meta)
