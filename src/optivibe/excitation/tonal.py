"""Deterministic tonal generators: single sine and multitone.

The sine source reproduces the S0 acceptance signal exactly (hello scenario,
SW-11); multitone generalizes it to a sum of tones with individual phases
(doc 11 §2.1). S-21 adds the opt-in AM/FM modulation of the sine carrier
(doc 11 §2.1.3) -- the carrier is the only kind that takes a modulator, because
``f_c`` must be a single, closed-form frequency for the sideband family (and
for its prediction in :mod:`optivibe.analysis.expected_peaks`); several
modulated carriers are expressed as a ``composite``. Amplitudes are specified in
g and converted to SI at this boundary via :data:`optivibe.core.units.G0_M_S2`
(10 §6, 01 §4.3).
"""

from __future__ import annotations

import numpy as np

from optivibe.core.config.models import AmModulation, ExcitationSpec, MultitoneSpec, SineSpec
from optivibe.core.types import Excitation
from optivibe.core.units import G0_M_S2
from optivibe.excitation._common import pack_on_axis, time_grid

__all__ = ["MultitoneExcitationSource", "SineExcitationSource"]


class SineExcitationSource:
    """Single-tone sinusoidal acceleration on one axis, optionally AM/FM modulated.

    Unmodulated it generates ``a(t) = a_c sin(2 pi f_c t + phi_c)`` on the chosen
    axis and zeros on the other two, with ``a_c = amplitude_g * g0``. With
    ``spec.modulation`` set (doc 11 §2.1.3):

    * ``am``: ``a(t) = a_c [1 + m cos(2 pi f_m t + phi_m)] sin(2 pi f_c t + phi_c)``
      -- sidebands ``m a_c / 2`` at ``f_c +- f_m``;
    * ``fm``: ``a(t) = a_c sin(2 pi f_c t + beta sin(2 pi f_m t + phi_m) + phi_c)``
      -- sidebands ``a_c |J_k(beta)|`` at ``f_c +- k f_m``.

    Deterministic (the seed is recorded for traceability but the waveform does
    not use randomness). The degenerate cases are exact, not approximate:
    ``m = 0``, ``beta = 0`` and ``phi_c = 0`` reduce the expressions to the
    unmodulated one bit-for-bit (IEEE-754: ``1 + 0*cos = 1`` and ``x + 0 = x``),
    which the S-21 regression goldens pin.
    """

    def generate(self, spec: ExcitationSpec, *, seed: int | None = None) -> Excitation:
        """Generate a sine acceleration time series (see :class:`SineSpec`)."""
        if not isinstance(spec, SineSpec):
            msg = f"'sine' source expects SineSpec, got kind={spec.kind!r}"
            raise TypeError(msg)
        t = time_grid(spec.fs_hz, spec.duration_s)
        amplitude = spec.amplitude_g * G0_M_S2
        carrier_phase = 2.0 * np.pi * spec.frequency_hz * t + spec.phase_rad
        modulation = spec.modulation
        meta: dict[str, object] = {
            "generator": "sine",
            "axis": spec.axis,
            "frequency_hz": spec.frequency_hz,
            "amplitude_g": spec.amplitude_g,
        }
        if modulation is None:
            wave = amplitude * np.sin(carrier_phase)
        else:
            modulator = 2.0 * np.pi * modulation.f_m_hz * t + modulation.phase_rad
            if isinstance(modulation, AmModulation):
                wave = (
                    amplitude * (1.0 + modulation.depth * np.cos(modulator)) * np.sin(carrier_phase)
                )
            else:
                wave = amplitude * np.sin(carrier_phase + modulation.beta * np.sin(modulator))
            meta["modulation"] = modulation.model_dump()
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
