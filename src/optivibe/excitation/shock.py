"""Shock (transient pulse) generator.

Implements the ``shock`` excitation of doc 11 §2.1. v-S1 ships the classical
half-sine pulse ``a(t) = peak * sin(pi * (t - delay) / T)`` on the interval
``delay <= t <= delay + T`` and zero elsewhere; other shapes (haversine,
trapezoid, sawtooth) extend the ``shape`` literal later without contract
changes.
"""

from __future__ import annotations

import numpy as np

from optivibe.core.config.models import ExcitationSpec, ShockSpec
from optivibe.core.types import Excitation
from optivibe.core.units import G0_M_S2
from optivibe.excitation._common import pack_on_axis, time_grid

__all__ = ["ShockExcitationSource"]


class ShockExcitationSource:
    """Single half-sine shock pulse on one axis. Deterministic."""

    def generate(self, spec: ExcitationSpec, *, seed: int | None = None) -> Excitation:
        """Generate a shock time series (see :class:`ShockSpec`)."""
        if not isinstance(spec, ShockSpec):
            msg = f"'shock' source expects ShockSpec, got kind={spec.kind!r}"
            raise TypeError(msg)
        t = time_grid(spec.fs_hz, spec.duration_s)
        pulse_s = spec.pulse_ms / 1.0e3
        local = t - spec.delay_s
        inside = (local >= 0.0) & (local <= pulse_s)
        wave = np.zeros_like(t)
        wave[inside] = (spec.peak_g * G0_M_S2) * np.sin(np.pi * local[inside] / pulse_s)
        meta: dict[str, object] = {
            "generator": "shock",
            "axis": spec.axis,
            "shape": spec.shape,
            "peak_g": spec.peak_g,
            "pulse_ms": spec.pulse_ms,
            "delay_s": spec.delay_s,
        }
        return pack_on_axis(wave, spec.axis, spec.fs_hz, seed, meta)
