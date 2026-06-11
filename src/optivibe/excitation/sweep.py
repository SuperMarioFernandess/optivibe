"""Constant-amplitude frequency sweep (chirp) generator.

Implements the ``sweep`` excitation of doc 11 §2.1 on top of
:func:`scipy.signal.chirp`. The instantaneous frequency runs ``f_start_hz ->
f_end_hz`` over the full spec duration, linearly or logarithmically; the
amplitude is constant.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import chirp

from optivibe.core.config.models import ExcitationSpec, SweepSpec
from optivibe.core.types import Excitation
from optivibe.core.units import G0_M_S2
from optivibe.excitation._common import pack_on_axis, time_grid

__all__ = ["SweepExcitationSource"]

_METHODS = {"linear": "linear", "log": "logarithmic"}


class SweepExcitationSource:
    """Linear or logarithmic chirp on one axis. Deterministic."""

    def generate(self, spec: ExcitationSpec, *, seed: int | None = None) -> Excitation:
        """Generate a sweep time series (see :class:`SweepSpec`)."""
        if not isinstance(spec, SweepSpec):
            msg = f"'sweep' source expects SweepSpec, got kind={spec.kind!r}"
            raise TypeError(msg)
        t = time_grid(spec.fs_hz, spec.duration_s)
        wave = (spec.amplitude_g * G0_M_S2) * np.asarray(
            chirp(
                t,
                f0=spec.f_start_hz,
                t1=spec.duration_s,
                f1=spec.f_end_hz,
                method=_METHODS[spec.method],
            ),
            dtype=np.float64,
        )
        meta: dict[str, object] = {
            "generator": "sweep",
            "axis": spec.axis,
            "f_start_hz": spec.f_start_hz,
            "f_end_hz": spec.f_end_hz,
            "amplitude_g": spec.amplitude_g,
            "method": spec.method,
        }
        return pack_on_axis(wave, spec.axis, spec.fs_hz, seed, meta)
