"""Shared helpers of the excitation generators (time grid, axis packing)."""

from __future__ import annotations

import numpy as np

from optivibe.core.types import Excitation, FloatArray


def time_grid(fs_hz: float, duration_s: float) -> FloatArray:
    """Build the sampling time grid ``t_k = k / fs``.

    Parameters
    ----------
    fs_hz : float
        Sampling frequency, Hz.
    duration_s : float
        Signal duration, s.

    Returns
    -------
    numpy.ndarray
        1-D time array of ``round(duration_s * fs_hz)`` samples, s.

    Raises
    ------
    ValueError
        If the grid would contain fewer than one sample.
    """
    n_samples = round(duration_s * fs_hz)
    if n_samples < 1:
        msg = f"duration_s * fs_hz must yield >= 1 sample, got {n_samples}"
        raise ValueError(msg)
    return np.arange(n_samples, dtype=np.float64) / fs_hz


def pack_on_axis(
    signal: FloatArray,
    axis: str,
    fs: float,
    seed: int | None,
    meta: dict[str, object],
) -> Excitation:
    """Wrap a 1-axis signal into the 3-axis :class:`Excitation` contract.

    Parameters
    ----------
    signal : numpy.ndarray
        Acceleration on the chosen axis, m/s^2.
    axis : {"x", "y", "z"}
        Target axis; the other two are zero (S1 semantics of ``axis``).
    fs : float
        Sampling frequency, Hz.
    seed : int or None
        Seed recorded for traceability.
    meta : dict
        Generator metadata.

    Returns
    -------
    Excitation
        The packed contract.
    """
    zeros = np.zeros_like(signal)
    channels: dict[str, FloatArray] = {"x": zeros, "y": zeros.copy(), "z": zeros.copy()}
    channels[axis] = signal
    return Excitation(
        a_x=channels["x"], a_y=channels["y"], a_z=channels["z"], fs=fs, seed=seed, meta=meta
    )
