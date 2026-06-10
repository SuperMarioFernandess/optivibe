"""Data contracts — the internal ICD (mirror of knowledge-base document 04).

Stages communicate *only* through these typed, immutable contracts (architecture
principle 09 §3, §4). The tip-state vector is exactly ``(dx, dy, dz, theta_x,
theta_y)`` from documents 00/01/04, with SI units; the optical/detector signal
follows ``S = R * P * (R1 + rho * eta)`` (route 2, doc 04 §4).

Contracts that carry large numpy arrays are lightweight ``frozen`` dataclasses,
not pydantic models: shape / ``fs`` / units are validated **once** on
construction rather than per sample (09 §5; 10 §9). Pydantic is reserved for
configuration and metadata (see :mod:`optivibe.core.config`).

Mapping to document 04
----------------------
====================  =========================================================
Contract              Role in ICD (doc 04)
====================  =========================================================
``Excitation``        system input: base acceleration a(t) on x/y/z
``TipState``          boundary variable q_tip = (dx, dy, dz, theta_x, theta_y)
``OpticalResponse``   eta(t) and bias eta0 (04 §4, matrix dS/dq)
``DetectorOutput``    digitized S = R*P*(R1 + rho*eta) (04 §4, §8)
``VibrationResult``   system output: a/v/x on target axis + cross residuals
``Spectrum``          derived spectral product
====================  =========================================================
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

__all__ = [
    "DetectorOutput",
    "Excitation",
    "OpticalResponse",
    "Spectrum",
    "TipState",
    "VibrationResult",
]


def _as_1d_f64(name: str, arr: npt.ArrayLike) -> FloatArray:
    """Coerce ``arr`` to a 1-D contiguous float64 array, validating its shape.

    Parameters
    ----------
    name : str
        Field name, used in error messages.
    arr : array_like
        Input data.

    Returns
    -------
    numpy.ndarray
        1-D float64 array.

    Raises
    ------
    ValueError
        If the array is not one-dimensional, is empty, or contains non-finite
        values.
    """
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if out.ndim != 1:
        msg = f"{name}: expected a 1-D array, got shape {out.shape}"
        raise ValueError(msg)
    if out.size == 0:
        msg = f"{name}: array must be non-empty"
        raise ValueError(msg)
    if not np.all(np.isfinite(out)):
        msg = f"{name}: array contains non-finite values"
        raise ValueError(msg)
    return out


def _check_fs(fs: float) -> None:
    """Validate a sampling frequency in Hz.

    Raises
    ------
    ValueError
        If ``fs`` is not strictly positive and finite.
    """
    if not (np.isfinite(fs) and fs > 0.0):
        msg = f"fs must be a positive, finite frequency in Hz, got {fs!r}"
        raise ValueError(msg)


@dataclass(frozen=True)
class Excitation:
    """Three-axis base acceleration time series (system input).

    Parameters
    ----------
    a_x, a_y, a_z : numpy.ndarray, shape (N,)
        Base acceleration along the x/y/z axes, in m/s^2 (SI). The target axis is
        ``x`` (doc 01 §1).
    fs : float
        Sampling frequency, in Hz.
    seed : int or None, optional
        Random seed of the generating scenario, for reproducibility (10 §8).
    meta : Mapping[str, object], optional
        Free-form metadata (e.g. generator name, units of the source file).

    Notes
    -----
    Acceptable input range is specified by document 00 (0.1 g - 50 g; behaviour
    above 50 g is investigated). Shapes and ``fs`` are validated on construction.
    """

    a_x: FloatArray
    a_y: FloatArray
    a_z: FloatArray
    fs: float
    seed: int | None = None
    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ax = _as_1d_f64("a_x", self.a_x)
        ay = _as_1d_f64("a_y", self.a_y)
        az = _as_1d_f64("a_z", self.a_z)
        if not (ax.size == ay.size == az.size):
            msg = f"a_x/a_y/a_z length mismatch: {ax.size}/{ay.size}/{az.size}"
            raise ValueError(msg)
        _check_fs(self.fs)
        object.__setattr__(self, "a_x", ax)
        object.__setattr__(self, "a_y", ay)
        object.__setattr__(self, "a_z", az)

    @property
    def n_samples(self) -> int:
        """Number of time samples."""
        return int(self.a_x.size)

    @property
    def duration_s(self) -> float:
        """Signal duration in seconds (``n_samples / fs``)."""
        return self.n_samples / self.fs


@dataclass(frozen=True)
class TipState:
    """Fiber tip state q_tip over time (mechanics -> optics boundary).

    Parameters
    ----------
    dx, dy, dz : numpy.ndarray, shape (N,)
        Transverse displacements ``dx, dy`` and gap change ``dz``, in metres.
        Sign of ``dz``: positive means the gap A grows (doc 01 §2).
    theta_x, theta_y : numpy.ndarray, shape (N,)
        Tip tilts about the x and y axes, in radians. Physically coupled to the
        displacements by ``theta = 1.377 * delta / L`` (doc 04 §2), enforced by
        the mechanics stage, not by this contract.
    fs : float
        Sampling frequency, in Hz.
    """

    dx: FloatArray
    dy: FloatArray
    dz: FloatArray
    theta_x: FloatArray
    theta_y: FloatArray
    fs: float

    def __post_init__(self) -> None:
        arrays = {
            "dx": _as_1d_f64("dx", self.dx),
            "dy": _as_1d_f64("dy", self.dy),
            "dz": _as_1d_f64("dz", self.dz),
            "theta_x": _as_1d_f64("theta_x", self.theta_x),
            "theta_y": _as_1d_f64("theta_y", self.theta_y),
        }
        sizes = {name: a.size for name, a in arrays.items()}
        if len(set(sizes.values())) != 1:
            msg = f"tip-state component length mismatch: {sizes}"
            raise ValueError(msg)
        _check_fs(self.fs)
        for name, a in arrays.items():
            object.__setattr__(self, name, a)

    @property
    def n_samples(self) -> int:
        """Number of time samples."""
        return int(self.dx.size)


@dataclass(frozen=True)
class OpticalResponse:
    """Back-coupling efficiency eta(t) at the working point (optics output).

    Parameters
    ----------
    eta : numpy.ndarray, shape (N,)
        Total coupling efficiency eta = eta_x * eta_y, dimensionless (doc 04 §4).
    bias : float
        Static working point eta0 (dimensionless) set by the intentional
        de-centering Δx0 (doc 04 §4).
    fs : float
        Sampling frequency, in Hz.
    eta_x, eta_y : numpy.ndarray or None, optional
        Per-plane factors of the astigmatic cylinder, dimensionless. When given,
        their length must match ``eta``.
    """

    eta: FloatArray
    bias: float
    fs: float
    eta_x: FloatArray | None = None
    eta_y: FloatArray | None = None

    def __post_init__(self) -> None:
        eta = _as_1d_f64("eta", self.eta)
        _check_fs(self.fs)
        object.__setattr__(self, "eta", eta)
        for name in ("eta_x", "eta_y"):
            value = getattr(self, name)
            if value is not None:
                coerced = _as_1d_f64(name, value)
                if coerced.size != eta.size:
                    msg = f"{name} length {coerced.size} != eta length {eta.size}"
                    raise ValueError(msg)
                object.__setattr__(self, name, coerced)

    @property
    def n_samples(self) -> int:
        """Number of time samples."""
        return int(self.eta.size)


@dataclass(frozen=True)
class DetectorOutput:
    """Digitized photodetector signal (detector output).

    Parameters
    ----------
    samples : numpy.ndarray, shape (N,)
        Digitized samples of ``S = R * P * (R1 + rho * eta)`` (doc 04 §4), in the
        units given by ``units``.
    fs : float
        Sampling frequency, in Hz.
    dc_level : float
        DC pedestal of the signal, in ``units`` (the ``R*P*R1`` term plus the eta
        bias contribution).
    units : {"A", "V"}, optional
        Physical units of ``samples`` (photocurrent or transimpedance voltage).
    noise : Mapping[str, object], optional
        Noise parameters used to synthesize the samples (shot/RIN/Johnson/...),
        for traceability (doc 04 §8; detailed model lands in S4).
    """

    samples: FloatArray
    fs: float
    dc_level: float
    units: Literal["A", "V"] = "A"
    noise: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        samples = _as_1d_f64("samples", self.samples)
        _check_fs(self.fs)
        object.__setattr__(self, "samples", samples)

    @property
    def n_samples(self) -> int:
        """Number of time samples."""
        return int(self.samples.size)


@dataclass(frozen=True)
class Spectrum:
    """A frequency-domain product (derived contract).

    Parameters
    ----------
    freq : numpy.ndarray, shape (M,)
        Frequency axis, in Hz (non-negative, ascending).
    values : numpy.ndarray, shape (M,)
        Amplitude (same units as the source signal) or PSD ((unit^2)/Hz),
        according to ``kind``.
    kind : {"amplitude", "psd"}
        Whether ``values`` is an amplitude spectrum or a power spectral density.
    window : str, optional
        Window function name used (e.g. ``"hann"``).
    method : str, optional
        Estimation method (e.g. ``"fft"`` or ``"welch"``).
    """

    freq: FloatArray
    values: FloatArray
    kind: Literal["amplitude", "psd"]
    window: str = "boxcar"
    method: str = "fft"

    def __post_init__(self) -> None:
        freq = _as_1d_f64("freq", self.freq)
        values = _as_1d_f64("values", self.values)
        if freq.size != values.size:
            msg = f"freq length {freq.size} != values length {values.size}"
            raise ValueError(msg)
        object.__setattr__(self, "freq", freq)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class VibrationResult:
    """Reconstructed vibration on the target axis (system output).

    Parameters
    ----------
    a, v, x : numpy.ndarray, shape (N,)
        Acceleration (m/s^2), velocity (m/s) and displacement (m) on the target
        axis, recovered by the inverse/DSP chain.
    fs : float
        Sampling frequency, in Hz.
    dominant_freqs_hz : tuple of float, optional
        Dominant frequencies, in Hz, ordered by prominence.
    rms : Mapping[str, float], optional
        Root-mean-square metrics keyed by quantity (``"a"``, ``"v"``, ``"x"``).
    cross_residual : Mapping[str, float], optional
        Residual cross-axis response on the non-target axes (``"y"``, ``"z"``),
        the cross-sensitivity metric required by document 00.
    spectrum : Spectrum or None, optional
        A representative spectrum of the target-axis signal.
    iso : Mapping[str, object] or None, optional
        ISO 10816/20816 vibration-severity assessment (filled in S5).
    """

    a: FloatArray
    v: FloatArray
    x: FloatArray
    fs: float
    dominant_freqs_hz: tuple[float, ...] = ()
    rms: Mapping[str, float] = field(default_factory=dict)
    cross_residual: Mapping[str, float] = field(default_factory=dict)
    spectrum: Spectrum | None = None
    iso: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        a = _as_1d_f64("a", self.a)
        v = _as_1d_f64("v", self.v)
        x = _as_1d_f64("x", self.x)
        if not (a.size == v.size == x.size):
            msg = f"a/v/x length mismatch: {a.size}/{v.size}/{x.size}"
            raise ValueError(msg)
        _check_fs(self.fs)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "x", x)

    @property
    def n_samples(self) -> int:
        """Number of time samples."""
        return int(self.a.size)
