"""Kinematics: integrate acceleration to velocity and displacement (task S5 §2).

Double integration is ill-conditioned at low frequency: any DC offset or slow
drift in ``a`` integrates to a ramp in ``v`` and a parabola in ``x``. Both
integrators therefore suppress content below a high-pass cut-off ``f_hp`` (by
default the lower band edge of the variant, doc 08). Two interchangeable methods
are registered (selected by ``DspOptions.integrator``):

``"frequency"``
    Spectral integration ``v(f) = a(f)/(j omega)``, ``x(f) = a(f)/(j omega)^2``
    with a smooth second-order high-pass mask ``|H_hp| = r^2 / sqrt(1 + r^4)``,
    ``r = f/f_hp`` (and the DC bin zeroed). Exact for (quasi-)stationary signals
    and phase-correct: for a tone well inside the band ``x = -a/omega^2`` lags
    ``a`` by ``pi`` (the golden phase check).

``"time"``
    Cumulative-trapezoid integration followed by a polynomial detrend (the
    ``"детренд"`` option) to remove the integration drift -- the DC offset that
    ``cumtrapz`` introduces grows to a ramp/parabola under the second
    integration, which a cubic detrend removes robustly. A Butterworth high-pass
    is additionally applied when its cut-off is numerically safe
    (``f_hp/f_Nyq >= 0.01``); at the sub-Hz cut-offs the band demands relative to
    a multi-kHz ``fs`` the Butterworth is ill-conditioned (its settling time
    exceeds the record), so the detrend carries the drift removal there. Agrees
    with the frequency method in band to a few percent.

Both return signals at the input sampling rate; the high-pass leaves the in-band
tone amplitude/phase unchanged (it only removes the sub-``f_hp`` drift).
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import polynomial as npoly
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, sosfiltfilt

from optivibe.core.registry import Registry
from optivibe.core.types import FloatArray

__all__ = [
    "INTEGRATOR_REGISTRY",
    "highpass_mask",
    "integrate_frequency",
    "integrate_time",
]

# Registry of a -> (v, x) integrators selected by ``DspOptions.integrator``
# (architecture SW-02: a swappable method behind a key). Factories are the
# integrator functions themselves; callers use ``.get(key)`` to retrieve one.
INTEGRATOR_REGISTRY: Registry[object] = Registry("dsp.integrator")

# Butterworth high-pass order and the smallest normalized cut-off it is applied
# at (below this the sub-Hz design is numerically ill-conditioned for filtfilt).
_BUTTER_ORDER = 2
_MIN_SAFE_WN = 0.01
# Polynomial order removed from each integrated channel to kill the cumulative
# DC/ramp/parabola drift of trapezoidal integration.
_DETREND_ORDER = 3


def highpass_mask(freq_hz: FloatArray, f_hp: float) -> FloatArray:
    """High-pass magnitude mask: zero below ``f_hp`` with a raised-cosine edge.

    Double integration multiplies by ``1/omega^2``, which amplifies any residual
    sub-band content enormously, so the mask hard-zeros bins below ``f_hp`` (with
    a short raised-cosine transition over ``[f_hp/2, f_hp]`` to limit ringing)
    rather than merely attenuating them. The in-band response is flat
    (``= 1`` for ``f >= f_hp``) and the DC bin is exactly zero. A non-positive
    ``f_hp`` disables the mask (returns ones, except DC).

    Parameters
    ----------
    freq_hz : numpy.ndarray
        Non-negative frequencies, Hz.
    f_hp : float
        High-pass cut-off, Hz; ``<= 0`` disables the mask.

    Returns
    -------
    numpy.ndarray
        Real magnitude mask in ``[0, 1]``, same shape as ``freq_hz``.
    """
    freq = np.asarray(freq_hz, dtype=np.float64)
    mask = np.ones_like(freq)
    if f_hp > 0.0:
        f_lo = 0.5 * f_hp
        mask = np.where(freq >= f_hp, 1.0, 0.0)
        transition = (freq > f_lo) & (freq < f_hp)
        # Raised-cosine rising edge from 0 at f_lo to 1 at f_hp.
        mask[transition] = 0.5 * (1.0 - np.cos(np.pi * (freq[transition] - f_lo) / (f_hp - f_lo)))
    mask[freq == 0.0] = 0.0
    return np.ascontiguousarray(mask, dtype=np.float64)


@INTEGRATOR_REGISTRY.register("frequency")
def integrate_frequency(accel: FloatArray, fs: float, f_hp: float) -> tuple[FloatArray, FloatArray]:
    """Spectral integration ``a -> v -> x`` with a high-pass mask (task S5 §2).

    ``V(f) = H_hp(f) a(f)/(j omega)``; ``X(f) = H_hp(f) a(f)/(j omega)^2``. The
    omega = 0 bin is set to zero (no DC integration), so a DC pedestal in the
    acceleration cannot ramp the result; the brick-wall mask additionally removes
    any bin-aligned sub-band content (e.g. a slow machine sway at a resolved
    frequency). Spectral integration is exact for periodic in-band content; a
    *non-periodic* drift (an arbitrary ramp/shelf) is not bin-aligned and leaks,
    so for that case use the time-domain integrator, whose polynomial detrend
    removes it. Returns real signals at the same length and rate as the input.

    Parameters
    ----------
    accel : numpy.ndarray
        Acceleration time series, m/s^2.
    fs : float
        Sampling frequency, Hz.
    f_hp : float
        High-pass cut-off, Hz.

    Returns
    -------
    velocity : numpy.ndarray
        Velocity, m/s.
    displacement : numpy.ndarray
        Displacement, m.
    """
    n = accel.size
    spectrum = np.fft.rfft(np.ascontiguousarray(accel, dtype=np.float64))
    freq = np.fft.rfftfreq(n, d=1.0 / fs).astype(np.float64)
    omega = 2.0 * np.pi * freq
    mask = highpass_mask(freq, f_hp)

    nonzero = omega != 0.0
    inv_jw = np.zeros_like(spectrum)
    inv_jw[nonzero] = 1.0 / (1j * omega[nonzero])

    vel_spec = spectrum * inv_jw * mask
    disp_spec = spectrum * (inv_jw**2) * mask
    velocity = np.fft.irfft(vel_spec, n=n).astype(np.float64)
    displacement = np.fft.irfft(disp_spec, n=n).astype(np.float64)
    return (
        np.ascontiguousarray(velocity, dtype=np.float64),
        np.ascontiguousarray(displacement, dtype=np.float64),
    )


def _detrend_poly(signal: FloatArray, order: int) -> FloatArray:
    """Remove a low-order polynomial trend (drift) from ``signal``.

    Fits and subtracts a degree-``order`` polynomial on a normalized abscissa
    (well-conditioned). Removes the cumulative DC/ramp/parabola drift of
    trapezoidal integration without touching an in-band tone (orthogonal to a
    low-order polynomial over many periods). Falls back to mean removal for very
    short records.
    """
    n = signal.size
    if n <= order + 1:
        return signal - float(np.mean(signal))
    grid = np.linspace(0.0, 1.0, n)
    coef = npoly.polyfit(grid, signal, order)
    trend = npoly.polyval(grid, coef)
    return np.ascontiguousarray(signal - trend, dtype=np.float64)


def _safe_highpass(signal: FloatArray, fs: float, f_hp: float) -> FloatArray:
    """Zero-phase Butterworth high-pass applied only at a numerically safe cut-off.

    Returns the input unchanged when ``f_hp`` is out of range or its normalized
    cut-off ``f_hp/f_Nyq`` is below :data:`_MIN_SAFE_WN` (sub-Hz designs against a
    multi-kHz ``fs`` are ill-conditioned for ``filtfilt``); the polynomial
    detrend handles the drift there.
    """
    nyquist = fs / 2.0
    wn = f_hp / nyquist
    if not (_MIN_SAFE_WN <= wn < 1.0) or signal.size <= 3 * (_BUTTER_ORDER + 1):
        return signal
    sos = butter(_BUTTER_ORDER, wn, btype="highpass", output="sos")
    out: FloatArray = sosfiltfilt(sos, signal)
    return np.ascontiguousarray(out, dtype=np.float64)


@INTEGRATOR_REGISTRY.register("time")
def integrate_time(accel: FloatArray, fs: float, f_hp: float) -> tuple[FloatArray, FloatArray]:
    """Time-domain integration ``a -> v -> x`` with a polynomial detrend (S5 §2).

    Cumulative-trapezoid integration with a cubic detrend after each stage to
    remove the integration drift, plus a Butterworth high-pass when its cut-off
    is numerically safe. Returns real signals at the same length and rate as the
    input.

    Parameters
    ----------
    accel : numpy.ndarray
        Acceleration time series, m/s^2.
    fs : float
        Sampling frequency, Hz.
    f_hp : float
        High-pass cut-off, Hz.

    Returns
    -------
    velocity : numpy.ndarray
        Velocity, m/s.
    displacement : numpy.ndarray
        Displacement, m.
    """
    dt = 1.0 / fs
    velocity = cumulative_trapezoid(accel, dx=dt, initial=0.0)
    velocity = _detrend_poly(np.ascontiguousarray(velocity, dtype=np.float64), _DETREND_ORDER)
    velocity = _safe_highpass(velocity, fs, f_hp)
    displacement = cumulative_trapezoid(velocity, dx=dt, initial=0.0)
    displacement = _detrend_poly(
        np.ascontiguousarray(displacement, dtype=np.float64), _DETREND_ORDER
    )
    displacement = _safe_highpass(displacement, fs, f_hp)
    return (
        np.ascontiguousarray(velocity, dtype=np.float64),
        np.ascontiguousarray(displacement, dtype=np.float64),
    )
