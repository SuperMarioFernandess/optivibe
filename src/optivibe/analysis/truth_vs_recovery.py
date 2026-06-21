"""Truth vs recovery: end-to-end error budget (task S6 §B5).

Compares the recovered target-axis acceleration ``VibrationResult.a`` with the
*known* forward input ``Excitation.a_x`` and reports an error budget: amplitude,
phase / group delay, spectral shape, dominant lines and RMS. It also reconciles
the acceleration error with the NEA floor and -- crucially -- keeps the **small
acceleration recovery error** separate from the **elevated displacement noise
floor** raised by double integration (``PSD_x = PSD_a / omega^4``, doc 07; task
§8). Conflating the two is the headline reporting hazard this module guards
against: the calibration recovers ``a`` to within the noise floor, while the
recovered ``x`` carries a far larger low-frequency-amplified floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate

from optivibe.analysis.nea_budget import nea_budget
from optivibe.core.config.loader import load_constants
from optivibe.core.config.models import Constants
from optivibe.core.types import Excitation, FloatArray, VibrationResult
from optivibe.dsp.metrics import rms
from optivibe.dsp.spectra import amplitude_spectrum, dominant_frequencies

__all__ = ["ErrorBudget", "truth_vs_recovery"]


@dataclass(frozen=True)
class ErrorBudget:
    """End-to-end ``truth vs recovery`` error budget (task S6 §B5).

    Attributes
    ----------
    rms_true_a, rms_recovered_a : float
        RMS of the true and recovered target-axis acceleration, m/s^2.
    amplitude_ratio : float
        ``rms_recovered_a / rms_true_a`` (1.0 is perfect scale).
    amplitude_rel_error : float
        ``|amplitude_ratio - 1|``.
    delay_samples, delay_s : float
        Best-fit lag of the recovered signal relative to the input.
    rms_error_a : float
        RMS of the lag-aligned acceleration residual, m/s^2 (the recovery error).
    rms_error_rel : float
        ``rms_error_a / rms_true_a``.
    phase_error_rad : float
        Phase error at the dominant line, rad.
    dominant_true_hz, dominant_recovered_hz : tuple of float
        Dominant frequencies of the input and the recovery, Hz.
    dominant_match : bool
        Whether the leading dominant agrees within one bin.
    spectral_rel_error : float
        In-band relative L2 error of the amplitude spectrum.
    nea_full_band : float or None
        Full-band NEA, m/s^2 (None for a noiseless detector).
    accel_error_over_nea : float or None
        ``rms_error_a / nea_full_band`` -- the recovery error in units of the
        noise floor (around 1 at the shot floor, << 1 for a strong tone).
    rms_displacement : float
        RMS of the recovered displacement, m (carries the elevated floor).
    displacement_floor_rms : float or None
        Displacement noise floor from ``PSD_x = PSD_a / omega^4``, m -- the
        elevated figure that must NOT be read as a calibration error.
    """

    rms_true_a: float
    rms_recovered_a: float
    amplitude_ratio: float
    amplitude_rel_error: float
    delay_samples: float
    delay_s: float
    rms_error_a: float
    rms_error_rel: float
    phase_error_rad: float
    dominant_true_hz: tuple[float, ...]
    dominant_recovered_hz: tuple[float, ...]
    dominant_match: bool
    spectral_rel_error: float
    nea_full_band: float | None
    accel_error_over_nea: float | None
    rms_displacement: float
    displacement_floor_rms: float | None

    def summary_text(self) -> str:
        """Render a compact, human-readable budget block (used by ``report``)."""
        nea = "-" if self.nea_full_band is None else f"{self.nea_full_band:.4g} m/s^2"
        over = "-" if self.accel_error_over_nea is None else f"{self.accel_error_over_nea:.3g}"
        floor = (
            "-" if self.displacement_floor_rms is None else f"{self.displacement_floor_rms:.4g} m"
        )
        dom_t = ", ".join(f"{f:.3f}" for f in self.dominant_true_hz) or "-"
        dom_r = ", ".join(f"{f:.3f}" for f in self.dominant_recovered_hz) or "-"
        lines = [
            "truth vs recovery (target axis a):",
            f"  amplitude ratio   : {self.amplitude_ratio:.6f} "
            f"(rel err {self.amplitude_rel_error:.2e})",
            f"  delay             : {self.delay_samples:.1f} samples ({self.delay_s:.3e} s)",
            f"  rms error (a)     : {self.rms_error_a:.4g} m/s^2 (rel {self.rms_error_rel:.2e})",
            f"  phase error       : {self.phase_error_rad:.3e} rad @ dominant",
            f"  spectral rel err  : {self.spectral_rel_error:.2e} (in band)",
            f"  dominant true     : {dom_t} Hz",
            f"  dominant recovered: {dom_r} Hz  (match: {self.dominant_match})",
            f"  NEA (full band)   : {nea}",
            f"  a-error / NEA      : {over}",
            "displacement (separate floor, PSD_x = PSD_a / omega^4):",
            f"  rms x (recovered) : {self.rms_displacement:.4g} m",
            f"  x noise floor     : {floor}  (low-frequency amplified; NOT a calibration error)",
        ]
        return "\n".join(lines)


def _best_lag(recovered: FloatArray, reference: FloatArray) -> int:
    """Integer lag (samples) maximizing the cross-correlation of two signals."""
    a = recovered - float(np.mean(recovered))
    b = reference - float(np.mean(reference))
    corr = correlate(a, b, mode="full", method="fft")
    lag = int(np.argmax(corr)) - (b.size - 1)
    return lag


def _phase_at(signal: FloatArray, fs: float, freq_hz: float) -> float:
    """Phase (rad) of ``signal`` at the bin nearest ``freq_hz``."""
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / fs)
    idx = int(np.argmin(np.abs(freqs - freq_hz)))
    return float(np.angle(spectrum[idx]))


def _spectral_rel_error(
    recovered: FloatArray, reference: FloatArray, fs: float, band_hz: tuple[float, float]
) -> float:
    """Relative L2 error of the in-band amplitude spectrum."""
    a_rec = amplitude_spectrum(recovered, fs)
    a_ref = amplitude_spectrum(reference, fs)
    mask = (a_ref.freq >= band_hz[0]) & (a_ref.freq <= band_hz[1])
    num = float(np.linalg.norm(a_rec.values[mask] - a_ref.values[mask]))
    den = float(np.linalg.norm(a_ref.values[mask])) or 1.0
    return num / den


def truth_vs_recovery(
    excitation: Excitation,
    result: VibrationResult,
    detector_output: object | None = None,
    *,
    variant: object | None = None,
    constants: Constants | None = None,
    band_hz: tuple[float, float] | None = None,
) -> ErrorBudget:
    """Build the end-to-end error budget for one run (task S6 §B5).

    Parameters
    ----------
    excitation : Excitation
        Forward input; the target-axis truth is ``a_x``.
    result : VibrationResult
        Reconstructed vibration (target-axis ``a``/``v``/``x``).
    detector_output : DetectorOutput or None, optional
        Detector signal; when a photodiode output (and ``variant`` is given) the
        NEA floor and displacement floor are filled in.
    variant : VariantConfig or None, optional
        Sensor variant (needed for the NEA floor).
    constants : Constants or None, optional
        Physical constants (default loaded when ``None``).
    band_hz : tuple of float or None, optional
        Assessment band for the spectral error; defaults to the full positive
        spectrum.

    Returns
    -------
    ErrorBudget
        The error budget (see attributes).
    """
    consts = load_constants() if constants is None else constants
    true_a = np.asarray(excitation.a_x, dtype=np.float64)
    rec_a = np.asarray(result.a, dtype=np.float64)
    fs = result.fs

    rms_true = rms(true_a)
    rms_rec = rms(rec_a)
    ratio = rms_rec / rms_true if rms_true > 0.0 else math.nan

    lag = _best_lag(rec_a, true_a)
    aligned = np.roll(rec_a, -lag)
    rms_err = float(np.std(aligned - true_a))

    dom_true = dominant_frequencies(amplitude_spectrum(true_a, fs))
    dom_rec = result.dominant_freqs_hz or dominant_frequencies(amplitude_spectrum(rec_a, fs))
    match = bool(dom_true and dom_rec and abs(dom_true[0] - dom_rec[0]) <= fs / true_a.size + 1.0)

    if dom_true:
        phase_err = _phase_at(rec_a, fs, dom_true[0]) - _phase_at(true_a, fs, dom_true[0])
        phase_err = math.atan2(math.sin(phase_err), math.cos(phase_err))  # wrap to (-pi, pi]
    else:
        phase_err = math.nan

    band = band_hz if band_hz is not None else (0.0, fs / 2.0)
    spectral_err = _spectral_rel_error(rec_a, true_a, fs, band)

    nea_full_band: float | None = None
    disp_floor: float | None = None
    from optivibe.core.config.models import VariantConfig
    from optivibe.core.types import DetectorOutput

    if isinstance(detector_output, DetectorOutput) and isinstance(variant, VariantConfig):
        budget = nea_budget(detector_output, variant, consts)
        if budget is not None:
            nea_full_band = budget.nea_full_band
            disp_floor = budget.displacement_floor_rms

    accel_over_nea = (
        rms_err / nea_full_band if (nea_full_band is not None and nea_full_band > 0.0) else None
    )

    return ErrorBudget(
        rms_true_a=rms_true,
        rms_recovered_a=rms_rec,
        amplitude_ratio=ratio,
        amplitude_rel_error=abs(ratio - 1.0) if math.isfinite(ratio) else math.nan,
        delay_samples=float(lag),
        delay_s=lag / fs,
        rms_error_a=rms_err,
        rms_error_rel=rms_err / rms_true if rms_true > 0.0 else math.nan,
        phase_error_rad=phase_err,
        dominant_true_hz=dom_true,
        dominant_recovered_hz=dom_rec,
        dominant_match=match,
        spectral_rel_error=spectral_err,
        nea_full_band=nea_full_band,
        accel_error_over_nea=accel_over_nea,
        rms_displacement=rms(np.asarray(result.x, dtype=np.float64)),
        displacement_floor_rms=disp_floor,
    )
