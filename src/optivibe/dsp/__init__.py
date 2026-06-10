"""Inverse/DSP stage: detector samples -> reconstructed vibration.

S0 ships a minimal but real inverse: it removes the DC pedestal, treats the AC
signal as the (uncalibrated) target-axis acceleration, integrates ``a -> v -> x``
by cumulative trapezoid with mean removal, computes an FFT amplitude spectrum and
the dominant frequency, and reports RMS metrics. Calibration against the
effective sensitivity, frequency-domain integration with an HP cut-off,
Welch/spectrogram, ISO 10816/20816 and the cross-axis metric arrive in S5
(documents 05/07).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

from optivibe.core.config.models import DspOptions, VariantConfig
from optivibe.core.registry import Registry
from optivibe.core.stages import DspStage
from optivibe.core.types import DetectorOutput, Spectrum, VibrationResult

DSP_REGISTRY: Registry[DspStage] = Registry("dsp")

__all__ = ["DSP_REGISTRY", "StubDsp"]


def _rms(values: np.ndarray) -> float:
    """Return the root-mean-square of ``values``."""
    return float(np.sqrt(np.mean(np.square(values))))


@DSP_REGISTRY.register("stub")
class StubDsp:
    """Minimal inverse chain (DC removal, integration, FFT spectrum).

    Warnings
    --------
    No physical calibration in S0: the AC photocurrent is used directly as the
    acceleration estimate. The calibrated reconstruction arrives in S5.
    """

    def run(
        self, detector: DetectorOutput, variant: VariantConfig, options: DspOptions
    ) -> VibrationResult:
        """Reconstruct a/v/x, an FFT spectrum and basic metrics."""
        fs = detector.fs
        dt = 1.0 / fs
        accel = detector.samples - detector.dc_level

        velocity = cumulative_trapezoid(accel, dx=dt, initial=0.0)
        velocity = velocity - float(np.mean(velocity))
        displacement = cumulative_trapezoid(velocity, dx=dt, initial=0.0)
        displacement = displacement - float(np.mean(displacement))

        spectrum, dominant = self._spectrum(accel, fs)

        return VibrationResult(
            a=accel,
            v=velocity,
            x=displacement,
            fs=fs,
            dominant_freqs_hz=dominant,
            rms={"a": _rms(accel), "v": _rms(velocity), "x": _rms(displacement)},
            cross_residual={},
            spectrum=spectrum,
            iso=None,
        )

    @staticmethod
    def _spectrum(accel: np.ndarray, fs: float) -> tuple[Spectrum, tuple[float, ...]]:
        """Return an FFT amplitude spectrum and the dominant frequency (excl. DC)."""
        n_samples = accel.size
        freq = np.fft.rfftfreq(n_samples, d=1.0 / fs).astype(np.float64)
        amplitude = np.abs(np.fft.rfft(accel)).astype(np.float64) * (2.0 / n_samples)
        if amplitude.size:
            amplitude[0] *= 0.5  # DC bin is not doubled
        spectrum = Spectrum(
            freq=freq, values=amplitude, kind="amplitude", window="boxcar", method="fft"
        )
        if amplitude.size > 1:
            peak_index = int(np.argmax(amplitude[1:])) + 1
            dominant: tuple[float, ...] = (float(freq[peak_index]),)
        else:
            dominant = ()
        return spectrum, dominant
