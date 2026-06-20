"""S0 stub inverse chain (kept for regression behind the ``"stub"`` key).

A minimal but real inverse: it removes the DC pedestal, treats the AC signal as
the (uncalibrated) target-axis acceleration, integrates ``a -> v -> x`` by
cumulative trapezoid with mean removal, computes an FFT amplitude spectrum and
the dominant frequency, and reports RMS metrics. The physically calibrated
reconstruction (calibration against ``s_target``, drift-suppressed integration,
Welch/spectrogram, ISO 10816/20816, NEA and the cross-axis metric) is
:class:`~optivibe.dsp.standard.StandardDsp` (S5). The stub stays the default so
the prior S1-S4 dominants are reproduced unchanged (decision SW-S5-01).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

from optivibe.core.config.models import DspOptions, VariantConfig
from optivibe.core.types import DetectorOutput, FloatArray, Spectrum, VibrationResult

__all__ = ["StubDsp"]


def _rms(values: FloatArray) -> float:
    """Return the root-mean-square of ``values``."""
    return float(np.sqrt(np.mean(np.square(values))))


class StubDsp:
    """Minimal inverse chain (DC removal, integration, FFT spectrum).

    Warnings
    --------
    No physical calibration: the AC photocurrent is used directly as the
    acceleration estimate. The calibrated reconstruction is
    :class:`~optivibe.dsp.standard.StandardDsp`.
    """

    def run(
        self, detector: DetectorOutput, variant: VariantConfig, options: DspOptions
    ) -> VibrationResult:
        """Reconstruct a/v/x, an FFT spectrum and basic metrics."""
        fs = detector.fs
        dt = 1.0 / fs
        accel = np.asarray(detector.samples, dtype=np.float64) - detector.dc_level

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
    def _spectrum(accel: FloatArray, fs: float) -> tuple[Spectrum, tuple[float, ...]]:
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
