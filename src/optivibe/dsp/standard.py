"""Physically calibrated inverse chain (``"standard"`` DSP key, task S5).

:class:`StandardDsp` is the S5 inverse stage. It calibrates the detector samples
to target-axis acceleration against the through sensitivity ``s_target``
(:mod:`~optivibe.dsp.calibration`), integrates ``a -> v -> x`` with drift
suppression (:mod:`~optivibe.dsp.kinematics`), computes the representative
spectrum and dominant lines (:mod:`~optivibe.dsp.spectra`), grades the vibration
severity per ISO 10816-3 / 20816-3 (:mod:`~optivibe.dsp.iso`,
:mod:`~optivibe.dsp.metrics`) and refers the detector noise to the input as the
NEA (:mod:`~optivibe.dsp.nea`). It satisfies the unchanged
:class:`~optivibe.core.stages.DspStage` protocol and works with both the stub
detector (no noise -> no NEA) and the photodiode detector.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optivibe.core.config.loader import load_constants
from optivibe.core.config.models import Constants, DspOptions, VariantConfig
from optivibe.core.types import DetectorOutput, FloatArray, VibrationResult
from optivibe.dsp.calibration import calibrate_acceleration
from optivibe.dsp.iso import iso_assessment
from optivibe.dsp.kinematics import INTEGRATOR_REGISTRY
from optivibe.dsp.metrics import band_rms_velocity, rms, second_harmonic_ratio
from optivibe.dsp.nea import nea_from_detector
from optivibe.dsp.spectra import amplitude_spectrum, dominant_frequencies, welch_psd
from optivibe.mechanics.cantilever import CantileverModel

__all__ = ["StandardDsp"]

# Callable signature of a registered integrator: (accel, fs, f_hp) -> (v, x).
_Integrator = Callable[[FloatArray, float, float], tuple[FloatArray, FloatArray]]


class StandardDsp:
    """Calibrated inverse chain: detector samples -> vibration on target axis (S5).

    Registered under the key ``"standard"``. The default stage remains ``"stub"``
    (SW-S5-01); this implementation is selected explicitly per scenario.

    Parameters
    ----------
    constants : Constants or None, optional
        Physical constants; loaded once from ``configs/constants.yaml`` when
        ``None`` (the only I/O, performed at construction, like the other
        physical stages).
    """

    def __init__(self, *, constants: Constants | None = None) -> None:
        self._constants = load_constants() if constants is None else constants

    def run(
        self, detector: DetectorOutput, variant: VariantConfig, options: DspOptions
    ) -> VibrationResult:
        """Reconstruct calibrated a/v/x, spectra, metrics, ISO and NEA (S5).

        Parameters
        ----------
        detector : DetectorOutput
            Digitized detector signal.
        variant : VariantConfig
            Sensor variant (calibration, band, ISO class).
        options : DspOptions
            Integrator, spectrum, high-pass and ISO options.

        Returns
        -------
        VibrationResult
            Calibrated acceleration/velocity/displacement (SI), dominant lines,
            RMS metrics, cross-axis residual, a representative spectrum and the
            ISO severity assessment (with the NEA summary attached when the
            detector carries noise).
        """
        fs = detector.fs

        # 1. Calibration: detector samples -> acceleration (signed s_target).
        accel, _s_target = calibrate_acceleration(detector, variant, self._constants)
        if options.deconvolve_hlat:
            accel = self._deconvolve_hlat(accel, fs, variant)

        # 2. Kinematics: a -> v -> x with drift suppression below f_hp.
        f_hp = options.f_hp_hz if options.f_hp_hz is not None else variant.band.f_min_hz
        integrator: _Integrator = INTEGRATOR_REGISTRY.get(options.integrator)  # type: ignore[assignment]
        velocity, displacement = integrator(accel, fs, f_hp)

        # 3. Spectra and dominant frequencies (computed here; viz only draws).
        if options.spectrum_method == "welch":
            spectrum = welch_psd(
                accel,
                fs,
                window=options.window,
                nperseg=options.welch_nperseg,
                noverlap=options.welch_noverlap,
            )
        else:
            spectrum = amplitude_spectrum(accel, fs)
        dominant = dominant_frequencies(spectrum)

        # 4. Metrics: RMS, cross-axis residual, ISO severity.
        rms_metrics = {"a": rms(accel), "v": rms(velocity), "x": rms(displacement)}
        amp_for_residual = amplitude_spectrum(accel, fs)
        cross_residual: dict[str, float] = {}
        if dominant:
            cross_residual["second_harmonic_ratio"] = second_harmonic_ratio(
                amp_for_residual, dominant[0]
            )

        band = (variant.band.f_min_hz, variant.band.f_max_hz)
        velocity_psd = welch_psd(
            velocity,
            fs,
            window=options.window,
            nperseg=options.welch_nperseg,
            noverlap=options.welch_noverlap,
        )
        v_rms_band = band_rms_velocity(velocity_psd, band)
        iso: dict[str, object] = iso_assessment(
            v_rms_band, machine_class=options.iso_machine_class, band_hz=band
        )

        # 5. NEA (stand metric): refer the detector noise to the input. The
        #    contract has no NEA field, so the summary rides in the iso bag for
        #    traceability; the authoritative NEA path is the dsp.nea helpers.
        nea = nea_from_detector(detector, variant, self._constants)
        if nea is not None:
            iso["nea"] = nea.as_dict()

        return VibrationResult(
            a=accel,
            v=velocity,
            x=displacement,
            fs=fs,
            dominant_freqs_hz=dominant,
            rms=rms_metrics,
            cross_residual=cross_residual,
            spectrum=spectrum,
            iso=iso,
        )

    def _deconvolve_hlat(self, accel: FloatArray, fs: float, variant: VariantConfig) -> FloatArray:
        """Divide out ``D(f)`` to flatten the mechanical roll-up near ``f1`` (S5 §1).

        After the plateau-scalar calibration, ``accel(f) = D(f) a(f)``; dividing
        by the single-mode dynamic factor recovers ``a(f)`` toward ``f1``. A
        no-op in band where ``|D| ~ 1``.
        """
        n = accel.size
        cantilever = CantileverModel.from_config(self._constants, variant)
        freq = np.fft.rfftfreq(n, d=1.0 / fs).astype(np.float64)
        factor = cantilever.dynamic_factor(freq)
        corrected = np.fft.rfft(accel) / factor
        out: FloatArray = np.fft.irfft(corrected, n=n).astype(np.float64)
        return np.ascontiguousarray(out, dtype=np.float64)
