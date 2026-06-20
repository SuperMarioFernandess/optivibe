"""Noise-equivalent acceleration NEA (task S5 §5; docs 07/05; O-SW-08).

The detector reports its one-sided current-noise PSD (white in v-S4: shot + RIN +
Johnson, doc 07 §1) in :attr:`DetectorOutput.noise`. Referring it to the input
divides by the through sensitivity (doc 05 §7):

``NEA(f) = sqrt(S_i(f)) / |s_target(f)|``   [(m/s^2)/sqrt(Hz)].

On the plateau ``s_target`` is the constant ``s_target^QS``; across the band the
complex ``s_target(f) = s_target^QS D(f)`` is used, so a white current floor maps
to a flat input NEA that dips by ``~1/Q`` toward ``f1``. The full-band figure is
``NEA * sqrt(B)`` with ``B`` the noise bandwidth.

**Convention (O-SW-08).** The reference-arm shot model (``"matched"`` doubles the
shot PSD, ``"bright"`` keeps the bare ``2 e I_DC``) is fixed *upstream* by the
detector and is already baked into ``psd_total``. The DSP is convention-neutral:
it consumes the PSD as given and propagates the convention from the metadata
(``noise["reference_arm"]``) -- it never re-picks or re-derives it. The analytic
cross-check re-assembles the PSD from the variant using the *same* convention
read from the metadata, so it agrees by construction (the engineering vs
datasheet number choice stays an upstream/physics decision).
"""

from __future__ import annotations

import math

import numpy as np

from optivibe.core.config.models import Constants, VariantConfig
from optivibe.core.types import DetectorOutput, FloatArray
from optivibe.detector.photodiode import noise_psd
from optivibe.dsp.calibration import dynamic_sensitivity, target_sensitivity

__all__ = ["NeaResult", "analytic_noise_psd", "nea_from_detector", "nea_spectrum"]


class NeaResult:
    """Noise-equivalent acceleration summary referred to the input (doc 05 §7).

    Attributes
    ----------
    plateau_psd_a2_hz : float
        Total one-sided current-noise PSD on the plateau, A^2/Hz (from the
        detector metadata).
    s_target : float
        Signed plateau sensitivity used, A/(m/s^2).
    nea_plateau : float
        Plateau noise-equivalent acceleration density, (m/s^2)/sqrt(Hz).
    bandwidth_hz : float
        Noise bandwidth ``B`` used for the full-band figure, Hz.
    nea_full_band : float
        Full-band NEA ``NEA_plateau * sqrt(B)``, m/s^2 (RMS).
    reference_arm : str
        The shot-model convention propagated from the detector (O-SW-08).
    """

    def __init__(
        self,
        *,
        plateau_psd_a2_hz: float,
        s_target: float,
        bandwidth_hz: float,
        reference_arm: str,
    ) -> None:
        self.plateau_psd_a2_hz = plateau_psd_a2_hz
        self.s_target = s_target
        self.bandwidth_hz = bandwidth_hz
        self.reference_arm = reference_arm
        self.nea_plateau = math.sqrt(plateau_psd_a2_hz) / abs(s_target)
        self.nea_full_band = self.nea_plateau * math.sqrt(bandwidth_hz)

    def as_dict(self) -> dict[str, object]:
        """Return the NEA summary as a plain mapping (for VibrationResult/metadata)."""
        return {
            "nea_plateau_m_s2_rthz": self.nea_plateau,
            "nea_full_band_m_s2": self.nea_full_band,
            "bandwidth_hz": self.bandwidth_hz,
            "s_target_a_per_m_s2": self.s_target,
            "plateau_psd_a2_hz": self.plateau_psd_a2_hz,
            "reference_arm": self.reference_arm,
        }


def nea_from_detector(
    detector: DetectorOutput,
    variant: VariantConfig,
    constants: Constants | None = None,
    *,
    bandwidth_hz: float | None = None,
) -> NeaResult | None:
    """NEA referred to the input from the detector noise metadata (S5 §5).

    Returns ``None`` when the detector carries no physical noise (the stub
    detector, ``noise["model"] != "photodiode"``) -- there is nothing to refer.

    Parameters
    ----------
    detector : DetectorOutput
        Digitized detector signal with its noise metadata.
    variant : VariantConfig
        Sensor variant (for ``s_target``).
    constants : Constants or None, optional
        Physical constants (default loaded when ``None``).
    bandwidth_hz : float or None, optional
        Noise bandwidth for the full-band figure; defaults to the detector's
        Nyquist bandwidth (``noise["nyquist_bw_hz"]`` or ``fs/2``).

    Returns
    -------
    NeaResult or None
        The NEA summary, or ``None`` for a noiseless (stub) detector.
    """
    noise = detector.noise
    if noise.get("model") != "photodiode":
        return None
    psd_total = float(noise["psd_total_a2_hz"])  # type: ignore[arg-type]
    s_target = target_sensitivity(variant, constants)
    if bandwidth_hz is None:
        bandwidth_hz = float(noise.get("nyquist_bw_hz", detector.fs / 2.0))  # type: ignore[arg-type]
    reference_arm = str(noise.get("reference_arm", "matched"))
    return NeaResult(
        plateau_psd_a2_hz=psd_total,
        s_target=s_target,
        bandwidth_hz=bandwidth_hz,
        reference_arm=reference_arm,
    )


def nea_spectrum(
    detector: DetectorOutput,
    variant: VariantConfig,
    freq_hz: FloatArray,
    constants: Constants | None = None,
) -> FloatArray:
    """NEA density across frequency ``sqrt(S_i)/|s_target(f)|`` (S5 §5; doc 05 §7).

    Uses the white current PSD from the metadata and the complex
    ``s_target(f) = s_target^QS D(f)`` so the curve dips toward ``f1``.

    Parameters
    ----------
    detector : DetectorOutput
        Digitized detector signal with noise metadata (must be a photodiode
        read-out).
    variant : VariantConfig
        Sensor variant.
    freq_hz : numpy.ndarray
        Frequencies, Hz.
    constants : Constants or None, optional
        Physical constants (default loaded when ``None``).

    Returns
    -------
    numpy.ndarray
        NEA density at each frequency, (m/s^2)/sqrt(Hz).

    Raises
    ------
    ValueError
        If the detector carries no physical noise PSD.
    """
    noise = detector.noise
    if noise.get("model") != "photodiode":
        msg = "nea_spectrum requires a photodiode detector with a noise PSD"
        raise ValueError(msg)
    psd_total = float(noise["psd_total_a2_hz"])  # type: ignore[arg-type]
    s_f = dynamic_sensitivity(variant, freq_hz, constants)
    out: FloatArray = math.sqrt(psd_total) / np.abs(s_f)
    return np.ascontiguousarray(out, dtype=np.float64)


def analytic_noise_psd(
    detector: DetectorOutput,
    variant: VariantConfig,
    constants: Constants,
) -> float:
    """Re-assemble the total current-noise PSD from the variant, A^2/Hz (S5 §5).

    Independent analytic cross-check of ``psd_total`` using the *same* DC current
    and the *same* reference-arm convention read from the detector metadata
    (O-SW-08), so it agrees with the simulated floor within the doc 11 §7
    tolerance (<= 15 %, as the detector self-check SW-29).

    Parameters
    ----------
    detector : DetectorOutput
        Digitized detector signal (provides ``I_DC`` and the convention).
    variant : VariantConfig
        Sensor variant (source RIN, electronics).
    constants : Constants
        Physical constants (``e``, ``kB``, ``T``).

    Returns
    -------
    float
        Analytic total current-noise PSD, A^2/Hz.
    """
    noise = detector.noise
    i_dc = float(noise["i_dc_a"])  # type: ignore[arg-type]
    balanced = bool(noise.get("balanced", True))
    reference_arm = str(noise.get("reference_arm", "matched"))
    psd = noise_psd(i_dc, variant, constants, balanced=balanced, reference_arm=reference_arm)
    return float(psd["total"])
