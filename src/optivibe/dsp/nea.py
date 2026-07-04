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
import numpy.typing as npt

from optivibe.core.config.models import Constants, VariantConfig
from optivibe.core.types import DetectorOutput, FloatArray, Spectrum
from optivibe.detector.photodiode import noise_psd
from optivibe.dsp.calibration import dynamic_sensitivity, target_sensitivity

__all__ = ["NeaResult", "analytic_noise_psd", "nea_from_detector", "nea_from_psd", "nea_spectrum"]


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


def nea_from_psd(psd: Spectrum, s_target_f: npt.ArrayLike) -> FloatArray:
    """Refer a *measured* current PSD to the input as NEA(f) (doc 05 §7; S-02).

    The same referral formula as :func:`nea_spectrum`,
    ``NEA(f) = sqrt(S_I(f)) / |s_target(f)|``, but taking a **measured** PSD of
    a recorded photocurrent (role S-02, doc 20 §5) instead of the white
    metadata floor of the synthetic detector. On a quiet record the result is
    the instrument's input-referred noise floor; on a driven record the signal
    lines appear on top of it (the PSD is referred as-is, honestly).

    Parameters
    ----------
    psd : Spectrum
        One-sided current PSD of the recorded photocurrent (``kind="psd"``),
        A^2/Hz (e.g. from :func:`~optivibe.dsp.spectra.welch_psd`).
    s_target_f : array_like
        Through sensitivity at each PSD frequency, A/(m/s^2): a signed scalar
        for the plateau, or a complex ``s_target^QS * D(f)`` array matching
        ``psd.freq`` for the dynamic referral (doc 05 §2b).

    Returns
    -------
    numpy.ndarray
        NEA density at each PSD frequency, (m/s^2)/sqrt(Hz).

    Raises
    ------
    ValueError
        If ``psd`` is not a PSD spectrum, the sensitivity magnitude vanishes
        anywhere (degenerate working point), or an array sensitivity does not
        match the PSD grid.
    """
    if psd.kind != "psd":
        msg = f"nea_from_psd expects a PSD spectrum, got kind={psd.kind!r}"
        raise ValueError(msg)
    s_abs = np.abs(np.atleast_1d(np.asarray(s_target_f)))
    if s_abs.size not in (1, psd.freq.size):
        msg = f"s_target_f length {s_abs.size} does not match the PSD grid {psd.freq.size}"
        raise ValueError(msg)
    if not np.all(s_abs > 0.0):
        msg = "s_target magnitude must be positive everywhere (degenerate working point)"
        raise ValueError(msg)
    out: FloatArray = np.sqrt(np.maximum(psd.values, 0.0)) / s_abs
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
