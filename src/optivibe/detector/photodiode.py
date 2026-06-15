"""Photodiode detector: optical response -> noisy, digitized photocurrent.

S4 replaces :class:`~optivibe.detector.StubDetector` by the physical read-out of
documents 07 (noise budget / NEA) and 05 §5.3 (normalizations). The forward
chain is route 2 (doc 07 §1): the photocurrent is

``I(t) = R * P * (R1 + rho * eta(t))``  [A],

with the DC pedestal ``I_DC = R * P * (R1 + rho * eta0)`` carrying both the
endface Fresnel reflection ``R1`` and the bias coupling ``rho * eta0`` (only
``rho * eta`` carries signal; the pedestal raises the shot noise without adding
signal -- doc 07 §1.1). The canonical signal current is ``I_AC = R*P*rho*Delta
eta`` and the conversion factor between coupling modulation and detected
modulation is ``(Delta eta / eta0) / (I_AC / I_DC) = 1 + R1/(rho eta0)``
(doc 05 §5.3): ~1.15 for metal (rho=0.98) and ~5.0 for bare (rho=0.036) at the
illustrative eta0 = 0.25.

Noise model (one-sided current PSD on the Nyquist band ``fs/2``; doc 07 §1):

* **shot** -- white, ``S_shot = 2 e I_DC`` [A^2/Hz] (doc 07 §1.1);
* **RIN** -- white at the configured level, ``S_RIN = I_DC^2 * RIN0`` with
  ``RIN0 = 10^(RIN[dB/Hz]/10)`` (doc 07 §1.2); a spectral shape is a recorded
  extension (``rin_shape``);
* **Johnson / electronics** -- ``S_J = 4 kB T / Rf`` referred to current, present
  only when a transimpedance ``Rf`` is configured (doc 07 §1.3).

**Balanced reference channel (R-23, doc 07 §1.2).** RIN is common-mode between
the signal and reference arms and is suppressed by the common-mode rejection
ratio, ``S_RIN_eff = S_RIN * 10^(-CMRR/10)`` (i.e. ``10^(-CMRR/20)`` in RMS).
Shot noise is uncorrelated between arms and is *not* suppressed; with the chosen
arm model (two matched arms, the signal arm kept at full power and the reference
an additional matched tap) the difference carries shot from both arms, so the
shot PSD is taken as ``2 * 2 e I_DC`` (the worst case ``<= sqrt(2)`` in RMS of
doc 07 §1.2). Splitting the power 50/50 instead would leave the shot PSD at
``2 e I_DC``; the conservative matched-tap model is documented and used so the
shot budget never flatters the floor.

**ADC (doc 07 §1.4, AC-coupled front-end).** The DC pedestal drifts on thermal
time scales (``< 0.01 Hz``) and is removed by AC-coupling before the gain/ADC
stage, so the converter's full scale spans only the AC modulation (the pedestal
``I_DC = 3 mA`` would otherwise force ~30 effective bits to resolve a nanoamp).
The returned ``samples`` therefore are ``dc_level + ADC(I(t) - I_DC + noise)``;
``dc_level`` is reported as the analog pedestal (the DSP removes it). Inputs
beyond the full scale are clipped and logged (doc 10 §7).

Reproducibility (10 §8). The :class:`~optivibe.core.stages.DetectorStage`
protocol ``run(optical, variant)`` carries no seed, so -- following the S2
options pattern -- the scenario seed is handed to the *constructor* by the
orchestrator and an independent, deterministic noise sub-stream is derived from
it (see :func:`detector_seed_sequence`). One scenario seed -> bit-identical
noise.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.signal import resample_poly

from optivibe.core.config.loader import default_config_dir, load_constants
from optivibe.core.config.models import Constants, DetectorConfig, VariantConfig
from optivibe.core.logging import get_logger
from optivibe.core.types import DetectorOutput, FloatArray, OpticalResponse

logger = get_logger(__name__)

__all__ = [
    "PhotodiodeDetector",
    "detector_seed_sequence",
    "johnson_psd",
    "noise_psd",
    "rin_psd",
    "shot_psd",
    "signal_multiplier",
]

# Fixed tag mixed into the scenario seed so the detector's noise stream is
# reproducible yet independent of the excitation stream (which consumes the raw
# scenario seed). Any fixed integer works; this one spells "DET0".
_DETECTOR_SEED_TAG = 0x44455430


def detector_seed_sequence(scenario_seed: int | None) -> np.random.SeedSequence:
    """Derive the detector's noise seed sequence from the scenario seed.

    A fixed tag is mixed in so the stream is deterministic for a given scenario
    seed and independent of the excitation stream (10 §8). When the scenario
    seed is ``None`` a fresh, non-reproducible sequence is returned.

    Parameters
    ----------
    scenario_seed : int or None
        The scenario-level seed (``scenario.seed``).

    Returns
    -------
    numpy.random.SeedSequence
        Seed sequence feeding the detector's :class:`numpy.random.Generator`.
    """
    if scenario_seed is None:
        return np.random.SeedSequence()
    return np.random.SeedSequence(entropy=[int(scenario_seed), _DETECTOR_SEED_TAG])


def signal_multiplier(endface_reflectivity: float, reflectivity: float, eta0: float) -> float:
    """Return the normalization factor ``1 + R1 / (rho * eta0)`` (doc 05 §5.3).

    This is the ratio between the relative coupling modulation ``Delta eta /
    eta0`` and the detected modulation depth ``I_AC / I_DC``; the endface
    pedestal ``R1`` dilutes the detected modulation. For the illustrative
    ``eta0 = 0.25`` it is ~1.15 (metal, ``rho = 0.98``) and 5.0 (bare,
    ``rho = 0.036``).

    Parameters
    ----------
    endface_reflectivity : float
        Fiber endface Fresnel reflectivity ``R1`` (dimensionless).
    reflectivity : float
        Mirror reflectivity ``rho`` (dimensionless).
    eta0 : float
        Optical working point ``eta0`` (dimensionless, > 0).

    Returns
    -------
    float
        The factor ``1 + R1 / (rho * eta0)`` (>= 1).

    Raises
    ------
    ValueError
        If ``reflectivity`` or ``eta0`` is not strictly positive.
    """
    if reflectivity <= 0.0 or eta0 <= 0.0:
        msg = f"reflectivity and eta0 must be positive, got {reflectivity!r}, {eta0!r}"
        raise ValueError(msg)
    return 1.0 + endface_reflectivity / (reflectivity * eta0)


def shot_psd(i_dc_a: float, elementary_charge_c: float, *, arm_factor: float = 1.0) -> float:
    """One-sided shot-noise current PSD ``arm_factor * 2 e I_DC`` [A^2/Hz] (07 §1.1).

    The bare single-detector shot PSD is ``2 e I_DC``. ``arm_factor`` carries the
    reference-arm model of the balanced channel (doc 07 §1.2): ``1.0`` for a
    single-ended / bright-reference channel (``P_ref >> P_sig``, the datasheet
    shot limit) and ``2.0`` for two matched arms (the conservative ``<= sqrt(2)``
    RMS case). It is *not* suppressed by the CMRR -- shot is uncorrelated.

    Parameters
    ----------
    i_dc_a : float
        DC photocurrent ``I_DC``, A.
    elementary_charge_c : float
        Elementary charge ``e``, C.
    arm_factor : float, optional
        Reference-arm shot multiplier (1.0 single/bright, 2.0 matched).

    Returns
    -------
    float
        Shot-noise PSD, A^2/Hz.
    """
    return arm_factor * 2.0 * elementary_charge_c * i_dc_a


def shot_arm_factor(*, balanced: bool, reference_arm: str) -> float:
    """Map the balanced channel + reference-arm model to the shot multiplier.

    Open question O-SW-08: which convention is "the" NEA is deferred to test-time
    evaluation. ``"matched"`` (two equal arms, signal arm at full power) doubles
    the shot PSD -- the conservative engineering floor; ``"bright"``
    (``P_ref >> P_sig`` / normalization) leaves it at ``2 e I_DC`` -- the
    datasheet/doc-08 shot limit. A single-ended channel (``balanced=False``) also
    uses the bare ``2 e I_DC``.

    Parameters
    ----------
    balanced : bool
        Whether the balanced reference channel is active.
    reference_arm : {"matched", "bright"}
        Reference-arm model (only meaningful when ``balanced``).

    Returns
    -------
    float
        Shot multiplier (1.0 or 2.0).
    """
    if balanced and reference_arm == "matched":
        return 2.0
    return 1.0


def rin_psd(i_dc_a: float, rin_db_hz: float, cmrr_db: float, *, balanced: bool) -> float:
    """One-sided RIN current PSD ``I_DC^2 * RIN0`` [A^2/Hz] (doc 07 §1.2).

    ``RIN0 = 10^(RIN[dB/Hz]/10)``. With the balanced channel the level is
    suppressed by ``10^(-CMRR/10)`` (common-mode rejection).

    Parameters
    ----------
    i_dc_a : float
        DC photocurrent ``I_DC``, A.
    rin_db_hz : float
        Source relative intensity noise, dB/Hz.
    cmrr_db : float
        Common-mode rejection ratio of the balanced channel, dB.
    balanced : bool
        Whether the balanced reference channel is active.

    Returns
    -------
    float
        RIN current PSD (effective), A^2/Hz.
    """
    rin0 = 10.0 ** (rin_db_hz / 10.0)
    suppression = 10.0 ** (-cmrr_db / 10.0) if balanced else 1.0
    return float(i_dc_a**2 * rin0 * suppression)


def johnson_psd(
    transimpedance_ohm: float | None, boltzmann_j_k: float, temperature_k: float
) -> float:
    """One-sided Johnson current PSD ``4 kB T / Rf`` [A^2/Hz] (doc 07 §1.3).

    Returns ``0`` when no transimpedance resistor is configured (the
    ``Rf -> inf`` electronics-noiseless limit).

    Parameters
    ----------
    transimpedance_ohm : float or None
        Feedback resistor ``Rf``, ohm; ``None`` means no electronics floor.
    boltzmann_j_k : float
        Boltzmann constant ``kB``, J/K.
    temperature_k : float
        Absolute temperature ``T``, K.

    Returns
    -------
    float
        Johnson-noise current PSD, A^2/Hz.
    """
    if transimpedance_ohm is None:
        return 0.0
    return 4.0 * boltzmann_j_k * temperature_k / transimpedance_ohm


def noise_psd(
    i_dc_a: float,
    variant: VariantConfig,
    constants: Constants,
    *,
    balanced: bool,
    reference_arm: str | None = None,
) -> dict[str, float]:
    """Assemble the one-sided current-noise PSD components (doc 07 §1).

    Parameters
    ----------
    i_dc_a : float
        DC photocurrent ``I_DC``, A.
    variant : VariantConfig
        Sensor variant (source RIN, detector electronics).
    constants : Constants
        Physical constants (``e``, ``kB``, ``T``).
    balanced : bool
        Whether the balanced reference channel is active.
    reference_arm : {"matched", "bright"} or None, optional
        Reference-arm shot model (O-SW-08); the variant's
        ``detector.reference_arm`` is used when None.

    Returns
    -------
    dict of str to float
        Components ``"shot"``, ``"rin"``, ``"johnson"`` and their ``"total"``
        (the sum, since the sources are independent), each in A^2/Hz.
    """
    det = variant.detector
    dc = constants.detector
    arm = det.reference_arm if reference_arm is None else reference_arm
    arm_factor = shot_arm_factor(balanced=balanced, reference_arm=arm)
    shot = shot_psd(i_dc_a, dc.elementary_charge_c, arm_factor=arm_factor)
    rin = rin_psd(i_dc_a, variant.source.rin_db_hz, det.cmrr_db, balanced=balanced)
    johnson = johnson_psd(det.transimpedance_ohm, dc.boltzmann_j_k, dc.temperature_k)
    return {"shot": shot, "rin": rin, "johnson": johnson, "total": shot + rin + johnson}


def _quantize(signal: FloatArray, full_scale: float, bits: int) -> tuple[FloatArray, float, int]:
    """Mid-tread uniform quantization of ``signal`` over ``[-FS, +FS]``.

    Parameters
    ----------
    signal : numpy.ndarray
        AC-coupled signal to quantize (same units as ``full_scale``).
    full_scale : float
        Converter full scale (the +/- range), > 0.
    bits : int
        Resolution in bits.

    Returns
    -------
    quantized : numpy.ndarray
        Quantized signal.
    lsb : float
        Least-significant-bit step ``2 FS / 2^bits``.
    n_clipped : int
        Number of samples that exceeded the full scale (clipped).
    """
    levels = 2**bits
    lsb = 2.0 * full_scale / levels
    n_clipped = int(np.count_nonzero(np.abs(signal) > full_scale))
    clipped = np.clip(signal, -full_scale, full_scale - lsb)
    quantized: FloatArray = np.round(clipped / lsb) * lsb
    return quantized, lsb, n_clipped


def _antialias_decimate(signal: FloatArray, fs: float, adc_fs: float) -> tuple[FloatArray, float]:
    """Anti-alias filter and resample ``signal`` from ``fs`` to ~``adc_fs``.

    Uses a rational polyphase resampler (FIR anti-alias built in). When
    ``adc_fs >= fs`` the signal is returned unchanged (identity).

    Parameters
    ----------
    signal : numpy.ndarray
        Input samples at ``fs``.
    fs : float
        Input sampling frequency, Hz.
    adc_fs : float
        Target sampling frequency, Hz.

    Returns
    -------
    out : numpy.ndarray
        Resampled samples.
    fs_out : float
        Achieved sampling frequency, Hz (the realized rational ratio).
    """
    if adc_fs >= fs:
        return signal, fs
    # Rational ratio up/down with a small common-denominator approximation.
    ratio = adc_fs / fs
    down = max(2, round(1.0 / ratio))
    up = 1
    out: FloatArray = np.ascontiguousarray(resample_poly(signal, up, down), dtype=np.float64)
    fs_out = fs * up / down
    return out, fs_out


class PhotodiodeDetector:
    """Physical photodiode read-out with noise and an AC-coupled ADC (S4).

    Registered under ``"photodiode"``. Implements the route-2 read-out and the
    noise budget of document 07 (shot / RIN / Johnson), the balanced reference
    channel (R-23) and the AC-coupled ADC, behind the unchanged
    :class:`~optivibe.core.stages.DetectorStage` protocol.

    Parameters
    ----------
    balanced : bool or None, optional
        Per-scenario override of the variant's balanced-channel flag; the
        variant value (``variant.detector.balanced``) is used when ``None``.
        Injected by the orchestrator from ``scenario.detector``.
    reference_arm : {"matched", "bright"} or None, optional
        Per-scenario override of the reference-arm shot model (O-SW-08); the
        variant value (``variant.detector.reference_arm``) is used when ``None``.
        ``"matched"`` doubles the shot PSD (conservative two-arm floor),
        ``"bright"`` keeps the bare ``2 e I_DC`` (datasheet/doc-08 shot limit).
    scenario_seed : int or None, optional
        The scenario seed; an independent, reproducible noise sub-stream is
        derived from it (:func:`detector_seed_sequence`). Injected by the
        orchestrator from ``scenario.seed``.
    constants : Constants or None, optional
        Physical constants; loaded once from ``configs/constants.yaml`` when
        ``None`` (the only I/O, performed at construction).
    """

    def __init__(
        self,
        *,
        balanced: bool | None = None,
        reference_arm: str | None = None,
        scenario_seed: int | None = None,
        constants: Constants | None = None,
    ) -> None:
        if constants is None:
            constants = load_constants(default_config_dir() / "constants.yaml")
        self._constants = constants
        self._balanced_override = balanced
        self._reference_arm_override = reference_arm
        self._rng = np.random.default_rng(detector_seed_sequence(scenario_seed))

    def run(self, optical: OpticalResponse, variant: VariantConfig) -> DetectorOutput:
        """Convert the coupling response to noisy, digitized detector samples.

        Parameters
        ----------
        optical : OpticalResponse
            Coupling efficiency ``eta(t)`` with the computed working point
            ``eta0`` in :attr:`OpticalResponse.bias`.
        variant : VariantConfig
            Sensor variant (provides P, R, rho, R1, source RIN, detector
            electronics and ADC).

        Returns
        -------
        DetectorOutput
            Digitized samples (current or transimpedance voltage), the analog DC
            pedestal and a noise-metadata mapping.
        """
        det: DetectorConfig = variant.detector
        balanced = self._balanced_override if self._balanced_override is not None else det.balanced
        reference_arm = (
            self._reference_arm_override
            if self._reference_arm_override is not None
            else det.reference_arm
        )

        gain = variant.responsivity_a_w * variant.source.power_w
        r1 = variant.endface_reflectivity
        rho = variant.reflector.reflectivity

        # Analog photocurrent I(t) and its DC pedestal (route 2; doc 07 §1).
        current: FloatArray = gain * (r1 + rho * optical.eta)
        i_dc = float(gain * (r1 + rho * optical.bias))

        # Stationary current-noise from the DC operating point (doc 07 §1). The
        # noise depends only on I_DC, so it is additive w.r.t. the signal.
        psd = noise_psd(
            i_dc, variant, self._constants, balanced=balanced, reference_arm=reference_arm
        )
        nyquist_bw = optical.fs / 2.0
        sigma_i = float(np.sqrt(psd["total"] * nyquist_bw))
        noise_current: FloatArray = self._rng.standard_normal(current.size) * sigma_i

        # Optional transimpedance: amperes -> volts (doc 07 §1.3).
        rf = det.transimpedance_ohm
        if det.output == "voltage":
            if rf is None:
                msg = "detector output 'voltage' requires a transimpedance_ohm"
                raise ValueError(msg)
            scale, units = rf, "V"
        else:
            scale, units = 1.0, "A"
        units_literal: Literal["A", "V"] = "V" if units == "V" else "A"

        dc_level = i_dc * scale
        ac = (current - i_dc + noise_current) * scale

        # AC-coupled ADC: quantize the modulation over the configured full scale
        # (doc 07 §1.4). The DC pedestal is reported separately, not digitized.
        quantized, lsb, n_clipped = _quantize(ac, det.adc_full_scale, det.adc_bits)
        fs_out = optical.fs
        if det.adc_fs_hz is not None and det.antialias:
            quantized, fs_out = _antialias_decimate(quantized, optical.fs, det.adc_fs_hz)
        elif det.adc_fs_hz is not None:
            # No anti-alias filter requested: plain (aliasing) downselection.
            step = max(1, round(optical.fs / det.adc_fs_hz))
            quantized = np.ascontiguousarray(quantized[::step], dtype=np.float64)
            fs_out = optical.fs / step

        samples: FloatArray = dc_level + quantized

        if n_clipped:
            logger.warning(
                "ADC saturation: %d/%d samples clipped at +/- %.3e %s full scale",
                n_clipped,
                ac.size,
                det.adc_full_scale,
                units_literal,
            )

        noise_meta: dict[str, object] = {
            "model": "photodiode",
            "balanced": balanced,
            "reference_arm": reference_arm,
            "i_dc_a": i_dc,
            "psd_shot_a2_hz": psd["shot"],
            "psd_rin_a2_hz": psd["rin"],
            "psd_johnson_a2_hz": psd["johnson"],
            "psd_total_a2_hz": psd["total"],
            "sigma_i_a": sigma_i,
            "nyquist_bw_hz": nyquist_bw,
            "adc_bits": det.adc_bits,
            "adc_lsb": lsb,
            "adc_full_scale": det.adc_full_scale,
            "n_clipped": n_clipped,
            "units": units_literal,
        }
        return DetectorOutput(
            samples=samples,
            fs=fs_out,
            dc_level=float(dc_level),
            units=units_literal,
            noise=noise_meta,
        )
