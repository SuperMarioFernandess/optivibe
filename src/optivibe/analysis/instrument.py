"""Role S-02: analyzer of the real instrument output (photocurrent records).

Turns a recorded photocurrent (TDMS/HDF5/CSV via :mod:`optivibe.io.records`)
plus the instrument config into the **same metric structures the forward tract
produces** -- doc 20 §5 (the S-02 statement of work), 18 G5, 17 §1/§7 -- by
running the record through the *existing* standard inverse chain
(:class:`~optivibe.dsp.standard.StandardDsp`); no parallel metric
implementations exist (17 §7).

Calibration paths (photocurrent -> acceleration; boundary G9, 18 §4.2):

``"model"``
    ``s_target`` derived from the instrument config through the optics/
    mechanics models (the S5 path). **Cylinder-only** -- the model scalar rests
    on :class:`~optivibe.optics.cylinder.CylinderOpticsModel`; a non-cylinder
    variant (e.g. ``proto_poc``, sphere) is rejected loudly with a pointer to
    the measured paths. The boundary is preserved, not generalized.
``"measured"``
    A signed bench scalar supplied by the user (experiment E-3, plan 20 §5;
    the cascade of 19 §3.3 takes the *quantitative-phase* ``s_target`` from
    the bench). Works for any reflector shape.
``"bench"``
    ``s_target`` estimated from the record's own paired reference channel via
    :func:`~optivibe.dsp.calibration.bench_sensitivity` (ISO 16063-21 style
    comparison calibration).

Records without calibration degrade gracefully: the scale-free metrics of
17 §1 -- spectral dominants, the 2f/1f THD proxy, the relative spectrum and
PSD, the AC-current RMS -- are always produced; the calibrated block
(a/v/x, ISO severity, NEA referral) is simply absent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from optivibe.core.config.loader import load_constants, load_variant
from optivibe.core.config.models import Constants, DspOptions, VariantConfig
from optivibe.core.logging import get_logger
from optivibe.core.types import FloatArray, Spectrum, VibrationResult
from optivibe.dsp.calibration import bench_sensitivity, detector_ac_current
from optivibe.dsp.metrics import rms, second_harmonic_ratio
from optivibe.dsp.nea import nea_from_psd
from optivibe.dsp.sensitivity import MeasuredSensitivity, SensitivityModel, build_sensitivity_model
from optivibe.dsp.spectra import amplitude_spectrum, dominant_frequencies, welch_psd
from optivibe.dsp.standard import StandardDsp
from optivibe.io.records import InstrumentRecord, RecordSpec, read_record
from optivibe.mechanics.cantilever import CantileverModel

logger = get_logger(__name__)

__all__ = [
    "AnalyzeSpec",
    "CalibrationSpec",
    "InstrumentAnalysis",
    "analyze_record",
    "load_analyze_spec",
]


class _Frozen(BaseModel):
    """Immutable, strictly-validated base (mirror of the config base)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class CalibrationSpec(_Frozen):
    """The photocurrent -> acceleration calibration choice (20 §5; G9).

    Attributes
    ----------
    source : {"model", "measured", "bench"}
        Where the plateau ``s_target`` comes from (module docstring). The
        choice is the user's (20 §5); the quantitative L3 phase takes the
        bench value of experiment E-3 (cascade 19 §3.3).
    s_target_a_per_m_s2 : float or None
        The signed measured plateau sensitivity, A/(m/s^2). Required for (and
        exclusive to) ``source="measured"``.
    note : str or None
        Free-form traceability note (E-3 protocol id, calibration certificate;
        GUM traceability, 17 §4).
    """

    source: Literal["model", "measured", "bench"]
    s_target_a_per_m_s2: float | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _check_scalar(self) -> CalibrationSpec:
        if self.source == "measured" and self.s_target_a_per_m_s2 is None:
            msg = "calibration source 'measured' requires s_target_a_per_m_s2 (bench E-3 figure)"
            raise ValueError(msg)
        if self.source != "measured" and self.s_target_a_per_m_s2 is not None:
            msg = (
                f"s_target_a_per_m_s2 is exclusive to source 'measured'; "
                f"source {self.source!r} derives its own scalar"
            )
            raise ValueError(msg)
        return self


class AnalyzeSpec(_Frozen):
    """One reproducible instrument-output analysis (role S-02, 20 §5).

    Attributes
    ----------
    kind : "analyze"
        Spec discriminator (mirrors the sweep/montecarlo spec convention).
    name : str
        Human-readable analysis name.
    variant : str
        Instrument-config (variant) name from ``configs/variants/`` (e.g.
        ``"proto_poc"`` once its placeholders are lifted, 20 §5).
    record : RecordSpec
        Photocurrent-record descriptor (path, format, units/R_f, fs,
        timestamp, optional paired reference channel).
    calibration : CalibrationSpec or None
        Calibration choice; ``None`` degrades to the scale-free metrics only.
    dsp : DspOptions
        Inverse/DSP options of the standard chain (integrator, spectrum
        estimator, ISO machine class, ...).
    """

    kind: Literal["analyze"] = "analyze"
    name: str = Field(min_length=1)
    variant: str = Field(min_length=1)
    record: RecordSpec
    calibration: CalibrationSpec | None = None
    dsp: DspOptions = DspOptions()


def load_analyze_spec(path: Path | str) -> AnalyzeSpec:
    """Load and validate an analyze spec from YAML (role S-02).

    Parameters
    ----------
    path : pathlib.Path or str
        Path to a spec YAML file with ``kind: analyze``.

    Returns
    -------
    AnalyzeSpec
        The validated spec.

    Raises
    ------
    ValueError
        If the ``kind`` field is missing or is not ``"analyze"``.
    """
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    kind = raw.get("kind")
    if kind != "analyze":
        msg = f"analyze spec needs kind 'analyze', got {kind!r}"
        raise ValueError(msg)
    return AnalyzeSpec.model_validate(raw)


@dataclass(frozen=True)
class InstrumentAnalysis:
    """Metrics of one recorded instrument output (role S-02; 17 §1).

    The scale-free photocurrent products are always present; the calibrated
    block (``result``/``nea_*``) is filled only when a calibration source was
    resolved (graceful degradation, 20 §5).

    Attributes
    ----------
    spectrum : Spectrum
        Representative photocurrent spectrum per the DSP options (amplitude
        rFFT or Welch PSD), in record units (A / A^2/Hz).
    psd : Spectrum
        Welch PSD of the AC photocurrent, A^2/Hz (basis of the NEA referral).
    dominant_freqs_hz : tuple of float
        Spectral dominants of the photocurrent, Hz (17 §1) -- scale-free.
    second_harmonic_ratio : float or None
        ``|X(2 f0)|/|X(f0)|`` at the leading dominant (the THD proxy of the
        tract, doc 04 §5) -- scale-free; ``None`` without a dominant.
    i_ac_rms_a : float
        RMS of the AC photocurrent, A.
    calibration : dict or None
        The resolved calibration (``source``, the ``s_target`` actually used,
        the traceability ``note``); ``None`` when uncalibrated.
    result : VibrationResult or None
        The full calibrated inverse-chain output -- **the same structure the
        forward tract produces** (17 §7): a/v/x (SI), dominants, RMS metrics,
        cross residual, spectrum and the ISO severity assessment.
    nea_freq_hz, nea_density : numpy.ndarray or None
        Measured-PSD NEA referral ``sqrt(S_I(f))/|s_target D(f)|``,
        (m/s^2)/sqrt(Hz) (doc 05 §7); requires calibration.
    nea_full_band_m_s2 : float or None
        ``sqrt(integral of NEA(f)^2 df)`` over the variant band, m/s^2 (the
        spectral-integral convention pinned by 18 G3).
    s_target_bench_estimate : float or None
        Informational bench estimate from the paired reference channel
        (:func:`~optivibe.dsp.calibration.bench_sensitivity`), A/(m/s^2);
        ``None`` without a reference channel.
    reference_rms_m_s2 : float or None
        RMS of the reference acceleration, m/s^2.
    meta : dict
        Record and variant metadata (path, fs, timestamp, units, n_samples).
    """

    spectrum: Spectrum
    psd: Spectrum
    dominant_freqs_hz: tuple[float, ...]
    second_harmonic_ratio: float | None
    i_ac_rms_a: float
    calibration: dict[str, object] | None = None
    result: VibrationResult | None = None
    nea_freq_hz: FloatArray | None = None
    nea_density: FloatArray | None = None
    nea_full_band_m_s2: float | None = None
    s_target_bench_estimate: float | None = None
    reference_rms_m_s2: float | None = None
    meta: dict[str, object] = field(default_factory=dict)


def _resolve_sensitivity(
    spec: AnalyzeSpec,
    variant: VariantConfig,
    constants: Constants,
    bench_estimate: float | None,
) -> tuple[SensitivityModel, float] | None:
    """Resolve the calibration choice to a sensitivity model and its scalar.

    Returns ``None`` when the spec carries no calibration (graceful
    degradation). Enforces boundary G9 for the ``"model"`` source.
    """
    calibration = spec.calibration
    if calibration is None:
        return None
    if calibration.source == "model":
        shape = variant.reflector.shape
        if shape != "cylinder":
            msg = (
                f"calibration source 'model' is cylinder-only (boundary G9, doc 18 §4.2): "
                f"variant {variant.name!r} has reflector.shape={shape!r}; supply the bench "
                f"scalar (source 'measured', experiment E-3 / plan 20 §5) or use source "
                f"'bench' with a paired reference channel"
            )
            raise ValueError(msg)
        model = build_sensitivity_model(variant, spec.dsp, constants)
        return model, model.plateau_value
    if calibration.source == "measured":
        assert calibration.s_target_a_per_m_s2 is not None  # spec validator
        s_target = calibration.s_target_a_per_m_s2
        return MeasuredSensitivity(variant, constants, s_target), s_target
    # source == "bench": estimated from the record's own reference channel.
    if bench_estimate is None:
        msg = (
            "calibration source 'bench' needs a paired reference channel in the record "
            "descriptor (record.reference; mandatory metadata of plan 20 §5)"
        )
        raise ValueError(msg)
    return MeasuredSensitivity(variant, constants, bench_estimate), bench_estimate


def _nea_referral(
    psd: Spectrum,
    s_target: float,
    variant: VariantConfig,
    constants: Constants,
) -> tuple[FloatArray, FloatArray, float | None]:
    """Refer the measured PSD to the input across frequency (doc 05 §7).

    ``|s_target(f)| = |s_target^QS| |D(f)|`` uses the variant's cantilever
    (shape-agnostic mechanics), so the referral dips toward ``f1`` for every
    calibration source. Returns the grid, the NEA density and the full-band
    figure ``sqrt(integral NEA^2 df)`` over the variant band (18 G3 convention),
    or ``None`` when fewer than two PSD points fall inside the band.
    """
    cantilever = CantileverModel.from_config(constants, variant)
    s_f = s_target * cantilever.dynamic_factor(psd.freq)
    density = nea_from_psd(psd, s_f)
    lo, hi = variant.band.f_min_hz, variant.band.f_max_hz
    in_band = (psd.freq >= lo) & (psd.freq <= hi)
    full_band: float | None = None
    if int(np.count_nonzero(in_band)) >= 2:
        power = float(np.trapezoid(density[in_band] ** 2, psd.freq[in_band]))
        full_band = math.sqrt(max(power, 0.0))
    return np.ascontiguousarray(psd.freq, dtype=np.float64), density, full_band


def analyze_record(
    spec: AnalyzeSpec,
    *,
    config_dir: Path | None = None,
    constants: Constants | None = None,
) -> InstrumentAnalysis:
    """Analyze one recorded instrument output through the standard tract (S-02).

    Loads the record (:func:`optivibe.io.records.read_record`), always computes
    the scale-free photocurrent metrics, resolves the calibration choice
    (module docstring; boundary G9) and, when calibrated, runs the record
    through the **same** :class:`~optivibe.dsp.standard.StandardDsp` chain the
    forward simulation uses (17 §7), plus the measured-PSD NEA referral
    (doc 05 §7).

    Parameters
    ----------
    spec : AnalyzeSpec
        The validated analysis spec (record + variant + calibration + DSP).
    config_dir : pathlib.Path or None, optional
        Override of the ``configs/`` directory (variant and constants lookup).
    constants : Constants or None, optional
        Physical constants; loaded from the config dir when ``None``.

    Returns
    -------
    InstrumentAnalysis
        The metric bundle (17 §1): scale-free products always, the calibrated
        block when a calibration source was resolved.

    Raises
    ------
    ValueError
        On a non-cylinder variant with the ``"model"`` source (boundary G9),
        on a ``"bench"`` source without a reference channel, or on unit /
        metadata problems surfaced by the record reader.
    """
    consts = (
        load_constants(config_dir / "constants.yaml" if config_dir is not None else None)
        if constants is None
        else constants
    )
    variant = load_variant(spec.variant, config_dir)
    record: InstrumentRecord = read_record(spec.record)
    detector = record.detector
    fs = detector.fs

    # Scale-free photocurrent products (always; 20 §5 graceful degradation).
    i_ac = detector_ac_current(detector, variant)
    amplitude = amplitude_spectrum(i_ac, fs)
    psd = welch_psd(
        i_ac,
        fs,
        window=spec.dsp.window,
        nperseg=spec.dsp.welch_nperseg,
        noverlap=spec.dsp.welch_noverlap,
    )
    spectrum = psd if spec.dsp.spectrum_method == "welch" else amplitude
    dominant = dominant_frequencies(spectrum)
    shr = second_harmonic_ratio(amplitude, dominant[0]) if dominant else None

    bench_estimate: float | None = None
    reference_rms: float | None = None
    if record.reference_accel is not None:
        bench_estimate = bench_sensitivity(detector, record.reference_accel, variant)
        reference_rms = rms(record.reference_accel)

    meta: dict[str, object] = {
        **record.meta,
        "variant": variant.name,
        "fs_hz": fs,
        "n_samples": detector.n_samples,
    }

    resolved = _resolve_sensitivity(spec, variant, consts, bench_estimate)
    if resolved is None:
        logger.info("analyze '%s': no calibration -- scale-free metrics only", spec.name)
        return InstrumentAnalysis(
            spectrum=spectrum,
            psd=psd,
            dominant_freqs_hz=dominant,
            second_harmonic_ratio=shr,
            i_ac_rms_a=rms(i_ac),
            s_target_bench_estimate=bench_estimate,
            reference_rms_m_s2=reference_rms,
            meta=meta,
        )

    model, s_target = resolved
    # The one and only inverse chain (17 §7): calibration -> kinematics ->
    # spectra/dominants -> RMS -> ISO. Injecting the resolved model keeps the
    # measured paths on the same code as the forward tract (SW-52).
    stage = StandardDsp(constants=consts, sensitivity_model=model)
    result = stage.run(detector, variant, spec.dsp)
    nea_freq, nea_density, nea_full_band = _nea_referral(psd, s_target, variant, consts)
    calibration_info: dict[str, object] = {
        "source": spec.calibration.source if spec.calibration is not None else "?",
        "s_target_a_per_m_s2": s_target,
        "note": spec.calibration.note if spec.calibration is not None else None,
    }
    logger.info(
        "analyze '%s': calibrated (%s, s_target=%.4g A/(m/s^2))",
        spec.name,
        calibration_info["source"],
        s_target,
    )
    return InstrumentAnalysis(
        spectrum=spectrum,
        psd=psd,
        dominant_freqs_hz=dominant,
        second_harmonic_ratio=shr,
        i_ac_rms_a=rms(i_ac),
        calibration=calibration_info,
        result=result,
        nea_freq_hz=nea_freq,
        nea_density=nea_density,
        nea_full_band_m_s2=nea_full_band,
        s_target_bench_estimate=bench_estimate,
        reference_rms_m_s2=reference_rms,
        meta=meta,
    )
