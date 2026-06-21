"""Parameter sweep engine: design and response maps (task S6 §B7).

*Design* sweeps recompute the analytic ``s_target``/NEA/modulation over ``L``,
``R_c``, ``P``, bias or ``FS`` (and across variants) -- cheap, no time-domain run
-- so they reproduce the family trends of doc 08 *and* the SW-26 KB refinement
(``R_c`` 62.5 -> 31 um lifts the absolute slope x1.31 and the NEA by x1.57, not
x2). *Response* sweeps run forward + inverse over amplitude (0.1 g - >50 g, where
the optical curvature and the ADC clip show as THD / gain droop / saturation,
doc 00) or frequency, and measure the recovered response.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optivibe.analysis.nea_budget import nea_budget
from optivibe.analysis.spec import SweepSpec
from optivibe.analysis.truth_vs_recovery import truth_vs_recovery
from optivibe.analysis.variant_tools import G0, analytic_point, with_overrides
from optivibe.core.config.loader import load_constants, load_variant
from optivibe.core.config.models import (
    Constants,
    DetectorOptions,
    DspOptions,
    ScenarioConfig,
    SineSpec,
    StageSelection,
    VariantConfig,
)
from optivibe.core.types import FloatArray
from optivibe.dsp.metrics import rms, second_harmonic_ratio
from optivibe.dsp.spectra import amplitude_spectrum
from optivibe.pipeline.orchestrator import Pipeline

__all__ = ["SweepResult", "run_sweep"]


@dataclass(frozen=True)
class SweepResult:
    """Result of a parameter sweep (task S6 §B7).

    Attributes
    ----------
    name : str
        Sweep name.
    mode : str
        ``"design"`` or ``"response"``.
    parameter : str
        Swept parameter name.
    variant : str
        Base variant identifier.
    axis_values : numpy.ndarray
        Numeric axis values (NaN-free); for the ``variant`` axis the labels are
        in ``axis_labels`` and ``axis_values`` is the index.
    axis_labels : list of str
        String labels of the axis points (variant names, or formatted numbers).
    metrics : Mapping[str, numpy.ndarray]
        Metric name -> values over the axis (aligned with ``axis_values``).
    meta : Mapping[str, object]
        Free-form metadata (units, fixed tone, seed).
    """

    name: str
    mode: str
    parameter: str
    variant: str
    axis_values: FloatArray
    axis_labels: list[str]
    metrics: dict[str, FloatArray]
    meta: dict[str, object] = field(default_factory=dict)


def _design_variants(spec: SweepSpec) -> tuple[FloatArray, list[str], list[VariantConfig]]:
    """Build the per-point variants and axis for a design sweep."""
    base = load_variant(spec.variant)
    if spec.parameter == "variant":
        names = spec.variant_values or []
        labels = [str(name) for name in names]
        variants = [load_variant(name) for name in names]
        axis = np.arange(len(labels), dtype=np.float64)
        return axis, labels, variants
    assert spec.grid is not None  # validated by the spec
    axis = spec.grid.values()
    variants = [with_overrides(base, **{spec.parameter: float(v)}) for v in axis]
    labels = [f"{v:.4g}" for v in axis]
    return axis, labels, variants


def _run_design(spec: SweepSpec, constants: Constants) -> SweepResult:
    """Recompute analytic sensitivity/NEA over the design axis."""
    axis, labels, variants = _design_variants(spec)
    keys = (
        "s_target",
        "nea_plateau_ug",
        "nea_full_band",
        "f1_hz",
        "eta0",
        "eta_peak",
        "eta_ratio",
        "i_dc_a",
        "modulation_at_fs",
        "dynamic_range_db",
    )
    accum: dict[str, list[float]] = {k: [] for k in keys}
    for variant in variants:
        point = analytic_point(variant, constants)
        for k in keys:
            accum[k].append(float(getattr(point, k)))
    metrics = {k: np.asarray(v, dtype=np.float64) for k, v in accum.items()}
    return SweepResult(
        name=spec.name,
        mode="design",
        parameter=spec.parameter,
        variant=spec.variant,
        axis_values=axis,
        axis_labels=labels,
        metrics=metrics,
        meta={"unit": _PARAM_UNITS.get(spec.parameter, ""), "nea_unit": "ug/sqrt(Hz)"},
    )


_PARAM_UNITS = {
    "length_m": "m",
    "radius_of_curvature_m": "m",
    "power_w": "W",
    "bias_offset_m": "m",
    "full_scale_g": "g",
    "amplitude_g": "g",
    "frequency_hz": "Hz",
}


def _response_point(
    variant: VariantConfig,
    *,
    frequency_hz: float,
    amplitude_g: float,
    fs_hz: float,
    duration_s: float,
    detector: DetectorOptions,
    dsp: DspOptions,
    seed: int | None,
) -> dict[str, float]:
    """Run forward + inverse for one tone and read out the response metrics."""
    excitation = SineSpec(
        kind="sine",
        axis="x",
        fs_hz=fs_hz,
        duration_s=duration_s,
        frequency_hz=frequency_hz,
        amplitude_g=amplitude_g,
    )
    stages = StageSelection(detector="photodiode", dsp="standard")
    scenario = ScenarioConfig(
        name="sweep-point",
        variant=variant.name,
        excitation=excitation,
        stages=stages,
        detector=detector,
        dsp=dsp,
        seed=seed,
    )
    artifacts = Pipeline(scenario, variant).run()
    rec = np.asarray(artifacts.result.a, dtype=np.float64)
    applied = np.asarray(artifacts.forward.excitation.a_x, dtype=np.float64)
    eta = np.asarray(artifacts.forward.optical.eta, dtype=np.float64)
    n_clipped = int(float(str(artifacts.forward.detector.noise.get("n_clipped", 0))))
    budget = nea_budget(artifacts.forward.detector, variant)
    err = truth_vs_recovery(
        artifacts.forward.excitation,
        artifacts.result,
        artifacts.forward.detector,
        variant=variant,
    )
    return {
        "applied_rms_g": rms(applied) / G0,
        "recovered_rms_g": rms(rec) / G0,
        "gain_ratio": (rms(rec) / rms(applied)) if rms(applied) > 0 else float("nan"),
        "thd_recovered_pct": 100.0
        * second_harmonic_ratio(amplitude_spectrum(rec, fs_hz), frequency_hz),
        "optical_thd_pct": 100.0
        * second_harmonic_ratio(amplitude_spectrum(eta - float(np.mean(eta)), fs_hz), frequency_hz),
        "peak_dx_um": float(np.max(np.abs(artifacts.forward.tip.dx))) * 1.0e6,
        "n_clipped": float(n_clipped),
        "saturated": 1.0 if n_clipped > 0 else 0.0,
        "nea_full_band_ug": (budget.nea_full_band / G0 * 1.0e6)
        if budget is not None
        else float("nan"),
        "recovery_rel_error": err.rms_error_rel,
    }


def _run_response(spec: SweepSpec, constants: Constants) -> SweepResult:
    """Run forward + inverse over the response axis (amplitude or frequency)."""
    assert spec.grid is not None  # validated by the spec
    variant = load_variant(spec.variant)
    axis = spec.grid.values()
    keys = (
        "applied_rms_g",
        "recovered_rms_g",
        "gain_ratio",
        "thd_recovered_pct",
        "optical_thd_pct",
        "peak_dx_um",
        "n_clipped",
        "saturated",
        "nea_full_band_ug",
        "recovery_rel_error",
    )
    accum: dict[str, list[float]] = {k: [] for k in keys}
    for value in axis:
        freq = float(value) if spec.parameter == "frequency_hz" else spec.frequency_hz
        amp = float(value) if spec.parameter == "amplitude_g" else spec.amplitude_g
        point = _response_point(
            variant,
            frequency_hz=freq,
            amplitude_g=amp,
            fs_hz=spec.fs_hz,
            duration_s=spec.duration_s,
            detector=spec.detector,
            dsp=spec.dsp,
            seed=spec.seed,
        )
        for k in keys:
            accum[k].append(point[k])
    metrics = {k: np.asarray(v, dtype=np.float64) for k, v in accum.items()}
    return SweepResult(
        name=spec.name,
        mode="response",
        parameter=spec.parameter,
        variant=spec.variant,
        axis_values=axis,
        axis_labels=[f"{v:.4g}" for v in axis],
        metrics=metrics,
        meta={
            "unit": _PARAM_UNITS.get(spec.parameter, ""),
            "frequency_hz": spec.frequency_hz,
            "amplitude_g": spec.amplitude_g,
            "fs_hz": spec.fs_hz,
            "seed": spec.seed,
        },
    )


def run_sweep(spec: SweepSpec, constants: Constants | None = None) -> SweepResult:
    """Execute a parameter sweep (design or response) from its spec (S6 §B7).

    Parameters
    ----------
    spec : SweepSpec
        Validated sweep specification.
    constants : Constants or None, optional
        Physical constants (default loaded when ``None``).

    Returns
    -------
    SweepResult
        Axis values, labels and the per-metric arrays.
    """
    consts = load_constants() if constants is None else constants
    if spec.mode == "design":
        return _run_design(spec, consts)
    return _run_response(spec, consts)
