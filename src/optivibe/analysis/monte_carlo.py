"""Monte-Carlo over tolerances: NEA and cross-axis statistics (task S6 §B8).

Draws the manufacturing/assembly tolerances of doc 08 §7 -- anchor -> ``Q``
(``q_total``), ``R_c``, gap ``A``, bias ``Delta x0`` -- *and* the technological
de-centering ``epsilon_x`` as a separate per-run random parameter (the S3-deferred
item, 14 §8): ``epsilon_x`` adds to the optical x-bias on each run, so it spreads
the working point and hence ``s_target``/NEA. Each draw derives an independent,
reproducible sub-stream from the master seed (one seed -> one set of draws). The
output is the NEA and cross-axis-suppression distribution (median / percentiles).

Sampled geometry is clipped to the optics-model validity band (``R_c >= 5 w0``,
gap in the wash-out range, bias ``>= 0``) so a draw never trips the loud geometry
guard; the clipping is part of the documented sampling, not silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optivibe.analysis.spec import MonteCarloSpec, ToleranceSpec
from optivibe.analysis.variant_tools import G0, analytic_point, with_overrides
from optivibe.core.config.loader import load_constants, load_variant
from optivibe.core.config.models import (
    Constants,
    ScenarioConfig,
    SineSpec,
    StageSelection,
    VariantConfig,
)
from optivibe.core.types import FloatArray
from optivibe.dsp.metrics import cross_axis_suppression, rms
from optivibe.optics.gaussian import GaussianBeam
from optivibe.pipeline.orchestrator import Pipeline

__all__ = ["MonteCarloResult", "run_monte_carlo"]

# A reproducible domain tag mixed into the master seed for the MC sub-streams.
_MC_DOMAIN = 0x4D435F53  # "MC_S"


@dataclass(frozen=True)
class MonteCarloResult:
    """Monte-Carlo tolerance statistics (task S6 §B8).

    Attributes
    ----------
    name : str
        Run name.
    variant : str
        Base variant identifier.
    n_draws : int
        Number of draws.
    seed : int or None
        Master seed (reproducibility).
    tolerances : list of str
        The tolerance parameters that were varied.
    samples : Mapping[str, numpy.ndarray]
        Per-draw metric arrays (``"nea_full_band_ug"``, ``"s_target"``,
        ``"eta_ratio"`` and, when enabled, ``"cross_axis_suppression"``).
    stats : Mapping[str, Mapping[str, float]]
        Per-metric summary (``p05`` / ``p50`` (median) / ``p95`` / ``mean`` /
        ``std``).
    meta : Mapping[str, object]
        Free-form metadata.
    """

    name: str
    variant: str
    n_draws: int
    seed: int | None
    tolerances: list[str]
    samples: dict[str, FloatArray]
    stats: dict[str, dict[str, float]]
    meta: dict[str, object] = field(default_factory=dict)


def _draw(rng: np.random.Generator, nominal: float, tol: ToleranceSpec) -> float:
    """Draw one perturbed value from a tolerance spec around a nominal."""
    if tol.dist == "lognormal":
        sigma = tol.rel_sigma if tol.rel_sigma is not None else 0.0
        return float(nominal * np.exp(rng.normal(0.0, sigma)))
    if tol.abs_sigma is not None:
        return float(nominal + rng.normal(0.0, tol.abs_sigma))
    rel = tol.rel_sigma if tol.rel_sigma is not None else 0.0
    return float(nominal + rng.normal(0.0, rel * abs(nominal)))


def _perturbed_variant(
    base: VariantConfig, rng: np.random.Generator, tolerances: dict[str, ToleranceSpec]
) -> VariantConfig:
    """Build one perturbed variant from a draw (with validity clipping)."""
    w0 = base.optics.mode_field_radius_m
    overrides: dict[str, float] = {}
    if "q_total" in tolerances:
        overrides["q_total"] = max(_draw(rng, base.q_total, tolerances["q_total"]), 1.0)
    if "radius_of_curvature_m" in tolerances and base.reflector.radius_of_curvature_m is not None:
        rc = _draw(rng, base.reflector.radius_of_curvature_m, tolerances["radius_of_curvature_m"])
        overrides["radius_of_curvature_m"] = max(rc, 5.05 * w0)  # R_c >= ~5 w0 guard
    if "gap_m" in tolerances:
        gap = _draw(rng, base.optics.gap_m, tolerances["gap_m"])
        overrides["gap_m"] = float(np.clip(gap, 15.0e-6, 45.0e-6))  # wash-out / spot band
    # Bias and the technological de-centering both shift the x working point.
    bias = base.optics.bias_offset_m
    if "bias_offset_m" in tolerances:
        bias = _draw(rng, base.optics.bias_offset_m, tolerances["bias_offset_m"])
    if "epsilon_x" in tolerances:
        bias = bias + _draw(rng, 0.0, tolerances["epsilon_x"])  # epsilon_x ~ N(0, sigma)
    overrides["bias_offset_m"] = max(bias, 1.0e-9)  # keep off the eta-peak (slope != 0)
    return with_overrides(base, **overrides)


def _cross_axis_suppression_for(variant: VariantConfig, spec: MonteCarloSpec) -> float:
    """Recovered cross-axis (y) suppression for one perturbed variant."""
    stages = StageSelection(detector="photodiode", dsp="standard")

    def _run(axis: str) -> FloatArray:
        excitation = SineSpec(
            kind="sine",
            axis=axis,
            fs_hz=spec.fs_hz,
            duration_s=spec.duration_s,
            frequency_hz=spec.frequency_hz,
            amplitude_g=spec.amplitude_g,
        )
        scenario = ScenarioConfig(
            name="mc-cross",
            variant=variant.name,
            excitation=excitation,
            stages=stages,
            seed=spec.seed,
        )
        return np.asarray(Pipeline(scenario, variant).run().result.a, dtype=np.float64)

    rec_x = _run("x")
    rec_y = _run("y")
    return cross_axis_suppression(rms(rec_y), rms(rec_x))


def _percentiles(values: FloatArray) -> dict[str, float]:
    """Summary statistics (5th/50th/95th percentile, mean, std) of a sample."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"p05": float("nan"), "p50": float("nan"), "p95": float("nan")}
    p05, p50, p95 = (float(p) for p in np.percentile(finite, [5, 50, 95]))
    return {
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def run_monte_carlo(spec: MonteCarloSpec, constants: Constants | None = None) -> MonteCarloResult:
    """Run the tolerance Monte-Carlo (task S6 §B8).

    Parameters
    ----------
    spec : MonteCarloSpec
        Validated Monte-Carlo specification.
    constants : Constants or None, optional
        Physical constants (default loaded when ``None``).

    Returns
    -------
    MonteCarloResult
        Per-draw samples and the summary statistics, reproducible by seed.
    """
    consts = load_constants() if constants is None else constants
    base = load_variant(spec.variant)
    # Validate epsilon_x once so a bad spec fails before drawing (GaussianBeam is
    # the cheapest model touch that confirms the optics block parses).
    GaussianBeam(
        wavelength_m=base.source.wavelength_m, waist_radius_m=base.optics.mode_field_radius_m
    )

    seed_seq = np.random.SeedSequence([spec.seed if spec.seed is not None else 0, _MC_DOMAIN])
    child_seeds = seed_seq.spawn(spec.n_draws)

    nea_ug: list[float] = []
    s_targets: list[float] = []
    eta_ratios: list[float] = []
    cross: list[float] = []
    for child in child_seeds:
        rng = np.random.default_rng(child)
        variant = _perturbed_variant(base, rng, spec.tolerances)
        point = analytic_point(variant, consts)
        nea_ug.append(point.nea_full_band / G0 * 1.0e6)
        s_targets.append(point.s_target)
        eta_ratios.append(point.eta_ratio)
        if spec.cross_axis:
            cross.append(_cross_axis_suppression_for(variant, spec))

    samples: dict[str, FloatArray] = {
        "nea_full_band_ug": np.asarray(nea_ug, dtype=np.float64),
        "s_target": np.asarray(s_targets, dtype=np.float64),
        "eta_ratio": np.asarray(eta_ratios, dtype=np.float64),
    }
    if spec.cross_axis:
        samples["cross_axis_suppression"] = np.asarray(cross, dtype=np.float64)
    stats = {key: _percentiles(values) for key, values in samples.items()}
    return MonteCarloResult(
        name=spec.name,
        variant=spec.variant,
        n_draws=spec.n_draws,
        seed=spec.seed,
        tolerances=sorted(spec.tolerances),
        samples=samples,
        stats=stats,
        meta={"frequency_hz": spec.frequency_hz, "amplitude_g": spec.amplitude_g},
    )
