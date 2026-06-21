"""Analysis spec models: parameter sweep and Monte-Carlo (task S6 §B7/§B8).

YAML specs (validated by pydantic, like scenarios) drive the ``optivibe sweep``
CLI. A :class:`SweepSpec` sweeps one parameter -- either *design* (recompute the
analytic ``s_target``/NEA over ``L``/``R_c``/``P``/bias/``FS``/variant) or
*response* (forward + inverse over amplitude 0.1 g - >50 g, or frequency). A
:class:`MonteCarloSpec` draws tolerance distributions (doc 08 §7) including the
technological de-centering ``epsilon_x`` as a per-run random parameter (deferred
from S3, 14 §8).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from optivibe.core.config.models import DetectorOptions, DspOptions, OutputSpec
from optivibe.core.types import FloatArray

__all__ = [
    "AxisGrid",
    "MonteCarloSpec",
    "SweepSpec",
    "ToleranceSpec",
]

_DesignParam = Literal[
    "length_m", "radius_of_curvature_m", "power_w", "bias_offset_m", "full_scale_g", "variant"
]
_ResponseParam = Literal["amplitude_g", "frequency_hz"]


class _Frozen(BaseModel):
    """Immutable, strictly-validated base (mirror of the config base)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class AxisGrid(_Frozen):
    """A numeric sweep axis (linear or log-spaced).

    Attributes
    ----------
    start, stop : float
        Inclusive endpoints (in the parameter's SI unit, or g for amplitude).
    num : int
        Number of points (>= 2).
    log : bool
        Log-spacing when ``True`` (requires positive endpoints).
    """

    start: float
    stop: float
    num: int = Field(default=11, ge=2)
    log: bool = False

    @model_validator(mode="after")
    def _check(self) -> AxisGrid:
        if self.log and (self.start <= 0.0 or self.stop <= 0.0):
            msg = "log axis requires positive start/stop"
            raise ValueError(msg)
        return self

    def values(self) -> FloatArray:
        """Return the grid points as a 1-D float64 array."""
        if self.log:
            return np.geomspace(self.start, self.stop, self.num).astype(np.float64)
        return np.linspace(self.start, self.stop, self.num).astype(np.float64)


class SweepSpec(_Frozen):
    """A one-parameter sweep specification (task S6 §B7).

    Attributes
    ----------
    kind : "sweep"
        Spec discriminator.
    name : str
        Sweep name.
    mode : {"design", "response"}
        ``design`` recomputes analytic sensitivity/NEA per point (no time-domain
        run); ``response`` runs forward + inverse and measures the response.
    variant : {"A", "B", "C", "D"}
        Base sensor variant.
    parameter : str
        Swept parameter. Design: ``length_m``, ``radius_of_curvature_m``,
        ``power_w``, ``bias_offset_m``, ``full_scale_g`` or ``variant``.
        Response: ``amplitude_g`` or ``frequency_hz``.
    grid : AxisGrid or None
        Grid for a numeric parameter (required unless ``parameter == "variant"``).
    variant_values : list of {"A", "B", "C", "D"} or None
        Variant list when ``parameter == "variant"``.
    frequency_hz, amplitude_g : float
        Fixed tone context for response runs (the non-swept one is held here).
    fs_hz, duration_s : float
        Sampling rate and record length for response runs.
    detector, dsp : DetectorOptions, DspOptions
        Stage options for response runs (the detector is forced to photodiode
        and the DSP to standard so the recovery and NEA exist).
    seed : int or None
        Reproducibility seed.
    output : OutputSpec
        Persistence options.
    """

    kind: Literal["sweep"] = "sweep"
    name: str
    mode: Literal["design", "response"] = "design"
    variant: Literal["A", "B", "C", "D"] = "B"
    parameter: _DesignParam | _ResponseParam = "length_m"
    grid: AxisGrid | None = None
    variant_values: list[Literal["A", "B", "C", "D"]] | None = None
    frequency_hz: float = Field(default=200.0, gt=0.0)
    amplitude_g: float = Field(default=1.0, gt=0.0)
    fs_hz: float = Field(default=5000.0, gt=0.0)
    duration_s: float = Field(default=1.0, gt=0.0)
    detector: DetectorOptions = DetectorOptions()
    dsp: DspOptions = DspOptions()
    seed: int | None = 7
    output: OutputSpec = OutputSpec()

    @model_validator(mode="after")
    def _check(self) -> SweepSpec:
        if self.parameter == "variant":
            if not self.variant_values:
                msg = "parameter 'variant' requires a non-empty variant_values list"
                raise ValueError(msg)
        elif self.grid is None:
            msg = f"parameter {self.parameter!r} requires a numeric grid"
            raise ValueError(msg)
        response_params = {"amplitude_g", "frequency_hz"}
        if self.mode == "design" and self.parameter in response_params:
            msg = f"parameter {self.parameter!r} is a response axis; set mode: response"
            raise ValueError(msg)
        return self


class ToleranceSpec(_Frozen):
    """A single tolerance distribution for a Monte-Carlo parameter (doc 08 §7).

    Exactly one of ``rel_sigma`` / ``abs_sigma`` is given. ``normal`` is additive
    Gaussian; ``lognormal`` is multiplicative (kept positive, e.g. ``Q``).

    Attributes
    ----------
    dist : {"normal", "lognormal"}
        Distribution family.
    rel_sigma : float or None
        Relative standard deviation (fraction of the nominal).
    abs_sigma : float or None
        Absolute standard deviation (parameter's SI unit).
    """

    dist: Literal["normal", "lognormal"] = "normal"
    rel_sigma: float | None = Field(default=None, gt=0.0)
    abs_sigma: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> ToleranceSpec:
        if (self.rel_sigma is None) == (self.abs_sigma is None):
            msg = "exactly one of rel_sigma / abs_sigma must be set"
            raise ValueError(msg)
        if self.dist == "lognormal" and self.abs_sigma is not None:
            msg = "lognormal tolerance uses rel_sigma (multiplicative spread)"
            raise ValueError(msg)
        return self


class MonteCarloSpec(_Frozen):
    """A tolerance Monte-Carlo specification (task S6 §B8).

    Attributes
    ----------
    kind : "montecarlo"
        Spec discriminator.
    name : str
        Run name.
    variant : {"A", "B", "C", "D"}
        Base sensor variant.
    n_draws : int
        Number of Monte-Carlo draws (>= 2).
    tolerances : Mapping[str, ToleranceSpec]
        Per-parameter tolerance. Recognised keys: ``q_total``,
        ``radius_of_curvature_m``, ``gap_m``, ``bias_offset_m`` and the
        technological de-centering ``epsilon_x`` (a per-run extra x-bias, the
        S3-deferred parameter; doc 08 §7 / 14 §8).
    cross_axis : bool
        Whether to estimate the cross-axis suppression per draw (needs two
        forward + inverse runs per draw; slower).
    frequency_hz, amplitude_g, fs_hz, duration_s : float
        Tone context for the cross-axis runs.
    seed : int or None
        Master seed; each draw derives an independent sub-stream from it.
    output : OutputSpec
        Persistence options.
    """

    kind: Literal["montecarlo"] = "montecarlo"
    name: str
    variant: Literal["A", "B", "C", "D"] = "B"
    n_draws: int = Field(default=256, ge=2)
    tolerances: dict[str, ToleranceSpec] = Field(default_factory=dict)
    cross_axis: bool = True
    frequency_hz: float = Field(default=120.0, gt=0.0)
    amplitude_g: float = Field(default=10.0, gt=0.0)
    fs_hz: float = Field(default=5000.0, gt=0.0)
    duration_s: float = Field(default=0.5, gt=0.0)
    seed: int | None = 7
    output: OutputSpec = OutputSpec()

    @model_validator(mode="after")
    def _check(self) -> MonteCarloSpec:
        allowed = {"q_total", "radius_of_curvature_m", "gap_m", "bias_offset_m", "epsilon_x"}
        unknown = set(self.tolerances) - allowed
        if unknown:
            msg = f"unknown Monte-Carlo tolerance parameter(s): {sorted(unknown)}"
            raise ValueError(msg)
        return self
