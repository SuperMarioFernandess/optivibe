"""Comparison bench for the DSP chain: one input, several chains (task S-22).

The bench answers one question -- *what changes in the numbers when a stage of
the inverse chain is swapped?* -- and it answers it under three rules fixed by
the coordination decision of 2026-07-29:

**1. The boundary of the verified.** The whole V&V apparatus (plans 18/19, the
golden set, the batch<->stream bit-identity) covers the **default** chain, i.e.
``DspOptions()``. Any deviation from it turns the run into an *experiment*, and
that verdict is not a cosmetic label: :func:`chain_status` computes it from the
options themselves, :func:`chain_provenance` ships it with the exported numbers
(discipline S-13), and the UI shows it. Exported experimental numbers must not
be mistakable for the output of the verified twin.

**2. Config-first.** A chain is a :class:`ChainSpec` -- a name plus a
:class:`~optivibe.core.config.models.DspOptions` -- and a whole comparison is a
``kind: compare`` YAML (:class:`CompareSpec`), the same way an analysis is a
``kind: analyze`` YAML. The GUI edits that config; ``optivibe compare`` runs it
head-less. An experiment that exists only as a widget state is neither
reproducible nor transferable.

**3. Cross-checking against analytics, not combinatorics.** Golden files for
every combination of knobs would explode; instead each alternative faces the
same analytical reference (a clean tone gives ``v = A/omega``, ``x = A/omega^2``
within the tolerances of 11 §7; windows are checked by amplitude correction and
Parseval). That is the style of the L2 cross-checks of 19 §2.

**No second data path.** The bench never loads a record its own way: a
synthetic input is the forward chain of a scenario (as
:class:`~optivibe.gui.workers.stream.ScenarioSource` builds it) and a recorded
input goes through :func:`~optivibe.io.records.read_record` plus
:func:`~optivibe.analysis.instrument.record_sensitivity_model` -- the very seam
:func:`~optivibe.analysis.instrument.analyze_record` and the live replay use.
Otherwise a difference in the numbers could come from the *input* rather than
from the chain, which is exactly what this bench exists to rule out; the
identity is pinned by tests rather than by this paragraph.

**No physics here either.** Every chain is executed by the unchanged
:class:`~optivibe.dsp.standard.StandardDsp`; this module only feeds it options
and tabulates the metrics of 17 §1.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from optivibe.analysis.instrument import load_analyze_spec, record_sensitivity_model
from optivibe.core.config.loader import load_constants, load_scenario, load_variant
from optivibe.core.config.models import Constants, DspOptions, ScenarioConfig, VariantConfig
from optivibe.core.logging import get_logger
from optivibe.core.types import DetectorOutput, VibrationResult
from optivibe.dsp.sensitivity import SensitivityModel
from optivibe.dsp.standard import StandardDsp
from optivibe.io.records import read_record
from optivibe.pipeline import Pipeline

logger = get_logger(__name__)

__all__ = [
    "CHAIN_APPLICABILITY",
    "DEFAULT_CHAIN",
    "EXPERIMENT_FIELDS",
    "ChainDelta",
    "ChainMetrics",
    "ChainOutcome",
    "ChainSpec",
    "CompareInput",
    "CompareSpec",
    "ComparisonResult",
    "MetricRow",
    "chain_deltas",
    "chain_provenance",
    "chain_status",
    "code_revision",
    "compare_chains",
    "input_from_analyze_spec",
    "input_from_scenario",
    "load_compare_spec",
    "provenance_yaml",
    "resolve_compare_input",
    "run_comparison",
]

#: The verified chain: everything the golden set and plans 18/19 cover.
DEFAULT_CHAIN = DspOptions()

#: Verdict of a chain relative to :data:`DEFAULT_CHAIN` (rule 1).
ChainStatus = Literal["verified", "experimental"]

#: Which computational path each option actually reaches.
#:
#: ``"batch"`` options are structurally unavailable to the causal streaming
#: layer -- a real-time chain cannot know the whole record, so a zero-phase
#: spectral integrator, a record-length rFFT or a Welch segmentation of it do
#: not exist there (theory-06 §3.6/§7.6); the streaming layer says so itself
#: (:mod:`optivibe.dsp.streaming`). Showing a knob that silently does nothing
#: would be a lie the UI tells, hence this map: the panel renders the label, it
#: does not invent it. Completeness against ``DspOptions`` is pinned by a test.
CHAIN_APPLICABILITY: dict[str, Literal["batch", "stream", "both"]] = {
    "integrator": "batch",
    "spectrum_method": "batch",
    "window": "both",
    "f_hp_hz": "both",
    "f_c_stream": "stream",
    "welch_nperseg": "batch",
    "welch_noverlap": "batch",
    "calibration": "batch",
    "sensitivity_model": "both",
    "sensitivity_freq": "batch",
    "deconvolve_hlat": "batch",
    "peak_interpolation": "both",
    "iso_machine_class": "both",
}

#: Options exposed by the experiment panel, in display order (backlog S-22 W-1).
#:
#: ``calibration`` is deliberately absent (bench calibration is task S-04) and
#: so is ``deconvolve_hlat`` (``sensitivity_freq="dynamic"`` drives the same
#: correction; two controls for one effect would be a trap rather than a
#: lesson).
EXPERIMENT_FIELDS: tuple[str, ...] = (
    "integrator",
    "spectrum_method",
    "window",
    "welch_nperseg",
    "welch_noverlap",
    "f_hp_hz",
    "f_c_stream",
    "sensitivity_model",
    "sensitivity_freq",
    "peak_interpolation",
)


@dataclass(frozen=True)
class ChainDelta:
    """One option that differs from the verified default.

    Attributes
    ----------
    field : str
        ``DspOptions`` field name (the config key, not a UI label).
    default : object
        Value in :data:`DEFAULT_CHAIN`.
    value : object
        Value actually used.
    """

    field: str
    default: object
    value: object

    def as_text(self) -> str:
        """Return the deviation as ``field: default -> value``."""
        return f"{self.field}: {self.default!r} -> {self.value!r}"


def chain_deltas(options: DspOptions) -> tuple[ChainDelta, ...]:
    """Return the options that deviate from the verified default chain.

    Parameters
    ----------
    options : DspOptions
        The chain to inspect.

    Returns
    -------
    tuple of ChainDelta
        Deviations in declaration order; empty for the default chain.
    """
    default = DEFAULT_CHAIN.model_dump()
    used = options.model_dump()
    return tuple(
        ChainDelta(field=key, default=default[key], value=used[key])
        for key in default
        if used[key] != default[key]
    )


def chain_status(options: DspOptions) -> ChainStatus:
    """Return ``"verified"`` for the default chain, ``"experimental"`` otherwise.

    Parameters
    ----------
    options : DspOptions
        The chain to grade.

    Returns
    -------
    {"verified", "experimental"}
        The verdict of rule 1. It is derived from the options, never set by
        hand, so a chain cannot be *declared* verified.
    """
    return "verified" if not chain_deltas(options) else "experimental"


def code_revision() -> dict[str, str]:
    """Return the code identity to record with exported numbers.

    Reads the installed package version and, when the source tree is a git
    checkout, the checked-out commit (from ``.git``, without invoking git).

    Returns
    -------
    dict of str
        ``{"optivibe_version": ..., "git_head": ...}``; unknown fields read
        ``"unknown"`` rather than being omitted, so a reader can tell that the
        question was asked.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        pkg_version = version("optivibe")
    except PackageNotFoundError:  # pragma: no cover - source-tree use
        pkg_version = "unknown"
    return {"optivibe_version": pkg_version, "git_head": _git_head() or "unknown"}


def _git_head() -> str | None:
    """Return the checked-out commit of the source tree, or ``None``."""
    for parent in Path(__file__).resolve().parents:
        git_dir = parent / ".git"
        if not git_dir.is_dir():
            continue
        try:
            head = (git_dir / "HEAD").read_text().strip()
            if head.startswith("ref:"):
                ref = git_dir / head.split(":", 1)[1].strip()
                return ref.read_text().strip() if ref.is_file() else None
            return head
        except OSError:  # pragma: no cover - unreadable checkout
            return None
    return None


def chain_provenance(
    options: DspOptions,
    *,
    input_label: str,
    name: str | None = None,
) -> dict[str, Any]:
    """Assemble the chain provenance that travels with exported numbers (rule 1).

    Parameters
    ----------
    options : DspOptions
        The chain that produced the numbers.
    input_label : str
        Human-readable description of the input (scenario or record).
    name : str or None, optional
        Chain name, when the export belongs to a named comparison chain.

    Returns
    -------
    dict
        Mapping ready for YAML: the verdict, the full chain, the complete
        deviation list and the code identity. Everything a reader needs to see
        that experimental numbers are *not* the output of the verified twin.
    """
    provenance: dict[str, Any] = {
        "kind": "dsp_chain",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "input": input_label,
        "status": chain_status(options),
        "chain": options.model_dump(mode="json"),
        "deviations_from_default": [
            {"field": delta.field, "default": delta.default, "value": delta.value}
            for delta in chain_deltas(options)
        ],
        "code": code_revision(),
    }
    if name is not None:
        provenance["chain_name"] = name
    return provenance


# --------------------------------------------------------------------------- #
# Config (rule 2)
# --------------------------------------------------------------------------- #
class _Frozen(BaseModel):
    """Immutable, strictly validated spec base (mirrors the other spec models)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ChainSpec(_Frozen):
    """One named chain of a comparison.

    Attributes
    ----------
    name : str
        Chain name shown in the overlay legend and the table header.
    dsp : DspOptions
        The inverse-chain options; the default value is the verified chain.
    """

    name: str = Field(min_length=1)
    dsp: DspOptions = DspOptions()


class CompareSource(_Frozen):
    """Where the common input of a comparison comes from.

    Exactly one field is set: the comparison feeds **one** input to every chain,
    so a difference in the numbers can only come from the chains.

    Attributes
    ----------
    scenario : str or None
        Path to a scenario YAML; the forward chain synthesizes the input.
    analyze : str or None
        Path to a ``kind: analyze`` spec; its record is the input, read through
        the same seam :func:`~optivibe.analysis.instrument.analyze_record` uses.
    """

    scenario: str | None = None
    analyze: str | None = None

    @model_validator(mode="after")
    def _check(self) -> CompareSource:
        """Require exactly one source."""
        if (self.scenario is None) == (self.analyze is None):
            msg = "compare source needs exactly one of 'scenario' / 'analyze'"
            raise ValueError(msg)
        return self


class CompareSpec(_Frozen):
    """One reproducible comparison of DSP chains (task S-22, rule 2).

    Attributes
    ----------
    kind : "compare"
        Spec discriminator (mirrors the sweep/montecarlo/analyze convention).
    name : str
        Human-readable comparison name.
    source : CompareSource
        The common input.
    chains : list of ChainSpec
        Two or more chains; the first is the reference the table diffs against
        (put the default chain first to read the table as "cost of leaving the
        verified path").
    """

    kind: Literal["compare"] = "compare"
    name: str = Field(min_length=1)
    source: CompareSource
    chains: list[ChainSpec] = Field(min_length=2)


def load_compare_spec(path: Path | str) -> CompareSpec:
    """Load and validate a comparison spec from YAML.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to a spec file with ``kind: compare``.

    Returns
    -------
    CompareSpec
        The validated spec.

    Raises
    ------
    ValueError
        If the ``kind`` field is missing or is not ``"compare"``.
    """
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    kind = raw.get("kind")
    if kind != "compare":
        msg = f"compare spec needs kind 'compare', got {kind!r}"
        raise ValueError(msg)
    return CompareSpec.model_validate(raw)


# --------------------------------------------------------------------------- #
# Input (one path, shared with analyze / the live replay)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CompareInput:
    """The single input every chain of a comparison consumes.

    Attributes
    ----------
    detector : DetectorOutput
        Digitized detector signal (synthesized or recorded).
    variant : VariantConfig
        Resolved sensor variant (calibration, band, ISO class).
    sensitivity_model : SensitivityModel or None
        Calibration to inject; ``None`` lets each chain resolve its own model
        from its options (the forward-tract default). A recorded input carries
        the calibration declared in its analyze spec, so on records the
        ``sensitivity_model`` knob is honoured by the *spec*, not by the panel.
    label : str
        Short human-readable input name (rides into the provenance).
    scenario : ScenarioConfig or None
        The scenario behind a synthetic input, ``None`` for a record.
    """

    detector: DetectorOutput
    variant: VariantConfig
    sensitivity_model: SensitivityModel | None = None
    label: str = "input"
    scenario: ScenarioConfig | None = None


def input_from_scenario(scenario: ScenarioConfig, variant: VariantConfig) -> CompareInput:
    """Synthesize the comparison input by running the forward chain.

    Parameters
    ----------
    scenario : ScenarioConfig
        Scenario to synthesize (its own ``dsp`` block is *not* used: the chains
        supply the inverse options).
    variant : VariantConfig
        Resolved sensor variant.

    Returns
    -------
    CompareInput
        The synthesized detector record and its context -- identical to what
        the live oscilloscope's scenario source produces (pinned by test).
    """
    forward = Pipeline(scenario, variant).forward()
    return CompareInput(
        detector=forward.detector,
        variant=variant,
        label=f"synthetic: {scenario.name}",
        scenario=scenario,
    )


def input_from_analyze_spec(
    spec_path: Path | str,
    *,
    config_dir: Path | None = None,
    constants: Constants | None = None,
) -> CompareInput:
    """Load a recorded input through the analyze seam (no second data path).

    Parameters
    ----------
    spec_path : pathlib.Path or str
        Path to a ``kind: analyze`` spec describing the record.
    config_dir : pathlib.Path or None, optional
        Override of the ``configs/`` directory.
    constants : Constants or None, optional
        Physical constants; loaded from the config dir when ``None``.

    Returns
    -------
    CompareInput
        The loaded record with the calibration its spec declares.
    """
    spec = load_analyze_spec(spec_path)
    consts = (
        load_constants(config_dir / "constants.yaml" if config_dir is not None else None)
        if constants is None
        else constants
    )
    variant = load_variant(spec.variant, config_dir)
    record = read_record(spec.record)
    model = record_sensitivity_model(spec, record, variant, consts)
    return CompareInput(
        detector=record.detector,
        variant=variant,
        sensitivity_model=model,
        label=f"record: {spec.name}",
    )


def resolve_compare_input(
    source: CompareSource,
    *,
    config_dir: Path | None = None,
    constants: Constants | None = None,
) -> CompareInput:
    """Open the input declared by a comparison spec.

    Parameters
    ----------
    source : CompareSource
        The declared input (scenario or analyze spec).
    config_dir : pathlib.Path or None, optional
        Override of the ``configs/`` directory.
    constants : Constants or None, optional
        Physical constants.

    Returns
    -------
    CompareInput
        The opened input.
    """
    if source.analyze is not None:
        return input_from_analyze_spec(source.analyze, config_dir=config_dir, constants=constants)
    assert source.scenario is not None  # validated by CompareSource
    scenario = load_scenario(Path(source.scenario))
    variant = load_variant(scenario.variant, config_dir)
    return input_from_scenario(scenario, variant)


# --------------------------------------------------------------------------- #
# Metrics (17 §1) and the comparison itself
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChainMetrics:
    """Metrics of one chain output, in the vocabulary of 17 §1.

    Attributes
    ----------
    rms_a, rms_v, rms_x : float
        RMS acceleration, velocity and displacement, m/s^2, m/s, m.
    dominant_freqs_hz : tuple of float
        Dominant spectral lines, Hz.
    second_harmonic_ratio : float or None
        THD proxy at the leading dominant (04 §5); ``None`` without a dominant.
    v_rms_band_m_s : float
        Band-limited RMS velocity behind the ISO grade, m/s.
    iso_zone : str or None
        ISO 10816-3 / 20816-3 evaluation zone.
    """

    rms_a: float
    rms_v: float
    rms_x: float
    dominant_freqs_hz: tuple[float, ...]
    second_harmonic_ratio: float | None
    v_rms_band_m_s: float
    iso_zone: str | None

    @classmethod
    def from_result(cls, result: VibrationResult) -> ChainMetrics:
        """Extract the comparison metrics from a chain output.

        Parameters
        ----------
        result : VibrationResult
            Output of one chain.

        Returns
        -------
        ChainMetrics
            The metrics; nothing is recomputed here, every number is read off
            the result the core produced.
        """
        iso = result.iso or {}
        zone = iso.get("zone")
        v_band = iso.get("v_rms_m_s")
        return cls(
            rms_a=float(result.rms["a"]),
            rms_v=float(result.rms["v"]),
            rms_x=float(result.rms["x"]),
            dominant_freqs_hz=tuple(float(f) for f in result.dominant_freqs_hz),
            second_harmonic_ratio=(
                float(result.cross_residual["second_harmonic_ratio"])
                if "second_harmonic_ratio" in result.cross_residual
                else None
            ),
            v_rms_band_m_s=float(v_band) if isinstance(v_band, (int, float)) else 0.0,
            iso_zone=str(zone) if zone is not None else None,
        )


@dataclass(frozen=True)
class ChainOutcome:
    """One chain of a comparison and what it produced.

    Attributes
    ----------
    name : str
        Chain name.
    options : DspOptions
        The chain itself.
    status : {"verified", "experimental"}
        Verdict of rule 1.
    deltas : tuple of ChainDelta
        Deviations from the default chain.
    result : VibrationResult
        Full chain output (traces and spectrum for the overlay).
    metrics : ChainMetrics
        The tabulated metrics.
    """

    name: str
    options: DspOptions
    status: ChainStatus
    deltas: tuple[ChainDelta, ...]
    result: VibrationResult
    metrics: ChainMetrics


@dataclass(frozen=True)
class MetricRow:
    """One metric across all chains, diffed against the reference chain.

    Attributes
    ----------
    key : str
        Metric key (``"rms_a"``, ``"f_dominant"``, ...).
    unit : str
        Unit string for display (empty for dimensionless metrics).
    values : tuple
        Value per chain (``None`` where the metric is undefined).
    abs_diff : tuple
        Absolute difference against chain 0 (``None`` for the reference itself
        and where either value is undefined).
    rel_diff : tuple
        Relative difference against chain 0 (fraction, not per cent);
        ``None`` where the reference is zero or a value is undefined.
    """

    key: str
    unit: str
    values: tuple[float | None, ...]
    abs_diff: tuple[float | None, ...]
    rel_diff: tuple[float | None, ...]


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of one comparison: the chains, their metrics and the diff table.

    Attributes
    ----------
    name : str
        Comparison name.
    input_label : str
        The common input.
    fs : float
        Sampling rate, Hz.
    chains : tuple of ChainOutcome
        Chains in spec order; ``chains[0]`` is the reference of the table.
    rows : tuple of MetricRow
        Metric rows (17 §1).
    """

    name: str
    input_label: str
    fs: float
    chains: tuple[ChainOutcome, ...]
    rows: tuple[MetricRow, ...]

    @property
    def status(self) -> ChainStatus:
        """Return ``"experimental"`` if *any* chain leaves the verified default."""
        return (
            "verified"
            if all(outcome.status == "verified" for outcome in self.chains)
            else "experimental"
        )

    def provenance(self) -> dict[str, Any]:
        """Return the provenance of the whole comparison (rule 1, S-13 style).

        Returns
        -------
        dict
            Per-chain verdicts and deviations plus the code identity, ready to
            be written next to exported numbers.
        """
        return {
            "kind": "dsp_chain_comparison",
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "name": self.name,
            "input": self.input_label,
            "status": self.status,
            "chains": [
                chain_provenance(outcome.options, input_label=self.input_label, name=outcome.name)
                for outcome in self.chains
            ],
        }

    def as_text(self) -> str:
        """Render the comparison as a fixed-width table (CLI and log output)."""
        head = [f"comparison : {self.name}", f"input      : {self.input_label}"]
        for outcome in self.chains:
            marks = ", ".join(delta.as_text() for delta in outcome.deltas) or "default chain"
            head.append(f"chain      : {outcome.name} [{outcome.status}] -- {marks}")
        names = "  ".join(f"{outcome.name:>16.16}" for outcome in self.chains)
        lines = [*head, "", f"{'metric':<22}{'unit':<10}{names}"]
        for row in self.rows:
            cells = "  ".join(_format_cell(value) for value in row.values)
            lines.append(f"{row.key:<22}{row.unit:<10}{cells}")
            if len(self.chains) > 1:
                rel = "  ".join(
                    " " * 16 if value is None else f"{value * 100.0:>15.3f}%"
                    for value in row.rel_diff
                )
                lines.append(f"{'  rel. to ' + self.chains[0].name:<32}{rel}")
        return "\n".join(lines)


def _format_cell(value: float | None) -> str:
    """Render one table cell."""
    return " " * 16 if value is None else f"{value:>16.6g}"


def _metric_rows(chains: Sequence[ChainOutcome]) -> tuple[MetricRow, ...]:
    """Build the diff table of 17 §1 metrics against the first chain."""
    extractors: tuple[tuple[str, str, Any], ...] = (
        ("rms_a", "m/s^2", lambda m: m.rms_a),
        ("rms_v", "m/s", lambda m: m.rms_v),
        ("rms_x", "m", lambda m: m.rms_x),
        ("f_dominant", "Hz", lambda m: m.dominant_freqs_hz[0] if m.dominant_freqs_hz else None),
        ("n_dominants", "", lambda m: float(len(m.dominant_freqs_hz))),
        ("second_harmonic_ratio", "", lambda m: m.second_harmonic_ratio),
        ("v_rms_band", "m/s", lambda m: m.v_rms_band_m_s),
    )
    rows: list[MetricRow] = []
    for key, unit, extract in extractors:
        values = tuple(_as_float(extract(outcome.metrics)) for outcome in chains)
        reference = values[0]
        abs_diff: list[float | None] = []
        rel_diff: list[float | None] = []
        for index, value in enumerate(values):
            if index == 0 or value is None or reference is None:
                abs_diff.append(None)
                rel_diff.append(None)
                continue
            delta = value - reference
            abs_diff.append(delta)
            rel_diff.append(delta / reference if reference != 0.0 else None)
        rows.append(
            MetricRow(
                key=key,
                unit=unit,
                values=values,
                abs_diff=tuple(abs_diff),
                rel_diff=tuple(rel_diff),
            )
        )
    return tuple(rows)


def _as_float(value: object) -> float | None:
    """Coerce a metric to float, keeping ``None`` for undefined metrics."""
    return None if value is None else float(value)  # type: ignore[arg-type]


def compare_chains(
    source: CompareInput,
    chains: Sequence[ChainSpec],
    *,
    name: str = "comparison",
    constants: Constants | None = None,
) -> ComparisonResult:
    """Run several DSP chains over one input and tabulate the differences.

    Every chain is executed by the unchanged
    :class:`~optivibe.dsp.standard.StandardDsp` (17 §7: one implementation of
    the metrics), so the default chain reproduces the ordinary run bit for bit.

    Parameters
    ----------
    source : CompareInput
        The common input (see the module docstring: one data path).
    chains : sequence of ChainSpec
        Chains to run; the first one is the reference of the diff table.
    name : str, optional
        Comparison name (rides into the provenance).
    constants : Constants or None, optional
        Physical constants; loaded once when ``None``.

    Returns
    -------
    ComparisonResult
        Chains, metrics and the diff table.

    Raises
    ------
    ValueError
        If fewer than one chain is given, or two chains share a name (the name
        keys the legend, the table and the provenance).
    """
    if not chains:
        msg = "a comparison needs at least one chain"
        raise ValueError(msg)
    names = [chain.name for chain in chains]
    if len(set(names)) != len(names):
        msg = f"chain names must be unique, got {names}"
        raise ValueError(msg)
    consts = load_constants() if constants is None else constants

    outcomes: list[ChainOutcome] = []
    for chain in chains:
        dsp = StandardDsp(constants=consts, sensitivity_model=source.sensitivity_model)
        result = dsp.run(source.detector, source.variant, chain.dsp)
        status = chain_status(chain.dsp)
        logger.debug("comparison '%s': chain '%s' is %s", name, chain.name, status)
        outcomes.append(
            ChainOutcome(
                name=chain.name,
                options=chain.dsp,
                status=status,
                deltas=chain_deltas(chain.dsp),
                result=result,
                metrics=ChainMetrics.from_result(result),
            )
        )
    return ComparisonResult(
        name=name,
        input_label=source.label,
        fs=source.detector.fs,
        chains=tuple(outcomes),
        rows=_metric_rows(outcomes),
    )


def run_comparison(
    spec: CompareSpec,
    *,
    config_dir: Path | None = None,
    constants: Constants | None = None,
) -> ComparisonResult:
    """Resolve a comparison spec and run it (the head-less entry point).

    Parameters
    ----------
    spec : CompareSpec
        The validated comparison spec.
    config_dir : pathlib.Path or None, optional
        Override of the ``configs/`` directory.
    constants : Constants or None, optional
        Physical constants.

    Returns
    -------
    ComparisonResult
        The comparison outcome.
    """
    source = resolve_compare_input(spec.source, config_dir=config_dir, constants=constants)
    return compare_chains(source, spec.chains, name=spec.name, constants=constants)


def provenance_yaml(provenance: dict[str, Any]) -> str:
    """Serialize a provenance mapping to YAML text.

    Parameters
    ----------
    provenance : dict
        Mapping from :func:`chain_provenance` or
        :meth:`ComparisonResult.provenance`.

    Returns
    -------
    str
        YAML document (block style, keys in insertion order).
    """
    return yaml.safe_dump(json.loads(json.dumps(provenance)), sort_keys=False, allow_unicode=True)
