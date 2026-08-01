"""Analysis layer: truth-vs-recovery, NEA budget, sweeps and Monte-Carlo (S6).

This package hosts the post-run analytics required by documents 00/07/08 and the
simulation spec (11 §4): the end-to-end ``truth vs recovery`` error budget, the
NEA budget with its contribution split and displacement floor, parameter sweeps
(design and response) and the tolerance Monte-Carlo. Spectra and metrics are
computed here (or in :mod:`optivibe.dsp`); ``viz`` only draws them (14 §8).
Role S-02 (doc 20 §5) adds :mod:`optivibe.analysis.instrument`: the analyzer of
recorded real-instrument output, reusing the same DSP tract (17 §7). Tasks
S-16/S-17 add :mod:`optivibe.analysis.expected_peaks`: the *predicted* peak set
of a run (doc 20 §3), computable from the configuration with no record at all.
Task S-22 adds :mod:`optivibe.analysis.compare`: the DSP-chain comparison bench
(one input, several chains, the metric diff of 17 §1) with the verified /
experimental verdict that rides with every exported number.
"""

from __future__ import annotations

from optivibe.analysis.compare import (
    CHAIN_APPLICABILITY,
    DEFAULT_CHAIN,
    EXPERIMENT_FIELDS,
    ChainDelta,
    ChainSpec,
    CompareInput,
    CompareSpec,
    ComparisonResult,
    chain_deltas,
    chain_provenance,
    chain_status,
    compare_chains,
    input_from_analyze_spec,
    input_from_scenario,
    load_compare_spec,
    provenance_yaml,
    run_comparison,
)
from optivibe.analysis.expected_peaks import (
    PEAK_KINDS,
    ExpectedPeak,
    ExpectedPeaks,
    PeakKind,
    predict_expected_peaks,
)
from optivibe.analysis.instrument import (
    AnalyzeSpec,
    CalibrationSpec,
    InstrumentAnalysis,
    analyze_record,
    load_analyze_spec,
)
from optivibe.analysis.io import (
    load_analysis_spec,
    save_monte_carlo_npz,
    save_sweep_npz,
)
from optivibe.analysis.monte_carlo import MonteCarloResult, run_monte_carlo
from optivibe.analysis.nea_budget import NeaBudget, nea_budget
from optivibe.analysis.spec import (
    AxisGrid,
    MonteCarloSpec,
    SweepSpec,
    ToleranceSpec,
)
from optivibe.analysis.sweep import SweepResult, run_sweep
from optivibe.analysis.truth_vs_recovery import ErrorBudget, truth_vs_recovery
from optivibe.analysis.variant_tools import AnalyticPoint, analytic_point, with_overrides

__all__ = [
    "CHAIN_APPLICABILITY",
    "DEFAULT_CHAIN",
    "EXPERIMENT_FIELDS",
    "PEAK_KINDS",
    "AnalyticPoint",
    "AnalyzeSpec",
    "AxisGrid",
    "CalibrationSpec",
    "ChainDelta",
    "ChainSpec",
    "CompareInput",
    "CompareSpec",
    "ComparisonResult",
    "ErrorBudget",
    "ExpectedPeak",
    "ExpectedPeaks",
    "InstrumentAnalysis",
    "MonteCarloResult",
    "MonteCarloSpec",
    "NeaBudget",
    "PeakKind",
    "SweepResult",
    "SweepSpec",
    "ToleranceSpec",
    "analytic_point",
    "analyze_record",
    "chain_deltas",
    "chain_provenance",
    "chain_status",
    "compare_chains",
    "input_from_analyze_spec",
    "input_from_scenario",
    "load_analysis_spec",
    "load_analyze_spec",
    "load_compare_spec",
    "nea_budget",
    "predict_expected_peaks",
    "provenance_yaml",
    "run_comparison",
    "run_monte_carlo",
    "run_sweep",
    "save_monte_carlo_npz",
    "save_sweep_npz",
    "truth_vs_recovery",
    "with_overrides",
]
