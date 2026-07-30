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
"""

from __future__ import annotations

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
    "PEAK_KINDS",
    "AnalyticPoint",
    "AnalyzeSpec",
    "AxisGrid",
    "CalibrationSpec",
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
    "load_analysis_spec",
    "load_analyze_spec",
    "nea_budget",
    "predict_expected_peaks",
    "run_monte_carlo",
    "run_sweep",
    "save_monte_carlo_npz",
    "save_sweep_npz",
    "truth_vs_recovery",
    "with_overrides",
]
