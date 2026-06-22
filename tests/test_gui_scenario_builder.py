"""Qt-free unit tests for the GUI scenario builder and jobs (task S7 §7).

These import the *Qt-free* seam only (``scenario_builder`` and ``jobs`` do not
pull in PySide6), so they run without a display and without the ``gui`` extra:
they check that GUI payloads assemble the correct, validated ``ScenarioConfig`` /
sweep / Monte-Carlo specs, that invalid payloads fail loudly, and that every job
drives the real core off-thread-ready (all physics lives in the core).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from optivibe.analysis import MonteCarloResult, SweepResult
from optivibe.core.config.models import ScenarioConfig
from optivibe.gui.controllers.scenario_builder import (
    build_excitation_spec,
    build_monte_carlo_spec,
    build_scenario_config,
    build_sweep_spec,
    demo_scenario_payload,
)
from optivibe.gui.workers.jobs import (
    MonteCarloJob,
    ReportBundle,
    ReportJob,
    ScenarioJob,
    SweepJob,
)
from optivibe.pipeline import RunArtifacts

_NOOP = lambda *_args, **_kwargs: None  # noqa: E731 - tiny test stub
_NO_CANCEL = lambda: False  # noqa: E731 - tiny test stub


def test_demo_payload_builds_valid_scenario() -> None:
    scenario = build_scenario_config(demo_scenario_payload())
    assert isinstance(scenario, ScenarioConfig)
    assert scenario.variant == "B"
    assert scenario.stages.detector == "photodiode"
    assert scenario.stages.dsp == "standard"
    assert scenario.dsp.sensitivity_model == "static"


def test_sine_payload_maps_to_scenario() -> None:
    payload = {
        "name": "t",
        "variant": "C",
        "excitation": {
            "kind": "sine",
            "axis": "y",
            "fs_hz": 4000.0,
            "duration_s": 1.0,
            "frequency_hz": 333.0,
            "amplitude_g": 2.0,
        },
        "stages": {"excitation": "sine", "optics": "stub", "detector": "stub", "dsp": "stub"},
        "seed": 11,
    }
    scenario = build_scenario_config(payload)
    assert scenario.variant == "C"
    assert scenario.excitation.kind == "sine"
    assert scenario.excitation.axis == "y"
    assert scenario.stages.optics == "stub"
    assert scenario.stages.detector == "stub"
    assert scenario.seed == 11


def test_multitone_payload_builds() -> None:
    spec = build_excitation_spec(
        {
            "kind": "multitone",
            "axis": "x",
            "fs_hz": 5000.0,
            "duration_s": 1.0,
            "tones": [[120.0, 1.0], [240.0, 0.5]],
        }
    )
    assert spec.kind == "multitone"
    assert len(spec.tones) == 2
    assert spec.tones[0].frequency_hz == pytest.approx(120.0)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "kind": "sine",
            "fs_hz": 5000.0,
            "duration_s": 1.0,
            "frequency_hz": 200.0,
            "amplitude_g": 1.0,
        },
        {
            "kind": "sweep",
            "fs_hz": 5000.0,
            "duration_s": 1.0,
            "f_start_hz": 20.0,
            "f_end_hz": 2000.0,
            "amplitude_g": 1.0,
            "method": "log",
        },
        {
            "kind": "random",
            "fs_hz": 5000.0,
            "duration_s": 1.0,
            "band_hz": [10.0, 2000.0],
            "g_rms": 1.0,
        },
        {
            "kind": "shock",
            "fs_hz": 5000.0,
            "duration_s": 1.0,
            "peak_g": 50.0,
            "pulse_ms": 2.0,
            "delay_s": 0.1,
        },
        {"kind": "csv", "path": "x.csv", "column": 1, "fs_hz": 5000.0, "units": "g"},
        {"kind": "wav", "path": "x.wav", "channel": 0, "full_scale_g": 10.0},
    ],
)
def test_excitation_kinds_validate(payload: dict[str, object]) -> None:
    spec = build_excitation_spec(payload)
    assert spec.kind == payload["kind"]


def test_invalid_excitation_raises() -> None:
    # random requires exactly one of g_rms / psd_g2_hz.
    with pytest.raises(ValueError):
        build_excitation_spec(
            {
                "kind": "random",
                "fs_hz": 5000.0,
                "duration_s": 1.0,
                "band_hz": [10.0, 2000.0],
                "g_rms": 1.0,
                "psd_g2_hz": 1.0,
            }
        )


def test_invalid_scenario_unknown_variant_raises() -> None:
    payload = demo_scenario_payload()
    payload["variant"] = "Z"
    with pytest.raises(ValueError):
        build_scenario_config(payload)


def test_sweep_spec_valid_and_invalid() -> None:
    spec = build_sweep_spec(
        {
            "kind": "sweep",
            "name": "s",
            "mode": "design",
            "variant": "B",
            "parameter": "length_m",
            "grid": {"start": 1.0e-3, "stop": 4.0e-3, "num": 6, "log": False},
        }
    )
    assert spec.mode == "design"
    # A response parameter under design mode is rejected.
    with pytest.raises(ValueError):
        build_sweep_spec(
            {
                "kind": "sweep",
                "name": "s",
                "mode": "design",
                "variant": "B",
                "parameter": "amplitude_g",
                "grid": {"start": 0.1, "stop": 10.0, "num": 4},
            }
        )


def test_monte_carlo_spec_valid_and_invalid() -> None:
    spec = build_monte_carlo_spec(
        {
            "kind": "montecarlo",
            "name": "m",
            "variant": "B",
            "n_draws": 8,
            "cross_axis": False,
            "tolerances": {"q_total": {"dist": "lognormal", "rel_sigma": 0.3}},
        }
    )
    assert spec.n_draws == 8
    with pytest.raises(ValueError):
        build_monte_carlo_spec(
            {
                "kind": "montecarlo",
                "name": "m",
                "variant": "B",
                "n_draws": 8,
                "tolerances": {"unknown_key": {"dist": "normal", "rel_sigma": 0.1}},
            }
        )


def test_scenario_job_runs_core(config_dir: Path) -> None:
    scenario = build_scenario_config(demo_scenario_payload())
    result = ScenarioJob(scenario=scenario, config_dir=config_dir).run(
        progress=_NOOP, is_cancelled=_NO_CANCEL
    )
    assert isinstance(result, RunArtifacts)
    assert result.result.dominant_freqs_hz[0] == pytest.approx(200.0, abs=1.0)


def test_report_job_runs_core(config_dir: Path) -> None:
    scenario = build_scenario_config(demo_scenario_payload())
    bundle = ReportJob(scenario=scenario, config_dir=config_dir).run(
        progress=_NOOP, is_cancelled=_NO_CANCEL
    )
    assert isinstance(bundle, ReportBundle)
    assert bundle.nea is not None  # photodiode -> NEA exists
    assert bundle.budget.amplitude_ratio == pytest.approx(1.0, abs=0.05)


def test_sweep_job_runs_core() -> None:
    spec = build_sweep_spec(
        {
            "kind": "sweep",
            "name": "s",
            "mode": "design",
            "variant": "B",
            "parameter": "length_m",
            "grid": {"start": 1.0e-3, "stop": 4.0e-3, "num": 5, "log": False},
        }
    )
    result = SweepJob(spec=spec).run(progress=_NOOP, is_cancelled=_NO_CANCEL)
    assert isinstance(result, SweepResult)
    assert len(result.axis_labels) == 5


def test_monte_carlo_job_runs_core() -> None:
    spec = build_monte_carlo_spec(
        {
            "kind": "montecarlo",
            "name": "m",
            "variant": "B",
            "n_draws": 12,
            "cross_axis": False,
            "tolerances": {"gap_m": {"dist": "normal", "abs_sigma": 5.0e-6}},
        }
    )
    result = MonteCarloJob(spec=spec).run(progress=_NOOP, is_cancelled=_NO_CANCEL)
    assert isinstance(result, MonteCarloResult)
    assert result.n_draws == 12
