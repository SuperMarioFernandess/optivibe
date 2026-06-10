"""End-to-end S0 acceptance: the hello scenario flows through the pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from optivibe.cli.main import main
from optivibe.core.config.models import (
    ExcitationSpec,
    ScenarioConfig,
    StageSelection,
)
from optivibe.core.registry import RegistryError
from optivibe.core.types import VibrationResult
from optivibe.pipeline import Pipeline, run_scenario

EXPECTED_FS = 5000.0
EXPECTED_N = 5000
EXPECTED_FREQ_HZ = 120.0


def test_hello_runs_and_recovers_frequency(hello_scenario: Path, config_dir: Path) -> None:
    artifacts = run_scenario(hello_scenario, config_dir=config_dir)

    # Forward intermediates are all the same length and sampling rate.
    fwd = artifacts.forward
    assert fwd.excitation.n_samples == EXPECTED_N
    assert fwd.tip.n_samples == EXPECTED_N
    assert fwd.optical.n_samples == EXPECTED_N
    assert fwd.detector.n_samples == EXPECTED_N
    assert fwd.detector.fs == pytest.approx(EXPECTED_FS)

    # Reconstructed output.
    result = artifacts.result
    assert isinstance(result, VibrationResult)
    assert result.fs == pytest.approx(EXPECTED_FS)
    assert result.n_samples == EXPECTED_N
    assert result.a.size == result.v.size == result.x.size == EXPECTED_N

    # The inverse stage recovers the injected tone as the dominant frequency.
    assert result.dominant_freqs_hz
    assert result.dominant_freqs_hz[0] == pytest.approx(EXPECTED_FREQ_HZ, abs=1.0)


def test_optical_response_stays_near_bias(hello_scenario: Path, config_dir: Path) -> None:
    artifacts = run_scenario(hello_scenario, config_dir=config_dir)
    optical = artifacts.forward.optical
    bias = optical.bias
    # The S0 stub keeps eta within +/-10 % of the bias working point.
    assert float(optical.eta.min()) >= 0.9 * bias - 1e-12
    assert float(optical.eta.max()) <= 1.1 * bias + 1e-12


def test_unknown_stage_key_raises(config_dir: Path) -> None:
    from optivibe.core.config import load_variant

    scenario = ScenarioConfig(
        name="bad",
        variant="B",
        excitation=ExcitationSpec(
            kind="sine", axis="x", fs_hz=1000.0, duration_s=0.1, frequency_hz=50.0, amplitude_g=1.0
        ),
        stages=StageSelection(mechanics="does-not-exist"),
    )
    variant = load_variant("B", config_dir=config_dir)
    with pytest.raises(RegistryError, match="does-not-exist"):
        Pipeline(scenario, variant)


def test_cli_run_returns_zero(
    hello_scenario: Path, config_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["run", str(hello_scenario), "--config-dir", str(config_dir)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "dominant" in out
    assert "120" in out


def test_cli_missing_file_returns_error(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    assert main(["run", str(missing)]) == 2
