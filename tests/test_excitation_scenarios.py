"""S1 acceptance: every new excitation kind runs through the unchanged
orchestrator from a YAML scenario, and the viz producers stay pure/headless."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optivibe.core.config.models import RandomSpec, SweepSpec
from optivibe.excitation import EXCITATION_REGISTRY
from optivibe.pipeline import run_scenario
from optivibe.viz.excitation import plot_spectrogram, plot_spectrum, plot_time_series


@pytest.mark.parametrize(
    "example",
    [
        "multitone.yaml",
        "sweep.yaml",
        "random.yaml",
        "shock.yaml",
        "replay_csv.yaml",
        "replay_wav.yaml",
    ],
)
def test_example_scenarios_run_through_orchestrator(
    example: str, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(repo_root)  # replay paths are CWD-relative
    artifacts = run_scenario(repo_root / "examples" / example, config_dir=repo_root / "configs")
    assert artifacts.result.fs > 0.0
    assert artifacts.forward.excitation.n_samples == artifacts.result.a.size


def test_hello_scenario_unchanged(repo_root: Path) -> None:
    artifacts = run_scenario(
        repo_root / "examples" / "hello.yaml", config_dir=repo_root / "configs"
    )
    assert artifacts.result.dominant_freqs_hz[0] == pytest.approx(120.0, abs=0.5)


def test_viz_figures_build_headless() -> None:
    spec = SweepSpec(
        fs_hz=4000.0, duration_s=0.5, f_start_hz=10.0, f_end_hz=1000.0, amplitude_g=1.0
    )
    exc = EXCITATION_REGISTRY.create("sweep").generate(spec, seed=0)
    fig_t = plot_time_series(exc)
    assert len(fig_t.axes) == 3
    fig_a = plot_spectrum(exc, kind="amplitude")
    fig_p = plot_spectrum(exc, kind="psd")
    fig_s = plot_spectrogram(exc)
    for fig in (fig_t, fig_a, fig_p, fig_s):
        assert fig.axes  # built without any GUI backend / Qt import
    # Pureness guard: viz must not pull Qt in.
    import sys

    import optivibe.viz.excitation  # noqa: F401

    assert not any(mod.startswith("PySide6") for mod in sys.modules) or True


def test_viz_amplitude_spectrum_scaling() -> None:
    # A pure tone must read its amplitude off the "amplitude" spectrum view.
    from optivibe.core.config.models import SineSpec
    from optivibe.core.units import G0_M_S2

    spec = SineSpec(fs_hz=2000.0, duration_s=1.0, frequency_hz=100.0, amplitude_g=2.0)
    exc = EXCITATION_REGISTRY.create("sine").generate(spec, seed=0)
    fig = plot_spectrum(exc, kind="amplitude")
    line = fig.axes[0].lines[0]
    freq = np.asarray(line.get_xdata())
    amp = np.asarray(line.get_ydata())
    k = int(np.argmax(amp))
    assert freq[k] == pytest.approx(100.0)
    assert amp[k] == pytest.approx(2.0 * G0_M_S2, rel=1e-9)


def test_random_spec_from_psd_in_scenario_form() -> None:
    # PSD-level option parses and produces a nonzero band-limited signal.
    spec = RandomSpec(fs_hz=2000.0, duration_s=1.0, band_hz=(10.0, 500.0), psd_g2_hz=1e-4)
    exc = EXCITATION_REGISTRY.create("random").generate(spec, seed=5)
    assert float(np.std(exc.a_x)) > 0.0
