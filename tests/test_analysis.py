"""Tests for the S6 analysis layer (task §B; docs 07/08/11).

Covers the end-to-end truth-vs-recovery error budget (and the displacement-floor
separation of §8), the NEA budget with its contribution split and analytic
cross-check (<= 15 %, SW-29), the design and response parameter sweeps (the
``NEA ~ L^-4`` law of doc 07 and the SW-26 ``R_c`` refinement -- x1.57 NEA gain,
not x2), the tolerance Monte-Carlo with ``epsilon_x`` and its reproducibility by
seed, the npz persistence, the spec validation guards, the CLI exit codes and the
head-less figures. Run sizes are kept small for speed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from optivibe.analysis import (
    AxisGrid,
    MonteCarloSpec,
    SweepSpec,
    ToleranceSpec,
    analytic_point,
    load_analysis_spec,
    nea_budget,
    run_monte_carlo,
    run_sweep,
    save_monte_carlo_npz,
    save_sweep_npz,
    truth_vs_recovery,
    with_overrides,
)
from optivibe.core.config.loader import load_constants, load_variant
from optivibe.core.config.models import (
    Constants,
    ScenarioConfig,
    SineSpec,
    StageSelection,
    VariantConfig,
)
from optivibe.pipeline.orchestrator import Pipeline

G0 = 9.80665


@pytest.fixture(scope="module")
def constants(config_dir: Path) -> Constants:
    """Physical constants bundle."""
    return load_constants(config_dir / "constants.yaml")


@pytest.fixture(scope="module")
def variant_b(config_dir: Path) -> VariantConfig:
    """Variant B preset."""
    return load_variant("B", config_dir)


def _run(variant: VariantConfig, f0: float, amp_g: float, axis: str = "x") -> object:
    """Full forward + inverse with photodiode + standard DSP for one tone."""
    excitation = SineSpec(
        kind="sine", axis=axis, fs_hz=5000.0, duration_s=1.0, frequency_hz=f0, amplitude_g=amp_g
    )
    scenario = ScenarioConfig(
        name="t",
        variant=variant.name,
        excitation=excitation,
        stages=StageSelection(detector="photodiode", dsp="standard"),
        seed=7,
    )
    return Pipeline(scenario, variant).run()


# --------------------------------------------------------------------------- #
# truth vs recovery.
# --------------------------------------------------------------------------- #
def test_truth_vs_recovery_strong_tone(variant_b: VariantConfig, constants: Constants) -> None:
    """A strong tone recovers a faithfully; the error sits near the NEA floor."""
    art = _run(variant_b, 200.0, 1.0)
    budget = truth_vs_recovery(
        art.forward.excitation, art.result, art.forward.detector, variant=variant_b
    )
    assert budget.amplitude_rel_error < 1e-2
    assert budget.dominant_match
    assert abs(budget.phase_error_rad) < 1e-2
    assert budget.nea_full_band is not None
    assert budget.accel_error_over_nea is not None
    assert budget.accel_error_over_nea < 5.0  # error within a few x the floor


def test_truth_vs_recovery_separates_displacement_floor(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """The displacement floor (low-frequency amplified) is reported and is much
    larger than the acceleration error (the §8 reporting hazard)."""
    art = _run(variant_b, 200.0, 1.0)
    budget = truth_vs_recovery(
        art.forward.excitation, art.result, art.forward.detector, variant=variant_b
    )
    assert budget.displacement_floor_rms is not None
    assert budget.displacement_floor_rms > 0.0
    # The acceleration recovers cleanly (tiny relative error) while the displacement
    # carries a large relative noise floor -- the two must not be conflated (§8).
    accel_rel = budget.rms_error_a / budget.rms_recovered_a
    disp_floor_rel = budget.displacement_floor_rms / budget.rms_displacement
    assert accel_rel < 0.05  # acceleration is clean
    assert disp_floor_rel > 0.05  # displacement floor is a sizeable fraction of x
    assert disp_floor_rel > accel_rel  # displacement is far more contaminated
    assert "NOT a calibration error" in budget.summary_text()


def test_truth_vs_recovery_no_nea_for_stub(variant_b: VariantConfig, constants: Constants) -> None:
    """With a stub detector (no noise) the NEA-related fields are None."""
    excitation = SineSpec(
        kind="sine", axis="x", fs_hz=5000.0, duration_s=1.0, frequency_hz=200.0, amplitude_g=1.0
    )
    scenario = ScenarioConfig(
        name="t",
        variant="B",
        excitation=excitation,
        stages=StageSelection(detector="stub", dsp="standard"),
        seed=7,
    )
    art = Pipeline(scenario, variant_b).run()
    budget = truth_vs_recovery(art.forward.excitation, art.result)
    assert budget.nea_full_band is None
    assert budget.displacement_floor_rms is None


# --------------------------------------------------------------------------- #
# NEA budget.
# --------------------------------------------------------------------------- #
def test_nea_budget_plateau_and_split(variant_b: VariantConfig, constants: Constants) -> None:
    """The plateau NEA is in the documented window and the split sums to the total."""
    art = _run(variant_b, 200.0, 0.01)
    budget = nea_budget(art.forward.detector, variant_b, constants)
    assert budget is not None
    nea_ug = budget.nea_plateau / G0 * 1e6
    assert 10.0 <= nea_ug <= 60.0  # matched B ~ 24.6 ug/sqrt(Hz) (SW-29 window)
    parts = budget.psd_components
    assert parts["shot"] + parts["rin"] + parts["johnson"] == pytest.approx(
        parts["total"], rel=1e-9
    )


def test_nea_budget_analytic_cross_check(variant_b: VariantConfig, constants: Constants) -> None:
    """The analytic PSD agrees with the simulated floor within 15 % (SW-29)."""
    art = _run(variant_b, 200.0, 0.01)
    budget = nea_budget(art.forward.detector, variant_b, constants)
    assert budget is not None
    assert budget.psd_rel_error <= 0.15


def test_nea_budget_displacement_floor_low_frequency_dominated(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """The displacement floor (PSD_x = PSD_a/omega^4) is dominated by the low band
    edge: most of its variance comes from the lowest decade (the §8 mechanism)."""
    art = _run(variant_b, 200.0, 0.01)
    budget = nea_budget(art.forward.detector, variant_b, constants)
    assert budget is not None
    assert budget.displacement_floor_rms > 0.0
    assert budget.velocity_floor_rms > 0.0
    omega = 2.0 * np.pi * budget.freq_hz
    psd_x = budget.nea_density**2 / omega**4
    total_var = float(np.trapezoid(psd_x, budget.freq_hz))
    low = budget.freq_hz <= 10.0 * budget.freq_hz[0]
    low_var = float(np.trapezoid(psd_x[low], budget.freq_hz[low]))
    assert low_var / total_var > 0.9  # > 90 % of the x-variance is in the lowest decade


def test_nea_budget_none_for_stub(variant_b: VariantConfig, constants: Constants) -> None:
    """A noiseless stub detector yields no NEA budget."""
    excitation = SineSpec(
        kind="sine", axis="x", fs_hz=5000.0, duration_s=0.5, frequency_hz=200.0, amplitude_g=1.0
    )
    scenario = ScenarioConfig(
        name="t",
        variant="B",
        excitation=excitation,
        stages=StageSelection(detector="stub", dsp="standard"),
        seed=7,
    )
    art = Pipeline(scenario, variant_b).run()
    assert nea_budget(art.forward.detector, variant_b, constants) is None


# --------------------------------------------------------------------------- #
# Parameter sweep: design.
# --------------------------------------------------------------------------- #
def test_design_sweep_nea_scales_inverse_l4(constants: Constants) -> None:
    """The plateau NEA follows ~ L^-4 over the length axis (doc 07 R-31)."""
    spec = SweepSpec(
        kind="sweep",
        name="L",
        mode="design",
        variant="B",
        parameter="length_m",
        grid=AxisGrid(start=1.0e-3, stop=2.0e-3, num=5),
    )
    result = run_sweep(spec, constants)
    nea = result.metrics["nea_plateau_ug"]
    assert np.all(np.diff(nea) < 0.0)  # strictly decreasing with L
    # Doubling L (1 -> 2 mm) drops NEA by ~ 2^4 = 16.
    assert nea[0] / nea[-1] == pytest.approx(16.0, rel=0.15)
    f1 = result.metrics["f1_hz"]
    assert f1[0] / f1[-1] == pytest.approx(4.0, rel=0.05)  # f1 ~ 1/L^2


@pytest.mark.golden
def test_sw26_rc_gain_is_157_not_2(constants: Constants) -> None:
    """At scale-equivalent bias, R_c 62.5 -> 31 um lifts NEA by ~1.57x (SW-26)."""
    base = load_variant("A")  # R_c = 62.5 um, bias 2 um
    scaled = with_overrides(
        base,
        radius_of_curvature_m=31.0e-6,
        bias_offset_m=base.optics.bias_offset_m * 31.0 / 62.5,  # sigma ~ R_c
    )
    p625 = analytic_point(base, constants)
    p31 = analytic_point(scaled, constants)
    slope_ratio = abs(p31.s_target) / abs(p625.s_target)
    nea_gain = p625.nea_plateau / p31.nea_plateau
    assert slope_ratio == pytest.approx(1.31, rel=0.05)  # absolute slope x1.31, not x2
    assert nea_gain == pytest.approx(1.57, rel=0.05)  # NEA gain x1.57


def test_design_sweep_over_variants(constants: Constants) -> None:
    """A variant-axis design sweep returns one point per named variant."""
    spec = SweepSpec(
        kind="sweep",
        name="fam",
        mode="design",
        parameter="variant",
        variant_values=["A", "B", "C"],
    )
    result = run_sweep(spec, constants)
    assert result.axis_labels == ["A", "B", "C"]
    assert result.metrics["f1_hz"].size == 3


# --------------------------------------------------------------------------- #
# Parameter sweep: response.
# --------------------------------------------------------------------------- #
def test_response_sweep_saturates_above_50g(constants: Constants) -> None:
    """The amplitude response is linear at low g and saturates (ADC clip + gain
    droop + rising THD) toward >50 g (doc 00)."""
    spec = SweepSpec(
        kind="sweep",
        name="amp",
        mode="response",
        variant="B",
        parameter="amplitude_g",
        grid=AxisGrid(start=1.0, stop=200.0, num=8, log=True),
        frequency_hz=200.0,
        fs_hz=5000.0,
        duration_s=0.3,
        seed=7,
    )
    result = run_sweep(spec, constants)
    gain = result.metrics["gain_ratio"]
    clipped = result.metrics["n_clipped"]
    optical_thd = result.metrics["optical_thd_pct"]
    assert gain[0] == pytest.approx(1.0, abs=0.05)  # linear at 1 g
    assert clipped[-1] > 0  # saturates at the top of the range
    assert gain[-1] < gain[0]  # gain droops under saturation
    assert optical_thd[-1] > optical_thd[0]  # optical THD grows with amplitude


# --------------------------------------------------------------------------- #
# Monte-Carlo.
# --------------------------------------------------------------------------- #
def _mc_spec(n: int = 16, cross: bool = False) -> MonteCarloSpec:
    return MonteCarloSpec(
        kind="montecarlo",
        name="mc",
        variant="B",
        n_draws=n,
        cross_axis=cross,
        fs_hz=5000.0,
        duration_s=0.2,
        tolerances={
            "q_total": ToleranceSpec(dist="lognormal", rel_sigma=0.3),
            "radius_of_curvature_m": ToleranceSpec(rel_sigma=0.05),
            "gap_m": ToleranceSpec(abs_sigma=5.0e-6),
            "bias_offset_m": ToleranceSpec(abs_sigma=0.1e-6),
            "epsilon_x": ToleranceSpec(abs_sigma=0.1e-6),
        },
        seed=7,
    )


def test_monte_carlo_stats_finite_and_spread(constants: Constants) -> None:
    """The Monte-Carlo produces finite NEA statistics with a non-zero spread."""
    result = run_monte_carlo(_mc_spec(n=32), constants)
    nea = result.samples["nea_full_band_ug"]
    assert np.all(np.isfinite(nea))
    stats = result.stats["nea_full_band_ug"]
    assert stats["p05"] < stats["p50"] < stats["p95"]
    assert "epsilon_x" in result.tolerances


def test_monte_carlo_reproducible_by_seed(constants: Constants) -> None:
    """The same seed reproduces the draws bit-for-bit (one seed -> one result)."""
    r1 = run_monte_carlo(_mc_spec(n=24), constants)
    r2 = run_monte_carlo(_mc_spec(n=24), constants)
    assert np.array_equal(r1.samples["nea_full_band_ug"], r2.samples["nea_full_band_ug"])
    assert np.array_equal(r1.samples["s_target"], r2.samples["s_target"])


def test_monte_carlo_cross_axis(constants: Constants) -> None:
    """The cross-axis suppression statistic is small and finite (y suppressed)."""
    result = run_monte_carlo(_mc_spec(n=16, cross=True), constants)
    cross = result.samples["cross_axis_suppression"]
    assert np.all(np.isfinite(cross))
    assert float(np.median(cross)) < 1e-2  # y strongly suppressed vs x


def test_epsilon_x_increases_s_target_spread(constants: Constants) -> None:
    """Adding epsilon_x widens the s_target spread vs the same draws without it."""
    without = MonteCarloSpec(
        kind="montecarlo",
        name="mc",
        variant="B",
        n_draws=64,
        cross_axis=False,
        tolerances={"bias_offset_m": ToleranceSpec(abs_sigma=0.1e-6)},
        seed=11,
    )
    with_eps = without.model_copy(
        update={
            "tolerances": {
                "bias_offset_m": ToleranceSpec(abs_sigma=0.1e-6),
                "epsilon_x": ToleranceSpec(abs_sigma=0.3e-6),
            }
        }
    )
    s_without = run_monte_carlo(without, constants).samples["s_target"]
    s_with = run_monte_carlo(with_eps, constants).samples["s_target"]
    assert float(np.std(s_with)) > float(np.std(s_without))


# --------------------------------------------------------------------------- #
# Spec validation (loud failures).
# --------------------------------------------------------------------------- #
def test_sweep_spec_requires_grid() -> None:
    """A numeric sweep parameter without a grid is rejected."""
    with pytest.raises(ValueError, match="numeric grid"):
        SweepSpec(kind="sweep", name="x", parameter="length_m")


def test_sweep_spec_response_param_needs_response_mode() -> None:
    """A response parameter in design mode is rejected."""
    with pytest.raises(ValueError, match="response axis"):
        SweepSpec(
            kind="sweep",
            name="x",
            mode="design",
            parameter="amplitude_g",
            grid=AxisGrid(start=0.1, stop=10.0, num=3),
        )


def test_tolerance_spec_exclusive_sigma() -> None:
    """A tolerance with both rel and abs sigma is rejected."""
    with pytest.raises(ValueError, match="exactly one"):
        ToleranceSpec(rel_sigma=0.1, abs_sigma=0.1)


def test_monte_carlo_spec_unknown_tolerance() -> None:
    """An unknown Monte-Carlo tolerance key is rejected."""
    with pytest.raises(ValueError, match="unknown"):
        MonteCarloSpec(
            kind="montecarlo", name="x", tolerances={"not_a_param": ToleranceSpec(rel_sigma=0.1)}
        )


# --------------------------------------------------------------------------- #
# IO and spec loading.
# --------------------------------------------------------------------------- #
def test_sweep_npz_roundtrip(constants: Constants, tmp_path: Path) -> None:
    """A sweep result persists to npz and reloads with the same metric arrays."""
    spec = SweepSpec(
        kind="sweep",
        name="L",
        mode="design",
        variant="B",
        parameter="length_m",
        grid=AxisGrid(start=1.0e-3, stop=3.0e-3, num=4),
    )
    result = run_sweep(spec, constants)
    path = save_sweep_npz(result, tmp_path / "L")
    assert path.exists()
    loaded = np.load(path, allow_pickle=True)
    assert np.array_equal(loaded["metric__nea_plateau_ug"], result.metrics["nea_plateau_ug"])


def test_monte_carlo_npz_roundtrip(constants: Constants, tmp_path: Path) -> None:
    """A Monte-Carlo result persists to npz and reloads the per-draw samples."""
    result = run_monte_carlo(_mc_spec(n=16), constants)
    path = save_monte_carlo_npz(result, tmp_path / "mc")
    assert path.exists()
    loaded = np.load(path, allow_pickle=True)
    assert np.array_equal(loaded["sample__nea_full_band_ug"], result.samples["nea_full_band_ug"])


def test_load_analysis_spec_dispatch(examples_dir: Path) -> None:
    """The spec loader dispatches on the kind field."""
    sweep = load_analysis_spec(examples_dir / "nea_vs_L.yaml")
    assert isinstance(sweep, SweepSpec)
    mc = load_analysis_spec(examples_dir / "montecarlo_tolerances.yaml")
    assert isinstance(mc, MonteCarloSpec)


def test_load_analysis_spec_rejects_unknown(tmp_path: Path) -> None:
    """A spec with an unknown kind is rejected."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("kind: nonsense\nname: x\n")
    with pytest.raises(ValueError, match="kind"):
        load_analysis_spec(bad)


# --------------------------------------------------------------------------- #
# Example specs run; CLI exit codes.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "spec_name", ["nea_vs_L.yaml", "sweep_amplitude.yaml", "montecarlo_tolerances.yaml"]
)
def test_example_specs_run(examples_dir: Path, spec_name: str, constants: Constants) -> None:
    """Each shipped analysis spec parses and runs end-to-end."""
    spec = load_analysis_spec(examples_dir / spec_name)
    if isinstance(spec, SweepSpec):
        result = run_sweep(spec, constants)
        assert result.axis_values.size > 0
    else:
        small = spec.model_copy(update={"n_draws": 8, "cross_axis": False})
        result = run_monte_carlo(small, constants)
        assert result.samples["nea_full_band_ug"].size == 8


def test_cli_report_exit_zero(examples_dir: Path) -> None:
    """`optivibe report` on recover_sine exits 0."""
    from optivibe.cli.main import main

    assert main(["report", str(examples_dir / "recover_sine.yaml")]) == 0


def test_cli_sweep_exit_zero(examples_dir: Path, tmp_path: Path) -> None:
    """`optivibe sweep` on a design spec exits 0 and writes outputs."""
    from optivibe.cli.main import main

    code = main(["sweep", str(examples_dir / "nea_vs_L.yaml"), "--out", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "nea_vs_L.npz").exists()


def test_cli_report_with_figures(examples_dir: Path, tmp_path: Path) -> None:
    """`optivibe report --figures` saves the truth and NEA-budget figures."""
    from optivibe.cli.main import main

    code = main(["report", str(examples_dir / "recover_sine.yaml"), "--figures", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "truth.png").exists()
    assert (tmp_path / "nea_budget.png").exists()


def test_cli_sweep_montecarlo(tmp_path: Path) -> None:
    """`optivibe sweep` on a Monte-Carlo spec runs and saves npz + figure."""
    from optivibe.cli.main import main

    spec_path = tmp_path / "mc.yaml"
    spec_path.write_text(
        "kind: montecarlo\n"
        "name: cli_mc\n"
        "variant: B\n"
        "n_draws: 8\n"
        "cross_axis: false\n"
        "tolerances:\n"
        "  q_total: {dist: lognormal, rel_sigma: 0.3}\n"
        "  epsilon_x: {dist: normal, abs_sigma: 0.1e-6}\n"
        "seed: 7\n"
    )
    code = main(["sweep", str(spec_path), "--out", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "cli_mc.npz").exists()
    assert (tmp_path / "cli_mc.png").exists()


def test_cli_sweep_bad_spec_exits_nonzero(tmp_path: Path) -> None:
    """An unparseable spec returns a non-zero exit code (no traceback)."""
    from optivibe.cli.main import main

    bad = tmp_path / "bad.yaml"
    bad.write_text("kind: nonsense\nname: x\n")
    assert main(["sweep", str(bad)]) == 2


# --------------------------------------------------------------------------- #
# Visualization smoke (head-less).
# --------------------------------------------------------------------------- #
def test_analysis_figures_build_headless(constants: Constants) -> None:
    """The analysis figures build without Qt/pyplot (SW-09)."""
    from optivibe.viz.analysis import (
        plot_monte_carlo,
        plot_nea_budget,
        plot_sweep,
        plot_truth_vs_recovery_avx,
    )

    variant = load_variant("B")
    art = _run(variant, 200.0, 1.0)
    budget = nea_budget(art.forward.detector, variant, constants)
    assert budget is not None
    sweep = run_sweep(
        SweepSpec(
            kind="sweep",
            name="L",
            mode="design",
            variant="B",
            parameter="length_m",
            grid=AxisGrid(start=1.0e-3, stop=3.0e-3, num=4),
        ),
        constants,
    )
    mc = run_monte_carlo(_mc_spec(n=12), constants)
    figures = (
        plot_truth_vs_recovery_avx(art.forward.excitation.a_x, art.result),
        plot_nea_budget(budget),
        plot_sweep(sweep),
        plot_monte_carlo(mc),
    )
    for figure in figures:
        assert figure.get_axes()
