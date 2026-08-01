"""Tests for the DSP-chain comparison bench (task S-22; Qt-free half).

Deliberately *not* combinatorial (rule 3 of the coordination decision of
2026-07-29): a golden file per combination of knobs would explode, so each
alternative faces the same analytical reference instead -- a clean tone must
give ``v = A/omega`` and ``x = A/omega^2`` inside the tolerances of 11 §7, and
every window must satisfy amplitude correction and Parseval. That is the style
of the L2 cross-checks of plan 19 §2 and pins formulas of the base, not the
current output of the code (18 §5, rule "g").

The other half of the file pins the two structural promises of the bench: the
default chain reproduces an ordinary run **bit for bit** (the acceptance check
of W-1), and the input is opened through the *same* path the live oscilloscope
and ``analyze`` use, so a difference in the numbers can only come from the
chains.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from optivibe.analysis.compare import (
    CHAIN_APPLICABILITY,
    DEFAULT_CHAIN,
    EXPERIMENT_FIELDS,
    ChainSpec,
    CompareSource,
    CompareSpec,
    chain_deltas,
    chain_provenance,
    chain_status,
    compare_chains,
    input_from_analyze_spec,
    input_from_scenario,
    load_compare_spec,
    run_comparison,
)
from optivibe.analysis.instrument import analyze_record, load_analyze_spec
from optivibe.core.config.loader import load_constants, load_variant
from optivibe.core.config.models import (
    Constants,
    DspOptions,
    ScenarioConfig,
    SineSpec,
    StageSelection,
    VariantConfig,
)
from optivibe.dsp.kinematics import INTEGRATOR_REGISTRY
from optivibe.dsp.spectra import amplitude_spectrum, dominant_frequencies, welch_psd
from optivibe.pipeline.orchestrator import Pipeline, RunArtifacts

F0 = 200.0
FS = 5000.0
TS = "2026-08-01T00:00:00"
WINDOWS = ("hann", "hamming", "blackman", "nuttall", "flattop", "boxcar")


@pytest.fixture(scope="module")
def constants(config_dir: Path) -> Constants:
    return load_constants(config_dir / "constants.yaml")


@pytest.fixture(scope="module")
def variant_b(config_dir: Path) -> VariantConfig:
    return load_variant("B", config_dir)


@pytest.fixture(scope="module")
def scenario_b() -> ScenarioConfig:
    return ScenarioConfig(
        name="s22-fixture",
        variant="B",
        excitation=SineSpec(
            kind="sine", axis="x", fs_hz=FS, duration_s=1.0, frequency_hz=F0, amplitude_g=1.0
        ),
        stages=StageSelection(detector="photodiode", dsp="standard"),
        seed=7,
    )


@pytest.fixture(scope="module")
def scenario_clean() -> ScenarioConfig:
    """Noiseless variant-B scenario (stub detector) for the integrator reper.

    The cross-check of two integrators has to be run on a *clean* input: with
    detector noise present the two routes legitimately disagree far more,
    because the spectral integrator brick-walls the sub-band noise that the
    time-domain one only detrends -- and 1/omega^2 magnifies exactly that band.
    That difference is a lesson of the bench, not a defect, so it is not what
    this reper measures.
    """
    return ScenarioConfig(
        name="s22-clean",
        variant="B",
        excitation=SineSpec(
            kind="sine", axis="x", fs_hz=FS, duration_s=1.0, frequency_hz=F0, amplitude_g=1.0
        ),
        stages=StageSelection(detector="stub", dsp="standard"),
        seed=7,
    )


@pytest.fixture(scope="module")
def run_b(scenario_b: ScenarioConfig, variant_b: VariantConfig) -> RunArtifacts:
    return Pipeline(scenario_b, variant_b).run()


def _write_record_csv(path: Path, artifacts: RunArtifacts) -> None:
    det = artifacts.forward.detector
    t = np.arange(det.n_samples) / det.fs
    np.savetxt(
        path,
        np.column_stack([t, det.samples]),
        delimiter=",",
        header="time_s,i_pd_a",
        comments="",
        fmt="%.17e",
    )


def _analyze_spec_yaml(record_csv: Path, spec_path: Path) -> None:
    spec_path.write_text(
        "\n".join(
            [
                "kind: analyze",
                "name: s22-record",
                "variant: B",
                "record:",
                "  format: csv",
                f"  path: {record_csv}",
                "  units: A",
                "  column: i_pd_a",
                "  time_column: time_s",
                f'  timestamp: "{TS}"',
                "calibration:",
                "  source: model",
            ]
        )
        + "\n"
    )


# --------------------------------------------------------------------------- #
# Rule 1: the boundary of the verified is computed, never declared
# --------------------------------------------------------------------------- #
def test_default_chain_is_the_only_verified_one() -> None:
    """``DspOptions()`` grades verified; any deviation flips it to experimental."""
    assert chain_status(DEFAULT_CHAIN) == "verified"
    assert chain_deltas(DEFAULT_CHAIN) == ()
    experimental = DspOptions(integrator="time")
    assert chain_status(experimental) == "experimental"
    deltas = chain_deltas(experimental)
    assert [d.field for d in deltas] == ["integrator"]
    assert deltas[0].default == "frequency"
    assert deltas[0].value == "time"


def test_provenance_carries_verdict_deviations_and_code_identity() -> None:
    """Rule 1: exported numbers must not be mistakable for the verified twin."""
    provenance = chain_provenance(
        DspOptions(integrator="time", window="flattop"), input_label="probe"
    )
    assert provenance["status"] == "experimental"
    fields = {entry["field"] for entry in provenance["deviations_from_default"]}
    assert fields == {"integrator", "window"}
    assert provenance["chain"]["integrator"] == "time"
    assert set(provenance["code"]) == {"optivibe_version", "git_head"}


def test_applicability_map_covers_every_dsp_option() -> None:
    """Every option is labelled batch / stream / both -- a new one cannot hide.

    The panel renders this map; if a future ``DspOptions`` field were missing
    here it would silently appear without an applicability tag, i.e. the UI
    would stop saying which path the knob reaches.
    """
    assert set(CHAIN_APPLICABILITY) == set(DEFAULT_CHAIN.model_dump())
    assert set(EXPERIMENT_FIELDS) <= set(CHAIN_APPLICABILITY)
    # The batch-only labels are the ones the streaming layer documents.
    assert CHAIN_APPLICABILITY["integrator"] == "batch"
    assert CHAIN_APPLICABILITY["spectrum_method"] == "batch"
    assert CHAIN_APPLICABILITY["sensitivity_freq"] == "batch"
    assert CHAIN_APPLICABILITY["f_c_stream"] == "stream"


# --------------------------------------------------------------------------- #
# The acceptance check of W-1: the default chain is bit-identical to a run
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_default_chain_reproduces_the_ordinary_run_bit_for_bit(
    scenario_b: ScenarioConfig, variant_b: VariantConfig, run_b: RunArtifacts
) -> None:
    """Comparing chains must not perturb the verified path by a single bit."""
    source = input_from_scenario(scenario_b, variant_b)
    result = compare_chains(source, [ChainSpec(name="default")])
    chain = result.chains[0]
    assert chain.status == "verified"
    np.testing.assert_array_equal(chain.result.a, run_b.result.a)
    np.testing.assert_array_equal(chain.result.v, run_b.result.v)
    np.testing.assert_array_equal(chain.result.x, run_b.result.x)
    assert chain.result.dominant_freqs_hz == run_b.result.dominant_freqs_hz


@pytest.mark.golden
def test_comparison_input_equals_the_live_scenario_source(
    scenario_b: ScenarioConfig, variant_b: VariantConfig, config_dir: Path
) -> None:
    """One data path: the bench and the live oscilloscope synthesize the same input.

    If the two ever diverged, a difference in the compared numbers could come
    from the *input* rather than from the chains -- the one thing this bench
    exists to rule out.
    """
    from optivibe.gui.workers.stream import ScenarioSource

    setup = ScenarioSource(scenario=scenario_b, config_dir=config_dir).open()
    source = input_from_scenario(scenario_b, variant_b)
    np.testing.assert_array_equal(source.detector.samples, setup.detector.samples)
    assert source.detector.fs == setup.detector.fs
    assert source.detector.dc_level == setup.detector.dc_level


@pytest.mark.golden
def test_record_input_equals_the_analyze_path(tmp_path: Path, run_b: RunArtifacts) -> None:
    """A recorded input is read exactly the way ``optivibe analyze`` reads it."""
    record = tmp_path / "rec.csv"
    spec_path = tmp_path / "spec.yaml"
    _write_record_csv(record, run_b)
    _analyze_spec_yaml(record, spec_path)

    source = input_from_analyze_spec(spec_path)
    compared = compare_chains(source, [ChainSpec(name="default")]).chains[0].result
    analyzed = analyze_record(load_analyze_spec(spec_path))
    assert analyzed.result is not None
    np.testing.assert_array_equal(compared.a, analyzed.result.a)
    np.testing.assert_array_equal(compared.v, analyzed.result.v)


# --------------------------------------------------------------------------- #
# Rule 3: every alternative against the same analytical reference (19 §2)
# --------------------------------------------------------------------------- #
@pytest.mark.golden
@pytest.mark.parametrize("key", ["frequency", "time", "leaky"])
def test_every_integrator_meets_the_kinematic_reference(key: str) -> None:
    """A clean tone must integrate to ``v = A/omega`` and ``x = A/omega^2``.

    The closed-form reference is the kinematics of a harmonic motion (doc 05 §2;
    tolerance band of 11 §7 for the DSP row), not the current output of any
    integrator -- so the check grades a *new* alternative the same way it grades
    the registered ones. That is rule 3 of the S-22 decision in practice: one
    analytical reper instead of a golden per combination.

    The measurement window is the settled tail of the record, because the causal
    ``leaky`` route has a warm-up: its filters start at rest and need a few leak
    time constants (``1/(2 pi f_c)``) before their output means anything. That is
    the same fact the live oscilloscope publishes as ``warmed`` (theory-06 §5.7),
    and it is a property of causality, not a defect -- so the reper is applied
    where the physics is, not across the transient.
    """
    fs, f0, amplitude, f_hp = 5000.0, 200.0, 3.0, 1.0
    n = int(4.0 * fs)
    t = np.arange(n, dtype=np.float64) / fs
    accel = amplitude * np.sin(2.0 * math.pi * f0 * t)
    omega = 2.0 * math.pi * f0

    integrate = INTEGRATOR_REGISTRY.get(key)
    velocity, displacement = integrate(accel, fs, f_hp)
    # Two chained causal stages need roughly twice the warm-up of one, so the
    # window opens at ten leak time constants (the streaming layer applies the
    # same logic to its own `warmed` flag).
    settled = slice(int(10.0 / (2.0 * math.pi * f_hp) * fs), None)

    # Compare RMS amplitudes (phase conventions differ between the routes).
    v_expected = amplitude / omega / math.sqrt(2.0)
    x_expected = amplitude / omega**2 / math.sqrt(2.0)
    assert float(np.std(velocity[settled])) == pytest.approx(v_expected, rel=0.02)
    assert float(np.std(displacement[settled])) == pytest.approx(x_expected, rel=0.02)


@pytest.mark.golden
def test_peak_interpolation_removes_the_bin_quantisation() -> None:
    """Sub-bin interpolation is worth its keep, and switching it off shows the grid.

    The reference is the bin width itself (``fs/N``): without interpolation a
    dominant can only be reported at a bin centre, so an off-bin tone is wrong by
    up to half a bin; with it, the quadratic fit recovers the true line. Both
    statements are checked against the tone, not against a stored number.
    """
    fs = 4096.0
    n = 4096
    bin_hz = fs / n
    f0 = 128.0 + 0.5 * bin_hz  # deliberately between two bins
    t = np.arange(n, dtype=np.float64) / fs
    signal = np.sin(2.0 * math.pi * f0 * t)
    spectrum = amplitude_spectrum(signal, fs)

    interpolated = dominant_frequencies(spectrum, interpolate=True)[0]
    raw = dominant_frequencies(spectrum, interpolate=False)[0]
    assert abs(interpolated - f0) < 0.25 * bin_hz
    assert abs(raw - f0) >= 0.25 * bin_hz
    assert raw / bin_hz == pytest.approx(round(raw / bin_hz), abs=1e-9)  # a bin centre


@pytest.mark.golden
def test_new_alternatives_are_selectable_and_marked_experimental(
    scenario_clean: ScenarioConfig, variant_b: VariantConfig
) -> None:
    """The W-3 alternatives reach the chain through the registry and the config.

    They are additions, never a change of the default: the verified chain still
    grades verified, and the causal route grades experimental like any other
    deviation.
    """
    source = input_from_scenario(scenario_clean, variant_b)
    result = compare_chains(
        source,
        [
            ChainSpec(name="default"),
            ChainSpec(name="causal", dsp=DspOptions(integrator="leaky")),
            ChainSpec(name="no-interp", dsp=DspOptions(peak_interpolation=False)),
        ],
    )
    assert [outcome.status for outcome in result.chains] == [
        "verified",
        "experimental",
        "experimental",
    ]
    # The acceleration is upstream of every one of these knobs.
    for outcome in result.chains[1:]:
        np.testing.assert_array_equal(outcome.result.a, result.chains[0].result.a)


@pytest.mark.golden
@pytest.mark.parametrize("window", WINDOWS)
def test_every_window_is_amplitude_corrected(window: str) -> None:
    """Each window reads a pure tone at its true amplitude (coherent-gain rule).

    Amplitude correction divides by the coherent gain ``sum(w)/N``; the pinned
    reference is the tone amplitude itself, so a window family member that
    forgot the correction fails here rather than in a snapshot.
    """
    fs, f0, amplitude = 4096.0, 128.0, 2.5
    n = 4096  # integer number of periods -> no leakage, exact bin
    t = np.arange(n, dtype=np.float64) / fs
    signal = amplitude * np.sin(2.0 * math.pi * f0 * t)
    spectrum = amplitude_spectrum(signal, fs, window=window)
    assert float(np.max(spectrum.values)) == pytest.approx(amplitude, rel=0.02)


@pytest.mark.golden
@pytest.mark.parametrize("window", WINDOWS)
def test_every_window_satisfies_parseval(window: str) -> None:
    """The PSD integral equals the signal power for every window of the family."""
    fs = 4096.0
    n = 8192
    rng = np.random.default_rng(11)
    signal = rng.standard_normal(n)
    psd = welch_psd(signal, fs, window=window, nperseg=1024)
    power_time = float(np.mean(signal**2))
    power_freq = float(np.trapezoid(psd.values, psd.freq))
    assert power_freq == pytest.approx(power_time, rel=0.05)


@pytest.mark.golden
def test_two_integrators_agree_in_band_on_the_same_input(
    scenario_clean: ScenarioConfig, variant_b: VariantConfig
) -> None:
    """The L2 cross-check of 19 §2 as the bench presents it: two routes, one input.

    The disagreement between the spectral and the time-domain integrator is a
    *measurement* of the numerical error of double integration; the reference is
    the documented few-per-cent agreement in band (``dsp/kinematics`` module
    docstring, doc 05 §2), so a regression that silently broke one route would
    widen this gap.
    """
    source = input_from_scenario(scenario_clean, variant_b)
    result = compare_chains(
        source,
        [ChainSpec(name="freq"), ChainSpec(name="time", dsp=DspOptions(integrator="time"))],
        name="integrators",
    )
    rows = {row.key: row for row in result.rows}
    for key in ("rms_v", "rms_x"):
        rel = rows[key].rel_diff[1]
        assert rel is not None
        assert abs(rel) < 0.05
    # Acceleration is upstream of the integrators: it must be bit-identical.
    np.testing.assert_array_equal(result.chains[0].result.a, result.chains[1].result.a)


# --------------------------------------------------------------------------- #
# Rule 2: the experiment is a config
# --------------------------------------------------------------------------- #
def test_compare_spec_requires_exactly_one_source() -> None:
    """A comparison feeds one input to every chain -- two sources is a mistake."""
    with pytest.raises(ValueError, match="exactly one"):
        CompareSource(scenario="a.yaml", analyze="b.yaml")
    with pytest.raises(ValueError, match="exactly one"):
        CompareSource()


def test_compare_spec_rejects_a_foreign_kind(tmp_path: Path) -> None:
    """The ``kind`` discriminator is checked, as for the analyze/sweep specs."""
    path = tmp_path / "spec.yaml"
    path.write_text("kind: analyze\nname: x\n")
    with pytest.raises(ValueError, match="kind 'compare'"):
        load_compare_spec(path)


def test_chain_names_must_be_unique(scenario_b: ScenarioConfig, variant_b: VariantConfig) -> None:
    """The name keys the legend, the table and the provenance."""
    source = input_from_scenario(scenario_b, variant_b)
    with pytest.raises(ValueError, match="unique"):
        compare_chains(source, [ChainSpec(name="a"), ChainSpec(name="a")])


@pytest.mark.golden
def test_example_compare_spec_runs_and_grades_its_chains(examples_dir: Path) -> None:
    """The shipped example is a working, reproducible experiment (rule 2)."""
    spec = load_compare_spec(examples_dir / "compare_integrators.yaml")
    assert isinstance(spec, CompareSpec)
    result = run_comparison(spec)
    statuses = {outcome.name: outcome.status for outcome in result.chains}
    assert statuses == {
        "default": "verified",
        "time-integrator": "experimental",
        "welch-spectrum": "experimental",
    }
    assert result.status == "experimental"
    # The reference chain of the table is the default one, and the table is
    # complete: every metric row carries one value per chain.
    assert all(len(row.values) == len(result.chains) for row in result.rows)
    assert "experimental" in result.as_text()


def test_cli_compare_prints_the_table(examples_dir: Path, tmp_path: Path, capsys) -> None:
    """``optivibe compare`` is the head-less half of the bench."""
    from optivibe.cli.main import main

    provenance = tmp_path / "prov.yaml"
    code = main(
        [
            "compare",
            str(examples_dir / "compare_integrators.yaml"),
            "--provenance",
            str(provenance),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "rms_a" in out
    assert "experimental" in out
    text = provenance.read_text()
    assert "status: experimental" in text
    assert "deviations_from_default" in text
