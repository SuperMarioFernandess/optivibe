"""Expected-peak layer: golden checks against the closed forms of the base.

Tasks S-16/S-17; plan 18 §5(g) -- every golden here pins a formula of the
knowledge base (02 §2 for ``f1``, 07 §2.3 for ``f1/Q``, 03 §e for ``HD2``,
07 §1/§3.1 for the NEA threshold), never the current output of the code.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from optivibe.analysis.expected_peaks import (
    PEAK_KINDS,
    PEAK_PREDICTOR_REGISTRY,
    amplitude_noise_threshold,
    predict_expected_peaks,
    second_harmonic_ratio_taylor,
)
from optivibe.analysis.nea_budget import nea_budget
from optivibe.core.config.loader import load_constants, load_variant
from optivibe.core.config.models import Constants, ScenarioConfig, VariantConfig
from optivibe.dsp.spectra import amplitude_spectrum
from optivibe.mechanics.cantilever import CantileverModel, first_mode_hz
from optivibe.optics.reflector import build_reflector_model
from optivibe.pipeline import Pipeline

G0 = 9.80665


@pytest.fixture(scope="module")
def constants() -> Constants:
    """Physical constants (doc 01 mirror)."""
    return load_constants()


@pytest.fixture(scope="module")
def variant_b() -> VariantConfig:
    """Wideband variant B (doc 08): L = 1.4 mm, f1 ~ 25 kHz, Q ~ 2606."""
    return load_variant("B")


def _sine_scenario(
    frequency_hz: float = 100.0,
    amplitude_g: float = 10.2,
    fs_hz: float = 51200.0,
    duration_s: float = 1.0,
) -> ScenarioConfig:
    """A single-tone scenario on variant B with the physical stages."""
    return ScenarioConfig(
        name="expected_peaks_probe",
        variant="B",
        excitation={
            "kind": "sine",
            "fs_hz": fs_hz,
            "duration_s": duration_s,
            "frequency_hz": frequency_hz,
            "amplitude_g": amplitude_g,
        },
        stages={"detector": "photodiode"},
        seed=7,
    )


# --------------------------------------------------------------------------- #
# Mode branch: position and width against docs 02 §2 / 07 §2.3.
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_expected_mode_matches_first_mode_hz(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """The mode marker sits exactly on ``first_mode_hz``, not on ``100/L^2``.

    Doc 02 §2 gives the Euler-Bernoulli closed form; ``f1 ~ 100/L[mm]^2`` kHz
    (doc 08, R-31) is only a scaling reference. The layer must publish the exact
    function value (task requirement), and that value must still land within the
    doc 11 §7 tolerance of the scaling reference.
    """
    expected = predict_expected_peaks(_sine_scenario(), variant_b, constants)
    modes = expected.of_kind("mode")
    assert len(modes) == 1
    f1_reference = first_mode_hz(constants, variant_b.length_m)
    assert modes[0].freq_hz == f1_reference
    assert expected.f1_hz == f1_reference

    # Scaling reference of doc 08 / 11 §7: f1 ~ 100 / L[mm]^2 kHz, within 5 %.
    length_mm = variant_b.length_m * 1.0e3
    f1_scaling_hz = 100.0e3 / length_mm**2
    assert f1_reference == pytest.approx(f1_scaling_hz, rel=0.05)


@pytest.mark.golden
def test_golden_expected_mode_width_is_f1_over_q(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """The shaded band is ``f1/Q`` and honours the scenario ``q_total`` override.

    Doc 07 §2.3 / 17 §1: the resonance half-power width is ``Delta f = f1 / Q``.
    Halving ``Q`` through ``mechanics.q_total`` must double the band, proving the
    prediction reads the *effective* Q of the run rather than the variant's.
    """
    scenario = _sine_scenario()
    expected = predict_expected_peaks(scenario, variant_b, constants)
    mode = expected.of_kind("mode")[0]
    assert mode.width_hz == pytest.approx(expected.f1_hz / variant_b.q_total, rel=1.0e-12)

    band = expected.band_hz
    assert band is not None
    assert band[1] - band[0] == pytest.approx(mode.width_hz, rel=1.0e-12)

    halved = ScenarioConfig(
        **{**scenario.model_dump(), "mechanics": {"q_total": variant_b.q_total / 2.0}}
    )
    mode_halved = predict_expected_peaks(halved, variant_b, constants).of_kind("mode")[0]
    assert mode_halved.width_hz == pytest.approx(2.0 * mode.width_hz, rel=1.0e-12)


@pytest.mark.golden
def test_golden_expected_mode_reproduces_the_variant_b_report(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """Variant B reproduces the documented ``f1 ~ 25 kHz``, ``Q ~ 2606``, ``~10 Hz``.

    These are the numbers doc 20 §3.1 quotes for the GUI experiment of
    2026-07-13 (the "mysterious" 25 kHz line on a multitone drive), so the layer
    that explains that line must reproduce them.
    """
    expected = predict_expected_peaks(_sine_scenario(), variant_b, constants)
    assert expected.f1_hz == pytest.approx(25.0e3, rel=0.02)
    assert expected.q_total == pytest.approx(2606.0, rel=0.02)
    assert expected.of_kind("mode")[0].width_hz == pytest.approx(10.0, rel=0.05)


@pytest.mark.golden
def test_expected_mode_height_is_not_claimed(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """The resonance height stays unknown -- it is not a property of the config.

    ``|D(f1)| = Q`` amplifies whatever density happens to sit at ``f1``, so the
    layer reports ``None`` (and ``significant is None``) rather than inventing a
    number; the threshold is still published.
    """
    mode = predict_expected_peaks(_sine_scenario(), variant_b, constants).of_kind("mode")[0]
    assert mode.amplitude_m_s2 is None
    assert mode.significant is None
    assert mode.threshold_m_s2 is not None and mode.threshold_m_s2 > 0.0


# --------------------------------------------------------------------------- #
# Harmonic branch: positions against k*f, height against the doc 03 §e Taylor.
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_expected_harmonics_sit_on_k_times_tone(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """Harmonic markers sit on ``k f`` of every drive tone, within one bin."""
    tones = (137.0, 240.0)
    scenario = ScenarioConfig(
        name="expected_peaks_multitone",
        variant="B",
        excitation={
            "kind": "multitone",
            "fs_hz": 51200.0,
            "duration_s": 1.0,
            "tones": [[tones[0], 1.0], [tones[1], 2.0]],
        },
        stages={"detector": "photodiode"},
    )
    expected = predict_expected_peaks(scenario, variant_b, constants)
    resolution = expected.resolution_hz
    assert resolution is not None

    predicted = {(peak.source_freq_hz, peak.order): peak for peak in expected.of_kind("harmonic")}
    assert set(predicted) == {(tones[0], 2), (tones[0], 3), (tones[1], 2), (tones[1], 3)}
    for (tone_hz, order), peak in predicted.items():
        assert tone_hz is not None
        assert abs(peak.freq_hz - order * tone_hz) <= resolution


@pytest.mark.golden
def test_golden_expected_hd2_matches_simulated_eta_second_harmonic(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """The predicted ``2f`` height matches a simulated ``eta(dx sine)`` spectrum.

    Independent reference, the same construction as the S3 golden
    ``test_golden_thd_matches_analytic_second_harmonic`` (doc 03 §e): drive the
    coupling model with the tip displacement the mechanics FRF gives for this
    tone (``d = |H_lat(f)| a``, doc 05 §1), measure ``2f/1f`` on the resulting
    ``eta`` series, and compare with the amplitude the layer predicts (which
    refers both lines through the same plateau scalar, so the ratio carries).
    """
    frequency_hz, amplitude_g = 100.0, 10.2
    scenario = _sine_scenario(frequency_hz=frequency_hz, amplitude_g=amplitude_g)
    expected = predict_expected_peaks(scenario, variant_b, constants)
    peak = next(p for p in expected.of_kind("harmonic") if p.order == 2)
    assert peak.amplitude_m_s2 is not None

    cantilever = CantileverModel.from_config(constants, variant_b)
    accel = amplitude_g * G0
    tip_m = abs(complex(cantilever.h_lat(frequency_hz)[0])) * accel
    gain = abs(complex(cantilever.dynamic_factor(frequency_hz)[0]))

    fs, n_cycles = 51200.0, 64
    t = np.arange(round(n_cycles * fs / frequency_hz)) / fs
    dx = tip_m * np.sin(2.0 * np.pi * frequency_hz * t)
    model = build_reflector_model(variant_b)
    eta = model.eta(dx=dx)
    spectrum = np.abs(np.fft.rfft(eta - np.mean(eta)))
    bin_1f = round(frequency_hz * eta.size / fs)
    hd2_simulated = float(spectrum[2 * bin_1f] / spectrum[bin_1f])

    hd2_predicted = peak.amplitude_m_s2 / (accel * gain)
    assert hd2_predicted == pytest.approx(hd2_simulated, rel=0.02)

    # Doc 20 §3.1 quotes THD ~ 0.03 % for exactly this drive.
    assert hd2_predicted == pytest.approx(3.0e-4, rel=0.15)


@pytest.mark.golden
def test_golden_expected_hd2_line_grows_as_amplitude_squared(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """The ``2f`` line scales as ``d^2`` -- the diagnostic key of doc 20 §3.2.

    ``HD2 ~ d`` and the fundamental ~ ``d``, so the absolute ``2f`` amplitude
    goes as ``d^2``: the experiment behind S-17 (10.2 g -> visible, 0.5 g ->
    gone, ratio ``(0.5/10.2)^2 ~ 1/416``).
    """
    loud, quiet = 10.2, 0.5

    def line_amplitude(amplitude_g: float) -> float:
        expected = predict_expected_peaks(
            _sine_scenario(amplitude_g=amplitude_g), variant_b, constants
        )
        peak = next(p for p in expected.of_kind("harmonic") if p.order == 2)
        assert peak.amplitude_m_s2 is not None
        return peak.amplitude_m_s2

    assert line_amplitude(loud) / line_amplitude(quiet) == pytest.approx(
        (loud / quiet) ** 2, rel=1.0e-9
    )


@pytest.mark.golden
def test_golden_third_harmonic_is_located_but_not_sized(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """``3f`` gets a position and a threshold, but no invented height."""
    expected = predict_expected_peaks(_sine_scenario(), variant_b, constants)
    third = next(p for p in expected.of_kind("harmonic") if p.order == 3)
    assert third.freq_hz == pytest.approx(300.0, rel=1.0e-12)
    assert third.amplitude_m_s2 is None
    assert third.significant is None


def test_expected_harmonics_absent_without_a_tonal_stimulus(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """Random noise has no closed-form line list, so no harmonics are claimed."""
    scenario = ScenarioConfig(
        name="expected_peaks_random",
        variant="B",
        excitation={
            "kind": "random",
            "fs_hz": 51200.0,
            "duration_s": 1.0,
            "band_hz": (20.0, 2000.0),
            "g_rms": 1.0,
        },
        stages={"excitation": "random", "detector": "photodiode"},
    )
    expected = predict_expected_peaks(scenario, variant_b, constants)
    assert expected.of_kind("harmonic") == ()
    assert len(expected.of_kind("mode")) == 1  # the resonance is still expected


def test_expected_harmonics_unsized_with_the_stub_optics(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """Without a physical optics stage there is no ``eta`` to differentiate."""
    base = _sine_scenario().model_dump()
    scenario = ScenarioConfig(
        **{**base, "stages": {**base["stages"], "optics": "stub", "detector": "photodiode"}}
    )
    expected = predict_expected_peaks(scenario, variant_b, constants)
    second = next(p for p in expected.of_kind("harmonic") if p.order == 2)
    assert second.amplitude_m_s2 is None


@pytest.mark.golden
def test_golden_second_harmonic_ratio_taylor_is_linear_in_displacement(
    variant_b: VariantConfig,
) -> None:
    """``HD2 = |eta''| d / (4 |eta'|)`` is exactly proportional to ``d`` (03 §e)."""
    model = build_reflector_model(variant_b)
    small = second_harmonic_ratio_taylor(model, 10.0e-9)
    large = second_harmonic_ratio_taylor(model, 40.0e-9)
    assert small is not None and large is not None
    assert large / small == pytest.approx(4.0, rel=1.0e-12)


# --------------------------------------------------------------------------- #
# Intermodulation branch.
# --------------------------------------------------------------------------- #
def test_expected_intermod_positions_are_sum_and_difference(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """Two tones give ``f_i + f_j`` and ``|f_i - f_j|``; one tone gives nothing."""
    scenario = ScenarioConfig(
        name="expected_peaks_im2",
        variant="B",
        excitation={
            "kind": "multitone",
            "fs_hz": 51200.0,
            "duration_s": 1.0,
            "tones": [[137.0, 1.0], [240.0, 1.0]],
        },
        stages={"detector": "photodiode"},
    )
    positions = {
        peak.freq_hz
        for peak in predict_expected_peaks(scenario, variant_b, constants).of_kind("intermod")
    }
    assert positions == {377.0, 103.0}

    single = predict_expected_peaks(_sine_scenario(), variant_b, constants)
    assert single.of_kind("intermod") == ()


# --------------------------------------------------------------------------- #
# Taxonomy: declared in full, empty branches stay empty and never raise.
# --------------------------------------------------------------------------- #
def test_expected_peaks_taxonomy_is_declared_in_full() -> None:
    """The whole family of doc 13 is declared; only some branches are wired."""
    assert PEAK_KINDS == (
        "mode",
        "harmonic",
        "intermod",
        "sideband",
        "mains",
        "alias",
        "f_mount",
    )
    assert set(PEAK_PREDICTOR_REGISTRY.keys()) == {"mode", "harmonic", "intermod"}


def test_expected_peaks_empty_branches_are_silent(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """Requesting an unimplemented branch yields nothing instead of an error.

    The slots ``sideband`` (S-21), ``mains``, ``alias`` and ``f_mount`` are
    declared so a caller can always ask for the whole taxonomy (doc 13).
    """
    empty = ("sideband", "mains", "alias", "f_mount")
    expected = predict_expected_peaks(_sine_scenario(), variant_b, constants, kinds=empty)
    assert expected.peaks == ()
    assert expected.kinds == empty
    for kind in empty:
        assert expected.of_kind(kind) == ()  # type: ignore[arg-type]


def test_expected_peaks_default_request_covers_every_kind(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """The default call requests the full taxonomy."""
    assert predict_expected_peaks(_sine_scenario(), variant_b, constants).kinds == PEAK_KINDS


# --------------------------------------------------------------------------- #
# Nyquist / resolution handling.
# --------------------------------------------------------------------------- #
def test_expected_peaks_drop_lines_above_nyquist(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """A line the run cannot show is not predicted (it would alias)."""
    slow = _sine_scenario(fs_hz=2000.0, duration_s=1.0)
    expected = predict_expected_peaks(slow, variant_b, constants)
    assert expected.nyquist_hz == 1000.0
    assert expected.of_kind("mode") == ()  # f1 ~ 25 kHz is far above Nyquist
    assert expected.band_hz is None
    assert all(peak.freq_hz < 1000.0 for peak in expected.peaks)


def test_expected_peaks_resolution_unknown_for_file_replay(
    constants: Constants, variant_b: VariantConfig, tmp_path
) -> None:
    """A replayed record has no config-known length, so thresholds stay ``None``."""
    csv_path = tmp_path / "record.csv"
    csv_path.write_text("t,a\n0.0,0.0\n")
    scenario = ScenarioConfig(
        name="expected_peaks_replay",
        variant="B",
        excitation={"kind": "csv", "path": str(csv_path), "fs_hz": 51200.0},
        stages={"excitation": "csv", "detector": "photodiode"},
    )
    expected = predict_expected_peaks(scenario, variant_b, constants)
    assert expected.resolution_hz is None
    mode = expected.of_kind("mode")[0]
    assert mode.threshold_m_s2 is None
    assert mode.significant is None


def test_expected_peaks_resolution_matches_the_rfft_grid(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """The assumed bin width equals the real ``amplitude_spectrum`` spacing."""
    fs_hz, duration_s = 51200.0, 0.5
    expected = predict_expected_peaks(
        _sine_scenario(fs_hz=fs_hz, duration_s=duration_s), variant_b, constants
    )
    n_samples = round(fs_hz * duration_s)
    spectrum = amplitude_spectrum(np.zeros(n_samples) + 1.0, fs_hz)
    assert expected.resolution_hz == pytest.approx(
        float(spectrum.freq[1] - spectrum.freq[0]), rel=1.0e-12
    )


def test_predict_expected_peaks_rejects_a_degenerate_harmonic_order(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """``max_harmonic < 2`` is a loud error, not a silent empty result (10 §7)."""
    with pytest.raises(ValueError, match="max_harmonic"):
        predict_expected_peaks(_sine_scenario(), variant_b, constants, max_harmonic=1)


# --------------------------------------------------------------------------- #
# Significance threshold: the NEA chain of doc 07 and its spectral mapping.
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_config_only_nea_matches_the_detector_path(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """The config-only plateau NEA equals the one the detector record reports.

    Cross-check of the two routes to the same doc 07 quantity: this layer
    rebuilds ``I_DC = R P (R1 + rho eta0)`` from the configuration, while
    :func:`~optivibe.analysis.nea_budget.nea_budget` reads it from the detector
    metadata of an actual run. They must agree to numerical precision -- if they
    ever drift, one of the two has silently changed convention (O-SW-08).
    """
    scenario = _sine_scenario(duration_s=0.05)
    artifacts = Pipeline(scenario, variant_b).run()
    budget = nea_budget(artifacts.forward.detector, variant_b)
    assert budget is not None

    expected = predict_expected_peaks(scenario, variant_b, constants)
    assert expected.nea_plateau_m_s2_rthz == pytest.approx(budget.nea_plateau, rel=1.0e-9)


@pytest.mark.golden
def test_golden_amplitude_noise_threshold_matches_a_white_floor() -> None:
    """``sqrt(2) NEA sqrt(df)`` is the RMS bin amplitude of a white floor.

    Verifies the mapping of :func:`amplitude_noise_threshold` against a
    simulated white series instead of against the code: with
    ``amplitude_spectrum``'s ``2/N`` scaling, ``E[A^2] = 2 S df``.
    """
    fs, n_samples = 4096.0, 1 << 16
    density = 3.0  # (m/s^2)/sqrt(Hz)
    rng = np.random.default_rng(20260730)
    sigma = density * math.sqrt(fs / 2.0)
    spectrum = amplitude_spectrum(rng.standard_normal(n_samples) * sigma, fs)
    resolution = float(spectrum.freq[1] - spectrum.freq[0])

    body = spectrum.values[1:-1]
    measured_rms = float(np.sqrt(np.mean(body**2)))
    one_sigma = amplitude_noise_threshold(density, resolution, sigma_factor=1.0)
    assert measured_rms == pytest.approx(one_sigma, rel=0.02)
    assert one_sigma == pytest.approx(math.sqrt(2.0 * resolution) * density, rel=1.0e-12)


def test_amplitude_noise_threshold_scales_with_sigma_and_resolution() -> None:
    """The threshold is linear in ``sigma_factor`` and goes as ``sqrt(df)``."""
    base = amplitude_noise_threshold(1.0, 1.0, sigma_factor=1.0)
    assert amplitude_noise_threshold(1.0, 1.0, sigma_factor=3.0) == pytest.approx(3.0 * base)
    assert amplitude_noise_threshold(1.0, 4.0, sigma_factor=1.0) == pytest.approx(2.0 * base)
    with pytest.raises(ValueError, match="resolution_hz"):
        amplitude_noise_threshold(1.0, 0.0)
    with pytest.raises(ValueError, match="sigma_factor"):
        amplitude_noise_threshold(1.0, 1.0, sigma_factor=0.0)


@pytest.mark.golden
def test_golden_threshold_dips_toward_the_resonance(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """The threshold near ``f1`` is far below the in-band one (doc 07 §2.4).

    The optical branch is divided by ``|s_target^QS D(f)|``, so it dips by
    ``~1/Q`` toward the resonance until the flat Brownian floor takes over.
    """
    expected = predict_expected_peaks(_sine_scenario(), variant_b, constants)
    in_band = next(p for p in expected.of_kind("harmonic") if p.order == 2)
    mode = expected.of_kind("mode")[0]
    assert in_band.threshold_m_s2 is not None and mode.threshold_m_s2 is not None
    assert mode.threshold_m_s2 < in_band.threshold_m_s2 / 10.0


def test_expected_peak_significance_compares_amplitude_with_threshold(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """The loud drive's ``2f`` clears the floor; a very quiet one does not."""
    loud = predict_expected_peaks(_sine_scenario(amplitude_g=10.2), variant_b, constants)
    quiet = predict_expected_peaks(_sine_scenario(amplitude_g=0.001), variant_b, constants)
    assert next(p for p in loud.of_kind("harmonic") if p.order == 2).significant is True
    assert next(p for p in quiet.of_kind("harmonic") if p.order == 2).significant is False


# --------------------------------------------------------------------------- #
# Artifact shape and provenance.
# --------------------------------------------------------------------------- #
def test_expected_peaks_artifact_is_sorted_and_carries_provenance(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """Peaks come out ordered by frequency with the prediction's provenance."""
    expected = predict_expected_peaks(_sine_scenario(), variant_b, constants)
    frequencies = [peak.freq_hz for peak in expected.peaks]
    assert frequencies == sorted(frequencies)
    assert len(expected) == len(expected.peaks)
    assert expected.variant_name == variant_b.name
    assert expected.scenario_name == "expected_peaks_probe"
    assert expected.nyquist_hz == 25600.0


def test_expected_peaks_do_not_touch_the_vibration_result_contract(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """The prediction stays a separate artifact (doc 13, coordination 2026-07-29).

    ``VibrationResult`` must gain no expected-peak field: measured and expected
    are different categories (doc 20 §3.1), and the prediction needs no record.
    """
    scenario = _sine_scenario(duration_s=0.05)
    result = Pipeline(scenario, variant_b).run().result
    assert not any("expect" in name for name in vars(result))
    predict_expected_peaks(scenario, variant_b, constants)  # needs no result at all


# --------------------------------------------------------------------------- #
# Figure overlay (viz/dsp) -- the matplotlib path, testable without Qt.
# --------------------------------------------------------------------------- #
def test_plot_spectrum_overlays_the_expected_peaks(
    constants: Constants, variant_b: VariantConfig
) -> None:
    """``plot_spectrum(expected=...)`` adds one line per peak plus the band.

    The matplotlib route is what the CLI ``report --figures`` export uses, so it
    is exercised head-less; the Qt overlay is covered by the ``gui`` tests.
    """
    from optivibe.viz.dsp import plot_spectrum

    scenario = _sine_scenario(duration_s=0.05)
    artifacts = Pipeline(scenario, variant_b).run()
    spectrum = artifacts.result.spectrum
    assert spectrum is not None
    expected = predict_expected_peaks(scenario, variant_b, constants)
    assert expected.peaks

    plain = plot_spectrum(spectrum, artifacts.result.dominant_freqs_hz)
    marked = plot_spectrum(spectrum, artifacts.result.dominant_freqs_hz, expected=expected)

    added = len(marked.get_axes()[0].lines) - len(plain.get_axes()[0].lines)
    assert added == len(expected.peaks)
    # The f1/Q band is a patch (axvspan), not a line.
    assert len(marked.get_axes()[0].patches) > len(plain.get_axes()[0].patches)


def test_plot_spectrum_is_unchanged_without_expected_peaks(
    variant_b: VariantConfig,
) -> None:
    """The default call is byte-for-byte the pre-S-16 figure (additive change)."""
    from optivibe.viz.dsp import plot_spectrum

    artifacts = Pipeline(_sine_scenario(duration_s=0.05), variant_b).run()
    spectrum = artifacts.result.spectrum
    assert spectrum is not None
    figure = plot_spectrum(spectrum, artifacts.result.dominant_freqs_hz)
    axis = figure.get_axes()[0]
    assert len(axis.lines) == 1 + len(artifacts.result.dominant_freqs_hz)
    assert len(axis.patches) == 0
