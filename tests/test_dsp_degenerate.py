"""Degenerate inputs of the computing core: metrics, spectra, ISO (task S-25, gap 18 G8).

Every test here answers one question: **when a metric cannot be computed, is the
answer distinguishable from an honest zero?** The rule under test is the one
fixed by `SW-77` and mirrored in doc 17 §1a -- an undefined scalar metric is
``None``, never ``0.0`` -- and it is an application of coding convention 10 §7
("no silent failures"), not a new invention. The reference for each assertion is
the *definition* of the metric (17 §1, 20 §3) or of the operation it rests on
(the trapezoid rule needs an interval; a ratio needs a denominator; prominence
needs a neighbourhood), never the current output of the code (18 §5(g)).

Degenerate inputs do not occur on synthetic scenarios and will occur on the
phase-0 records of plan 20 (short record, narrow ISO band, unidentified
fundamental, spectrum of a few bins), which is why they are P1 rather than
coverage hygiene.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from optivibe.core.config.loader import load_constants, load_scenario, load_variant
from optivibe.core.config.models import Constants, DspOptions, VariantConfig
from optivibe.core.types import DetectorOutput, Spectrum
from optivibe.dsp.iso import ISO_10816_3_ZONES, iso_assessment
from optivibe.dsp.metrics import (
    DEGENERATE_REASONS,
    band_rms_velocity,
    cross_axis_suppression,
    second_harmonic_ratio,
)
from optivibe.dsp.spectra import amplitude_spectrum, dominant_frequencies
from optivibe.dsp.streaming import StreamingDsp
from optivibe.pipeline.orchestrator import Pipeline

FS = 10_000.0


@pytest.fixture(scope="module")
def constants(config_dir: Path) -> Constants:
    """Physical constants bundle."""
    return load_constants(config_dir / "constants.yaml")


@pytest.fixture(scope="module")
def variant_b(config_dir: Path) -> VariantConfig:
    """Variant B (the numeric reference)."""
    return load_variant("B", config_dir)


def _psd(freq: list[float], values: list[float]) -> Spectrum:
    """Build a PSD spectrum contract from explicit bins."""
    return Spectrum(
        freq=np.asarray(freq, dtype=np.float64),
        values=np.asarray(values, dtype=np.float64),
        kind="psd",
        window="hann",
        method="welch",
    )


def _amplitude(freq: list[float], values: list[float]) -> Spectrum:
    """Build an amplitude spectrum contract from explicit bins."""
    return Spectrum(
        freq=np.asarray(freq, dtype=np.float64),
        values=np.asarray(values, dtype=np.float64),
        kind="amplitude",
    )


# --------------------------------------------------------------------------- #
# band_rms_velocity (17 §1: the broadband velocity ISO 10816-3 grades).
# --------------------------------------------------------------------------- #
def test_band_rms_velocity_rejects_a_non_psd_spectrum() -> None:
    """An amplitude spectrum is a wrong-kind input, i.e. a loud error (10 §7).

    The metric is defined as the integral of a *power spectral density* over the
    band (17 §1); handing it an amplitude spectrum is a programming error, not a
    degenerate measurement, so it raises rather than returning ``None``.
    """
    with pytest.raises(ValueError, match="expects a PSD spectrum"):
        band_rms_velocity(_amplitude([0.0, 1.0, 2.0], [0.0, 1.0, 0.0]), (0.0, 2.0))


def test_band_rms_velocity_is_none_when_the_band_holds_one_bin() -> None:
    """One bin in the band gives no interval to integrate -> undefined, not zero.

    ``v_band = sqrt(int_band S_v(f) df)`` (17 §1) is a trapezoid over the in-band
    bins; a single node spans no interval. With ``fs / N`` coarser than the band
    width -- a short record against the narrow lower ISO band -- this is the
    ordinary case on a real record, not an exotic one.
    """
    psd = _psd([0.0, 10.0, 20.0, 30.0], [0.0, 4.0, 4.0, 4.0])
    assert band_rms_velocity(psd, (9.0, 11.0)) is None


def test_band_rms_velocity_is_none_when_the_band_holds_no_bin() -> None:
    """A band entirely between bins holds nothing to integrate."""
    psd = _psd([0.0, 10.0, 20.0], [1.0, 1.0, 1.0])
    assert band_rms_velocity(psd, (12.0, 18.0)) is None


def test_band_rms_velocity_honest_zero_is_distinct_from_undefined() -> None:
    """A silent band with enough bins reads 0.0; a degenerate band reads ``None``.

    This is the whole point of S-25: "the band carries no velocity" and "the band
    could not be evaluated" are different statements about the instrument, and
    the pre-`SW-77` code rendered both as ``0.0``.
    """
    silent = _psd([0.0, 10.0, 20.0, 30.0], [0.0, 0.0, 0.0, 0.0])
    degenerate = _psd([0.0, 10.0, 20.0, 30.0], [0.0, 0.0, 0.0, 0.0])

    honest = band_rms_velocity(silent, (0.0, 30.0))
    undefined = band_rms_velocity(degenerate, (9.0, 11.0))

    assert honest == 0.0
    assert undefined is None
    assert honest is not undefined


def test_band_rms_velocity_matches_the_trapezoid_definition() -> None:
    """The defined path is unchanged: flat PSD S over width W gives sqrt(S*W).

    Reference: the definition in 17 §1 (`v_band = sqrt(int S_v df)`), with the
    trapezoid rule exact on a constant integrand.
    """
    psd = _psd([0.0, 10.0, 20.0, 30.0, 40.0], [2.0, 2.0, 2.0, 2.0, 2.0])
    value = band_rms_velocity(psd, (10.0, 30.0))
    assert value is not None
    assert value == pytest.approx(math.sqrt(2.0 * 20.0), rel=1e-12)


# --------------------------------------------------------------------------- #
# second_harmonic_ratio (17 §1 THD proxy; doc 04 §5 quadratic cross channel).
# --------------------------------------------------------------------------- #
def test_second_harmonic_ratio_is_none_on_a_single_bin_spectrum() -> None:
    """One bin cannot hold both f0 and 2 f0 -> undefined ("spectrum_too_short")."""
    assert second_harmonic_ratio(_amplitude([0.0], [1.0]), 50.0) is None


@pytest.mark.parametrize("fundamental_hz", [0.0, -50.0])
def test_second_harmonic_ratio_is_none_without_a_positive_fundamental(
    fundamental_hz: float,
) -> None:
    """No fundamental, no ratio: ``2 f0`` is not a frequency for ``f0 <= 0``.

    This is the path a real record takes when the dominant search returns
    nothing identifiable and the caller has no ``f0`` to offer.
    """
    spectrum = _amplitude([0.0, 50.0, 100.0], [0.0, 1.0, 0.01])
    assert second_harmonic_ratio(spectrum, fundamental_hz) is None


def test_second_harmonic_ratio_is_none_when_the_harmonic_is_above_the_spectrum() -> None:
    """``2 f0`` above the last bin means the harmonic was never observed.

    Reporting ``0`` here would claim "no second harmonic" on the strength of a
    band that never contained it -- the aliasing/Nyquist case of doc 20 §3.1,
    where the acceptance ``linearity_ramp`` (dominant 25001.5 Hz at fs = 100 kHz)
    already lands: ``2 f0`` sits above Nyquist.
    """
    spectrum = _amplitude([0.0, 50.0, 100.0], [0.0, 1.0, 0.1])
    assert second_harmonic_ratio(spectrum, 80.0) is None


def test_second_harmonic_ratio_is_none_without_amplitude_at_the_fundamental() -> None:
    """An empty fundamental bin gives the ratio no denominator."""
    spectrum = _amplitude([0.0, 50.0, 100.0], [0.0, 0.0, 0.5])
    assert second_harmonic_ratio(spectrum, 50.0) is None


def test_second_harmonic_ratio_honest_zero_is_distinct_from_undefined() -> None:
    """A clean tone reads exactly 0.0; a missing harmonic band reads ``None``."""
    clean = _amplitude([0.0, 50.0, 100.0], [0.0, 1.0, 0.0])
    honest = second_harmonic_ratio(clean, 50.0)
    undefined = second_harmonic_ratio(clean, 80.0)  # 2 f0 = 160 Hz > last bin

    assert honest == 0.0
    assert undefined is None


def test_second_harmonic_ratio_matches_its_definition() -> None:
    """The defined path is unchanged: ratio = |X(2 f0)| / |X(f0)| (17 §1, 04 §5)."""
    spectrum = _amplitude([0.0, 50.0, 100.0], [0.0, 4.0, 1.0])
    assert second_harmonic_ratio(spectrum, 50.0) == pytest.approx(0.25, rel=1e-12)


# --------------------------------------------------------------------------- #
# cross_axis_suppression (17 §1 C_xy; ISO 16063-31 on the bench).
# --------------------------------------------------------------------------- #
def test_cross_axis_suppression_is_none_without_off_axis_input() -> None:
    """No applied off-axis excitation, no cross-sensitivity measurement.

    ``C_xy`` is defined as a *ratio of responses* (17 §1); with a zero
    denominator nothing was measured. The pre-`SW-77` ``0.0`` announced perfect
    suppression -- the best number the metric can produce -- on no measurement.
    """
    assert cross_axis_suppression(1.0e-3, 0.0) is None


def test_cross_axis_suppression_honest_zero_is_distinct_from_undefined() -> None:
    """Zero leakage under real off-axis drive is 0.0; no drive at all is ``None``."""
    assert cross_axis_suppression(0.0, 1.0) == 0.0
    assert cross_axis_suppression(0.0, 0.0) is None


def test_cross_axis_suppression_matches_its_definition() -> None:
    """The defined path is unchanged: recovered target RMS over applied off-axis RMS."""
    assert cross_axis_suppression(2.0e-3, 1.0) == pytest.approx(2.0e-3, rel=1e-12)


# --------------------------------------------------------------------------- #
# Reason vocabulary (17 §1a).
# --------------------------------------------------------------------------- #
def test_degenerate_reasons_are_a_closed_vocabulary() -> None:
    """The reason strings are a closed, unique set mirrored by doc 17 §1a."""
    assert len(set(DEGENERATE_REASONS)) == len(DEGENERATE_REASONS)
    assert set(DEGENERATE_REASONS) == {
        "band_has_fewer_than_two_bins",
        "spectrum_too_short",
        "fundamental_not_positive",
        "second_harmonic_outside_spectrum",
        "fundamental_amplitude_not_positive",
        "offaxis_input_is_zero",
        "spectrum_not_ready",
    }


# --------------------------------------------------------------------------- #
# ISO assessment on an undefined velocity (17 §1; ISO 10816-3 zones).
# --------------------------------------------------------------------------- #
def test_iso_assessment_refuses_to_grade_an_undefined_velocity() -> None:
    """No velocity, no zone -- but the traceable context survives.

    ISO 10816-3 grades a *measured* broadband velocity; with none measured there
    is no zone to award. Everything independent of the measurement (standard,
    machine class, band, published boundaries) is still true and stays in the
    bag, so the metric remains expressible in a report (17 §1).
    """
    band = (10.0, 1000.0)
    bag = iso_assessment(
        None, machine_class="group2_rigid", band_hz=band, undefined_reason="spectrum_not_ready"
    )
    limits = ISO_10816_3_ZONES["group2_rigid"]

    assert bag["zone"] is None
    assert bag["v_rms_mm_s"] is None
    assert bag["v_rms_m_s"] is None
    assert bag["undefined_reason"] == "spectrum_not_ready"
    assert bag["machine_class"] == limits.machine_class
    assert bag["band_hz"] == band
    assert bag["zone_boundaries_mm_s"] == {
        "A/B": limits.a_b_mm_s,
        "B/C": limits.b_c_mm_s,
        "C/D": limits.c_d_mm_s,
    }


def test_iso_assessment_undefined_is_distinct_from_a_quiet_zone_a() -> None:
    """A degenerate zero would grade as zone A -- the best verdict of the standard.

    Pinning the contrast, because this is the failure mode S-25 exists to remove:
    a zero velocity is a perfectly valid, *good* reading, so an instrument that
    substitutes zero for "unknown" reports excellent machine health when it in
    fact measured nothing.
    """
    graded_zero = iso_assessment(0.0)
    ungraded = iso_assessment(None, undefined_reason="band_has_fewer_than_two_bins")

    assert graded_zero["zone"] == "A"
    assert ungraded["zone"] is None


def test_iso_assessment_defined_path_carries_no_undefined_key() -> None:
    """The graded bag keeps exactly its pre-`SW-77` keys (no silent contract growth)."""
    bag = iso_assessment(2.0e-3, band_hz=(10.0, 1000.0))
    assert set(bag) == {
        "standard",
        "machine_class",
        "v_rms_mm_s",
        "v_rms_m_s",
        "zone",
        "zone_boundaries_mm_s",
        "band_hz",
    }


# --------------------------------------------------------------------------- #
# The live path: an unwarmed stream must not grade the machine.
# --------------------------------------------------------------------------- #
def test_streaming_snapshot_before_warmup_is_not_graded(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """Before the first segment there is no velocity spectrum -> no ISO zone.

    This is where a user meets the degeneracy first (live mode, doc 20 §5), and
    it is the same provenance rule as ``warmed`` / ``dropped_samples`` of
    `SW-72`: undefined is reported as undefined, not as a reassuring zero. The
    pre-`SW-77` snapshot graded the substituted 0.0 and displayed zone A.
    """
    n_short = 16  # far fewer than one segment: the spectra cannot have folded yet
    detector = DetectorOutput(
        samples=np.zeros(n_short, dtype=np.float64),
        fs=FS,
        dc_level=0.0,
        units="A",
    )
    stream = StreamingDsp(detector, variant_b, DspOptions(), constants=constants, nperseg=1024)
    stream.process(detector.samples)
    snapshot = stream.snapshot()

    assert not stream.warmed
    assert snapshot.iso is not None
    assert snapshot.iso["zone"] is None
    assert snapshot.iso["undefined_reason"] == "spectrum_not_ready"


def test_streaming_snapshot_after_warmup_is_graded(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """Once a segment has been folded in, the ordinary graded bag comes back."""
    n = 8192
    t = np.arange(n, dtype=np.float64) / FS
    detector = DetectorOutput(
        samples=1.0e-6 * np.sin(2.0 * math.pi * 120.0 * t),
        fs=FS,
        dc_level=0.0,
        units="A",
    )
    stream = StreamingDsp(detector, variant_b, DspOptions(), constants=constants, nperseg=1024)
    stream.process(detector.samples)
    snapshot = stream.snapshot()

    assert snapshot.iso is not None
    assert snapshot.iso["zone"] in {"A", "B", "C", "D"}
    assert "undefined_reason" not in snapshot.iso


def test_streaming_snapshot_omits_an_undefined_second_harmonic(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """A dominant above fs/4 puts ``2 f0`` past Nyquist -> the key is left out.

    ``VibrationResult.cross_residual`` carries floats (04 / core.types), so an
    undefined ratio is expressed the way an absent dominant already is: no key.
    Consumers (``analysis/compare``, ``analysis/instrument``, the CLI) already
    read a missing key as ``None``, so the contract does not move.
    """
    n = 8192
    f0 = 0.4 * FS  # 2 f0 = 0.8 fs is beyond the one-sided spectrum
    t = np.arange(n, dtype=np.float64) / FS
    detector = DetectorOutput(
        samples=1.0e-6 * np.sin(2.0 * math.pi * f0 * t), fs=FS, dc_level=0.0, units="A"
    )
    stream = StreamingDsp(detector, variant_b, DspOptions(), constants=constants, nperseg=1024)
    stream.process(detector.samples)
    snapshot = stream.snapshot()

    assert snapshot.dominant_freqs_hz
    assert snapshot.dominant_freqs_hz[0] == pytest.approx(f0, rel=0.02)
    assert "second_harmonic_ratio" not in snapshot.cross_residual


def test_batch_run_omits_the_ratio_when_the_harmonic_is_above_nyquist(
    examples_dir: Path,
) -> None:
    """The ``shock`` acceptance: dominant ~25 kHz at fs = 50 kHz has no observable 2 f0.

    The scenario is the one acceptance of 18 §5 whose second harmonic falls
    outside the recorded band. Its dominant is unchanged (that is the pinned
    quantity); what changes is that the run no longer reports a ``2f/1f`` of
    ``0``, which would have read as "this shock excites no second harmonic"
    while the harmonic was simply never in the spectrum (doc 20 §3.1, aliasing
    and Nyquist headroom).
    """
    base = load_scenario(examples_dir / "shock.yaml")
    stages = base.stages.model_copy(update={"dsp": "standard"})
    scenario = base.model_copy(update={"stages": stages})
    result = Pipeline(scenario, load_variant(scenario.variant)).run().result

    assert result.dominant_freqs_hz[0] == pytest.approx(24995.0, abs=1.5)
    assert 2.0 * result.dominant_freqs_hz[0] > scenario.excitation.fs_hz / 2.0
    assert "second_harmonic_ratio" not in result.cross_residual


# --------------------------------------------------------------------------- #
# dominant_frequencies and the spectral helpers (doc 20 §3: peaks are prominence).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_bins", [1, 2])
def test_dominant_frequencies_reports_nothing_on_a_too_short_spectrum(n_bins: int) -> None:
    """Fewer than three bins carry no prominence, hence no dominant.

    Every peak category of the interpretation protocol (20 §3.1) is a statement
    about a peak standing out from its neighbourhood; with one or two bins there
    is no neighbourhood, so the honest answer is "none found" -- the empty tuple
    consumers already test for -- rather than the bare ``argmax``, which would
    hand the bench protocol a line to classify that no measurement supports.
    """
    freq = [10.0 * i for i in range(n_bins)]
    values = [1.0 + i for i in range(n_bins)]
    assert dominant_frequencies(_amplitude(freq, values)) == ()


def test_dominant_frequencies_reports_nothing_on_a_silent_spectrum() -> None:
    """An all-zero body (DC excluded) has no peak to report."""
    assert dominant_frequencies(_amplitude([0.0, 10.0, 20.0, 30.0], [5.0, 0.0, 0.0, 0.0])) == ()


def test_dominant_frequencies_falls_back_to_the_bin_at_the_spectrum_edge() -> None:
    """A peak in the last bin has no right-hand neighbour: report the bin centre.

    Quadratic interpolation fits three points around the peak; at the array edge
    the third point does not exist, so the sub-bin refinement is undefined and
    the bare bin frequency is the correct (and documented) answer.
    """
    freq = [0.0, 10.0, 20.0, 30.0, 40.0]
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert dominant_frequencies(_amplitude(freq, values)) == (40.0,)


def test_dominant_frequencies_falls_back_to_the_bin_on_a_flat_peak() -> None:
    """A flat-topped peak makes the parabola degenerate (denominator 0).

    Three equal magnitudes are collinear, so the fitted parabola has no vertex;
    the bin centre is the only defined answer. The plateau is placed away from
    the global maximum so that it is reported as a secondary line.
    """
    freq = [float(i) for i in range(9)]
    values = [0.0, 10.0, 0.0, 0.0, 3.0, 3.0, 3.0, 0.0, 0.0]
    assert dominant_frequencies(_amplitude(freq, values)) == (1.0, 5.0)


def test_amplitude_spectrum_rejects_an_empty_signal() -> None:
    """An empty record fails loudly in the transform, never yielding an empty spectrum.

    Pins *why* the module carries no empty-array branch (S-25, `SW-77`): the
    rFFT rejects ``n = 0`` outright, so an empty amplitude array is unreachable
    and the former guard for it was dead code, not a handled case.
    """
    with pytest.raises(ValueError, match="FFT data points"), np.errstate(invalid="ignore"):
        amplitude_spectrum(np.empty(0, dtype=np.float64), FS)


def test_amplitude_spectrum_odd_length_has_no_nyquist_bin_to_halve() -> None:
    """For odd ``N`` the rFFT has no Nyquist bin, so no bin is halved.

    Reference: the one-sided scaling convention of the module (``2/N`` with DC
    and Nyquist not doubled) -- a tone of amplitude ``A`` must read ``A`` at its
    bin for odd ``N`` exactly as it does for even ``N``.
    """
    n = 1001
    fs = 1000.0
    f0 = fs * 100.0 / n  # exactly on bin 100 -> no leakage, the tone reads its amplitude
    t = np.arange(n, dtype=np.float64) / fs
    spectrum = amplitude_spectrum(3.0 * np.sin(2.0 * math.pi * f0 * t), fs)

    assert spectrum.freq.size == (n + 1) // 2  # no Nyquist bin for odd N
    peak = int(np.argmax(spectrum.values))
    assert spectrum.freq[peak] == pytest.approx(f0, abs=fs / n)
    assert float(spectrum.values[peak]) == pytest.approx(3.0, rel=0.01)
