"""Tests for the S-03 real-time streaming layer (doc 06 §5/§7, SW-67).

Increment 1 covers the causal :class:`~optivibe.dsp.streaming.LeakyIntegrator`:
seam-invariance (feeding a signal in arbitrary frames returns the same samples
as one call -- the bit-exact criterion doc 06 §7.6-ii at the integrator level),
in-band amplitude/phase against the analytic ``1/(j omega)`` integral (doc 11 §7
inverse <= 2 %), and drift suppression (the leak keeps a DC offset from ramping
away). The batch path is untouched; these are additive.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from optivibe.core.config.loader import load_constants, load_variant
from optivibe.core.config.models import Constants, DspOptions, VariantConfig
from optivibe.core.types import DetectorOutput
from optivibe.dsp import StandardDsp, target_sensitivity
from optivibe.dsp.metrics import band_rms_velocity
from optivibe.dsp.spectra import dominant_frequencies
from optivibe.dsp.streaming import (
    LeakyIntegrator,
    StreamingDsp,
    StreamingSpectrum,
    replay_record,
)

FS = 10_000.0
F_C = 1.0


def _tone(freq: float, amp: float, fs: float, n: int, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / fs
    return amp * np.cos(2.0 * math.pi * freq * t + phase)


# --------------------------------------------------------------------------- #
# Construction guards.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("fs", "f_c"),
    [(0.0, 1.0), (-1.0, 1.0), (FS, 0.0), (FS, -1.0), (FS, FS / 2.0), (FS, FS)],
)
def test_leaky_rejects_bad_params(fs: float, f_c: float) -> None:
    """Non-positive fs/f_c and f_c at/above Nyquist are rejected."""
    with pytest.raises(ValueError):
        LeakyIntegrator(fs, f_c)


def test_leaky_pole_and_dc_gain_are_finite() -> None:
    """The leak places the pole inside the unit circle -> bounded DC gain."""
    integ = LeakyIntegrator(FS, F_C)
    assert 0.0 < integ.alpha < 1.0
    assert math.isfinite(integ.dc_gain)
    # dt / (1 - alpha), with alpha = exp(-2 pi f_c / fs).
    expected = (1.0 / FS) / (1.0 - math.exp(-2.0 * math.pi * F_C / FS))
    assert integ.dc_gain == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------- #
# Seam-invariance (doc 06 §7.6-ii, at the integrator level -- bit-exact).
# --------------------------------------------------------------------------- #
def test_leaky_seam_invariance_two_chunks() -> None:
    """One call == split into two carried calls, bit-for-bit."""
    rng = np.random.default_rng(20260725)
    x = rng.standard_normal(4096)

    whole = LeakyIntegrator(FS, F_C).process(x)

    framed = LeakyIntegrator(FS, F_C)
    part = np.concatenate([framed.process(x[:1000]), framed.process(x[1000:])])

    np.testing.assert_array_equal(whole, part)


@settings(max_examples=25, deadline=None)
@given(frame=st.integers(min_value=1, max_value=997))
def test_leaky_frame_size_invariant(frame: int) -> None:
    """Streaming in frames of any size reproduces the single-call output exactly."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal(3000)

    whole = LeakyIntegrator(FS, F_C).process(x)

    framed = LeakyIntegrator(FS, F_C)
    chunks = [framed.process(x[i : i + frame]) for i in range(0, x.size, frame)]
    streamed = np.concatenate(chunks)

    np.testing.assert_array_equal(whole, streamed)


def test_leaky_reset_returns_to_start() -> None:
    """``reset`` clears carried state so a re-fed signal matches a fresh instance."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(512)
    integ = LeakyIntegrator(FS, F_C)
    first = integ.process(x)
    integ.reset()
    second = integ.process(x)
    np.testing.assert_array_equal(first, second)


# --------------------------------------------------------------------------- #
# In-band correctness against the analytic integral (doc 11 §7: <= 2 %).
# --------------------------------------------------------------------------- #
def test_leaky_integrates_tone_amplitude_in_band() -> None:
    """A tone well above f_c integrates to v = a / omega (amplitude), <= 2 %."""
    f0, amp = 200.0, 1.0  # f0 >> f_c=1 Hz -> high-pass is a no-op in band
    n = int(4.0 * FS)
    a = _tone(f0, amp, FS, n)

    v = LeakyIntegrator(FS, F_C).process(a)

    # Discard warm-up (several 1/(2 pi f_c) ~ 0.16 s); measure steady state.
    steady = v[int(1.5 * FS) :]
    v_rms = float(np.sqrt(np.mean(steady**2)))
    expected_rms = amp / (2.0 * math.pi * f0) / math.sqrt(2.0)
    assert v_rms == pytest.approx(expected_rms, rel=0.02)


def test_leaky_integrates_tone_phase_lag() -> None:
    """Integrating a cosine yields ~ +sin (a -90 deg / -pi/2 phase lag), in band."""
    f0, amp = 200.0, 1.0
    n = int(4.0 * FS)
    a = _tone(f0, amp, FS, n)  # cos
    v = LeakyIntegrator(FS, F_C).process(a)

    steady = slice(int(1.5 * FS), int(3.5 * FS))
    t = np.arange(n, dtype=np.float64) / FS
    # Ideal: v(t) = A/omega * sin(omega t); correlate against sin to check phase.
    ref_sin = np.sin(2.0 * math.pi * f0 * t)
    corr = float(np.dot(v[steady], ref_sin[steady]))
    # Positive projection onto +sin confirms the -pi/2 lag of a single integrator.
    assert corr > 0.0
    # And a near-zero projection onto the input cos (quadrature).
    ref_cos = np.cos(2.0 * math.pi * f0 * t)
    quad = abs(float(np.dot(v[steady], ref_cos[steady]))) / abs(corr)
    assert quad < 0.02


# --------------------------------------------------------------------------- #
# Drift suppression: the leak keeps a DC offset from ramping (doc 06 §3.2).
# --------------------------------------------------------------------------- #
def test_leaky_suppresses_dc_drift() -> None:
    """A DC offset produces a *bounded* pedestal, not an unbounded ramp.

    An ideal integrator would ramp a DC offset as ``dc * t`` (>= 2.0 at t = 4 s
    and growing). The leak instead bounds the DC response to the finite
    ``dc_gain = dt / (1 - alpha)`` (doc 06 §3.6) -- a constant, settled pedestal
    that does not grow. It sits below the assessment band, so it does not affect
    ``band_rms_velocity``; for real acceleration (DC ~ 0) it is negligible.
    """
    n = int(6.0 * FS)
    dc = 0.5
    a = _tone(200.0, 1.0, FS, n) + dc
    integ = LeakyIntegrator(FS, F_C)
    v = integ.process(a)

    steady = v[int(2.0 * FS) :]  # well past warm-up (1/(2 pi f_c) ~ 0.16 s)
    expected_pedestal = dc * integ.dc_gain
    assert float(np.mean(steady)) == pytest.approx(expected_pedestal, rel=0.05)

    # No secular growth: the pedestal is settled -- first vs last second agree.
    first_s = v[int(2.0 * FS) : int(3.0 * FS)]
    last_s = v[int(5.0 * FS) : int(6.0 * FS)]
    assert abs(float(np.mean(last_s) - np.mean(first_s))) < 0.01 * expected_pedestal + 1e-9


# --------------------------------------------------------------------------- #
# StreamingSpectrum (doc 06 §5.2): ring buffer + exponential-averaged PSD.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"fs": 0.0, "nperseg": 256}, "fs"),
        ({"fs": FS, "nperseg": 1}, "nperseg"),
        ({"fs": FS, "nperseg": 256, "noverlap": 256}, "noverlap"),
        ({"fs": FS, "nperseg": 256, "beta": 0.0}, "beta"),
        ({"fs": FS, "nperseg": 256, "beta": 1.0}, "beta"),
        ({"fs": FS, "nperseg": 256, "avg_segments": 0}, "avg_segments"),
    ],
)
def test_streaming_spectrum_rejects_bad_params(kwargs: dict, match: str) -> None:
    """Invalid fs/nperseg/noverlap/beta/avg_segments are rejected."""
    with pytest.raises(ValueError, match=match):
        StreamingSpectrum(**kwargs)


def test_streaming_spectrum_not_ready_before_first_segment() -> None:
    """No estimate until a full nperseg window has been seen."""
    spec = StreamingSpectrum(FS, nperseg=512)
    spec.process(np.ones(100))
    assert not spec.ready
    assert spec.spectrum() is None
    assert spec.n_segments == 0
    spec.process(np.ones(500))  # now >= 512 total -> one segment
    assert spec.ready
    assert spec.n_segments >= 1
    assert spec.spectrum() is not None


def test_streaming_spectrum_seam_invariance() -> None:
    """Segmentation depends only on the total stream, not the frame chunking."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(3000)

    a = StreamingSpectrum(FS, nperseg=512, noverlap=256)
    for i in range(0, x.size, 100):
        a.process(x[i : i + 100])

    b = StreamingSpectrum(FS, nperseg=512, noverlap=256)
    for i in range(0, x.size, 1000):
        b.process(x[i : i + 1000])

    assert a.n_segments == b.n_segments
    sa, sb = a.spectrum(), b.spectrum()
    assert sa is not None and sb is not None
    np.testing.assert_array_equal(sa.values, sb.values)
    np.testing.assert_array_equal(sa.freq, sb.freq)


def test_streaming_spectrum_recovers_tone_and_power() -> None:
    """A stationary tone lands on the right line with the right band power."""
    f0, amp = 200.0, 1.0
    n = int(6.0 * FS)
    x = _tone(f0, amp, FS, n)

    spec = StreamingSpectrum(FS, nperseg=2048, noverlap=1024)
    spec.process(x)
    psd = spec.spectrum()
    assert psd is not None

    # Dominant line within one bin (Δf = fs / L).
    df = FS / 2048
    dom = dominant_frequencies(psd)
    assert dom and abs(dom[0] - f0) <= df

    # Band power around the tone == A^2/2 -> band RMS = A/sqrt(2) (Parseval).
    band = (f0 - 20.0, f0 + 20.0)
    v_rms_band = band_rms_velocity(psd, band)
    assert v_rms_band == pytest.approx(amp / math.sqrt(2.0), rel=0.05)


# --------------------------------------------------------------------------- #
# StreamingDsp orchestrator (doc 06 §5.4) + replay_record driver.
# --------------------------------------------------------------------------- #
STREAM_FS = 40_000.0  # > 2 * f_max (variant B band to 10 kHz)


@pytest.fixture(scope="module")
def constants(config_dir: Path) -> Constants:
    """Physical constants bundle."""
    return load_constants(config_dir / "constants.yaml")


@pytest.fixture(scope="module")
def variant_b(config_dir: Path) -> VariantConfig:
    """Variant B (the numeric reference)."""
    return load_variant("B", config_dir)


def _tone_detector(
    variant: VariantConfig,
    constants: Constants,
    *,
    fs: float = STREAM_FS,
    f0: float = 200.0,
    amp: float = 1.0,
    seconds: float = 4.0,
    dc_level: float = 1.0e-3,
) -> DetectorOutput:
    """Synthesize a detector record whose calibrated acceleration is a known tone."""
    s_target = target_sensitivity(variant, constants)
    n = int(seconds * fs)
    a = _tone(f0, amp, fs, n)
    samples = dc_level + s_target * a  # I_AC = samples - dc_level = s_target * a
    return DetectorOutput(samples=samples, fs=fs, dc_level=dc_level, units="A")


def test_streaming_calibration_bit_exact(variant_b: VariantConfig, constants: Constants) -> None:
    """Streaming calibration (samples -> a) is bit-identical to batch (doc 06 §7.6-i)."""
    det = _tone_detector(variant_b, constants)
    opts = DspOptions()
    batch = StandardDsp(constants=constants).run(det, variant_b, opts)
    stream = replay_record(det, variant_b, opts, block_size=997, constants=constants, nperseg=4096)
    np.testing.assert_array_equal(stream.a, batch.a)


def test_streaming_seam_invariance_orchestrator(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """Replaying in different block sizes yields identical a/v/x and metrics (doc 06 §7.6-ii)."""
    det = _tone_detector(variant_b, constants)
    opts = DspOptions()
    r100 = replay_record(det, variant_b, opts, block_size=100, constants=constants, nperseg=4096)
    r1000 = replay_record(det, variant_b, opts, block_size=1000, constants=constants, nperseg=4096)

    np.testing.assert_array_equal(r100.a, r1000.a)
    np.testing.assert_array_equal(r100.v, r1000.v)
    np.testing.assert_array_equal(r100.x, r1000.x)
    assert r100.dominant_freqs_hz == r1000.dominant_freqs_hz
    # The a/v/x samples are bit-exact; the running RMS is a reduction whose float
    # summation order depends on the chunking, so it agrees only to round-off.
    for key in ("a", "v", "x"):
        assert r100.rms[key] == pytest.approx(r1000.rms[key], rel=1e-12)
    # Spectral metrics come from the (bit-exact) streaming spectra -> bit-exact.
    assert r100.iso is not None and r1000.iso is not None
    assert r100.iso["zone"] == r1000.iso["zone"]
    assert r100.iso["v_rms_mm_s"] == r1000.iso["v_rms_mm_s"]


def test_streaming_snapshot_recovers_tone(variant_b: VariantConfig, constants: Constants) -> None:
    """The snapshot lands the dominant line, matches batch RMS(a) and grades ISO."""
    det = _tone_detector(variant_b, constants)
    opts = DspOptions()
    batch = StandardDsp(constants=constants).run(det, variant_b, opts)
    stream = replay_record(det, variant_b, opts, block_size=1024, constants=constants, nperseg=4096)

    df = STREAM_FS / 4096
    assert stream.dominant_freqs_hz
    assert abs(stream.dominant_freqs_hz[0] - 200.0) <= df
    # a is bit-exact, so RMS(a) matches batch to summation round-off.
    assert stream.rms["a"] == pytest.approx(batch.rms["a"], rel=1e-9)
    assert stream.iso is not None
    assert stream.iso["zone"] in {"A", "B", "C", "D"}


def test_streaming_dropped_and_warmed(variant_b: VariantConfig, constants: Constants) -> None:
    """Provenance: warm-up flag settles and the dropped-sample counter records gaps."""
    det = _tone_detector(variant_b, constants, seconds=2.0)
    stream = StreamingDsp(det, variant_b, DspOptions(), constants=constants, nperseg=4096)
    assert not stream.warmed
    assert stream.dropped_samples == 0

    stream.note_dropped(128)
    assert stream.dropped_samples == 128
    with pytest.raises(ValueError, match="non-negative"):
        stream.note_dropped(-1)

    stream.process(det.samples)
    assert stream.warmed
    assert stream.n_samples == det.samples.size


def test_streaming_uses_f_c_stream_override(variant_b: VariantConfig, constants: Constants) -> None:
    """``DspOptions.f_c_stream`` sets the causal cut-off independently of f_hp."""
    det = _tone_detector(variant_b, constants, seconds=1.0)
    opts_default = DspOptions()
    opts_override = DspOptions(f_c_stream=5.0)
    s_default = StreamingDsp(det, variant_b, opts_default, constants=constants)
    s_override = StreamingDsp(det, variant_b, opts_override, constants=constants)
    # Default falls back to f_hp = band lower edge (1 Hz for B); override is honoured.
    assert s_default.f_c == variant_b.band.f_min_hz
    assert s_override.f_c == 5.0


# --------------------------------------------------------------------------- #
# MAIN ACCEPTANCE GOLDEN: batch <-> stream equivalence (doc 06 §7.6).
#
# (i)   calibration bit-exact               -> test_streaming_calibration_bit_exact
# (ii)  seam-invariance bit-exact           -> test_streaming_seam_invariance_*
# (iii) in-band metrics within doc 11 §7, after warm-up (this test), tied to the
#       analytic v = A/omega, x = A/omega^2 formulas (base formulas, 18 §5(g)).
# The integration stage is NOT bit-exact vs batch (batch is non-causal); that is
# excluded by construction (doc 06 §3.6/§7.6).
# --------------------------------------------------------------------------- #
def test_batch_stream_equivalence_golden(variant_b: VariantConfig, constants: Constants) -> None:
    """Streaming metrics equal batch and the analytic tone, in band, after warm-up."""
    f0, amp, seconds = 200.0, 1.0, 4.0
    det = _tone_detector(variant_b, constants, f0=f0, amp=amp, seconds=seconds)
    opts = DspOptions(welch_nperseg=4096, welch_noverlap=2048)  # same L both sides

    batch = StandardDsp(constants=constants).run(det, variant_b, opts)
    stream = replay_record(det, variant_b, opts, block_size=1024, constants=constants, nperseg=4096)

    # (i) Calibration bit-exact -> RMS(a) matches batch to summation round-off.
    assert stream.rms["a"] == pytest.approx(batch.rms["a"], rel=1e-9)

    # (iii) ISO-graded band RMS velocity: stream == batch == analytic, <= 2 %.
    v_rms_analytic_mm_s = amp / (2.0 * math.pi * f0) / math.sqrt(2.0) * 1.0e3
    assert stream.iso is not None and batch.iso is not None
    assert float(stream.iso["v_rms_mm_s"]) == pytest.approx(v_rms_analytic_mm_s, rel=0.02)
    assert float(stream.iso["v_rms_mm_s"]) == pytest.approx(
        float(batch.iso["v_rms_mm_s"]), rel=0.02
    )
    assert stream.iso["zone"] == batch.iso["zone"]

    # Dominant line within one bin of the analytic tone.
    df = STREAM_FS / 4096
    assert stream.dominant_freqs_hz
    assert abs(stream.dominant_freqs_hz[0] - f0) <= df

    # After warm-up, the causal v and x match the analytic integrals (and batch)
    # within doc 11 §7 (<= 2 %); the whole-signal RMS(x) differs only by the
    # double-integration start-up transient (doc 06 §7.6: results before warm-up
    # are not graded).
    warm = int(2.0 * STREAM_FS)
    v_rms_stream = float(np.sqrt(np.mean(stream.v[warm:] ** 2)))
    x_rms_stream = float(np.sqrt(np.mean(stream.x[warm:] ** 2)))
    v_rms_analytic = amp / (2.0 * math.pi * f0) / math.sqrt(2.0)
    x_rms_analytic = amp / (2.0 * math.pi * f0) ** 2 / math.sqrt(2.0)
    assert v_rms_stream == pytest.approx(v_rms_analytic, rel=0.02)
    assert x_rms_stream == pytest.approx(x_rms_analytic, rel=0.02)
    # And equal to batch on the same settled window.
    assert x_rms_stream == pytest.approx(float(np.sqrt(np.mean(batch.x[warm:] ** 2))), rel=0.02)


def test_streaming_keep_history_false_bounded_trace(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """``keep_history=False`` keeps only the last nperseg samples as a live trace."""
    det = _tone_detector(variant_b, constants, seconds=2.0)
    nperseg = 2048
    stream = StreamingDsp(
        det, variant_b, DspOptions(), constants=constants, nperseg=nperseg, keep_history=False
    )
    stream.process(det.samples)
    snap = stream.snapshot()
    # Bounded memory: the trace is capped at one segment, yet metrics/ISO still fill.
    assert snap.a.size == nperseg
    assert snap.v.size == nperseg
    assert snap.iso is not None
    assert stream.n_samples == det.samples.size  # running stats still see everything


# --------------------------------------------------------------------------- #
# Guards / edge cases (empty and non-1-D blocks, bad block_size).
# --------------------------------------------------------------------------- #
def test_streaming_empty_and_2d_guards(variant_b: VariantConfig, constants: Constants) -> None:
    """Empty blocks are no-ops; non-1-D blocks raise; bad block_size raises."""
    integ = LeakyIntegrator(FS, F_C)
    assert integ.process(np.empty(0)).size == 0
    with pytest.raises(ValueError, match="1-D"):
        integ.process(np.zeros((2, 2)))

    spec = StreamingSpectrum(FS, nperseg=256)
    spec.process(np.empty(0))  # no-op
    assert not spec.ready
    with pytest.raises(ValueError, match="1-D"):
        spec.process(np.zeros((2, 2)))
    spec.reset()
    assert spec.n_segments == 0

    det = _tone_detector(variant_b, constants, seconds=0.5)
    stream = StreamingDsp(det, variant_b, DspOptions(), constants=constants, nperseg=256)
    stream.process(np.empty(0))  # no-op
    assert stream.n_samples == 0
    with pytest.raises(ValueError, match="1-D"):
        stream.process(np.zeros((2, 2)))

    with pytest.raises(ValueError, match="block_size"):
        replay_record(det, variant_b, DspOptions(), block_size=0, constants=constants)


def test_leaky_reset_and_dc_gain_property() -> None:
    """LeakyIntegrator reset clears state; alpha/dc_gain properties are exposed."""
    integ = LeakyIntegrator(FS, F_C)
    integ.process(np.ones(100))
    integ.reset()
    # After reset the first output equals a fresh instance's on the same input.
    fresh = LeakyIntegrator(FS, F_C)
    np.testing.assert_array_equal(integ.process(np.ones(50)), fresh.process(np.ones(50)))
    assert 0.0 < integ.alpha < 1.0
