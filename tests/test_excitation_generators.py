"""S1 tests: signal properties and seed reproducibility of the generators.

Checks follow the acceptance items of 12 §S1 / 11 §7: tonal peaks at the
requested frequencies and amplitudes, sweep instantaneous frequency running
f_start -> f_end, random noise hitting the target PSD/RMS in band (Welch),
shock peak/width, Parseval's identity, and one-seed-one-result (10 §8).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.signal import welch

from optivibe.core.config.models import (
    MultitoneSpec,
    RandomSpec,
    ShockSpec,
    SineSpec,
    SweepSpec,
)
from optivibe.core.units import G0_M_S2
from optivibe.excitation import EXCITATION_REGISTRY

# ---------------------------------------------------------------------------
# Registry / dispatch
# ---------------------------------------------------------------------------


def test_registry_has_all_kinds() -> None:
    for kind in ("sine", "multitone", "sweep", "random", "shock", "csv", "wav"):
        assert EXCITATION_REGISTRY.get(kind) is not None


def test_kind_mismatch_raises() -> None:
    spec = SineSpec(fs_hz=1000.0, duration_s=0.1, frequency_hz=10.0, amplitude_g=1.0)
    with pytest.raises(TypeError, match="expects MultitoneSpec"):
        EXCITATION_REGISTRY.create("multitone").generate(spec, seed=1)


# ---------------------------------------------------------------------------
# Tonal: peaks at requested frequencies and amplitudes
# ---------------------------------------------------------------------------


def _amplitude_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    n = x.size
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    amp = np.abs(np.fft.rfft(x)) * 2.0 / n
    return freq, amp


def test_multitone_peaks_at_requested_tones() -> None:
    fs, dur = 8000.0, 1.0  # df = 1 Hz -> integer tones land exactly on bins
    tones = [(120.0, 1.0, 0.0), (440.0, 0.5, 1.0), (1000.0, 0.25, -0.5)]
    spec = MultitoneSpec(fs_hz=fs, duration_s=dur, tones=tones)
    exc = EXCITATION_REGISTRY.create("multitone").generate(spec, seed=0)
    freq, amp = _amplitude_spectrum(exc.a_x, fs)
    for f_i, a_i, _ in tones:
        k = round(f_i * dur)
        assert freq[k] == pytest.approx(f_i)
        assert amp[k] == pytest.approx(a_i * G0_M_S2, rel=1e-9)
    # Off-tone bins are empty (signal is exactly the three tones).
    mask = np.ones_like(amp, dtype=bool)
    for f_i, _, _ in tones:
        mask[round(f_i * dur)] = False
    assert np.max(amp[mask]) < 1e-9 * G0_M_S2


def test_multitone_compact_tone_form_and_zero_other_axes() -> None:
    spec = MultitoneSpec(
        fs_hz=2000.0, duration_s=0.5, axis="y", tones=[(50.0, 1.0), (75.0, 0.5, 0.25)]
    )
    assert spec.tones[0].phase_rad == 0.0
    assert spec.tones[1].phase_rad == 0.25
    exc = EXCITATION_REGISTRY.create("multitone").generate(spec, seed=0)
    assert np.all(exc.a_x == 0.0)
    assert np.all(exc.a_z == 0.0)
    assert np.max(np.abs(exc.a_y)) > 0.0


# ---------------------------------------------------------------------------
# Sweep: instantaneous frequency runs f_start -> f_end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["linear", "log"])
def test_sweep_instantaneous_frequency_endpoints(method: str) -> None:
    fs, dur, f0, f1 = 20000.0, 2.0, 50.0, 2000.0
    spec = SweepSpec(
        fs_hz=fs, duration_s=dur, f_start_hz=f0, f_end_hz=f1, amplitude_g=1.0, method=method
    )
    exc = EXCITATION_REGISTRY.create("sweep").generate(spec, seed=0)
    x = exc.a_x
    n_seg = int(0.1 * fs)  # 0.1 s segments at both ends

    def dominant(segment: np.ndarray) -> float:
        freq, amp = _amplitude_spectrum(segment * np.hanning(segment.size), fs)
        return float(freq[np.argmax(amp)])

    f_head = dominant(x[:n_seg])
    f_tail = dominant(x[-n_seg:])
    # The segment averages the running frequency, so compare loosely but
    # directionally: head near f0, tail near f1, strictly increasing.
    assert f_head < 1.5 * (f0 + (f1 - f0) * 0.05) if method == "linear" else f_head < 4 * f0
    assert f_tail > 0.8 * f1 * (0.9 if method == "linear" else 0.7)
    assert f_tail > 10.0 * f_head

    # Constant amplitude: envelope (analytic-free check) via max over windows.
    win = int(fs / f0)
    peaks = [np.max(np.abs(x[i : i + win])) for i in range(0, x.size - win, win)]
    assert np.min(peaks) > 0.95 * G0_M_S2
    assert np.max(peaks) < 1.05 * G0_M_S2


# ---------------------------------------------------------------------------
# Random: PSD level in band and band RMS (Welch), out-of-band rejection
# ---------------------------------------------------------------------------


def test_random_rms_mode_hits_target_exactly() -> None:
    spec = RandomSpec(fs_hz=8000.0, duration_s=4.0, band_hz=(20.0, 2000.0), g_rms=1.0)
    exc = EXCITATION_REGISTRY.create("random").generate(spec, seed=42)
    rms = float(np.sqrt(np.mean(exc.a_x**2)))
    assert rms == pytest.approx(1.0 * G0_M_S2, rel=1e-12)
    assert float(np.mean(exc.a_x)) == pytest.approx(0.0, abs=1e-9)


def test_random_psd_mode_welch_in_tolerance() -> None:
    fs, dur = 8000.0, 16.0
    band = (100.0, 2000.0)
    target = 1.0e-3  # g^2/Hz
    spec = RandomSpec(fs_hz=fs, duration_s=dur, band_hz=band, psd_g2_hz=target)
    exc = EXCITATION_REGISTRY.create("random").generate(spec, seed=7)
    freq, psd = welch(exc.a_x, fs=fs, nperseg=4096)
    in_band = (freq > band[0] * 1.1) & (freq < band[1] * 0.9)
    out_band = freq > band[1] * 1.2
    psd_g = psd / G0_M_S2**2
    assert float(np.mean(psd_g[in_band])) == pytest.approx(target, rel=0.15)
    assert float(np.mean(psd_g[out_band])) < 1e-6 * target
    # Flat-shape link: band RMS ~ sqrt(S0 * BW).
    expected_rms_g = np.sqrt(target * (band[1] - band[0]))
    rms_g = float(np.sqrt(np.mean(exc.a_x**2))) / G0_M_S2
    assert rms_g == pytest.approx(expected_rms_g, rel=0.1)


def test_random_band_without_bins_raises() -> None:
    spec = RandomSpec(fs_hz=1000.0, duration_s=0.01, band_hz=(1.0, 2.0), g_rms=1.0)
    with pytest.raises(ValueError, match="no FFT bins"):
        EXCITATION_REGISTRY.create("random").generate(spec, seed=0)


def test_random_spec_level_validation() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        RandomSpec(fs_hz=1000.0, duration_s=1.0, band_hz=(10.0, 100.0))
    with pytest.raises(ValueError, match="exactly one"):
        RandomSpec(fs_hz=1000.0, duration_s=1.0, band_hz=(10.0, 100.0), g_rms=1.0, psd_g2_hz=1.0)
    with pytest.raises(ValueError, match="Nyquist"):
        RandomSpec(fs_hz=1000.0, duration_s=1.0, band_hz=(10.0, 600.0), g_rms=1.0)


# ---------------------------------------------------------------------------
# Shock: peak value and pulse width
# ---------------------------------------------------------------------------


def test_shock_peak_and_width() -> None:
    fs, peak_g, pulse_ms, delay = 100000.0, 50.0, 1.0, 0.01
    spec = ShockSpec(fs_hz=fs, duration_s=0.05, peak_g=peak_g, pulse_ms=pulse_ms, delay_s=delay)
    exc = EXCITATION_REGISTRY.create("shock").generate(spec, seed=0)
    x = exc.a_x
    assert float(np.max(x)) == pytest.approx(peak_g * G0_M_S2, rel=1e-4)
    assert np.all(x >= 0.0)  # half-sine is non-negative
    nonzero = np.nonzero(x > 1e-12)[0]
    width_s = (nonzero[-1] - nonzero[0]) / fs
    assert width_s == pytest.approx(pulse_ms / 1e3, abs=2.0 / fs)
    assert nonzero[0] / fs == pytest.approx(delay, abs=2.0 / fs)
    # Quiet head and tail.
    assert np.all(x[: int(delay * fs) - 1] == 0.0)


def test_shock_pulse_must_fit_duration() -> None:
    with pytest.raises(ValueError, match="exceeds duration_s"):
        ShockSpec(fs_hz=10000.0, duration_s=0.001, peak_g=1.0, pulse_ms=5.0)


# ---------------------------------------------------------------------------
# Parseval: time-domain energy equals rFFT-spectrum energy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2])
def test_parseval_for_random_signal(seed: int) -> None:
    spec = RandomSpec(fs_hz=4000.0, duration_s=2.0, band_hz=(10.0, 1000.0), g_rms=0.5)
    exc = EXCITATION_REGISTRY.create("random").generate(spec, seed=seed)
    x = exc.a_x
    n = x.size
    spectrum = np.fft.rfft(x)
    weights = np.full(spectrum.size, 2.0)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[-1] = 1.0
    energy_freq = float(np.sum(weights * np.abs(spectrum) ** 2)) / n
    energy_time = float(np.sum(x**2))
    assert energy_freq == pytest.approx(energy_time, rel=1e-10)


# ---------------------------------------------------------------------------
# Reproducibility: one seed -> identical arrays; different seeds differ (10 §8)
# ---------------------------------------------------------------------------


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_random_same_seed_identical(seed: int) -> None:
    spec = RandomSpec(fs_hz=2000.0, duration_s=0.5, band_hz=(10.0, 500.0), g_rms=1.0)
    src = EXCITATION_REGISTRY.create("random")
    a = src.generate(spec, seed=seed)
    b = src.generate(spec, seed=seed)
    assert np.array_equal(a.a_x, b.a_x)


@settings(max_examples=20, deadline=None)
@given(
    seeds=st.tuples(
        st.integers(min_value=0, max_value=2**31 - 1),
        st.integers(min_value=0, max_value=2**31 - 1),
    ).filter(lambda s: s[0] != s[1])
)
def test_random_different_seeds_differ(seeds: tuple[int, int]) -> None:
    spec = RandomSpec(fs_hz=2000.0, duration_s=0.5, band_hz=(10.0, 500.0), g_rms=1.0)
    src = EXCITATION_REGISTRY.create("random")
    a = src.generate(spec, seed=seeds[0])
    b = src.generate(spec, seed=seeds[1])
    assert not np.array_equal(a.a_x, b.a_x)


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_deterministic_generators_ignore_seed(seed: int) -> None:
    spec = SineSpec(fs_hz=1000.0, duration_s=0.25, frequency_hz=100.0, amplitude_g=1.0)
    src = EXCITATION_REGISTRY.create("sine")
    a = src.generate(spec, seed=seed)
    b = src.generate(spec, seed=None)
    assert np.array_equal(a.a_x, b.a_x)
    assert a.seed == seed
