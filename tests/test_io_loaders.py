"""S1 tests of the io loader registry (SW-08): CSV/WAV round-trips, units, fs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from optivibe.core.config.models import CsvSpec, SineSpec, WavSpec
from optivibe.core.units import G0_M_S2
from optivibe.excitation import EXCITATION_REGISTRY
from optivibe.io.loaders import LOADER_REGISTRY


def _write_csv(path: Path, t: np.ndarray, a: np.ndarray, header: bool = True) -> None:
    with path.open("w") as fh:
        if header:
            fh.write("time_s,accel\n")
        for ti, ai in zip(t, a, strict=True):
            fh.write(f"{ti:.9f},{ai:.12e}\n")


def test_csv_round_trip_with_time_column(tmp_path: Path) -> None:
    fs = 2000.0
    t = np.arange(1000) / fs
    a = np.sin(2 * np.pi * 50.0 * t)  # m/s^2
    path = tmp_path / "rec.csv"
    _write_csv(path, t, a)
    spec = CsvSpec(path=str(path), time_column="time_s", column="accel", units="m/s^2")
    exc = LOADER_REGISTRY.create("csv").load(spec, seed=3)
    assert exc.fs == pytest.approx(fs, rel=1e-9)
    assert np.allclose(exc.a_x, a, atol=1e-10)
    assert np.all(exc.a_y == 0.0)
    assert exc.seed == 3


def test_csv_units_g_converted_to_si(tmp_path: Path) -> None:
    t = np.arange(100) / 1000.0
    a_g = np.full(100, 2.0)
    path = tmp_path / "rec_g.csv"
    _write_csv(path, t, a_g)
    spec = CsvSpec(path=str(path), time_column=0, column=1, units="g")
    exc = LOADER_REGISTRY.create("csv").load(spec)
    assert np.allclose(exc.a_x, 2.0 * G0_M_S2)


def test_csv_explicit_fs_without_time_column(tmp_path: Path) -> None:
    a = np.linspace(-1.0, 1.0, 50)
    path = tmp_path / "no_time.csv"
    with path.open("w") as fh:
        for ai in a:
            fh.write(f"{ai:.9f}\n")
    spec = CsvSpec(path=str(path), column=0, fs_hz=512.0)
    exc = LOADER_REGISTRY.create("csv").load(spec)
    assert exc.fs == 512.0
    assert np.allclose(exc.a_x, a, atol=1e-8)


def test_csv_spec_requires_rate_source() -> None:
    with pytest.raises(ValueError, match="time_column or fs_hz"):
        CsvSpec(path="x.csv")


def test_csv_unknown_header_raises(tmp_path: Path) -> None:
    path = tmp_path / "rec.csv"
    _write_csv(path, np.arange(3) / 10.0, np.zeros(3))
    spec = CsvSpec(path=str(path), time_column="time_s", column="nope")
    with pytest.raises(ValueError, match="not found in CSV header"):
        LOADER_REGISTRY.create("csv").load(spec)


def test_wav_round_trip_int16_full_scale(tmp_path: Path) -> None:
    fs = 8000
    t = np.arange(800) / fs
    x = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    path = tmp_path / "rec.wav"
    wavfile.write(path, fs, (x * 32767).astype(np.int16))
    full_scale_g = 5.0
    spec = WavSpec(path=str(path), full_scale_g=full_scale_g)
    exc = LOADER_REGISTRY.create("wav").load(spec)
    assert exc.fs == float(fs)
    expected = x * (32767.0 / 32768.0) * full_scale_g * G0_M_S2
    assert np.allclose(exc.a_x, expected, atol=full_scale_g * G0_M_S2 / 32768.0)


def test_wav_float32_and_channel_select(tmp_path: Path) -> None:
    fs = 4000
    t = np.arange(400) / fs
    left = np.sin(2 * np.pi * 100.0 * t).astype(np.float32)
    right = np.cos(2 * np.pi * 100.0 * t).astype(np.float32)
    path = tmp_path / "stereo.wav"
    wavfile.write(path, fs, np.stack([left, right], axis=1))
    spec = WavSpec(path=str(path), channel=1, full_scale_g=1.0, axis="z")
    exc = LOADER_REGISTRY.create("wav").load(spec)
    assert np.allclose(exc.a_z, right.astype(np.float64) * G0_M_S2, atol=1e-6)
    assert np.all(exc.a_x == 0.0)
    with pytest.raises(ValueError, match="out of range"):
        LOADER_REGISTRY.create("wav").load(WavSpec(path=str(path), channel=2, full_scale_g=1.0))


def test_resampling_preserves_tone(tmp_path: Path) -> None:
    fs_in, fs_out = 8000.0, 4000.0
    t = np.arange(int(fs_in)) / fs_in
    a = np.sin(2 * np.pi * 200.0 * t)
    path = tmp_path / "rs.csv"
    _write_csv(path, t, a)
    spec = CsvSpec(path=str(path), time_column="time_s", column="accel", resample_hz=fs_out)
    exc = LOADER_REGISTRY.create("csv").load(spec)
    assert exc.fs == fs_out
    assert exc.n_samples == pytest.approx(int(fs_in) * fs_out / fs_in, abs=2)
    freq = np.fft.rfftfreq(exc.n_samples, d=1.0 / fs_out)
    amp = np.abs(np.fft.rfft(exc.a_x)) * 2.0 / exc.n_samples
    assert freq[np.argmax(amp)] == pytest.approx(200.0, abs=1.0)


def test_file_excitation_source_dispatches_by_kind(tmp_path: Path) -> None:
    t = np.arange(100) / 1000.0
    path = tmp_path / "rec.csv"
    _write_csv(path, t, np.ones(100))
    spec = CsvSpec(path=str(path), time_column=0, column=1)
    exc = EXCITATION_REGISTRY.create("csv").generate(spec, seed=11)
    assert exc.seed == 11
    sine = SineSpec(fs_hz=100.0, duration_s=0.1, frequency_hz=10.0, amplitude_g=1.0)
    with pytest.raises(TypeError, match="CsvSpec or WavSpec"):
        EXCITATION_REGISTRY.create("wav").generate(sine)
