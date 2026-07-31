"""Qt-free tests of the live-stream sources and loop (task O-SW-03).

These exercise :mod:`optivibe.gui.workers.stream` **without a display**: the
module imports no Qt, which is the property that lets the loop be verified here
and moved into the core untouched should a second consumer appear (CLI live
mode, backlog S-22).

Three claims are load-bearing and each has a test:

* the live path runs the *verified* math -- its final state matches
  :func:`~optivibe.dsp.streaming.replay_record`, the finite driver the
  batch<->stream acceptance golden pins (theory-06 §7.6);
* dropped samples are produced only where they can exist -- real-time pacing
  that falls behind (theory-06 §5.7) -- and are reported as "undefined" rather
  than zero when there is no clock to fall behind;
* a looped record never pretends to be continuous: the wrap is flagged.

The wall clock is injected, so nothing here waits in real time.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from optivibe.core.config.loader import load_constants, load_scenario, load_variant
from optivibe.core.config.models import Constants, DspOptions, VariantConfig
from optivibe.core.types import DetectorOutput
from optivibe.dsp.sensitivity import target_sensitivity
from optivibe.dsp.streaming import replay_record
from optivibe.gui.workers.stream import (
    ScenarioSource,
    StreamConfig,
    StreamFrame,
    StreamSetup,
    default_block_size,
    run_stream,
)

FS = 10_000.0
SECONDS = 1.0


class _Clock:
    """A virtual monotonic clock: time moves only when the loop asks it to.

    ``step`` is added on every reading: the loop gates its frames on the wall
    clock, so a clock that moved only on ``sleep`` would freeze the gate in the
    accelerated mode (which never sleeps). A large ``step`` is how a *lagging*
    consumer is simulated -- the loop's own bookkeeping then shows it behind
    the record.
    """

    def __init__(self, step: float = 0.005) -> None:
        self.now = 0.0
        self.step = step
        self.slept = 0.0

    def monotonic(self) -> float:
        """Return the current virtual time, advancing it by ``step``."""
        self.now += self.step
        return self.now

    def sleep(self, seconds: float) -> None:
        """Advance the virtual clock instead of blocking."""
        self.slept += seconds
        self.now += seconds


class _Collector:
    """Collects frames and stops the loop after ``limit`` of them."""

    def __init__(self, limit: int = 10_000) -> None:
        self.frames: list[StreamFrame] = []
        self.limit = limit

    def on_frame(self, frame: StreamFrame) -> None:
        """Record one frame."""
        self.frames.append(frame)

    def is_stopped(self) -> bool:
        """Stop once enough frames have been seen."""
        return len(self.frames) >= self.limit


@pytest.fixture(scope="module")
def constants(config_dir: Path) -> Constants:
    """Physical constants from the repository config."""
    return load_constants(config_dir / "constants.yaml")


@pytest.fixture(scope="module")
def variant_b(config_dir: Path) -> VariantConfig:
    """The canonical wideband variant B."""
    return load_variant("B", config_dir=config_dir)


def _tone_detector(variant: VariantConfig, constants: Constants) -> DetectorOutput:
    """A detector record whose calibrated acceleration is a known 200 Hz tone."""
    n = int(SECONDS * FS)
    t = np.arange(n, dtype=np.float64) / FS
    a = np.cos(2.0 * math.pi * 200.0 * t)
    samples = 1.0e-3 + target_sensitivity(variant, constants) * a
    return DetectorOutput(samples=samples, fs=FS, dc_level=1.0e-3, units="A")


def _setup(variant: VariantConfig, constants: Constants) -> StreamSetup:
    """A ready setup over the synthetic tone record."""
    return StreamSetup(
        detector=_tone_detector(variant, constants),
        variant=variant,
        options=DspOptions(),
        label="test tone",
    )


# --------------------------------------------------------------------------- #
# The live path is the verified path.
# --------------------------------------------------------------------------- #
def test_run_stream_final_state_matches_replay_record(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """The loop's last frame equals the accepted finite driver on the same record.

    ``replay_record`` is what the batch<->stream golden runs (theory-06 §7.6);
    the live loop calls the same ``process()``/``snapshot()`` on the same
    blocks, so the running metrics must agree exactly. Any divergence would
    mean the oscilloscope had grown a second, unverified arithmetic.
    """
    setup = _setup(variant_b, constants)
    block = 512
    reference = replay_record(
        setup.detector,
        variant_b,
        setup.options,
        block_size=block,
        constants=constants,
        nperseg=1024,
    )
    collector = _Collector()
    clock = _Clock()
    run_stream(
        setup,
        StreamConfig(rate_hz=10.0, speed="max", loop=False, nperseg=1024, block_size=block),
        on_frame=collector.on_frame,
        is_stopped=collector.is_stopped,
        constants=constants,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    final = collector.frames[-1].result
    assert final.dominant_freqs_hz == reference.dominant_freqs_hz
    assert final.rms == reference.rms
    assert final.spectrum is not None
    assert reference.spectrum is not None
    np.testing.assert_array_equal(final.spectrum.values, reference.spectrum.values)


def test_run_stream_emits_and_ends_on_a_fresh_frame(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """Frames arrive while streaming and the last one is the state it ended in."""
    setup = _setup(variant_b, constants)
    collector = _Collector()
    clock = _Clock(step=0.0)  # real-time mode: the clock moves by sleeping
    run_stream(
        setup,
        StreamConfig(rate_hz=10.0, speed="realtime", loop=False),
        on_frame=collector.on_frame,
        is_stopped=collector.is_stopped,
        constants=constants,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert collector.frames
    last = collector.frames[-1]
    # One second of record at real time -> the loop slept about a second and the
    # final frame accounts for the whole record.
    assert clock.slept == pytest.approx(SECONDS, abs=0.05)
    assert last.elapsed_s == pytest.approx(SECONDS, abs=0.01)
    assert last.n_samples == setup.detector.samples.size
    assert last.warmed is True
    assert last.source_label == "test tone"
    # The frame rate is honoured: ~10 frames for a second of record, not one per
    # block (the producer gates itself so a consumer cannot be flooded).
    assert 5 <= len(collector.frames) <= 13


def test_run_stream_window_is_bounded_and_reported(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """The frame carries a bounded trace and says how long it is."""
    setup = _setup(variant_b, constants)
    collector = _Collector()
    clock = _Clock()
    config = StreamConfig(rate_hz=10.0, speed="max", window_s=0.25, nperseg=1024)
    run_stream(
        setup,
        config,
        on_frame=collector.on_frame,
        is_stopped=collector.is_stopped,
        constants=constants,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    last = collector.frames[-1]
    assert last.result.a.size == int(FS * 0.25)
    assert last.window_s == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Provenance: drops exist only where a clock can be missed.
# --------------------------------------------------------------------------- #
def test_run_stream_drops_samples_when_it_falls_behind(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """Real-time pacing that lags keeps the newest and counts the gap.

    theory-06 §5.7: a live monitor cannot catch up by processing faster, so it
    jumps the gap -- and the gap must appear in the provenance, or the user
    reads an interrupted stream as a continuous one.
    """
    setup = _setup(variant_b, constants)
    collector = _Collector()
    # Each clock reading jumps 40 ms while a block is worth 25 ms -> the loop is
    # structurally unable to keep up.
    clock = _Clock(step=0.04)
    run_stream(
        setup,
        StreamConfig(rate_hz=10.0, speed="realtime", loop=False),
        on_frame=collector.on_frame,
        is_stopped=collector.is_stopped,
        constants=constants,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    last = collector.frames[-1]
    assert last.dropped_samples is not None
    assert last.dropped_samples > 0
    # Dropped samples are skipped, not processed: the two must not double-count.
    assert last.n_samples + last.dropped_samples <= setup.detector.samples.size
    assert last.paced is True


def test_run_stream_reports_no_drop_counter_without_a_clock(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """Accelerated replay reports ``None``, not a reassuring zero.

    With no obligation to a wall clock a drop cannot happen, so the counter is
    undefined rather than zero -- the same honesty rule that puts ``warmed`` on
    screen (owner decision, 2026-07-31).
    """
    setup = _setup(variant_b, constants)
    collector = _Collector()
    clock = _Clock(step=1.0)  # wildly lagging -- and still no drops, by construction
    run_stream(
        setup,
        StreamConfig(rate_hz=10.0, speed="max", loop=False),
        on_frame=collector.on_frame,
        is_stopped=collector.is_stopped,
        constants=constants,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert collector.frames
    assert all(frame.dropped_samples is None for frame in collector.frames)
    assert all(frame.paced is False for frame in collector.frames)
    assert collector.frames[-1].n_samples == setup.detector.samples.size


# --------------------------------------------------------------------------- #
# Looping is never silently continuous.
# --------------------------------------------------------------------------- #
def test_run_stream_flags_the_loop_seam(variant_b: VariantConfig, constants: Constants) -> None:
    """A wrap is counted and the frame straddling the splice is flagged.

    The splice is a genuine discontinuity: its wide-band splatter would read as
    physics on the very screen built to tell physics from artifacts, so looping
    announces itself instead of being smoothed over.
    """
    setup = _setup(variant_b, constants)
    collector = _Collector(limit=25)
    clock = _Clock()
    run_stream(
        setup,
        StreamConfig(rate_hz=10.0, speed="max", loop=True),
        on_frame=collector.on_frame,
        is_stopped=collector.is_stopped,
        constants=constants,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert collector.frames[-1].loops >= 1
    assert any(frame.seam for frame in collector.frames)
    # Streaming past the end is what looping is for.
    assert collector.frames[-1].n_samples > setup.detector.samples.size


def test_run_stream_stops_at_the_end_without_looping(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """A finite source ends by itself: the record is played exactly once."""
    setup = _setup(variant_b, constants)
    collector = _Collector(limit=10_000)
    clock = _Clock()
    run_stream(
        setup,
        StreamConfig(rate_hz=10.0, speed="max", loop=False),
        on_frame=collector.on_frame,
        is_stopped=collector.is_stopped,
        constants=constants,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    last = collector.frames[-1]
    assert last.loops == 0
    assert not last.seam
    assert last.n_samples == setup.detector.samples.size


def test_run_stream_stops_when_asked(variant_b: VariantConfig, constants: Constants) -> None:
    """The cooperative stop is honoured mid-record, with a final frame."""
    setup = _setup(variant_b, constants)
    collector = _Collector(limit=2)
    clock = _Clock()
    run_stream(
        setup,
        StreamConfig(rate_hz=1000.0, speed="max", loop=True),
        on_frame=collector.on_frame,
        is_stopped=collector.is_stopped,
        constants=constants,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert len(collector.frames) == 3  # two before the stop poll bites, plus the final one
    assert collector.frames[-1].n_samples < setup.detector.samples.size


def test_run_stream_stopped_before_the_first_block_emits_nothing(
    variant_b: VariantConfig, constants: Constants
) -> None:
    """An immediate stop ends quietly rather than snapshotting an empty stream.

    There is nothing to show before the first block, and a snapshot over empty
    traces violates the ``VibrationResult`` contract -- which would surface to
    the user as a failed stream instead of a stopped one.
    """
    setup = _setup(variant_b, constants)
    frames: list[StreamFrame] = []
    run_stream(
        setup,
        StreamConfig(),
        on_frame=frames.append,
        is_stopped=lambda: True,
        constants=constants,
        monotonic=_Clock().monotonic,
        sleep=_Clock().sleep,
    )
    assert frames == []


# --------------------------------------------------------------------------- #
# Guards and sources.
# --------------------------------------------------------------------------- #
def test_run_stream_rejects_a_bad_rate(variant_b: VariantConfig, constants: Constants) -> None:
    """A non-positive frame rate is refused loudly.

    (An empty record needs no guard here: the ``DetectorOutput`` contract in
    ``core/types`` already rejects one, so the loop cannot receive it.)
    """
    setup = _setup(variant_b, constants)
    collector = _Collector()
    with pytest.raises(ValueError, match="rate_hz"):
        run_stream(
            setup,
            StreamConfig(rate_hz=0.0),
            on_frame=collector.on_frame,
            is_stopped=collector.is_stopped,
        )


def test_default_block_size_is_a_fraction_of_a_frame() -> None:
    """Blocks are shorter than a frame, so pacing and stopping stay responsive."""
    assert default_block_size(100_000.0, 10.0) == 2500
    assert default_block_size(10.0, 10.0) == 16  # never degenerates to one sample


def test_scenario_source_opens_a_finite_record(hello_scenario: Path, config_dir: Path) -> None:
    """The synthetic source resolves its variant and synthesizes the record.

    It is finite by construction -- exactly the scenario's duration -- which is
    why endless play needs an explicit, flagged loop rather than an implicit
    one (backlog S-24 covers a phase-continuous generator).
    """
    scenario = load_scenario(hello_scenario)
    source = ScenarioSource(scenario=scenario, config_dir=config_dir)
    setup = source.open()
    expected = int(scenario.excitation.fs_hz * scenario.excitation.duration_s)
    assert setup.detector.samples.size == expected
    assert setup.detector.fs == scenario.excitation.fs_hz
    assert setup.variant.name == load_variant(scenario.variant, config_dir=config_dir).name
    assert setup.scenario is scenario  # carried for the expected-peak overlay
    assert "hello" in source.label
