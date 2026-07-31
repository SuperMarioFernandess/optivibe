"""Qt-free sources and loop of the real-time oscilloscope (task O-SW-03).

The job family next door (:mod:`optivibe.gui.workers.jobs`) is *one-shot*: run,
return one result, finish. The live oscilloscope has the opposite shape -- an
endless loop that emits a snapshot ``N`` times a second until the user stops it
-- so it gets its own worker family instead of overloading the job contract
(decision 13 §4.5; the architecture fork was settled in favour of a separate
streaming worker, 09 §9 untouched).

This module is the **Qt-free half**: it knows about records, blocks, wall-clock
pacing and provenance, and nothing about Qt, so it is unit-testable without a
display (10 §10). :mod:`optivibe.gui.workers.stream_worker` wraps it in a
``QObject``.

Three properties are deliberate and load-bearing:

**No DSP here.** Every number on screen comes from
:class:`~optivibe.dsp.streaming.StreamingDsp` -- the same causal layer the
batch<->stream acceptance golden pins (theory-06 §7.6). The loop only slices
blocks and decides *when* to hand one over;
:func:`~optivibe.dsp.streaming.replay_record` is the finite driver of the same
``process()``/``snapshot()`` calls, so the live path executes verified code
rather than a copy of it.

**The producer paces itself.** Frames are gated on the wall clock inside the
loop, so an accelerated replay cannot flood the consumer no matter how fast the
record is chewed through.

**Provenance is not decorative.** ``warmed`` comes straight from the streaming
layer. Dropped samples are *produced* only where they can exist: in real-time
mode the loop measures its lag and, when it falls behind by more than a block,
skips forward and records the gap (drop-to-newest, theory-06 §5.7). In
accelerated mode there is no obligation to a clock, so a drop is impossible by
construction and the counter is reported as ``None`` -- "not defined here", not
a reassuring zero.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from optivibe.analysis.instrument import AnalyzeSpec, record_sensitivity_model
from optivibe.core.config.loader import default_config_dir, load_constants, load_variant
from optivibe.core.config.models import Constants, DspOptions, ScenarioConfig, VariantConfig
from optivibe.core.config.subsystems import SystemConfig
from optivibe.core.logging import get_logger
from optivibe.core.types import DetectorOutput, VibrationResult
from optivibe.dsp.sensitivity import SensitivityModel
from optivibe.dsp.streaming import StreamingDsp
from optivibe.gui.controllers.system_builder import resolve_system_variant
from optivibe.io.records import read_record
from optivibe.pipeline import Pipeline

logger = get_logger(__name__)

__all__ = [
    "RecordSource",
    "ScenarioSource",
    "StreamConfig",
    "StreamFrame",
    "StreamSetup",
    "StreamSource",
    "default_block_size",
    "run_stream",
]

#: Replay pace: follow the record's own sampling rate, or go as fast as possible.
Speed = Literal["realtime", "max"]

#: Blocks fed per emitted frame (finer than the frame rate so pacing and the
#: cooperative stop poll stay responsive between frames).
_BLOCKS_PER_FRAME = 4

#: Smallest block the loop will feed (very slow records would otherwise tick
#: one sample at a time).
_MIN_BLOCK = 16


def default_block_size(fs: float, rate_hz: float) -> int:
    """Return the default block length for a source at ``fs`` and ``rate_hz``.

    Sized to a fraction of one frame period so the loop can pace itself and
    notice a stop request between frames, without slicing so finely that the
    per-block overhead dominates.

    Parameters
    ----------
    fs : float
        Sampling rate, Hz.
    rate_hz : float
        Frame (snapshot) rate, Hz.

    Returns
    -------
    int
        Block length in samples (``>= 16``).
    """
    return max(_MIN_BLOCK, int(fs / (rate_hz * _BLOCKS_PER_FRAME)))


@dataclass(frozen=True)
class StreamSetup:
    """Everything the loop needs, assembled by a source off the UI thread.

    Attributes
    ----------
    detector : DetectorOutput
        The finite record to stream (a file capture or a freshly synthesized
        one). It doubles as the stream template: ``fs``, ``dc_level``, ``units``
        and the noise metadata come from it.
    variant : VariantConfig
        The resolved sensor variant (calibration, band, ISO class).
    options : DspOptions
        Inverse/DSP options; ``f_c_stream`` sets the causal cut-off.
    scenario : ScenarioConfig or None
        The scenario behind a synthetic source, or ``None`` for a file replay.
        Carried so the view can predict expected peaks (config-only, SW-70);
        a file record has no scenario, hence no prediction.
    sensitivity_model : SensitivityModel or None
        Calibration model to inject; ``None`` lets the streaming layer resolve
        the model scalar from the options (the forward-tract default).
    label : str
        Short human-readable source name, shown with the provenance.
    """

    detector: DetectorOutput
    variant: VariantConfig
    options: DspOptions
    scenario: ScenarioConfig | None = None
    sensitivity_model: SensitivityModel | None = None
    label: str = "stream"


@dataclass(frozen=True)
class StreamConfig:
    """Live-loop settings (the minimal control set of decision 13 §4.5).

    Attributes
    ----------
    rate_hz : float
        Frames per second handed to the consumer; the reference profile is
        ``N = 10/s`` (theory-06 §5.5, SW-67).
    speed : {"realtime", "max"}
        Replay pace. ``"realtime"`` follows the record's own clock (and can
        therefore drop samples when it falls behind); ``"max"`` runs flat out.
    loop : bool
        Restart the record when it ends. Every wrap is a genuine discontinuity
        and is flagged, never smoothed over (:attr:`StreamFrame.seam`).
    nperseg : int
        Streaming FFT length ``L``.
    window_s : float
        Requested oscilloscope window for the ``a/v/x`` traces, seconds; turned
        into ``history_samples`` once ``fs`` is known (never shorter than one
        spectral frame).
    block_size : int or None
        Samples fed per iteration; ``None`` uses :func:`default_block_size`.
    """

    rate_hz: float = 10.0
    speed: Speed = "realtime"
    loop: bool = False
    nperseg: int = 1024
    window_s: float = 0.5
    block_size: int | None = None


@dataclass(frozen=True)
class StreamFrame:
    """One emitted snapshot plus the provenance that makes it readable.

    Attributes
    ----------
    result : VibrationResult
        The streaming snapshot: bounded ``a/v/x`` traces, running spectrum,
        dominants, RMS, ISO and NEA -- all computed by the core.
    warmed : bool
        Whether the causal filters and the running spectrum have settled. Until
        they have, the numbers are **not final** (theory-06 §5.7).
    dropped_samples : int or None
        Samples the loop skipped to stay on the wall clock, or ``None`` in
        accelerated mode, where the counter is not defined by construction.
    n_samples : int
        Samples actually processed so far.
    elapsed_s : float
        Record time advanced so far (processed + skipped), seconds.
    loops : int
        Completed wraps of the record (``0`` unless looping).
    seam : bool
        ``True`` on the first frame after a wrap: its window straddles the
        splice, so the wide-band splatter it shows is an artifact of looping,
        not physics.
    paced : bool
        Whether the loop is following the record's clock.
    source_label : str
        Name of the source that produced the frame.
    """

    result: VibrationResult
    warmed: bool
    dropped_samples: int | None
    n_samples: int
    elapsed_s: float
    loops: int
    seam: bool
    paced: bool
    source_label: str

    @property
    def window_s(self) -> float:
        """Time span of the trace currently on screen, seconds."""
        fs = self.result.fs
        return float(self.result.a.size / fs) if fs > 0.0 else 0.0


@runtime_checkable
class StreamSource(Protocol):
    """A thing that can be opened into a :class:`StreamSetup`.

    :meth:`open` does the heavy work -- file I/O, composition resolution, a
    forward run -- and therefore always runs on the worker thread (SW-06).
    """

    @property
    def label(self) -> str:
        """Short human-readable source name."""
        ...

    def open(self) -> StreamSetup:
        """Load/synthesize the record and resolve everything the loop needs."""
        ...


@dataclass(frozen=True)
class ScenarioSource:
    """Synthetic source: run the forward chain, stream its detector output.

    The record is finite -- it lasts exactly the scenario's duration -- and the
    loop stops at its end unless looping is on. Stitching a longer signal out
    of repeats is *not* silently continuous, so it is never done implicitly: a
    phase-continuous chunked generator is a change to the excitation contract
    (backlog S-24), not something a view may improvise.

    Parameters
    ----------
    scenario : ScenarioConfig
        The scenario to synthesize (config-first: assembled from the panel, the
        same payload the Run action uses).
    system : SystemConfig or None, optional
        Edited composition to resolve instead of loading the named variant;
        resolution reads presets off disk and so belongs on this thread (SW-06).
    config_dir : pathlib.Path or None, optional
        Override for the ``configs/`` directory.
    """

    scenario: ScenarioConfig
    system: SystemConfig | None = None
    config_dir: Path | None = None

    @property
    def label(self) -> str:
        """Short human-readable source name."""
        return f"synthetic: {self.scenario.name}"

    def open(self) -> StreamSetup:
        """Resolve the variant and run the forward chain into a record.

        Returns
        -------
        StreamSetup
            The synthesized detector record and its context.
        """
        variant = _resolve_variant(
            self.scenario.variant, system=self.system, config_dir=self.config_dir
        )
        forward = Pipeline(self.scenario, variant).forward()
        logger.debug("stream source '%s': %d samples", self.label, forward.detector.n_samples)
        return StreamSetup(
            detector=forward.detector,
            variant=variant,
            options=self.scenario.dsp,
            scenario=self.scenario,
            label=self.label,
        )


@dataclass(frozen=True)
class RecordSource:
    """File source: replay a recorded photocurrent through the causal chain.

    The descriptor is an **analyze spec** (``kind: analyze``) -- the very file
    ``optivibe analyze`` consumes -- so the record's units, rate, timestamp,
    variant and calibration are declared once, in config, and the live view
    adds no metadata form of its own (config-first; headless parity, 09 §9).
    The calibration choice is honoured through
    :func:`~optivibe.analysis.instrument.record_sensitivity_model`: a record
    calibrated on a bench must not be rendered with the model scalar just
    because it is being watched live.

    Parameters
    ----------
    spec : AnalyzeSpec
        The validated analyze spec (record + variant + calibration + DSP).
    config_dir : pathlib.Path or None, optional
        Override for the ``configs/`` directory.
    """

    spec: AnalyzeSpec
    config_dir: Path | None = None

    @property
    def label(self) -> str:
        """Short human-readable source name."""
        return f"record: {self.spec.name}"

    def open(self) -> StreamSetup:
        """Load the record and resolve its variant and calibration.

        Returns
        -------
        StreamSetup
            The loaded record and its context.
        """
        record = read_record(self.spec.record)
        constants = load_constants(
            self.config_dir / "constants.yaml" if self.config_dir is not None else None
        )
        variant = load_variant(self.spec.variant, config_dir=self.config_dir)
        model = record_sensitivity_model(self.spec, record, variant, constants)
        logger.debug("stream source '%s': %d samples", self.label, record.detector.n_samples)
        return StreamSetup(
            detector=record.detector,
            variant=variant,
            options=self.spec.dsp,
            scenario=None,
            sensitivity_model=model,
            label=self.label,
        )


def _resolve_variant(
    name: str,
    *,
    system: SystemConfig | None,
    config_dir: Path | None,
) -> VariantConfig:
    """Resolve an edited composition, or load the variant by name."""
    if system is not None:
        return resolve_system_variant(system, config_dir or default_config_dir())
    return load_variant(name, config_dir=config_dir)


def run_stream(
    setup: StreamSetup,
    config: StreamConfig,
    *,
    on_frame: Callable[[StreamFrame], None],
    is_stopped: Callable[[], bool],
    constants: Constants | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Feed the record to a :class:`StreamingDsp` and emit frames at ``rate_hz``.

    The loop owns no physics: it slices blocks, paces itself against the wall
    clock and calls ``process()``/``snapshot()`` -- the same calls
    :func:`~optivibe.dsp.streaming.replay_record` makes over a finite record.

    Parameters
    ----------
    setup : StreamSetup
        The opened source.
    config : StreamConfig
        Loop settings (rate, pace, looping, window).
    on_frame : Callable[[StreamFrame], None]
        Consumer of each emitted frame. In the GUI this is a Qt signal, so the
        frame crosses to the UI thread through the event queue.
    is_stopped : Callable[[], bool]
        Cooperative stop poll, checked once per block.
    constants : Constants or None, optional
        Physical constants (loaded by the streaming layer when ``None``).
    monotonic, sleep : callable, optional
        Clock and sleep injection points (tests drive a virtual clock instead
        of waiting in real time).

    Raises
    ------
    ValueError
        If ``rate_hz`` is not positive.
    """
    if config.rate_hz <= 0.0:
        msg = f"rate_hz must be positive, got {config.rate_hz}"
        raise ValueError(msg)
    samples = setup.detector.samples  # non-empty by the DetectorOutput contract
    fs = setup.detector.fs
    block_size = config.block_size or default_block_size(fs, config.rate_hz)
    history = max(config.nperseg, round(fs * config.window_s))
    stream = StreamingDsp(
        setup.detector,
        setup.variant,
        setup.options,
        constants=constants,
        sensitivity_model=setup.sensitivity_model,
        nperseg=config.nperseg,
        keep_history=False,
        history_samples=history,
    )
    paced = config.speed == "realtime"
    period = 1.0 / config.rate_hz
    started = monotonic()
    next_emit = started + period
    position = 0
    advanced = 0
    loops = 0
    seam = False

    def frame(seam_flag: bool) -> StreamFrame:
        """Assemble the payload for the current stream state."""
        return StreamFrame(
            result=stream.snapshot(),
            warmed=stream.warmed,
            dropped_samples=stream.dropped_samples if paced else None,
            n_samples=stream.n_samples,
            elapsed_s=advanced / fs,
            loops=loops,
            seam=seam_flag,
            paced=paced,
            source_label=setup.label,
        )

    while not is_stopped():
        block = samples[position : position + block_size]
        stream.process(block)
        position += int(block.size)
        advanced += int(block.size)

        if paced:
            skipped = _pace(
                started=started,
                advanced=advanced,
                fs=fs,
                block_size=block_size,
                remaining=samples.size - position,
                monotonic=monotonic,
                sleep=sleep,
            )
            if skipped:
                stream.note_dropped(skipped)
                position += skipped
                advanced += skipped

        exhausted = position >= samples.size
        if exhausted and config.loop:
            position = 0
            loops += 1
            seam = True
        now = monotonic()
        if now >= next_emit and not (exhausted and not config.loop):
            on_frame(frame(seam))
            seam = False
            next_emit = now + period
        if exhausted and not config.loop:
            break

    # Finish on a fresh frame: the last state the user saw must be the state
    # the stream actually ended in (whether it ran out or was stopped). A stop
    # honoured before the first block leaves nothing to snapshot -- emitting
    # there would build a VibrationResult over empty arrays and fail the
    # contract, turning "stopped immediately" into a spurious error.
    if stream.n_samples:
        on_frame(frame(seam))


def _pace(
    *,
    started: float,
    advanced: int,
    fs: float,
    block_size: int,
    remaining: int,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> int:
    """Hold the record's own clock; return the samples to skip when behind.

    Sleeping is the normal case (the consumer is faster than real time). When
    the loop is *behind* by more than one block, catching up by processing
    faster is impossible by definition, so the policy of theory-06 §5.7 applies:
    keep the newest, jump the gap and count it. The skip is capped by what is
    left of the record; a wrap re-times naturally on the next iteration.

    Returns
    -------
    int
        Samples to skip (``0`` when on time).
    """
    due = started + advanced / fs
    now = monotonic()
    if now < due:
        sleep(due - now)
        return 0
    lag = now - due
    if lag <= block_size / fs:
        return 0
    return min(int(lag * fs), max(remaining, 0))
