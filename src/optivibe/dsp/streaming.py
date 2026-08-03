"""Causal streaming kinematics and DSP for the real-time layer (S-03 / H-3).

The batch integrators in :mod:`optivibe.dsp.kinematics` are *non-causal* -- the
frequency method takes a full-record rFFT and the time method a whole-record
cubic detrend plus a zero-phase ``sosfiltfilt`` -- so neither can emit a sample
until the entire record is known (theory-06 §3.6). The real-time layer therefore
uses a **causal scheme**: a leaky integrator, i.e. a first-order recursive
high-pass/integrator that carries its filter state ``zi`` across frames and runs
sample-by-sample with bounded latency.

The causal scheme is *not* bit-exact with the batch integrators on the
integration stage -- that is impossible by construction (the batch path is
non-causal, theory-06 §7.6). It agrees with the batch path in band, after warm-up,
within the doc 11 §7 tolerances. The batch chain and its frozen numbers are
untouched; this module is strictly additive.

References
----------
Block 06 of the theory slice (``docs/theory/06_dsp_algorithm.md``) §3.6, §5, §7;
decision SW-67.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from optivibe.core.config.loader import load_constants
from optivibe.core.config.models import Constants, DspOptions, VariantConfig
from optivibe.core.types import DetectorOutput, FloatArray, Spectrum, VibrationResult
from optivibe.dsp.calibration import calibrate_acceleration
from optivibe.dsp.iso import iso_assessment
from optivibe.dsp.metrics import band_rms_velocity, second_harmonic_ratio
from optivibe.dsp.nea import nea_from_detector
from optivibe.dsp.sensitivity import SensitivityModel, build_sensitivity_model
from optivibe.dsp.spectra import amplitude_spectrum, dominant_frequencies, welch_psd

__all__ = ["LeakyIntegrator", "StreamingDsp", "StreamingSpectrum", "replay_record"]


class _TraceRing:
    """Fixed-size ring buffer over the most recent ``capacity`` samples.

    Backs the *display* trace of :class:`StreamingDsp` when the history is
    bounded. The storage is one pre-allocated array written in place, so the
    memory of an endless stream is constant by construction -- the whole point
    of the bounded mode (theory-06 §5.7: a live monitor keeps the newest, it
    does not accumulate). Never grow this into a list of blocks: that silently
    restores the unbounded cost the bounded mode exists to remove.

    Parameters
    ----------
    capacity : int
        Number of samples retained (``>= 1``).
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = int(capacity)
        self._buf = np.zeros(self._capacity, dtype=np.float64)
        self._start = 0
        self._filled = 0

    def push(self, block: FloatArray) -> None:
        """Append ``block``, overwriting the oldest samples once full.

        Parameters
        ----------
        block : numpy.ndarray, shape (n,)
            New contiguous samples.
        """
        cap = self._capacity
        x = block if block.size <= cap else block[-cap:]
        n = int(x.size)
        end = (self._start + self._filled) % cap
        first = min(n, cap - end)
        self._buf[end : end + first] = x[:first]
        if n > first:
            self._buf[: n - first] = x[first:]
        if self._filled + n < cap:
            self._filled += n
        else:
            self._filled = cap
            self._start = (end + n) % cap

    def values(self) -> FloatArray:
        """Return the retained samples in chronological order (a fresh copy)."""
        if self._filled == 0:
            return np.empty(0, dtype=np.float64)
        if self._filled < self._capacity:
            return np.ascontiguousarray(self._buf[: self._filled])
        return np.ascontiguousarray(
            np.concatenate([self._buf[self._start :], self._buf[: self._start]])
        )


class LeakyIntegrator:
    r"""Causal first-order leaky integrator (one ``a -> v`` or ``v -> x`` stage).

    The ideal trapezoidal integrator ``H(z) = (dt/2)(1 + z^-1)/(1 - z^-1)`` has a
    pole at ``z = 1`` (marginally stable -> accumulates DC -> drift). The leak
    moves the pole inside the unit circle,

    .. math::

        H_{\text{leaky}}(z) = \frac{dt}{2}\,
        \frac{1 + z^{-1}}{1 - \alpha z^{-1}},\qquad
        \alpha = e^{-2\pi f_c / f_s} < 1,

    giving a finite DC gain ``dt / (1 - alpha)`` and forgetting old offsets with
    time constant ``1 / (2 pi f_c)`` -- equivalently an integrator followed by a
    causal first-order high-pass at ``f_c`` (theory-06 §3.6). The cut-off ``f_c`` is
    the streaming high-pass ``DspOptions.f_c_stream`` (independent of the batch
    ``f_hp``, theory-06 §9.3-2).

    The instance carries the filter state ``zi`` between :meth:`process` calls,
    so streaming a signal in frames of any size yields exactly the same samples
    as a single call -- the internal seam-invariance the acceptance golden
    checks (theory-06 §7.6-ii). Chain two instances for ``a -> v -> x``.

    Parameters
    ----------
    fs : float
        Sampling rate, Hz (``> 0``).
    f_c : float
        Streaming high-pass / leak cut-off, Hz (``0 < f_c < fs / 2``).

    Attributes
    ----------
    alpha : float
        The leak pole ``exp(-2 pi f_c / fs)``.
    """

    def __init__(self, fs: float, f_c: float) -> None:
        if fs <= 0.0:
            msg = f"fs must be positive, got {fs}"
            raise ValueError(msg)
        if f_c <= 0.0:
            msg = f"f_c must be positive, got {f_c}"
            raise ValueError(msg)
        if f_c >= fs / 2.0:
            msg = f"f_c ({f_c}) must be below Nyquist fs/2 ({fs / 2.0})"
            raise ValueError(msg)
        dt = 1.0 / fs
        alpha = float(np.exp(-2.0 * np.pi * f_c / fs))
        self._fs = float(fs)
        self._f_c = float(f_c)
        self._b = np.array([dt / 2.0, dt / 2.0], dtype=np.float64)
        self._a = np.array([1.0, -alpha], dtype=np.float64)
        # An integrator at rest starts with zero delay state.
        self._zi = np.zeros(1, dtype=np.float64)

    @property
    def fs(self) -> float:
        """Sampling rate, Hz."""
        return self._fs

    @property
    def f_c(self) -> float:
        """Leak / high-pass cut-off, Hz."""
        return self._f_c

    @property
    def alpha(self) -> float:
        """Leak pole ``alpha = exp(-2 pi f_c / fs)``."""
        return float(-self._a[1])

    @property
    def dc_gain(self) -> float:
        """Bounded DC gain ``dt / (1 - alpha)`` (finite thanks to the leak)."""
        return float(self._b.sum() / self._a.sum())

    def reset(self) -> None:
        """Clear the carried state, returning the integrator to rest."""
        self._zi = np.zeros(1, dtype=np.float64)

    def process(self, block: FloatArray) -> FloatArray:
        """Integrate one contiguous ``block`` causally, carrying state.

        Parameters
        ----------
        block : numpy.ndarray, shape (n,)
            Input samples (e.g. acceleration for the ``a -> v`` stage).

        Returns
        -------
        numpy.ndarray, shape (n,)
            Integrated output. The instance advances its internal state ``zi`` so
            the next call continues seamlessly -- feeding a signal split into any
            sequence of blocks returns the same concatenated output as one call.
        """
        x = np.ascontiguousarray(block, dtype=np.float64)
        if x.ndim != 1:
            msg = f"block must be 1-D, got shape {x.shape}"
            raise ValueError(msg)
        if x.size == 0:
            return np.empty(0, dtype=np.float64)
        y, self._zi = signal.lfilter(self._b, self._a, x, zi=self._zi)
        return np.ascontiguousarray(y, dtype=np.float64)


class StreamingSpectrum:
    r"""Continuous one-sided PSD from a ring buffer with exponential averaging.

    The batch spectrum (:func:`~optivibe.dsp.spectra.welch_psd`) needs the whole
    record. The streaming spectrum instead keeps the last ``nperseg`` samples in
    a ring buffer and, every ``hop = nperseg - noverlap`` samples, forms one
    windowed periodogram and folds it into a running estimate (theory-06 §5.2)

    .. math::

        S[m] = \beta\, S[m-1] + (1 - \beta)\,\lvert X[m]\rvert^2 ,\qquad
        0 < \beta < 1,

    with effective averaging depth ``~ 1 / (1 - beta)`` frames. Each per-segment
    periodogram is produced by the *same* :func:`~optivibe.dsp.spectra.welch_psd`
    the batch path uses (single segment, ``noverlap=0``), so the PSD
    normalization is identical to batch (units ``u^2/Hz``, one-sided) -- a single
    implementation (17 §7). The result is a plain
    :class:`~optivibe.core.types.Spectrum` (``kind="psd"``) and therefore feeds
    the unchanged :func:`~optivibe.dsp.spectra.dominant_frequencies` and
    :func:`~optivibe.dsp.metrics.band_rms_velocity`.

    Segmentation depends only on the *total* number of samples seen, not on how
    they are chunked across :meth:`process` calls, so feeding a signal in frames
    of any size yields the same estimate (seam-invariance, theory-06 §7.6-ii).

    Parameters
    ----------
    fs : float
        Sampling rate, Hz (``> 0``).
    nperseg : int
        Segment/FFT length ``L`` (``>= 2``); sets the resolution ``fs / L`` and
        the frame-fill latency ``L / fs`` (theory-06 §5.5).
    window : str, optional
        Window name (default ``"hann"``), passed through to the periodogram.
    noverlap : int or None, optional
        Segment overlap; ``None`` uses ``nperseg // 2`` (50 %, theory-06 §2.4).
    beta : float or None, optional
        Exponential forgetting factor ``0 < beta < 1``. ``None`` derives it from
        ``avg_segments`` as ``1 - 1 / avg_segments``.
    avg_segments : int, optional
        Effective averaging depth when ``beta`` is ``None`` (default ``8``).
    """

    def __init__(
        self,
        fs: float,
        nperseg: int,
        *,
        window: str = "hann",
        noverlap: int | None = None,
        beta: float | None = None,
        avg_segments: int = 8,
    ) -> None:
        if fs <= 0.0:
            msg = f"fs must be positive, got {fs}"
            raise ValueError(msg)
        if nperseg < 2:
            msg = f"nperseg must be >= 2, got {nperseg}"
            raise ValueError(msg)
        ov = nperseg // 2 if noverlap is None else int(noverlap)
        if not 0 <= ov < nperseg:
            msg = f"noverlap must be in [0, nperseg), got {ov}"
            raise ValueError(msg)
        if beta is None:
            if avg_segments < 1:
                msg = f"avg_segments must be >= 1, got {avg_segments}"
                raise ValueError(msg)
            beta = 1.0 - 1.0 / float(avg_segments)
        if not 0.0 < beta < 1.0:
            msg = f"beta must be in (0, 1), got {beta}"
            raise ValueError(msg)
        self._fs = float(fs)
        self._nperseg = int(nperseg)
        self._window = window
        self._noverlap = ov
        self._hop = nperseg - ov
        self._beta = float(beta)
        self._buf = np.empty(0, dtype=np.float64)
        self._freq: FloatArray | None = None
        self._psd: FloatArray | None = None
        self._n_segments = 0

    @property
    def fs(self) -> float:
        """Sampling rate, Hz."""
        return self._fs

    @property
    def nperseg(self) -> int:
        """Segment/FFT length ``L``."""
        return self._nperseg

    @property
    def hop(self) -> int:
        """Advance between segments, ``nperseg - noverlap``."""
        return self._hop

    @property
    def beta(self) -> float:
        """Exponential forgetting factor."""
        return self._beta

    @property
    def n_segments(self) -> int:
        """Number of periodograms folded into the running estimate."""
        return self._n_segments

    @property
    def ready(self) -> bool:
        """Whether at least one full segment has been processed."""
        return self._psd is not None

    def reset(self) -> None:
        """Clear the ring buffer and the running estimate."""
        self._buf = np.empty(0, dtype=np.float64)
        self._freq = None
        self._psd = None
        self._n_segments = 0

    def process(self, block: FloatArray) -> None:
        """Append ``block`` and fold every newly completed segment into ``S[m]``.

        Parameters
        ----------
        block : numpy.ndarray, shape (n,)
            New contiguous samples (e.g. acceleration or velocity).
        """
        x = np.ascontiguousarray(block, dtype=np.float64)
        if x.ndim != 1:
            msg = f"block must be 1-D, got shape {x.shape}"
            raise ValueError(msg)
        if x.size == 0:
            return
        self._buf = x.copy() if self._buf.size == 0 else np.concatenate([self._buf, x])
        while self._buf.size >= self._nperseg:
            segment = self._buf[: self._nperseg]
            spec = welch_psd(
                segment, self._fs, window=self._window, nperseg=self._nperseg, noverlap=0
            )
            if self._psd is None:
                self._freq = spec.freq
                self._psd = spec.values.copy()
            else:
                self._psd = self._beta * self._psd + (1.0 - self._beta) * spec.values
            self._n_segments += 1
            self._buf = self._buf[self._hop :]

    def spectrum(self) -> Spectrum | None:
        """Return the current running PSD, or ``None`` before the first segment.

        Returns
        -------
        Spectrum or None
            A one-sided PSD (``kind="psd"``, units ``u^2/Hz``) with the same
            normalization as :func:`~optivibe.dsp.spectra.welch_psd`; ``None``
            until at least one full ``nperseg`` window has been seen (warm-up).
        """
        if self._psd is None or self._freq is None:
            return None
        return Spectrum(
            freq=self._freq,
            values=self._psd,
            kind="psd",
            window=self._window,
            method="welch",
        )


class StreamingDsp:
    r"""Causal streaming orchestrator: detector-sample blocks -> vibration (S-03).

    The real-time counterpart of :class:`~optivibe.dsp.standard.StandardDsp`. It
    runs the same five stages, but each carries state across frames so it can
    consume an unbounded stream of detector-sample blocks (theory-06 §5.4):

    1. **Calibration** ``samples -> a`` -- the *same*
       :func:`~optivibe.dsp.calibration.calibrate_acceleration` per block; it is
       per-sample memoryless, so block-by-block output is **bit-identical** to
       the batch calibration (theory-06 §7.6-i).
    2. **Kinematics** ``a -> v -> x`` -- two chained
       :class:`LeakyIntegrator` stages (causal, state ``zi`` carried), the
       causal scheme of theory-06 §3.6 (cut-off ``f_c_stream``).
    3. **Spectrum** -- a :class:`StreamingSpectrum` on acceleration (dominant
       lines) and one on velocity (band RMS).
    4. **Metrics** -- running RMS, band-RMS velocity -> ISO severity, second
       harmonic ``2f`` -- reusing the unchanged metric functions (17 §7).
    5. **NEA** -- signal-independent, computed once from the detector metadata.

    The integrators/spectra carry state, so streaming a signal in frames of any
    size gives -- after warm-up -- the same result (seam-invariance, doc
    §7.6-ii). It is **not** bit-exact with batch on the integration stage (the
    batch path is non-causal); in band it agrees within doc 11 §7 (§7.6-iii).

    Parameters
    ----------
    template : DetectorOutput
        Carries the stream's ``fs``, ``dc_level``, ``units`` and ``noise`` (its
        ``samples`` are ignored). Blocks fed to :meth:`process` share these.
    variant : VariantConfig
        Sensor variant (calibration, band, ISO class).
    options : DspOptions
        DSP options; ``f_c_stream`` (falling back to ``f_hp``) sets the causal
        cut-off. ``deconvolve_hlat`` / ``sensitivity_freq="dynamic"`` are batch
        only (non-causal) and are ignored here.
    constants : Constants or None, optional
        Physical constants (loaded when ``None``).
    sensitivity_model : SensitivityModel or None, optional
        Injected calibration model (role S-02); ``None`` resolves it from the
        options, reproducing the v1 static-plateau scalar.
    nperseg : int, optional
        Streaming FFT length ``L`` (default ``1024``).
    noverlap : int or None, optional
        Segment overlap (default ``nperseg // 2``).
    avg_segments : int, optional
        Exponential averaging depth for the streaming spectra (default ``8``).
    keep_history : bool, optional
        Accumulate the full ``a/v/x`` for the snapshot (default ``True``; the
        finite replay/verification path). ``False`` keeps only the last
        ``nperseg`` samples as a live oscilloscope trace (bounded memory).
    history_samples : int or None, optional
        Length of the retained ``a/v/x`` *display* trace when
        ``keep_history=False``; ``None`` (default) keeps ``nperseg`` samples,
        which is bit-identical to the pre-O-SW-03 behaviour. The parameter
        exists because one spectral frame is a poor oscilloscope window for a
        human: at the reference profile (``fs = 100 kHz``, ``L = 4096``,
        theory-06 §5.5) it is 41 ms. It moves **nothing but the trace** -- the
        spectra, the running metrics and the integrator states are untouched,
        so no number changes with it. Memory stays bounded: the trace lives in a
        pre-allocated ring (:class:`_TraceRing`), never a growing list. Invalid
        together with ``keep_history=True`` (the unbounded mode has no window).
    """

    def __init__(
        self,
        template: DetectorOutput,
        variant: VariantConfig,
        options: DspOptions,
        *,
        constants: Constants | None = None,
        sensitivity_model: SensitivityModel | None = None,
        nperseg: int = 1024,
        noverlap: int | None = None,
        avg_segments: int = 8,
        keep_history: bool = True,
        history_samples: int | None = None,
    ) -> None:
        if history_samples is not None:
            if keep_history:
                msg = "history_samples applies only to the bounded mode (keep_history=False)"
                raise ValueError(msg)
            if history_samples < 1:
                msg = f"history_samples must be >= 1, got {history_samples}"
                raise ValueError(msg)
        self._fs = template.fs
        self._dc_level = template.dc_level
        self._units = template.units
        self._variant = variant
        self._options = options
        self._constants = load_constants() if constants is None else constants
        self._model = (
            sensitivity_model
            if sensitivity_model is not None
            else build_sensitivity_model(variant, options, self._constants)
        )
        f_hp = options.f_hp_hz if options.f_hp_hz is not None else variant.band.f_min_hz
        f_c = options.f_c_stream if options.f_c_stream is not None else f_hp
        self._f_c = f_c
        self._integ_v = LeakyIntegrator(self._fs, f_c)
        self._integ_x = LeakyIntegrator(self._fs, f_c)
        self._spec_a = StreamingSpectrum(
            self._fs, nperseg, window=options.window, noverlap=noverlap, avg_segments=avg_segments
        )
        self._spec_v = StreamingSpectrum(
            self._fs, nperseg, window=options.window, noverlap=noverlap, avg_segments=avg_segments
        )
        self._band = (variant.band.f_min_hz, variant.band.f_max_hz)
        # Running RMS accumulators (sum of squares + count).
        self._ss = {"a": 0.0, "v": 0.0, "x": 0.0}
        self._n = 0
        self._keep_history = keep_history
        self._hist_a: list[FloatArray] = []
        self._hist_v: list[FloatArray] = []
        self._hist_x: list[FloatArray] = []
        capacity = nperseg if history_samples is None else int(history_samples)
        self._trace_capacity = None if keep_history else capacity
        self._ring_a = _TraceRing(capacity)
        self._ring_v = _TraceRing(capacity)
        self._ring_x = _TraceRing(capacity)
        # NEA is signal-independent -> compute once from the template.
        self._nea = nea_from_detector(template, variant, self._constants)
        self._dropped = 0
        # Warm-up: a few leak time-constants (1/(2 pi f_c)) and one segment.
        warmup_tc = int(np.ceil(5.0 / (2.0 * np.pi * f_c) * self._fs))
        self._warmup_samples = max(nperseg, warmup_tc)

    @property
    def fs(self) -> float:
        """Sampling rate, Hz."""
        return self._fs

    @property
    def f_c(self) -> float:
        """Causal streaming cut-off actually used, Hz."""
        return self._f_c

    @property
    def n_samples(self) -> int:
        """Total samples processed so far."""
        return self._n

    @property
    def dropped_samples(self) -> int:
        """Samples flagged as dropped by the source (provenance, theory-06 §5.7)."""
        return self._dropped

    @property
    def warmed(self) -> bool:
        """Whether the causal filters and spectra have settled (theory-06 §5.7)."""
        return self._spec_a.ready and self._n >= self._warmup_samples

    def note_dropped(self, count: int) -> None:
        """Record ``count`` samples the source dropped (a stream discontinuity).

        Increments the provenance counter; snapshots taken across a gap should be
        treated as not fully continuous (theory-06 §5.7).
        """
        if count < 0:
            msg = f"count must be non-negative, got {count}"
            raise ValueError(msg)
        self._dropped += int(count)

    def process(self, sample_block: FloatArray) -> None:
        """Consume one block of detector samples, advancing all stage state.

        Parameters
        ----------
        sample_block : numpy.ndarray, shape (n,)
            Detector samples in the template's ``units`` (photocurrent A or
            transimpedance V), same ``dc_level``.
        """
        block = np.ascontiguousarray(sample_block, dtype=np.float64)
        if block.ndim != 1:
            msg = f"sample_block must be 1-D, got shape {block.shape}"
            raise ValueError(msg)
        if block.size == 0:
            return
        detector = DetectorOutput(
            samples=block, fs=self._fs, dc_level=self._dc_level, units=self._units
        )
        accel, _ = calibrate_acceleration(
            detector, self._variant, self._constants, model=self._model
        )
        velocity = self._integ_v.process(accel)
        displacement = self._integ_x.process(velocity)

        self._ss["a"] += float(np.dot(accel, accel))
        self._ss["v"] += float(np.dot(velocity, velocity))
        self._ss["x"] += float(np.dot(displacement, displacement))
        self._n += accel.size

        self._spec_a.process(accel)
        self._spec_v.process(velocity)

        if self._keep_history:
            self._hist_a.append(accel)
            self._hist_v.append(velocity)
            self._hist_x.append(displacement)
        else:
            self._ring_a.push(accel)
            self._ring_v.push(velocity)
            self._ring_x.push(displacement)

    @property
    def trace_samples(self) -> int | None:
        """Length of the retained display trace, or ``None`` when unbounded."""
        return self._trace_capacity

    def _traces(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return the current ``a/v/x`` traces (full history or bounded window)."""
        if self._keep_history:
            return (
                np.concatenate(self._hist_a) if self._hist_a else np.empty(0),
                np.concatenate(self._hist_v) if self._hist_v else np.empty(0),
                np.concatenate(self._hist_x) if self._hist_x else np.empty(0),
            )
        return self._ring_a.values(), self._ring_v.values(), self._ring_x.values()

    def _running_rms(self) -> dict[str, float]:
        if self._n == 0:
            return {"a": 0.0, "v": 0.0, "x": 0.0}
        return {k: float(np.sqrt(v / self._n)) for k, v in self._ss.items()}

    def snapshot(self) -> VibrationResult:
        """Assemble the current :class:`~optivibe.core.types.VibrationResult`.

        The ``a/v/x`` traces are the accumulated stream (or the last
        ``history_samples`` / ``nperseg`` window when ``keep_history=False``);
        the RMS, spectrum, dominant lines, ISO and NEA
        are the running streaming estimates. The ``2f`` residual is taken from
        the amplitude spectrum of the (bit-exact) acceleration.
        """
        a, v, x = self._traces()

        spec_a = self._spec_a.spectrum()
        dominant = (
            dominant_frequencies(spec_a, interpolate=self._options.peak_interpolation)
            if spec_a is not None
            else ()
        )

        cross_residual: dict[str, float] = {}
        if dominant and a.size:
            amp = amplitude_spectrum(a, self._fs)
            shr = second_harmonic_ratio(amp, dominant[0])
            if shr is not None:
                cross_residual["second_harmonic_ratio"] = shr

        # Before the first full segment there is no velocity spectrum at all, so
        # there is no band RMS to grade. The pre-`SW-77` code substituted 0.0 and
        # graded it, which put "zone A" on screen for every stream that had not
        # warmed up yet -- the live path is where a user meets this first, so the
        # provenance rule of `SW-72` (`warmed`, `dropped_samples`) applies here
        # too: undefined is reported as undefined, not as a reassuring zero.
        spec_v = self._spec_v.spectrum()
        if spec_v is None:
            v_rms_band: float | None = None
            reason: str | None = "spectrum_not_ready"
        else:
            v_rms_band = band_rms_velocity(spec_v, self._band)
            reason = None if v_rms_band is not None else "band_has_fewer_than_two_bins"
        iso: dict[str, object] = iso_assessment(
            v_rms_band,
            machine_class=self._options.iso_machine_class,
            band_hz=self._band,
            undefined_reason=reason,
        )
        if self._nea is not None:
            iso["nea"] = self._nea.as_dict()

        return VibrationResult(
            a=a,
            v=v,
            x=x,
            fs=self._fs,
            dominant_freqs_hz=dominant,
            rms=self._running_rms(),
            cross_residual=cross_residual,
            spectrum=spec_a,
            iso=iso,
        )


def replay_record(
    detector: DetectorOutput,
    variant: VariantConfig,
    options: DspOptions,
    *,
    block_size: int,
    constants: Constants | None = None,
    sensitivity_model: SensitivityModel | None = None,
    nperseg: int = 1024,
    noverlap: int | None = None,
    avg_segments: int = 8,
) -> VibrationResult:
    """Replay a recorded (or synthesized) detector output through the stream.

    Feeds ``detector.samples`` to a :class:`StreamingDsp` in ``block_size``
    chunks -- the same path a live DAQ or the synthetic generator would take --
    and returns the final snapshot. This is the finite-record driver the
    batch<->stream acceptance golden runs (theory-06 §7.6).

    Parameters
    ----------
    detector : DetectorOutput
        The record to replay (from a file or the forward generator).
    variant, options :
        As for :class:`StreamingDsp`.
    block_size : int
        Samples per fed block (``>= 1``); the result is invariant to it after
        warm-up.
    constants, sensitivity_model, nperseg, noverlap, avg_segments :
        Forwarded to :class:`StreamingDsp`.

    Returns
    -------
    VibrationResult
        The streaming snapshot over the whole record.
    """
    if block_size < 1:
        msg = f"block_size must be >= 1, got {block_size}"
        raise ValueError(msg)
    stream = StreamingDsp(
        detector,
        variant,
        options,
        constants=constants,
        sensitivity_model=sensitivity_model,
        nperseg=nperseg,
        noverlap=noverlap,
        avg_segments=avg_segments,
    )
    samples = detector.samples
    for start in range(0, samples.size, block_size):
        stream.process(samples[start : start + block_size])
    return stream.snapshot()
