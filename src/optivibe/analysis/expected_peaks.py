"""Expected spectral peaks: what the twin *predicts* a spectrum must contain.

Tasks S-16 / S-17 (doc 16); consumer: the spectrum-interpretation protocol of
doc 20 §3. The layer answers one question -- *before* any data exists: given
this sensor composition and this stimulus, which lines is the instrument
obliged to show, and how tall must they be to matter?

Why a separate artifact (coordination 2026-07-29, doc 13)
--------------------------------------------------------
Measured and expected are different epistemic categories (doc 20 §3.1), so the
prediction is **not** a field of
:class:`~optivibe.core.types.VibrationResult`: that contract carries what the
inverse chain *recovered* from a record, while :class:`ExpectedPeaks` is
computable from the configuration and the scenario **with no record at all**.
Keeping them apart also keeps the frozen ``VibrationResult`` contract (and its
GUI / ``analyze`` / golden consumers) untouched.

Config-first (doc 13, coordination 2026-07-29)
---------------------------------------------
:func:`predict_expected_peaks` takes a :class:`ScenarioConfig` and a
:class:`VariantConfig` and nothing else, so the CLI, the GUI and a test all
reach the same prediction through the same call; the GUI is never able to show
a marker the API cannot produce.

Taxonomy (open, doc 13)
-----------------------
:data:`PEAK_KINDS` declares the full family up front; predictors live in
:data:`PEAK_PREDICTOR_REGISTRY` (the doc 09 §6 registry pattern). Implemented
here: ``"mode"`` (the cantilever resonance ``f1`` and its ``f1/Q`` band, docs 02
§2 / 07 §2.3), ``"harmonic"`` (``k*f`` of the drive tones from the optical
nonlinearity ``eta(x)``, doc 03 §e) and ``"intermod"`` (``f_i +- f_j``
positions). Declared but deliberately **empty** branches: ``"sideband"``
(AM/FM ``f_c +- k*f_m`` -- lands with the S-21 synthesizer), ``"mains"``
(``50*k`` Hz), ``"alias"`` and ``"f_mount"``. Asking for an unimplemented kind
yields no peaks rather than an error, so a caller may always request the whole
taxonomy.

Significance threshold
----------------------
A predicted line only matters if it can clear the instrument's own noise. The
threshold is referred to the input through the NEA of doc 07 (the same
``sqrt(S_i)/|s_target(f)| (+) NEA_th`` chain as :mod:`optivibe.dsp.nea`, but
assembled from the *configuration* instead of a detector record) and then
mapped into the amplitude-spectrum domain by
:func:`amplitude_noise_threshold`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from optivibe.core.config.loader import load_constants
from optivibe.core.config.models import Constants, ScenarioConfig, VariantConfig
from optivibe.core.registry import Registry
from optivibe.detector.photodiode import noise_psd
from optivibe.dsp.calibration import target_sensitivity
from optivibe.mechanics.cantilever import CantileverModel
from optivibe.mechanics.thermal import nea_thermal
from optivibe.optics.reflector import ReflectorModel, build_reflector_model

__all__ = [
    "PEAK_KINDS",
    "PEAK_PREDICTOR_REGISTRY",
    "ExpectedPeak",
    "ExpectedPeaks",
    "PeakKind",
    "PredictionContext",
    "amplitude_noise_threshold",
    "predict_expected_peaks",
    "second_harmonic_ratio_taylor",
]

PeakKind = Literal["mode", "harmonic", "intermod", "sideband", "mains", "alias", "f_mount"]

PEAK_KINDS: tuple[PeakKind, ...] = (
    "mode",
    "harmonic",
    "intermod",
    "sideband",
    "mains",
    "alias",
    "f_mount",
)
"""The full peak taxonomy (doc 13, coordination 2026-07-29).

Declared in one place so consumers (doc 20 §3, the GUI legend) can enumerate the
family; only the kinds present in :data:`PEAK_PREDICTOR_REGISTRY` produce peaks
today.
"""

#: Sigma multiple a predicted line must clear to be called significant.
DEFAULT_SIGMA_FACTOR = 3.0

#: Central-difference step for the ``eta`` derivatives, m (doc 03 §e; the same
#: 1 nm step as the repository golden ``test_golden_thd_...``).
_ETA_STEP_M = 1.0e-9


@dataclass(frozen=True)
class ExpectedPeak:
    """One line the twin expects in the spectrum of a run (doc 20 §3.1).

    Attributes
    ----------
    freq_hz : float
        Predicted line position, Hz.
    kind : PeakKind
        Which taxonomy branch produced it (:data:`PEAK_KINDS`).
    label : str
        Short UI label (e.g. ``"resonance f1"``), English msgid by the GUI
        convention (doc 13, SW-65).
    explanation : str
        One-line "why this line is here" for a tooltip / figure annotation.
    amplitude_m_s2 : float or None
        Expected amplitude in the recovered-acceleration amplitude spectrum,
        m/s^2, or ``None`` when the height is not a property of the
        configuration alone (see the per-predictor notes).
    threshold_m_s2 : float or None
        Significance threshold at ``freq_hz``, m/s^2
        (:func:`amplitude_noise_threshold`); ``None`` when the spectral
        resolution is unknown (a file-replay stimulus).
    width_hz : float or None
        Expected line width, Hz (``f1/Q`` for a mode; ``None`` for the
        coherent lines, which are bin-limited).
    order : int or None
        Harmonic / intermodulation order ``k`` where it applies.
    source_freq_hz : float or None
        Parent drive frequency the line derives from, Hz, where it applies.
    """

    freq_hz: float
    kind: PeakKind
    label: str
    explanation: str
    amplitude_m_s2: float | None = None
    threshold_m_s2: float | None = None
    width_hz: float | None = None
    order: int | None = None
    source_freq_hz: float | None = None

    @property
    def significant(self) -> bool | None:
        """Whether the line is expected to clear the noise threshold.

        Returns
        -------
        bool or None
            ``True``/``False`` when both the amplitude and the threshold are
            known, ``None`` when the comparison is not defined (an unmodelled
            height or an unknown resolution) -- an honest "cannot say" rather
            than a default.
        """
        if self.amplitude_m_s2 is None or self.threshold_m_s2 is None:
            return None
        return self.amplitude_m_s2 >= self.threshold_m_s2


@dataclass(frozen=True)
class ExpectedPeaks:
    """The predicted peak set of one run plus the provenance of the prediction.

    Attributes
    ----------
    peaks : tuple of ExpectedPeak
        The predicted lines, ordered by frequency.
    variant_name : str
        Composition the prediction was made for.
    scenario_name : str
        Scenario the prediction was made for.
    f1_hz : float
        First bending-mode frequency used, Hz
        (:func:`~optivibe.mechanics.cantilever.first_mode_hz`).
    q_total : float
        Quality factor used (scenario override or variant value).
    resolution_hz : float or None
        Amplitude-spectrum bin width the thresholds assume, Hz; ``None`` when
        the record length is not known from the configuration.
    nyquist_hz : float or None
        Nyquist frequency of the stimulus, Hz; predictions above it are dropped
        (they would alias, taxonomy branch ``"alias"``).
    nea_plateau_m_s2_rthz : float
        Plateau NEA density the thresholds are built on, (m/s^2)/sqrt(Hz).
    kinds : tuple of PeakKind
        Taxonomy branches that were requested.
    """

    peaks: tuple[ExpectedPeak, ...]
    variant_name: str
    scenario_name: str
    f1_hz: float
    q_total: float
    resolution_hz: float | None
    nyquist_hz: float | None
    nea_plateau_m_s2_rthz: float
    kinds: tuple[PeakKind, ...]

    def __len__(self) -> int:
        return len(self.peaks)

    def of_kind(self, kind: PeakKind) -> tuple[ExpectedPeak, ...]:
        """Return the predicted peaks of one taxonomy branch.

        Parameters
        ----------
        kind : PeakKind
            Taxonomy branch (:data:`PEAK_KINDS`).

        Returns
        -------
        tuple of ExpectedPeak
            The matching peaks (possibly empty).
        """
        return tuple(peak for peak in self.peaks if peak.kind == kind)

    @property
    def band_hz(self) -> tuple[float, float] | None:
        """The ``f1/Q`` resonance band ``(lo, hi)``, Hz, or ``None`` if absent."""
        modes = self.of_kind("mode")
        if not modes:
            return None
        mode = modes[0]
        half = 0.5 * (mode.width_hz or 0.0)
        return (mode.freq_hz - half, mode.freq_hz + half)


@dataclass(frozen=True)
class PredictionContext:
    """Everything a predictor needs, resolved once from the configuration.

    Attributes
    ----------
    variant : VariantConfig
        Sensor composition.
    scenario : ScenarioConfig
        Run description (stimulus, stage keys, overrides).
    constants : Constants
        Physical constants (doc 01 mirror).
    cantilever : CantileverModel
        Derived mechanics of the composition (``f1``, ``Q``, ``H_lat(f)``).
    tones : tuple of tuple of float
        Drive tones as ``(frequency_hz, amplitude_m_s2)``; empty for stimuli
        with no closed-form line list (random / sweep / shock / file replay).
    resolution_hz : float or None
        Amplitude-spectrum bin width, Hz.
    nyquist_hz : float or None
        Nyquist frequency of the stimulus, Hz.
    current_psd_a2_hz : float
        Total plateau current-noise PSD of the composition, A^2/Hz.
    nea_plateau : float
        Plateau NEA density, (m/s^2)/sqrt(Hz).
    optics : ReflectorModel or None
        Reflector coupling model, or ``None`` when the scenario selects the stub
        optics (then no nonlinearity is modelled and harmonic heights are
        reported as unknown).
    sigma_factor : float
        Sigma multiple used for the significance thresholds.
    max_harmonic : int
        Highest harmonic order to predict.
    """

    variant: VariantConfig
    scenario: ScenarioConfig
    constants: Constants
    cantilever: CantileverModel
    tones: tuple[tuple[float, float], ...]
    resolution_hz: float | None
    nyquist_hz: float | None
    current_psd_a2_hz: float
    nea_plateau: float
    optics: ReflectorModel | None
    sigma_factor: float
    max_harmonic: int

    def nea_at(self, freq_hz: float) -> float:
        """NEA density at ``freq_hz``, (m/s^2)/sqrt(Hz) (docs 05 §7, 07 §3.1).

        Reproduces the :func:`optivibe.dsp.nea.nea_spectrum` chain
        ``sqrt(S_i)/|s_target^QS D(f)| (+) NEA_th`` from the configuration: the
        white current floor is referred through the dynamic sensitivity, so the
        optical branch dips by ``~1/Q`` toward ``f1`` and the total settles on
        the flat Brownian floor there (doc 07 §2.4).

        Parameters
        ----------
        freq_hz : float
            Frequency, Hz.

        Returns
        -------
        float
            NEA density, (m/s^2)/sqrt(Hz).
        """
        s_qs = target_sensitivity(self.variant, self.constants)
        gain = abs(complex(self.cantilever.dynamic_factor(freq_hz)[0]))
        optical = math.sqrt(self.current_psd_a2_hz) / (abs(s_qs) * gain)
        thermal = nea_thermal(self.constants, self.cantilever.length_m, self.cantilever.q_total)
        return math.hypot(optical, thermal)

    def threshold_at(self, freq_hz: float) -> float | None:
        """Significance threshold at ``freq_hz``, m/s^2, or ``None`` if unknown."""
        if self.resolution_hz is None:
            return None
        return amplitude_noise_threshold(
            self.nea_at(freq_hz), self.resolution_hz, sigma_factor=self.sigma_factor
        )

    def in_range(self, freq_hz: float) -> bool:
        """Whether ``freq_hz`` is a positive frequency below Nyquist."""
        if freq_hz <= 0.0:
            return False
        return self.nyquist_hz is None or freq_hz < self.nyquist_hz


#: Predictor registry: taxonomy key -> ``(context) -> peaks`` (doc 09 §6).
PEAK_PREDICTOR_REGISTRY: Registry[tuple[ExpectedPeak, ...]] = Registry("analysis.expected-peak")

Predictor = Callable[[PredictionContext], tuple[ExpectedPeak, ...]]


# --------------------------------------------------------------------------- #
# Threshold and nonlinearity helpers (closed-form, doc-referenced).
# --------------------------------------------------------------------------- #
def amplitude_noise_threshold(
    nea_density: float, resolution_hz: float, *, sigma_factor: float = DEFAULT_SIGMA_FACTOR
) -> float:
    """Map a noise *density* onto the amplitude-spectrum bin floor, m/s^2.

    :func:`~optivibe.dsp.spectra.amplitude_spectrum` scales an rFFT by ``2/N``,
    so for a white input of one-sided PSD ``S`` the expected squared bin
    amplitude is ``E[A^2] = (2/N)^2 N sigma^2 = 4 sigma^2 / N = 2 S df`` (using
    ``sigma^2 = S fs / 2`` and ``df = fs / N``). The RMS amplitude a white floor
    of density ``NEA`` produces in one bin is therefore
    ``sqrt(2) NEA sqrt(df)``, and a line is called significant when it clears
    ``sigma_factor`` times that.

    Parameters
    ----------
    nea_density : float
        Input-referred noise density at the frequency of interest,
        (m/s^2)/sqrt(Hz).
    resolution_hz : float
        Amplitude-spectrum bin width ``df``, Hz.
    sigma_factor : float, optional
        Sigma multiple (default :data:`DEFAULT_SIGMA_FACTOR`).

    Returns
    -------
    float
        Significance threshold in the amplitude spectrum, m/s^2.

    Raises
    ------
    ValueError
        If ``resolution_hz`` or ``sigma_factor`` is not positive.
    """
    if resolution_hz <= 0.0:
        msg = f"resolution_hz must be positive, got {resolution_hz!r}"
        raise ValueError(msg)
    if sigma_factor <= 0.0:
        msg = f"sigma_factor must be positive, got {sigma_factor!r}"
        raise ValueError(msg)
    return sigma_factor * math.sqrt(2.0 * resolution_hz) * nea_density


def second_harmonic_ratio_taylor(
    optics: ReflectorModel, tip_amplitude_m: float, *, step_m: float = _ETA_STEP_M
) -> float | None:
    """Small-signal ``HD2 = |eta''| d / (4 |eta'|)`` at the working point.

    The second-order Taylor term ``eta'' x^2 / 2`` of a sine ``x = d sin(wt)``
    folds into ``cos 2wt`` (doc 03 §e), which is exactly the reference the
    repository golden ``test_golden_thd_matches_analytic_second_harmonic``
    pins; the derivatives are taken by central differences about the configured
    working point, with the same convention as that golden (pure ``dx``, tilt
    excluded).

    Parameters
    ----------
    optics : ReflectorModel
        Reflector coupling model of the composition.
    tip_amplitude_m : float
        Tip-displacement amplitude ``d`` at the drive frequency, m.
    step_m : float, optional
        Central-difference step, m.

    Returns
    -------
    float or None
        The ``2f/1f`` amplitude ratio, dimensionless, or ``None`` at a
        degenerate working point (vanishing first derivative -- the symmetric
        point, where the response is quadratic and the ratio is not defined).
    """
    plus = _eta_scalar(optics, +step_m)
    zero = _eta_scalar(optics, 0.0)
    minus = _eta_scalar(optics, -step_m)
    slope = (plus - minus) / (2.0 * step_m)
    curvature = (plus - 2.0 * zero + minus) / step_m**2
    if slope == 0.0:
        return None
    return abs(curvature) * tip_amplitude_m / (4.0 * abs(slope))


def _eta_scalar(optics: ReflectorModel, dx_m: float) -> float:
    """Coupling efficiency ``eta`` at a static transverse offset ``dx_m``, m."""
    values = optics.eta(dx=dx_m)
    return float(values[0]) if getattr(values, "ndim", 0) else float(values)


def _plateau_current_psd(
    variant: VariantConfig, scenario: ScenarioConfig, constants: Constants
) -> float:
    """Total plateau current-noise PSD from the configuration alone, A^2/Hz.

    Rebuilds the DC operating point the detector stage would reach,
    ``I_DC = R P (R1 + rho eta0)`` (doc 04 §4; the same expression as
    :meth:`optivibe.detector.photodiode.Photodiode.run`), and feeds
    :func:`~optivibe.detector.photodiode.noise_psd` with the *same*
    balanced / reference-arm convention the scenario resolves (never re-picked,
    O-SW-08).
    """
    detector = variant.detector
    balanced = (
        scenario.detector.balanced
        if scenario.detector.balanced is not None
        else (detector.balanced)
    )
    reference_arm = (
        scenario.detector.reference_arm
        if scenario.detector.reference_arm is not None
        else detector.reference_arm
    )
    eta0 = _working_point(variant, scenario)
    i_dc = (
        variant.responsivity_a_w
        * variant.source.power_w
        * (variant.endface_reflectivity + variant.reflector.reflectivity * eta0)
    )
    psd = noise_psd(i_dc, variant, constants, balanced=balanced, reference_arm=reference_arm)
    return float(psd["total"])


def _working_point(variant: VariantConfig, scenario: ScenarioConfig) -> float:
    """Return the static coupling ``eta0`` the selected optics stage would report."""
    if scenario.stages.optics == "stub":
        return variant.eta_bias
    return float(build_reflector_model(variant).eta_working_point())


def _optics_model(variant: VariantConfig, scenario: ScenarioConfig) -> ReflectorModel | None:
    """Reflector model of the composition, or ``None`` for the stub optics."""
    if scenario.stages.optics == "stub":
        return None
    return build_reflector_model(variant)


def _drive_tones(scenario: ScenarioConfig, constants: Constants) -> tuple[tuple[float, float], ...]:
    """Drive tones as ``(frequency_hz, amplitude_m_s2)`` (doc 11 §2).

    Only the stimuli with a closed-form line list contribute: ``sine`` and
    ``multitone``. ``sweep`` / ``random`` / ``shock`` and the file-replay kinds
    have no fixed tone set, so they yield an empty tuple -- the harmonic branch
    then predicts nothing rather than guessing.
    """
    spec = scenario.excitation
    g0 = constants.universal.g0_m_s2
    kind = getattr(spec, "kind", "")
    if kind == "sine":
        return ((float(spec.frequency_hz), float(spec.amplitude_g) * g0),)  # type: ignore[union-attr]
    if kind == "multitone":
        return tuple(
            (float(tone.frequency_hz), float(tone.amplitude_g) * g0)
            for tone in spec.tones  # type: ignore[union-attr]
        )
    return ()


def _grid(scenario: ScenarioConfig) -> tuple[float | None, float | None]:
    """Return ``(resolution_hz, nyquist_hz)`` of the stimulus, or ``None``s.

    The generated stimuli carry ``fs_hz`` and ``duration_s``, so the rFFT bin
    width ``fs / N`` with ``N = round(duration fs)`` is known before the run;
    the replay kinds take their rate from the file, so both are unknown.
    """
    spec = scenario.excitation
    fs_hz = getattr(spec, "fs_hz", None)
    duration_s = getattr(spec, "duration_s", None)
    if fs_hz is None:
        return (None, None)
    nyquist = float(fs_hz) / 2.0
    if duration_s is None:
        return (None, nyquist)
    n_samples = max(1, round(float(duration_s) * float(fs_hz)))
    return (float(fs_hz) / n_samples, nyquist)


# --------------------------------------------------------------------------- #
# Predictors (implemented taxonomy branches).
# --------------------------------------------------------------------------- #
@PEAK_PREDICTOR_REGISTRY.register("mode")
def predict_mode(context: PredictionContext) -> tuple[ExpectedPeak, ...]:
    """Predict the cantilever resonance ``f1`` and its ``f1/Q`` band (02 §2, 07 §2.3).

    ``f1`` comes from :func:`~optivibe.mechanics.cantilever.first_mode_hz` (the
    Euler-Bernoulli closed form), *not* from the ``100/L^2`` scaling reference,
    and the width is ``f1/Q`` with the effective ``Q`` of the run. The height
    is intentionally left unknown: the line is a ``|D(f1)| = Q`` amplification
    of whatever spectral density happens to sit at ``f1`` (broadband stimulus,
    or the noise floor itself), so it is not a property of the configuration.

    Parameters
    ----------
    context : PredictionContext
        Resolved prediction context.

    Returns
    -------
    tuple of ExpectedPeak
        One peak, or empty when ``f1`` lies above Nyquist.
    """
    f1 = context.cantilever.f1_hz
    if not context.in_range(f1):
        return ()
    q = context.cantilever.q_total
    peak = ExpectedPeak(
        freq_hz=f1,
        kind="mode",
        label="resonance f1",
        explanation=(
            "cantilever mode 1: |D(f1)| = Q amplification, width f1/Q; "
            "does not move when the stimulus changes (doc 20 §3.1)"
        ),
        amplitude_m_s2=None,
        threshold_m_s2=context.threshold_at(f1),
        width_hz=f1 / q,
    )
    return (peak,)


@PEAK_PREDICTOR_REGISTRY.register("harmonic")
def predict_harmonics(context: PredictionContext) -> tuple[ExpectedPeak, ...]:
    """Harmonics ``k f`` of the drive tones from the optical nonlinearity.

    ``eta(x)`` is only locally linear about the biased working point, so a tone
    of tip amplitude ``d`` folds a second harmonic with ratio
    ``HD2 = |eta''| d / (4 |eta'|) ~ d^2`` (doc 03 §e). The tip amplitude is
    ``d = |H_lat(f)| a`` from the mechanics FRF (doc 05 §1), and the plateau
    calibration refers both lines through the same scalar, so the predicted
    height in the recovered-acceleration spectrum is ``HD2`` times the
    recovered fundamental ``a |D(f)|``.

    Orders above two are located but not sized: ``HD_k ~ d^(k-1)`` needs the
    ``k``-th derivative of ``eta``, which is outside this task's scope, so the
    height is reported as unknown instead of guessed.

    Parameters
    ----------
    context : PredictionContext
        Resolved prediction context.

    Returns
    -------
    tuple of ExpectedPeak
        The predicted harmonic lines.
    """
    peaks: list[ExpectedPeak] = []
    for freq_hz, amplitude in context.tones:
        gain = abs(complex(context.cantilever.dynamic_factor(freq_hz)[0]))
        tip_m = float(abs(complex(context.cantilever.h_lat(freq_hz)[0]))) * amplitude
        hd2 = (
            second_harmonic_ratio_taylor(context.optics, tip_m)
            if context.optics is not None
            else None
        )
        for order in range(2, context.max_harmonic + 1):
            harmonic_hz = order * freq_hz
            if not context.in_range(harmonic_hz):
                continue
            height = hd2 * amplitude * gain if (order == 2 and hd2 is not None) else None
            peaks.append(
                ExpectedPeak(
                    freq_hz=harmonic_hz,
                    kind="harmonic",
                    label=f"HD{order} of {freq_hz:.4g} Hz",
                    explanation=(
                        "HD2 = |eta''| d / (4 |eta'|), so the line falls as d^2 with the "
                        "drive amplitude (optical nonlinearity eta, doc 03 §e)"
                        if order == 2
                        else f"harmonic {order}f of the drive tone; height ~ d^(k-1), "
                        "not modelled here (doc 03 §e)"
                    ),
                    amplitude_m_s2=height,
                    threshold_m_s2=context.threshold_at(harmonic_hz),
                    order=order,
                    source_freq_hz=freq_hz,
                )
            )
    return tuple(peaks)


@PEAK_PREDICTOR_REGISTRY.register("intermod")
def predict_intermod(context: PredictionContext) -> tuple[ExpectedPeak, ...]:
    """Second-order intermodulation positions ``f_i +- f_j`` (doc 20 §3.1).

    The same curvature that folds a second harmonic mixes any two drive tones
    into their sum and difference. Positions only: the height needs the joint
    second-order term of ``eta`` for a two-tone drive, which is not modelled in
    this task, so it is reported as unknown.

    Parameters
    ----------
    context : PredictionContext
        Resolved prediction context.

    Returns
    -------
    tuple of ExpectedPeak
        The predicted intermodulation lines (empty for a single-tone drive).
    """
    peaks: list[ExpectedPeak] = []
    tones = context.tones
    for i, (f_i, _a_i) in enumerate(tones):
        for f_j, _a_j in tones[i + 1 :]:
            for freq_hz, sign in ((f_i + f_j, "+"), (abs(f_i - f_j), "-")):
                if not context.in_range(freq_hz):
                    continue
                peaks.append(
                    ExpectedPeak(
                        freq_hz=freq_hz,
                        kind="intermod",
                        label=f"IM2 {f_i:.4g} {sign} {f_j:.4g} Hz",
                        explanation=(
                            "second-order intermodulation of two drive tones through the "
                            "curvature of eta; height not modelled (doc 03 §e)"
                        ),
                        amplitude_m_s2=None,
                        threshold_m_s2=context.threshold_at(freq_hz),
                        order=2,
                        source_freq_hz=f_i,
                    )
                )
    return tuple(peaks)


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #
def predict_expected_peaks(
    scenario: ScenarioConfig,
    variant: VariantConfig,
    constants: Constants | None = None,
    *,
    kinds: Sequence[PeakKind] | None = None,
    max_harmonic: int = 3,
    sigma_factor: float = DEFAULT_SIGMA_FACTOR,
) -> ExpectedPeaks:
    """Predict the spectral lines a run is obliged to show (tasks S-16/S-17).

    Step 1 of the interpretation protocol of doc 20 §3.2: run the *stimulus and
    the composition*, not a record, through the twin and obtain the list of
    mandatory lines. No time series is touched, so the call is equally
    available to the CLI, the GUI and a test.

    The signature is the guarantee: taking only a scenario and a variant, this
    function cannot be handed a record even by mistake, so its cost never grows
    with the data. That is what makes it safe to call from the GUI thread under
    the S7 invariant (doc 13, ``SW-06``/``SW-70``); keep it that way -- adding a
    record-shaped parameter would silently void the guarantee.

    Parameters
    ----------
    scenario : ScenarioConfig
        Run description (stimulus, stage keys, mechanics/detector overrides).
    variant : VariantConfig
        Resolved sensor composition.
    constants : Constants or None, optional
        Physical constants (loaded when ``None``).
    kinds : sequence of PeakKind or None, optional
        Taxonomy branches to predict; ``None`` requests all of
        :data:`PEAK_KINDS`. Branches without a predictor contribute nothing.
    max_harmonic : int, optional
        Highest harmonic order to locate (default 3, i.e. ``2f`` and ``3f``).
    sigma_factor : float, optional
        Sigma multiple for the significance thresholds.

    Returns
    -------
    ExpectedPeaks
        The predicted lines, ordered by frequency, plus the provenance of the
        prediction (``f1``, ``Q``, resolution, plateau NEA).

    Raises
    ------
    ValueError
        If ``max_harmonic`` is below 2.
    """
    if max_harmonic < 2:
        msg = f"max_harmonic must be >= 2, got {max_harmonic!r}"
        raise ValueError(msg)
    consts = load_constants() if constants is None else constants
    requested = tuple(PEAK_KINDS if kinds is None else kinds)
    cantilever = CantileverModel.from_config(consts, variant, q_total=scenario.mechanics.q_total)
    resolution_hz, nyquist_hz = _grid(scenario)
    current_psd = _plateau_current_psd(variant, scenario, consts)
    context = PredictionContext(
        variant=variant,
        scenario=scenario,
        constants=consts,
        cantilever=cantilever,
        tones=_drive_tones(scenario, consts),
        resolution_hz=resolution_hz,
        nyquist_hz=nyquist_hz,
        current_psd_a2_hz=current_psd,
        nea_plateau=math.hypot(
            math.sqrt(current_psd) / abs(target_sensitivity(variant, consts)),
            nea_thermal(consts, cantilever.length_m, cantilever.q_total),
        ),
        optics=_optics_model(variant, scenario),
        sigma_factor=sigma_factor,
        max_harmonic=max_harmonic,
    )
    peaks: list[ExpectedPeak] = []
    for kind in requested:
        if kind not in PEAK_PREDICTOR_REGISTRY:
            continue  # declared-but-empty taxonomy branch (doc 13)
        predictor: Predictor = PEAK_PREDICTOR_REGISTRY.get(kind)
        peaks.extend(predictor(context))
    peaks.sort(key=lambda peak: peak.freq_hz)
    return ExpectedPeaks(
        peaks=tuple(peaks),
        variant_name=variant.name,
        scenario_name=scenario.name,
        f1_hz=cantilever.f1_hz,
        q_total=cantilever.q_total,
        resolution_hz=resolution_hz,
        nyquist_hz=nyquist_hz,
        nea_plateau_m_s2_rthz=context.nea_plateau,
        kinds=requested,
    )
