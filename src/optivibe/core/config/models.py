"""Pydantic configuration models (constants, variants, scenarios).

These models validate the three configuration levels of document 09 §7 and
mirror the numbers of the knowledge base: :class:`Constants` reflects document
01 §4, :class:`VariantConfig` reflects document 08 §6, and :class:`ScenarioConfig`
describes one reproducible run (09 §8). Numbers are *not* duplicated by hand in
code: they live in ``configs/*.yaml`` and a consistency test checks them against
the base references (SW-03; see ``tests/test_constants_golden.py``).

All models are frozen and forbid unknown keys, so a typo in a YAML file fails
loudly at load time (10 §7, no silent failures).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Stage-selection keys default to the S0 stub/identity implementations; later
# chats register physical implementations under their own keys (09 §6).
StageKey = str


class _Frozen(BaseModel):
    """Base model: immutable, rejects unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #
# Constants (mirror of document 01 §4).
# --------------------------------------------------------------------------- #
class FiberConstants(_Frozen):
    """Fiber/fused-silica mechanical constants (doc 01 §4.1)."""

    diameter_m: float = Field(gt=0.0, description="Outer (cladding) diameter D, m")
    radius_m: float = Field(gt=0.0, description="Radius R = D/2, m")
    area_m2: float = Field(gt=0.0, description="Cross-section area S = pi R^2, m^2")
    inertia_m4: float = Field(gt=0.0, description="Second moment I = pi R^4 / 4, m^4")
    youngs_modulus_pa: float = Field(gt=0.0, description="Young's modulus E, Pa")
    density_kg_m3: float = Field(gt=0.0, description="Density rho, kg/m^3")
    poisson_ratio: float = Field(ge=0.0, lt=0.5, description="Poisson ratio nu")
    bar_velocity_m_s: float = Field(gt=0.0, description="Bar velocity c = sqrt(E/rho), m/s")


class AirConstants(_Frozen):
    """Air properties at ~20 C (doc 01 §4.2)."""

    density_kg_m3: float = Field(gt=0.0, description="Air density rho_f, kg/m^3")
    dynamic_viscosity_pa_s: float = Field(gt=0.0, description="Dynamic viscosity mu_f, Pa s")


class UniversalConstants(_Frozen):
    """Universal/geometric constants of the cantilever (doc 01 §4.3)."""

    g0_m_s2: float = Field(gt=0.0, description="Standard gravity g0, m/s^2")
    beta1_l: float = Field(gt=0.0, description="First eigenvalue beta_1 * L (dimensionless)")
    beta2_l: float = Field(gt=0.0, description="Second eigenvalue beta_2 * L (doc 02 §2)")
    phi1_at_tip: float = Field(description="Mode-1 shape at tip phi_1(1)")
    phi1_dd_at_root: float = Field(
        description="Mode-1 curvature at root phi_1''(0), for the doc-01 normalization phi_1(1)=2"
    )


class Constants(_Frozen):
    """Top-level physical constants bundle (doc 01).

    Attributes
    ----------
    fiber : FiberConstants
        Fiber/silica mechanical constants.
    air : AirConstants
        Air properties for damping.
    universal : UniversalConstants
        Universal and cantilever geometric constants.
    tilt_displacement_coupling_per_l : float
        Dimensionless rigid coupling ``theta * L / delta = 1.377`` (doc 01 §0,
        04 §2); divide by ``L`` to obtain ``theta / delta`` in 1/m.
    """

    fiber: FiberConstants
    air: AirConstants
    universal: UniversalConstants
    tilt_displacement_coupling_per_l: float = Field(
        gt=0.0, description="theta * L / delta coupling (1.377), dimensionless"
    )


# --------------------------------------------------------------------------- #
# Variant configuration (mirror of document 08 §6).
# --------------------------------------------------------------------------- #
class BandConfig(_Frozen):
    """Target frequency band of a variant (doc 08 §6)."""

    f_min_hz: float = Field(gt=0.0, description="Lower band edge, Hz")
    f_max_hz: float = Field(gt=0.0, description="Upper band edge, Hz")

    @model_validator(mode="after")
    def _check_order(self) -> BandConfig:
        if self.f_max_hz <= self.f_min_hz:
            msg = f"f_max_hz ({self.f_max_hz}) must exceed f_min_hz ({self.f_min_hz})"
            raise ValueError(msg)
        return self


class ReflectorConfig(_Frozen):
    """Convex reflector geometry/coating (doc 08 §6; R-12, R-16, R-24)."""

    shape: Literal["cylinder", "sphere", "wedge", "plane"] = "cylinder"
    radius_of_curvature_m: float = Field(gt=0.0, description="Reflector radius R_c, m")
    reflectivity: float = Field(
        gt=0.0, le=1.0, description="Mirror reflectivity rho (0.98 metallized)"
    )


class SourceConfig(_Frozen):
    """Optical source parameters (doc 08 §6; R-13, R-15, R-30)."""

    kind: Literal["SLD", "DFB"] = "SLD"
    wavelength_m: float = Field(gt=0.0, description="Laser wavelength lambda, m")
    power_w: float = Field(gt=0.0, description="Optical power P, W")
    rin_db_hz: float = Field(description="Relative intensity noise, dB/Hz")


class VariantConfig(_Frozen):
    """Full parameter set of one sensor-family variant (doc 08 §6).

    Attributes
    ----------
    name : {"A", "B", "C", "D"}
        Variant identifier.
    description : str
        Short human-readable description / class.
    mode : {"offresonance", "resonance"}
        Operating regime. A/B/C are off-resonance (R-21); D is resonant (R-39).
    band : BandConfig
        Target frequency band.
    line_freq_hz : float or None
        Resonant line frequency for variant D (None for off-resonance variants).
    length_m : float
        Cantilever length L, m.
    full_scale_g : float
        Full-scale acceleration FS, in g (placeholder per O-09 / doc 08 §1.3).
    reflector : ReflectorConfig
        Reflector geometry and coating.
    source : SourceConfig
        Optical source.
    route : {1, 2}
        Endface treatment route (2 = coherent wash-out; 1 = AR + DFB).
    responsivity_a_w : float
        Photodetector responsivity R, A/W.
    endface_reflectivity : float
        Fiber endface Fresnel reflectivity R1.
    eta_bias : float
        Optical working-point efficiency eta0.
    q_total : float
        Total quality factor of mode 1 at this variant's length (docs 07/08).
    target_nea_ug_rthz : float or None
        Placeholder target noise-equivalent acceleration, ug/sqrt(Hz) (O-09).
    vacuum : bool
        Whether the variant is operated under vacuum (A/D option).
    """

    name: Literal["A", "B", "C", "D"]
    description: str
    mode: Literal["offresonance", "resonance"] = "offresonance"
    band: BandConfig
    line_freq_hz: float | None = None
    length_m: float = Field(gt=0.0, description="Cantilever length L, m")
    full_scale_g: float = Field(gt=0.0, description="Full-scale acceleration FS, g")
    reflector: ReflectorConfig
    source: SourceConfig
    route: Literal[1, 2] = 2
    responsivity_a_w: float = Field(gt=0.0, description="PD responsivity R, A/W")
    endface_reflectivity: float = Field(ge=0.0, le=1.0, description="Endface Fresnel R1")
    eta_bias: float = Field(gt=0.0, le=1.0, description="Optical bias eta0")
    q_total: float = Field(
        gt=0.0,
        description=(
            "Total mechanical quality factor Q of mode 1 at this variant's L "
            "(docs 07 §4.3 / 08; overridable per scenario via mechanics.q_total)"
        ),
    )
    target_nea_ug_rthz: float | None = Field(default=None, gt=0.0)
    vacuum: bool = False

    @model_validator(mode="after")
    def _check_mode(self) -> VariantConfig:
        if self.mode == "resonance" and self.line_freq_hz is None:
            msg = "resonant variant requires line_freq_hz"
            raise ValueError(msg)
        return self


# --------------------------------------------------------------------------- #
# Scenario configuration (one reproducible run; doc 09 §8, 11 §6).
# --------------------------------------------------------------------------- #
class _ExcitationBase(_Frozen):
    """Fields shared by every excitation spec.

    The signal is generated on (or mapped to) the single axis ``axis``; the two
    remaining axes are zero. Different per-axis signals are a planned extension
    (a future composite kind), kept out of v-S1 so the default semantics of
    ``axis`` stay unchanged (task S1; doc 11 §2.1).
    """

    axis: Literal["x", "y", "z"] = "x"


class _GeneratedBase(_ExcitationBase):
    """Fields shared by synthetic generators (sampling grid is user-defined)."""

    fs_hz: float = Field(gt=0.0, description="Sampling frequency, Hz")
    duration_s: float = Field(gt=0.0, description="Signal duration, s")


class SineSpec(_GeneratedBase):
    """Single-tone sine excitation (the S0 acceptance signal)."""

    kind: Literal["sine"] = "sine"
    frequency_hz: float = Field(gt=0.0, description="Sine frequency, Hz")
    amplitude_g: float = Field(gt=0.0, description="Sine amplitude, g")


class Tone(_Frozen):
    """One tone of a multitone signal.

    Accepts the compact sequence form ``[frequency_hz, amplitude_g]`` or
    ``[frequency_hz, amplitude_g, phase_rad]`` in YAML, as well as the explicit
    mapping form.
    """

    frequency_hz: float = Field(gt=0.0, description="Tone frequency, Hz")
    amplitude_g: float = Field(gt=0.0, description="Tone amplitude, g")
    phase_rad: float = Field(default=0.0, description="Initial phase, rad")

    @model_validator(mode="before")
    @classmethod
    def _coerce_sequence(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            if not 2 <= len(value) <= 3:
                msg = f"tone sequence must be (frequency_hz, amplitude_g[, phase_rad]), got {value}"
                raise ValueError(msg)
            keys = ("frequency_hz", "amplitude_g", "phase_rad")
            return dict(zip(keys, value, strict=False))
        return value


class MultitoneSpec(_GeneratedBase):
    """Sum of sine tones with individual frequency/amplitude/phase."""

    kind: Literal["multitone"] = "multitone"
    tones: tuple[Tone, ...] = Field(min_length=1, description="Tones of the signal")


class SweepSpec(_GeneratedBase):
    """Constant-amplitude frequency sweep (chirp), linear or logarithmic."""

    kind: Literal["sweep"] = "sweep"
    f_start_hz: float = Field(gt=0.0, description="Start frequency, Hz")
    f_end_hz: float = Field(gt=0.0, description="End frequency, Hz")
    amplitude_g: float = Field(gt=0.0, description="Sweep amplitude, g")
    method: Literal["linear", "log"] = "linear"


class RandomSpec(_GeneratedBase):
    """Band-limited random noise with a target RMS level or one-sided PSD.

    Exactly one of ``g_rms`` (band RMS, g) or ``psd_g2_hz`` (flat one-sided PSD
    level, g^2/Hz) must be given. The synthesis shapes the spectrum directly in
    the frequency domain (see :mod:`optivibe.excitation.random_noise`).
    """

    kind: Literal["random"] = "random"
    band_hz: tuple[float, float] = Field(description="Band (f_lo, f_hi), Hz")
    g_rms: float | None = Field(default=None, gt=0.0, description="Target band RMS, g")
    psd_g2_hz: float | None = Field(
        default=None, gt=0.0, description="Target one-sided PSD level, g^2/Hz"
    )
    shape: Literal["flat"] = "flat"

    @model_validator(mode="after")
    def _check(self) -> RandomSpec:
        f_lo, f_hi = self.band_hz
        if not 0.0 <= f_lo < f_hi:
            msg = f"band_hz must satisfy 0 <= f_lo < f_hi, got {self.band_hz}"
            raise ValueError(msg)
        if f_hi > self.fs_hz / 2.0:
            msg = f"band upper edge {f_hi} Hz exceeds Nyquist {self.fs_hz / 2.0} Hz"
            raise ValueError(msg)
        if (self.g_rms is None) == (self.psd_g2_hz is None):
            msg = "exactly one of g_rms or psd_g2_hz must be set"
            raise ValueError(msg)
        return self


class ShockSpec(_GeneratedBase):
    """Single shock pulse (half-sine in v-S1) with optional pre-delay."""

    kind: Literal["shock"] = "shock"
    shape: Literal["half_sine"] = "half_sine"
    peak_g: float = Field(gt=0.0, description="Peak acceleration, g")
    pulse_ms: float = Field(gt=0.0, description="Pulse duration, ms")
    delay_s: float = Field(default=0.0, ge=0.0, description="Quiet time before pulse, s")

    @model_validator(mode="after")
    def _check_fits(self) -> ShockSpec:
        if self.delay_s + self.pulse_ms / 1.0e3 > self.duration_s:
            msg = (
                f"delay_s + pulse ({self.delay_s} s + {self.pulse_ms} ms) "
                f"exceeds duration_s = {self.duration_s} s"
            )
            raise ValueError(msg)
        return self


class CsvSpec(_ExcitationBase):
    """Replay of a measured acceleration record from a CSV file (seam SW-08).

    The sampling rate comes from the file: either a time column is given (fs is
    inferred from the median time step) or ``fs_hz`` must be set explicitly.
    ``resample_hz`` optionally resamples to a new rate (polyphase).
    """

    kind: Literal["csv"] = "csv"
    path: str = Field(description="Path to the CSV file")
    column: int | str = Field(default=1, description="Data column: 0-based index or header name")
    time_column: int | str | None = Field(
        default=None, description="Time column (s) to infer fs; index or header name"
    )
    fs_hz: float | None = Field(
        default=None, gt=0.0, description="Sampling rate, Hz (required if no time column)"
    )
    units: Literal["g", "m/s^2"] = Field(default="m/s^2", description="Units of the data column")
    delimiter: str = Field(default=",", description="Field delimiter")
    skiprows: int = Field(default=0, ge=0, description="Rows to skip before header/data")
    resample_hz: float | None = Field(default=None, gt=0.0, description="Target rate, Hz")

    @model_validator(mode="after")
    def _check_rate(self) -> CsvSpec:
        if self.time_column is None and self.fs_hz is None:
            msg = "csv excitation needs either time_column or fs_hz"
            raise ValueError(msg)
        return self


class WavSpec(_ExcitationBase):
    """Replay of a measured acceleration record from a WAV file (seam SW-08).

    Integer PCM samples are normalized to [-1, 1]; ``full_scale_g`` maps the
    normalized full scale (|1.0|) to acceleration in g. The sampling rate comes
    from the file header; ``resample_hz`` optionally resamples (polyphase).
    """

    kind: Literal["wav"] = "wav"
    path: str = Field(description="Path to the WAV file")
    channel: int = Field(default=0, ge=0, description="0-based channel index")
    full_scale_g: float = Field(gt=0.0, description="Acceleration at normalized full scale, g")
    resample_hz: float | None = Field(default=None, gt=0.0, description="Target rate, Hz")


ExcitationSpec = Annotated[
    SineSpec | MultitoneSpec | SweepSpec | RandomSpec | ShockSpec | CsvSpec | WavSpec,
    Field(discriminator="kind"),
]
"""Discriminated union of all excitation specs, selected by ``kind`` (S1).

The S0 form ``ExcitationSpec(kind="sine", ...)`` maps one-to-one onto
:class:`SineSpec`, so existing scenarios (``examples/hello.yaml``) parse
unchanged.
"""


class DspOptions(_Frozen):
    """Inverse/DSP options (full behaviour lands in S5)."""

    integrator: Literal["frequency", "time"] = "frequency"
    spectrum_method: Literal["fft", "welch"] = "fft"
    window: str = "hann"


class MechanicsOptions(_Frozen):
    """Per-scenario overrides of the mechanics stage (S2).

    Attributes
    ----------
    q_total : float or None
        Override of the variant's quality factor ``q_total`` (docs 07/08); the
        variant value is used when None. Forwarded to the mechanics
        implementation constructor by the orchestrator.
    """

    q_total: float | None = Field(
        default=None, gt=0.0, description="Quality-factor override (variant value if None)"
    )


class StageSelection(_Frozen):
    """Registry keys selecting the implementation of each stage (09 §6).

    Defaults point to the physical implementations where they exist ("modal"
    mechanics since S2) and to the S0 stubs elsewhere; the stubs remain
    registered for regression under their explicit keys.
    """

    excitation: StageKey = "sine"
    mechanics: StageKey = "modal"
    optics: StageKey = "stub"
    detector: StageKey = "stub"
    dsp: StageKey = "stub"


class OutputSpec(_Frozen):
    """Where/whether to persist run artifacts (io lands later)."""

    directory: str | None = None
    save: bool = False


class ScenarioConfig(_Frozen):
    """A complete, reproducible run description (09 §8, 11 §6).

    Attributes
    ----------
    name : str
        Scenario name.
    variant : {"A", "B", "C", "D"}
        Which variant preset to load from ``configs/variants``.
    excitation : ExcitationSpec
        Input-signal description.
    stages : StageSelection
        Registry keys selecting each stage implementation.
    mechanics : MechanicsOptions
        Per-scenario mechanics overrides (S2).
    dsp : DspOptions
        Inverse/DSP options.
    seed : int or None
        Random seed; one seed -> one result (10 §8).
    output : OutputSpec
        Output/persistence options.
    """

    name: str
    variant: Literal["A", "B", "C", "D"]
    excitation: ExcitationSpec
    stages: StageSelection = StageSelection()
    mechanics: MechanicsOptions = MechanicsOptions()
    dsp: DspOptions = DspOptions()
    seed: int | None = None
    output: OutputSpec = OutputSpec()
