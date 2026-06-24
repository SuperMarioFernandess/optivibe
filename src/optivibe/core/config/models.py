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


class DetectorConstants(_Frozen):
    """Universal/environment constants for the detector noise budget (doc 07).

    The elementary charge and Boltzmann constant are the exact SI-2019 values;
    the temperature mirrors document 07 §2.5 (``T = 293 K``). These are not in
    the document-01 table (mechanics did not need them), so they are added here
    as a detector-specific constants block. All values have ``Field`` defaults,
    so a ``constants.yaml`` without a ``detector`` block still validates.

    Attributes
    ----------
    elementary_charge_c : float
        Elementary charge ``e``, C (shot noise ``2 e I_DC``; doc 07 §1.1).
    boltzmann_j_k : float
        Boltzmann constant ``kB``, J/K (Johnson noise ``4 kB T / Rf``; 07 §1.3).
    temperature_k : float
        Absolute temperature ``T``, K (doc 07 §2.5).
    """

    elementary_charge_c: float = Field(default=1.602176634e-19, gt=0.0, description="e, C")
    boltzmann_j_k: float = Field(default=1.380649e-23, gt=0.0, description="kB, J/K")
    temperature_k: float = Field(default=293.0, gt=0.0, description="T, K (doc 07 §2.5)")


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
    detector : DetectorConstants
        Universal/environment constants for the detector noise budget (doc 07).
    tilt_displacement_coupling_per_l : float
        Dimensionless rigid coupling ``theta * L / delta = 1.377`` (doc 01 §0,
        04 §2); divide by ``L`` to obtain ``theta / delta`` in 1/m.
    """

    fiber: FiberConstants
    air: AirConstants
    universal: UniversalConstants
    detector: DetectorConstants = DetectorConstants()
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


class OpticsConfig(_Frozen):
    """Optical working-point parameters (mirror of docs 03 §1/§6 and 08 §2).

    Attributes
    ----------
    gap_m : float
        Nominal air gap A between the fiber endface and the reflector, m.
        Documented parametric band 20-40 um (doc 03 §6); the default 31 um is
        the S3 calibration anchoring eta0/eta_peak/slope to the doc 04/05
        references within 5 % (journal entry of 2026-06-12).
    bias_offset_m : float
        Intentional static de-centering Delta x0 setting the working point
        eta0 on the slope (docs 03 §5, 04 §4). The SNR-optimum rule
        eta0 ~ 0.37 * eta_peak (doc 08, R-40/O-05) is exposed as a helper
        (:meth:`optivibe.optics.cylinder.CylinderOpticsModel.bias_for_eta_ratio`);
        in v1 the bias is a static config value.
    mode_field_radius_m : float
        Gaussian mode-field radius w0 of the fiber at the source wavelength, m
        (doc 03 §1: w0 = 5.2 um at 1550 nm for SMF-28). Lives in the variant
        (not in constants.yaml) because it is tied to the source wavelength.
    """

    gap_m: float = Field(default=31.0e-6, gt=0.0, description="Air gap A, m (doc 03 §6)")
    bias_offset_m: float = Field(
        default=2.0e-6, ge=0.0, description="Working-point de-centering Delta x0, m (doc 03 §5)"
    )
    mode_field_radius_m: float = Field(
        default=5.2e-6, gt=0.0, description="Mode-field radius w0, m (doc 03 §1)"
    )


class DetectorConfig(_Frozen):
    """Detector front-end and ADC parameters (mirror of docs 07/08; S4).

    Mirrors the new detector-electronics numbers of documents 07 (noise budget)
    and 08 §6 (per-variant front-end). The optical numbers feeding the read-out
    (``P``, ``R``, ``rho``, ``R1``, source ``RIN``) already live on
    :class:`VariantConfig`/:class:`SourceConfig`/:class:`ReflectorConfig`; this
    block adds only what the detector stage introduces.

    Attributes
    ----------
    balanced : bool
        Whether the balanced reference channel is active (R-23, doc 07 §1.2).
        Default ``True``: RIN is common-mode and suppressed by ``cmrr_db``.
    reference_arm : {"matched", "bright"}
        Reference-arm shot model of the balanced channel (open question
        O-SW-08). ``"matched"`` (default) -- two equal arms, the signal arm at
        full power, so the difference carries shot from both arms and the shot
        PSD doubles (the conservative ``<= sqrt(2)`` RMS floor of doc 07 §1.2).
        ``"bright"`` -- ``P_ref >> P_sig`` / normalization, leaving the bare
        ``2 e I_DC`` (the datasheet / doc-08 shot limit). Ignored when
        ``balanced`` is ``False``. Flip this to compare the two NEA conventions
        at test time (see the journal O-SW-08).
    cmrr_db : float
        Common-mode rejection ratio of the balanced channel, dB. Default 40 dB:
        within the documented auto-balanced range 30-70 dB (doc 07 §1.2) and
        enough to keep the read-out shot-limited at the variant's operating
        power (30 dB suffices only near 1 mW; RIN does not improve with power).
    transimpedance_ohm : float or None
        Feedback resistor ``Rf``, ohm. Sets the Johnson-noise floor
        ``4 kB T / Rf`` referred to current (doc 07 §1.3); ``None`` removes the
        electronics floor (the ``Rf -> inf`` limit). Also the voltage gain when
        ``output == "voltage"``. The DC pedestal is AC-coupled away before the
        gain stage (doc 07 §1.4), so a large ``Rf`` does not saturate the TIA.
    output : {"current", "voltage"}
        Whether ``samples`` are photocurrent (A) or transimpedance voltage (V).
    adc_bits : int
        ADC resolution in bits. Default 24 (precision sigma-delta vibration DAQ),
        so quantization stays well below the electronic noise floor.
    adc_full_scale : float
        ADC full scale (the +/- range) for the AC-coupled modulation, in the
        output units (A or V). The DC pedestal is reported separately and not
        digitized (doc 07 §1.4). Inputs beyond +/- full scale are clipped.
    adc_fs_hz : float or None
        Optional ADC/decimation rate, Hz; ``None`` keeps the optical ``fs``
        (identity). When below ``fs`` the signal is resampled (anti-aliased when
        ``antialias`` is set).
    antialias : bool
        Whether to anti-alias-filter before decimation (doc 11 §2). Ignored when
        ``adc_fs_hz`` is ``None`` or ``>= fs``.
    rin_shape : {"white"}
        Spectral form of the RIN PSD. Only the white level is modelled in v-S4;
        a frequency-dependent shape is a recorded extension (doc 07 §1.2).
    """

    balanced: bool = True
    reference_arm: Literal["matched", "bright"] = "matched"
    cmrr_db: float = Field(default=40.0, description="Balanced-channel CMRR, dB (doc 07 §1.2)")
    transimpedance_ohm: float | None = Field(
        default=1.0e5, gt=0.0, description="Feedback resistor Rf, ohm (Johnson floor; doc 07 §1.3)"
    )
    output: Literal["current", "voltage"] = "current"
    adc_bits: int = Field(default=24, ge=1, le=32, description="ADC resolution, bits")
    adc_full_scale: float = Field(
        default=1.0e-4, gt=0.0, description="AC full scale (+/- range) in output units"
    )
    adc_fs_hz: float | None = Field(
        default=None, gt=0.0, description="ADC rate, Hz (None=identity)"
    )
    antialias: bool = True
    rin_shape: Literal["white"] = "white"


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
        Optical working-point efficiency eta0 used by the *stub* optics (S0);
        the physical "cylinder" optics computes its own eta0 from
        ``optics.bias_offset_m`` (S3) and reports it in
        ``OpticalResponse.bias``.
    optics : OpticsConfig
        Optical working-point parameters (gap A, bias Delta x0, mode-field
        radius w0; docs 03/08, S3).
    detector : DetectorConfig
        Detector front-end and ADC parameters (balanced channel, CMRR, Rf, ADC;
        docs 07/08, S4).
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
    eta_bias: float = Field(gt=0.0, le=1.0, description="Optical bias eta0 (stub optics)")
    optics: OpticsConfig = OpticsConfig()
    detector: DetectorConfig = DetectorConfig()
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


class _InstrumentBase(_ExcitationBase):
    """Fields shared by instrument-format replay specs (TDMS/UFF/MAT/HDF5, S8).

    These formats carry their own sampling metadata, so ``fs`` is read from the
    file (each subclass documents its source) unless overridden. ``units``
    selects how the stored channel maps onto acceleration in SI: ``"g"`` and
    ``"m/s^2"`` are engineering units; ``"V"`` is a raw voltage that needs an
    accelerometer ``sensitivity`` to become acceleration; ``"auto"`` reads the
    unit label embedded in the file and fails loudly if it is missing or
    unrecognized (10 §7). The actual conversion happens in the loader, at the
    input boundary (10 §6).
    """

    units: Literal["g", "m/s^2", "V", "auto"] = Field(
        default="m/s^2",
        description="Stored channel unit; 'auto' reads the file's unit label",
    )
    sensitivity: float | None = Field(
        default=None,
        gt=0.0,
        description="Accelerometer sensitivity (required only for voltage records)",
    )
    sensitivity_unit: Literal["mV/g", "V/g", "mV/(m/s^2)", "V/(m/s^2)"] = Field(
        default="mV/g", description="Units of `sensitivity` (voltage records only)"
    )
    resample_hz: float | None = Field(
        default=None, gt=0.0, description="Target sampling rate, Hz (polyphase resample)"
    )


class TdmsSpec(_InstrumentBase):
    """Replay of a measured record from an NI TDMS file (seam SW-08, S8).

    The sampling rate is taken from the channel's ``wf_increment`` waveform
    property unless ``fs_hz`` is set; ``units="auto"`` reads the channel's
    ``unit_string`` property.
    """

    kind: Literal["tdms"] = "tdms"
    path: str = Field(description="Path to the .tdms file")
    group: str | None = Field(
        default=None, description="TDMS group name (the first group is used if None)"
    )
    channel: int | str = Field(
        default=0, description="Channel within the group: 0-based index or channel name"
    )
    fs_hz: float | None = Field(
        default=None, gt=0.0, description="Sampling rate, Hz (from wf_increment if None)"
    )


class UffSpec(_InstrumentBase):
    """Replay of a measured record from a UFF/UNV dataset-58 file (S8).

    The sampling rate is taken from the function's ``abscissa_inc`` (the even
    abscissa step) unless ``fs_hz`` is set; ``units="auto"`` reads the
    ordinate's unit label.
    """

    kind: Literal["uff"] = "uff"
    path: str = Field(description="Path to the .uff/.unv file")
    dataset_index: int = Field(
        default=0, ge=0, description="Which dataset-58 record to read (0-based among them)"
    )
    fs_hz: float | None = Field(
        default=None, gt=0.0, description="Sampling rate, Hz (from abscissa_inc if None)"
    )


class MatSpec(_InstrumentBase):
    """Replay of a measured record from a MATLAB .mat file (v4/v5/v7; S8).

    A v7.3 ``.mat`` file is HDF5-based and is not read here -- use the ``hdf5``
    loader (or re-save as v7). The acceleration array lives in the variable
    ``data_key``; a 2-D array selects a channel with ``column``. MAT files carry
    no standard unit label, so ``units`` must be explicit (``"auto"`` is
    rejected).
    """

    kind: Literal["mat"] = "mat"
    path: str = Field(description="Path to the .mat file (v7.3 -> use the hdf5 loader)")
    data_key: str = Field(description="Name of the variable holding the acceleration array")
    column: int | None = Field(
        default=None, ge=0, description="Column to read for 2-D data (the first if None)"
    )
    fs_hz: float | None = Field(default=None, gt=0.0, description="Sampling rate, Hz")
    fs_key: str | None = Field(
        default=None, description="Name of a variable holding the scalar sampling rate"
    )

    @model_validator(mode="after")
    def _check_rate(self) -> MatSpec:
        if self.fs_hz is None and self.fs_key is None:
            msg = "mat excitation needs either fs_hz or fs_key"
            raise ValueError(msg)
        return self


class Hdf5Spec(_InstrumentBase):
    """Replay of a measured record from an HDF5 (.h5/.hdf5) file (S8).

    The signal lives at the dataset path ``dataset``; a 2-D dataset selects a
    channel with ``column``. The sampling rate comes from ``fs_hz`` or from the
    dataset attribute named by ``fs_attr``; ``units="auto"`` reads the unit
    string from the attribute named by ``units_attr``.
    """

    kind: Literal["hdf5"] = "hdf5"
    path: str = Field(description="Path to the .h5/.hdf5 file")
    dataset: str = Field(description="Path of the dataset inside the file (e.g. '/accel/x')")
    column: int | None = Field(
        default=None, ge=0, description="Column to read for 2-D data (the first if None)"
    )
    fs_hz: float | None = Field(default=None, gt=0.0, description="Sampling rate, Hz")
    fs_attr: str | None = Field(
        default=None, description="Dataset attribute holding the scalar sampling rate"
    )
    units_attr: str | None = Field(
        default=None, description="Dataset attribute holding the unit string (for units='auto')"
    )

    @model_validator(mode="after")
    def _check_rate(self) -> Hdf5Spec:
        if self.fs_hz is None and self.fs_attr is None:
            msg = "hdf5 excitation needs either fs_hz or fs_attr"
            raise ValueError(msg)
        return self


ExcitationSpec = Annotated[
    SineSpec
    | MultitoneSpec
    | SweepSpec
    | RandomSpec
    | ShockSpec
    | CsvSpec
    | WavSpec
    | TdmsSpec
    | UffSpec
    | MatSpec
    | Hdf5Spec,
    Field(discriminator="kind"),
]
"""Discriminated union of all excitation specs, selected by ``kind``.

The S0 form ``ExcitationSpec(kind="sine", ...)`` maps one-to-one onto
:class:`SineSpec`, so existing scenarios (``examples/hello.yaml``) parse
unchanged. S1 added CSV/WAV replay; S8 adds the instrument formats
(:class:`TdmsSpec`, :class:`UffSpec`, :class:`MatSpec`, :class:`Hdf5Spec`)
behind the same loader registry (seam SW-08).
"""


class DspOptions(_Frozen):
    """Inverse/DSP options (S5).

    Attributes
    ----------
    integrator : {"frequency", "time"}
        Kinematic integrator method ``a -> v -> x`` (registry key, S5 §2).
        ``"frequency"`` is spectral ``1/(j omega)`` with a high-pass mask;
        ``"time"`` is cumulative-trapezoid with a Butterworth detrend.
    spectrum_method : {"fft", "welch"}
        Spectral estimator for the representative spectrum and dominant-peak
        search (S5 §3): single rFFT amplitude or a Welch PSD.
    window : str
        Window name for the Welch PSD and windowed amplitude spectrum
        (default ``"hann"``).
    f_hp_hz : float or None
        High-pass cut-off for the integrators, Hz; ``None`` uses the variant's
        lower band edge (doc 08). Removes the double-integration drift.
    welch_nperseg : int or None
        Welch segment length; ``None`` lets the estimator choose from the record
        length.
    welch_noverlap : int or None
        Welch segment overlap; ``None`` uses ``nperseg // 2``.
    calibration : {"ideal", "bench"}
        Calibration source (S5 §1, SW-33 axis A): ``"ideal"`` computes
        ``s_target`` from the config and the S3/S2 models (known exactly, the v1
        default); ``"bench"`` is the stand-estimated sensitivity (helper for S6;
        live-pipeline bench is deferred, 14 §8).
    sensitivity_model : {"static", "operating_point", "nonlinear_curve"}
        Operating-point binding strategy (SW-33 axis B; key into
        ``optivibe.dsp.sensitivity.SENSITIVITY_REGISTRY``). ``"static"`` (the v1
        default) is the scalar at the nominal bias; ``"operating_point"``
        recomputes ``s_target`` at the SNR-optimum bias (0.37 rule);
        ``"nonlinear_curve"`` inverts ``eta(dx)`` point-wise for the >50 g study.
    sensitivity_freq : {"plateau", "dynamic"}
        Frequency treatment of the sensitivity (SW-33 axis C): ``"plateau"`` (the
        v1 default) uses the QS scalar; ``"dynamic"`` rolls it up by ``D(f)`` near
        ``f1`` (applied via the ``deconvolve_hlat`` mechanism).
    deconvolve_hlat : bool
        Whether to divide out ``|H_lat(f)|`` to approach ``f1`` (S5 §1); the
        default ``False`` uses the flat plateau scalar (the off-resonance mode).
        ``sensitivity_freq="dynamic"`` enables the same correction.
    iso_machine_class : str
        ISO 10816-3 machine class for the severity assessment (key into
        ``optivibe.dsp.iso.ISO_10816_3_ZONES``; default ``"group2_rigid"``).
    """

    integrator: Literal["frequency", "time"] = "frequency"
    spectrum_method: Literal["fft", "welch"] = "fft"
    window: str = "hann"
    f_hp_hz: float | None = Field(default=None, gt=0.0, description="HP cut-off, Hz (band if None)")
    welch_nperseg: int | None = Field(default=None, gt=0, description="Welch segment length")
    welch_noverlap: int | None = Field(default=None, ge=0, description="Welch segment overlap")
    calibration: Literal["ideal", "bench"] = "ideal"
    sensitivity_model: Literal["static", "operating_point", "nonlinear_curve"] = "static"
    sensitivity_freq: Literal["plateau", "dynamic"] = "plateau"
    deconvolve_hlat: bool = False
    iso_machine_class: str = "group2_rigid"


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


class DetectorOptions(_Frozen):
    """Per-scenario overrides of the detector stage (S4).

    Mirrors the :class:`MechanicsOptions` pattern: only explicitly set fields are
    forwarded to the detector implementation by the orchestrator; the
    option-less ``"stub"`` detector keeps constructing with no arguments. The
    noise seed is *not* a field here -- it is derived from the scenario-level
    ``seed`` and injected by the orchestrator (the ``run()`` protocol carries no
    seed; doc 10 §8).

    Attributes
    ----------
    balanced : bool or None
        Override of the variant's balanced-channel flag
        (``variant.detector.balanced``); the variant value is used when ``None``.
        Setting ``balanced: false`` exposes the unsuppressed RIN (doc 07 §1.2).
    reference_arm : {"matched", "bright"} or None
        Override of the variant's reference-arm shot model
        (``variant.detector.reference_arm``); the variant value is used when
        ``None``. Flip between ``"matched"`` (conservative two-arm floor) and
        ``"bright"`` (datasheet shot limit) to compare NEA conventions at test
        time (open question O-SW-08).
    """

    balanced: bool | None = Field(
        default=None, description="Balanced-channel override (variant value if None)"
    )
    reference_arm: Literal["matched", "bright"] | None = Field(
        default=None, description="Reference-arm shot model override (variant value if None)"
    )


class StageSelection(_Frozen):
    """Registry keys selecting the implementation of each stage (09 §6).

    Defaults point to the physical implementations where they exist ("modal"
    mechanics since S2, "cylinder" optics since S3) and to the S0 stubs
    elsewhere; the stubs remain registered for regression under their explicit
    keys.
    """

    excitation: StageKey = "sine"
    mechanics: StageKey = "modal"
    optics: StageKey = "cylinder"
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
    detector : DetectorOptions
        Per-scenario detector overrides (S4).
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
    detector: DetectorOptions = DetectorOptions()
    dsp: DspOptions = DspOptions()
    seed: int | None = None
    output: OutputSpec = OutputSpec()
