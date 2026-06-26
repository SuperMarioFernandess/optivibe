"""Editable per-subsystem configuration models and variant composition (S9-A).

This module introduces the **editable building blocks** of a sensor variant:
one pydantic model per physical subsystem (source, fiber line, cantilever,
reflector, detector) and a :class:`SystemConfig` that *composes* a variant from
named subsystem presets plus per-subsystem overrides (task S9-A; doc 09 §7).

Two layers, on purpose
----------------------
* **Editable layer (this module).** ``SourceConfig`` / ``FiberConfig`` /
  ``CantileverConfig`` / ``ReflectorConfig`` / ``DetectorConfig`` group the
  parameters by the subsystem that physically owns them. These are what users
  edit and what presets store (``configs/presets/<subsystem>/*.yaml``).
* **Resolved layer (:mod:`optivibe.core.config.models`).** Every stage reads its
  inputs off :class:`~optivibe.core.config.models.VariantConfig` (e.g.
  ``variant.responsivity_a_w``, ``variant.optics.gap_m``). That model is the
  *flat internal config* of task S9-A §2 and is **left byte-for-byte
  unchanged**, so :meth:`SystemConfig.resolve` re-flattens the subsystem blocks
  back into a ``VariantConfig`` and the physical stages need **no edits**. This
  is the most conservative way to guarantee the A/B/C/D bit-identity required by
  the task.

Field provenance
----------------
Each field carries its knowledge-base reference and SI unit in the docstring.
The redistribution relative to the S8 flat ``VariantConfig`` is:

==========================  ===================  ====================================
Subsystem field             SI unit              Resolved ``VariantConfig`` target
==========================  ===================  ====================================
``SourceConfig.source_kind``  --                 ``source.kind`` (doc 08 §6)
``SourceConfig.wavelength_m`` m                  ``source.wavelength_m`` (doc 03 §1)
``SourceConfig.power_w``      W                  ``source.power_w`` (doc 07 §2)
``SourceConfig.rin_db_hz``    dB/Hz              ``source.rin_db_hz`` (doc 07 §1.2)
``FiberConfig.mode_field_radius_m``  m           ``optics.mode_field_radius_m`` (03 §1)
``FiberConfig.fresnel_R1``    --                 ``endface_reflectivity`` (doc 04 §4)
``FiberConfig.clad_diameter_m``  m               informational (see note)
``CantileverConfig.length_m`` m                  ``length_m`` (doc 02/08)
``CantileverConfig.material`` --                 informational (see note)
``ReflectorConfig.shape``     --                 ``reflector.shape`` (doc 08 §6)
``ReflectorConfig.curvature_radius_m``  m        ``reflector.radius_of_curvature_m``
``ReflectorConfig.metallization_rho``  --        ``reflector.reflectivity`` (rho, 0.98)
``ReflectorConfig.gap_m``     m                  ``optics.gap_m`` (doc 03 §6)
``ReflectorConfig.bias_offset_m``  m             ``optics.bias_offset_m`` (doc 03 §5)
``DetectorConfig.responsivity``  A/W             ``responsivity_a_w`` (doc 07 §2)
``DetectorConfig.*`` (front-end/ADC)  --         ``detector.*`` (docs 07/08; S4)
==========================  ===================  ====================================

Note (informational fields). In v1 the mechanics stage reads the cantilever
material and the cladding diameter from the **global** constants
(``configs/constants.yaml``, doc 01 §4), not from the variant. ``material`` and
``clad_diameter_m`` therefore document the assumed values of the building block
and are not consumed by the resolved ``VariantConfig``; wiring per-subsystem
material/geometry into the mechanics constants path is a deferred loop (doc 14
§8, S9-B). They are validated for positivity so a typo still fails loudly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:  # avoid importing the heavy resolved model at runtime import time
    from optivibe.core.config.models import VariantConfig
    from optivibe.core.config.presets import PresetStore

# Reflector shapes with a registered optics implementation (doc 03; S3/S9-B).
# S9-B added the sphere/plane/wedge family behind the optics shape layer
# (:mod:`optivibe.optics.reflector`); each maps to a registered model factory in
# :data:`optivibe.optics.REFLECTOR_MODEL_REGISTRY`. A composition naming any
# other shape is rejected at composition time.
REGISTERED_REFLECTOR_SHAPES: frozenset[str] = frozenset({"cylinder", "sphere", "plane", "wedge"})

# Paraxial validity guards of the cylinder optics (doc 03 §6). Mirrors the
# runtime constants of :mod:`optivibe.optics.cylinder` so a bad composition
# fails *early*, at config time, with the same numeric thresholds.
_MIN_RADIUS_PER_WAIST = 5.0
_MAX_SPOT_PER_RADIUS = 1.0 / 3.0

# Paraxial range of the built-in wedge angle (doc 03 §6). Mirrors
# :data:`optivibe.optics.wedge.MAX_WEDGE_ANGLE_RAD` (kept in sync; the config
# layer does not import the optics layer, see the module note).
_MAX_WEDGE_ANGLE_RAD = 0.15


class _SubsystemBase(BaseModel):
    """Base model for all subsystem blocks: immutable, rejects unknown fields.

    ``extra="forbid"`` means an override with a misspelled key (or a YAML typo)
    raises at load time rather than being silently dropped (10 §7).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #
# Subsystem building blocks (editable; one model per physical subsystem).
# --------------------------------------------------------------------------- #
class SourceConfig(_SubsystemBase):
    """Optical source subsystem (doc 08 §6; R-13/R-15/R-30).

    Attributes
    ----------
    source_kind : {"SLD", "DFB"}
        Source type. A/B/C use a broadband SLD (route 2, coherent wash-out); D
        uses a narrow DFB (route 1, doc 08 §6).
    wavelength_m : float
        Centre wavelength lambda, m (doc 03 §1; 1550 nm common platform R-40).
    power_w : float
        Optical power P delivered to the fiber, W (doc 07 §2; 16-100 mW range).
    rin_db_hz : float
        Relative intensity noise, dB/Hz (doc 07 §1.2).
    """

    source_kind: Literal["SLD", "DFB"] = "SLD"
    wavelength_m: float = Field(gt=0.0, description="Source wavelength lambda, m (doc 03 §1)")
    power_w: float = Field(gt=0.0, description="Optical power P, W (doc 07 §2)")
    rin_db_hz: float = Field(description="Relative intensity noise, dB/Hz (doc 07 §1.2)")


class FiberMaterial(_SubsystemBase):
    """Fiber/cantilever bulk material (fused silica; doc 01 §4.1).

    Attributes
    ----------
    young_modulus_Pa : float
        Young's modulus E, Pa (doc 01 §4.1; 72 GPa for fused silica).
    density_kg_m3 : float
        Mass density rho, kg/m^3 (doc 01 §4.1; 2201 kg/m^3).
    """

    young_modulus_Pa: float = Field(gt=0.0, description="Young's modulus E, Pa (doc 01 §4.1)")
    density_kg_m3: float = Field(gt=0.0, description="Density rho, kg/m^3 (doc 01 §4.1)")


class FiberConfig(_SubsystemBase):
    """Fiber-line subsystem: the waveguide that carries and re-collects the beam.

    Attributes
    ----------
    clad_diameter_m : float
        Cladding (outer) diameter D, m (doc 01 §4.1; 125 um SMF-28).
        Informational in v1 -- the mechanics reads ``constants.fiber.diameter_m``
        (see the module note).
    mode_field_radius_m : float
        Gaussian mode-field radius w0 at the source wavelength, m (doc 03 §1;
        5.2 um at 1550 nm). Tied to the fiber mode, hence owned by the fiber
        subsystem; the resolver feeds it to ``optics.mode_field_radius_m``.
    fresnel_R1 : float
        Endface Fresnel reflectivity R1 (doc 04 §4; 0.036 bare, lower with AR).
        Feeds ``VariantConfig.endface_reflectivity``.
    """

    clad_diameter_m: float = Field(
        default=125.0e-6, gt=0.0, description="Cladding diameter D, m (doc 01 §4.1)"
    )
    mode_field_radius_m: float = Field(
        default=5.2e-6, gt=0.0, description="Mode-field radius w0, m (doc 03 §1)"
    )
    fresnel_R1: float = Field(
        default=0.036, ge=0.0, le=1.0, description="Endface Fresnel reflectivity R1 (doc 04 §4)"
    )


class CantileverConfig(_SubsystemBase):
    """Cantilever subsystem: the fiber clamped-free beam (doc 02; doc 08 §6).

    Attributes
    ----------
    length_m : float
        Free length L of the cantilever, m (doc 02; sets f1 ~ 1/L^2). Feeds
        ``VariantConfig.length_m``.
    material : FiberMaterial
        Bulk material (E, rho). Informational in v1 -- the mechanics reads the
        global ``constants.fiber`` block (see the module note).
    """

    length_m: float = Field(gt=0.0, description="Cantilever length L, m (doc 02)")
    material: FiberMaterial = FiberMaterial(young_modulus_Pa=72.0e9, density_kg_m3=2201.0)


class ReflectorConfig(_SubsystemBase):
    """Reflector subsystem: convex mirror + air gap working point (docs 03/08).

    Groups the mirror geometry/coating with the optical *working point* it sets
    (the air gap A and the static de-centering Delta x0), because both are
    properties of the reflector-and-gap assembly rather than of the fiber.

    Attributes
    ----------
    shape : {"cylinder", "sphere", "wedge", "plane"}
        Reflector profile. Each has a registered optics model
        (:data:`REGISTERED_REFLECTOR_SHAPES`; S9-B): cylinder (1-axis, curved in
        x), sphere (isotropic, curved in both planes), wedge (tilted plane,
        angular bias) and plane (the ``R_c -> inf`` reference). Other values are
        rejected at composition time.
    curvature_radius_m : float or None
        Radius of curvature R_c, m (doc 08 §6). Required for cylinder/sphere;
        ``None`` for the flat plane/wedge (they ignore curvature). Feeds
        ``reflector.radius_of_curvature_m``.
    metallization_rho : float
        Mirror reflectivity rho (doc 08 §6; 0.98 metallized). Feeds
        ``reflector.reflectivity``.
    gap_m : float
        Nominal one-way air gap A between endface and mirror, m (doc 03 §6;
        parametric band 20-40 um). Feeds ``optics.gap_m``.
    bias_offset_m : float
        Intentional static de-centering Delta x0 setting the working point on
        the slope, m (doc 03 §5). Feeds ``optics.bias_offset_m``. Ignored by the
        flat plane/wedge (no displacement coupling); the sphere applies it
        radially.
    wedge_angle_rad : float or None
        Built-in wedge face-tilt angle alpha_w, rad (doc 03 §c). Required for
        the wedge, ``None`` otherwise. Feeds ``optics.wedge_angle_rad`` (the
        shape parameter flows through the optics block, task S9-B §3).
    """

    shape: Literal["cylinder", "sphere", "wedge", "plane"] = "cylinder"
    curvature_radius_m: float | None = Field(
        default=None, gt=0.0, description="Radius of curvature R_c, m (doc 08 §6; None for plane)"
    )
    metallization_rho: float = Field(
        gt=0.0, le=1.0, description="Mirror reflectivity rho (doc 08 §6)"
    )
    gap_m: float = Field(default=31.0e-6, gt=0.0, description="Air gap A, m (doc 03 §6)")
    bias_offset_m: float = Field(
        default=2.0e-6, ge=0.0, description="Working-point de-centering Delta x0, m (doc 03 §5)"
    )
    wedge_angle_rad: float | None = Field(
        default=None,
        description="Built-in wedge face-tilt angle alpha_w, rad (wedge only; doc 03 §c)",
    )

    @model_validator(mode="after")
    def _check_shape_params(self) -> ReflectorConfig:
        """Per-shape parameter requirements, failing loudly on bad configs (10 §7).

        * cylinder / sphere need a finite ``curvature_radius_m`` (they are
          curved); plane and wedge are flat and leave it ``None``;
        * wedge needs ``wedge_angle_rad``; every other shape must leave it
          ``None`` (a wedge angle on a non-wedge is almost certainly a mistake).
        """
        needs_radius = self.shape in ("cylinder", "sphere")
        if needs_radius and self.curvature_radius_m is None:
            msg = f"reflector.shape {self.shape!r} requires curvature_radius_m (doc 08 §6)"
            raise ValueError(msg)
        if self.shape == "wedge" and self.wedge_angle_rad is None:
            msg = "reflector.shape 'wedge' requires wedge_angle_rad (doc 03 §c)"
            raise ValueError(msg)
        if self.shape != "wedge" and self.wedge_angle_rad is not None:
            msg = f"wedge_angle_rad is only valid for shape 'wedge', not {self.shape!r}"
            raise ValueError(msg)
        return self


class DetectorConfig(_SubsystemBase):
    """Detector subsystem: photodiode responsivity, balanced front-end, ADC.

    Holds the read-out responsivity (which the S8 flat model kept on the variant
    top level) together with the balanced-channel / transimpedance / ADC
    parameters of the S4 detector stage (docs 07/08).

    Attributes
    ----------
    responsivity : float
        Photodiode responsivity R, A/W (doc 07 §2; 1.0 A/W reference). Feeds
        ``VariantConfig.responsivity_a_w``.
    balanced : bool
        Whether the balanced reference channel is active (doc 07 §1.2).
    reference_arm : {"matched", "bright"}
        Reference-arm shot model of the balanced channel (open question
        O-SW-08; doc 07 §1.2).
    cmrr_db : float
        Common-mode rejection ratio of the balanced channel, dB (doc 07 §1.2).
    transimpedance_ohm : float or None
        Feedback resistor Rf, ohm; ``None`` removes the Johnson floor (doc 07
        §1.3).
    output : {"current", "voltage"}
        Whether digitized samples are photocurrent (A) or TIA voltage (V).
    adc_bits : int
        ADC resolution, bits (doc 07 §1.4).
    adc_full_scale : float
        AC full scale (+/- range) in the output units (doc 07 §1.4).
    adc_fs_hz : float or None
        ADC/decimation rate, Hz; ``None`` keeps the optical fs (identity).
    antialias : bool
        Whether to anti-alias-filter before decimation (doc 11 §2).
    rin_shape : {"white"}
        Spectral form of the RIN PSD (doc 07 §1.2).
    """

    responsivity: float = Field(gt=0.0, description="PD responsivity R, A/W (doc 07 §2)")
    balanced: bool = True
    reference_arm: Literal["matched", "bright"] = "matched"
    cmrr_db: float = Field(default=40.0, description="Balanced-channel CMRR, dB (doc 07 §1.2)")
    transimpedance_ohm: float | None = Field(
        default=1.0e5, gt=0.0, description="Feedback resistor Rf, ohm (doc 07 §1.3)"
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


# --------------------------------------------------------------------------- #
# Composition: a variant = subsystem presets + per-subsystem overrides.
# --------------------------------------------------------------------------- #
class SubsystemRef(_SubsystemBase):
    """Reference to one subsystem preset plus inline overrides (task S9-A §2).

    Attributes
    ----------
    preset : str
        Name of a built-in or user preset for this subsystem (file stem under
        ``configs/presets/<subsystem>/`` or ``configs/user/presets/<subsystem>/``).
    overrides : dict
        Field-level overrides merged on top of the preset. Keys must be valid
        fields of the subsystem model (unknown keys fail loudly via
        ``extra="forbid"`` when the merged block is validated).
    """

    preset: str = Field(min_length=1, description="Subsystem preset name")
    overrides: dict[str, Any] = Field(default_factory=dict, description="Field overrides")


class BandRange(_SubsystemBase):
    """Target frequency band of a composed variant (doc 08 §6).

    Attributes
    ----------
    f_min_hz : float
        Lower band edge, Hz.
    f_max_hz : float
        Upper band edge, Hz.
    """

    f_min_hz: float = Field(gt=0.0, description="Lower band edge, Hz")
    f_max_hz: float = Field(gt=0.0, description="Upper band edge, Hz")


class SystemConfig(_SubsystemBase):
    """Composed sensor configuration: subsystem presets + system-level scalars.

    A ``SystemConfig`` is the editable, composable description of a variant. It
    names one preset (with optional overrides) per subsystem and carries the
    handful of variant-level scalars that do not belong to a single subsystem
    (identity, operating regime, full scale, route, quality factor, ...).
    :meth:`resolve` turns it into the flat
    :class:`~optivibe.core.config.models.VariantConfig` that the stages read.

    Attributes
    ----------
    name : str
        Composition identity. For the four built-ins this is ``"A"``..``"D"``;
        user compositions may use any non-empty name.
    description : str
        Human-readable description / class.
    mode : {"offresonance", "resonance"}
        Operating regime (doc 08 §6).
    band : BandRange
        Target frequency band.
    line_freq_hz : float or None
        Resonant line frequency for ``mode == "resonance"`` (else ``None``).
    full_scale_g : float
        Full-scale acceleration FS, g (doc 08 §1.3).
    route : {1, 2}
        Endface treatment route (2 = coherent wash-out; 1 = AR + DFB; doc 08).
    eta_bias : float
        Optical working-point efficiency eta0 used by the *stub* optics (S0);
        the physical cylinder optics computes its own eta0 (doc 03 §5).
    q_total : float
        Total mechanical quality factor Q of mode 1 at this variant's length.
        Variant-specific (depends on L, vacuum, mounting), so it lives here
        rather than in the reusable cantilever preset (docs 07/08).
    target_nea_ug_rthz : float or None
        Target noise-equivalent acceleration, ug/sqrt(Hz) (placeholder, O-09).
    vacuum : bool
        Whether the variant is operated under vacuum (A/D option).
    source, fiber, cantilever, reflector, detector : SubsystemRef
        Per-subsystem preset references with overrides.
    """

    name: str = Field(min_length=1, description="Composition name")
    description: str = ""
    mode: Literal["offresonance", "resonance"] = "offresonance"
    band: BandRange
    line_freq_hz: float | None = None
    full_scale_g: float = Field(gt=0.0, description="Full-scale acceleration FS, g")
    route: Literal[1, 2] = 2
    eta_bias: float = Field(gt=0.0, le=1.0, description="Optical bias eta0 (stub optics)")
    q_total: float = Field(gt=0.0, description="Total quality factor Q of mode 1")
    target_nea_ug_rthz: float | None = Field(default=None, gt=0.0)
    vacuum: bool = False

    source: SubsystemRef
    fiber: SubsystemRef
    cantilever: SubsystemRef
    reflector: SubsystemRef
    detector: SubsystemRef

    @model_validator(mode="after")
    def _check_mode(self) -> SystemConfig:
        if self.mode == "resonance" and self.line_freq_hz is None:
            msg = "resonant composition requires line_freq_hz"
            raise ValueError(msg)
        if self.band.f_max_hz <= self.band.f_min_hz:
            msg = f"f_max_hz ({self.band.f_max_hz}) must exceed f_min_hz ({self.band.f_min_hz})"
            raise ValueError(msg)
        return self

    # ----------------------------------------------------------------- #
    # Resolution to the flat VariantConfig read by the stages.
    # ----------------------------------------------------------------- #
    def resolve(self, store: PresetStore) -> VariantConfig:
        """Resolve presets + overrides into a flat :class:`VariantConfig`.

        The subsystem blocks are built (preset then overrides), the
        cross-subsystem geometry guards are checked, and the values are
        re-flattened into the exact field layout the stages read -- so the
        result is bit-identical to the equivalent S8 flat variant.

        Parameters
        ----------
        store : PresetStore
            Resolver for ``{subsystem, preset_name} -> subsystem model``.

        Returns
        -------
        VariantConfig
            The flat, validated variant configuration.

        Raises
        ------
        ValueError
            If a preset is unknown, an override key is invalid, or the composed
            geometry violates the paraxial guards (R_c >= 5 w0, w(A) <= R_c/3)
            or names an unregistered reflector shape.
        """
        from optivibe.core.config.models import VariantConfig

        src = store.build_source(self.source)
        fib = store.build_fiber(self.fiber)
        can = store.build_cantilever(self.cantilever)
        ref = store.build_reflector(self.reflector)
        det = store.build_detector(self.detector)

        _check_composition_geometry(src, fib, ref)

        variant_dict: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "mode": self.mode,
            "band": {"f_min_hz": self.band.f_min_hz, "f_max_hz": self.band.f_max_hz},
            "line_freq_hz": self.line_freq_hz,
            "length_m": can.length_m,
            "full_scale_g": self.full_scale_g,
            "reflector": {
                "shape": ref.shape,
                "radius_of_curvature_m": ref.curvature_radius_m,
                "reflectivity": ref.metallization_rho,
            },
            "source": {
                "kind": src.source_kind,
                "wavelength_m": src.wavelength_m,
                "power_w": src.power_w,
                "rin_db_hz": src.rin_db_hz,
            },
            "route": self.route,
            "responsivity_a_w": det.responsivity,
            "endface_reflectivity": fib.fresnel_R1,
            "eta_bias": self.eta_bias,
            "optics": {
                "gap_m": ref.gap_m,
                "bias_offset_m": ref.bias_offset_m,
                "mode_field_radius_m": fib.mode_field_radius_m,
                "wedge_angle_rad": ref.wedge_angle_rad,
            },
            "detector": {
                "balanced": det.balanced,
                "reference_arm": det.reference_arm,
                "cmrr_db": det.cmrr_db,
                "transimpedance_ohm": det.transimpedance_ohm,
                "output": det.output,
                "adc_bits": det.adc_bits,
                "adc_full_scale": det.adc_full_scale,
                "adc_fs_hz": det.adc_fs_hz,
                "antialias": det.antialias,
                "rin_shape": det.rin_shape,
            },
            "q_total": self.q_total,
            "target_nea_ug_rthz": self.target_nea_ug_rthz,
            "vacuum": self.vacuum,
        }
        return VariantConfig.model_validate(variant_dict)


# --------------------------------------------------------------------------- #
# Cross-subsystem geometry guards (doc 03 §6; mirror of optics.cylinder).
# --------------------------------------------------------------------------- #
def _spot_radius_m(waist_radius_m: float, wavelength_m: float, gap_m: float) -> float:
    """Gaussian spot radius ``w(A) = w0 sqrt(1 + (A/zR)^2)``, m (doc 03 §1).

    Replicates :meth:`optivibe.optics.gaussian.GaussianBeam.spot_radius_m` with
    ``zR = pi w0^2 / lambda`` so the composition-time guard uses the exact same
    formula as the runtime optics model (no import cycle into the optics layer).

    Parameters
    ----------
    waist_radius_m : float
        Mode-field radius w0, m.
    wavelength_m : float
        Wavelength lambda, m.
    gap_m : float
        One-way gap A, m.

    Returns
    -------
    float
        Spot radius w(A), m.
    """
    rayleigh_range_m = math.pi * waist_radius_m**2 / wavelength_m
    return waist_radius_m * math.sqrt(1.0 + (gap_m / rayleigh_range_m) ** 2)


def _check_composition_geometry(
    source: SourceConfig, fiber: FiberConfig, reflector: ReflectorConfig
) -> None:
    """Validate the composed reflector+fiber+source geometry per shape (doc 03 §6).

    Enforces, at composition time, the same guards the optics models enforce at
    run time, specialised by reflector shape:

    * **cylinder / sphere** (curved): the shape must be registered, the mirror
      wide relative to the mode (``R_c >= 5 w0``) and the spot must fit the
      mirror (``w(A) <= R_c/3``);
    * **wedge** (tilted plane): only the paraxial wedge-angle range
      (``|alpha_w| <= 0.15 rad``); the gap positivity is already a field guard;
    * **plane** (``R_c -> inf``): only the gap guard (a field guard) -- the
      finite-aperture guard is vacuous for an infinite plane.

    Cross-subsystem because it combines the reflector (R_c, A, alpha_w), the
    fiber (w0) and the source (lambda).

    Parameters
    ----------
    source : SourceConfig
        Resolved source block (provides lambda).
    fiber : FiberConfig
        Resolved fiber block (provides w0).
    reflector : ReflectorConfig
        Resolved reflector block (provides shape, R_c, A, alpha_w).

    Raises
    ------
    ValueError
        If the shape is unregistered, or a per-shape geometry guard fails.
    """
    shape = reflector.shape
    if shape not in REGISTERED_REFLECTOR_SHAPES:
        registered = ", ".join(sorted(REGISTERED_REFLECTOR_SHAPES))
        msg = f"reflector.shape {shape!r} is not registered; available: {registered} (doc 14 §8)"
        raise ValueError(msg)

    w0 = fiber.mode_field_radius_m

    if shape in ("cylinder", "sphere"):
        radius = reflector.curvature_radius_m
        if radius is None:  # defensive: the ReflectorConfig validator enforces this
            msg = f"reflector.shape {shape!r} requires curvature_radius_m (doc 08 §6)"
            raise ValueError(msg)
        if radius < _MIN_RADIUS_PER_WAIST * w0:
            msg = (
                f"R_c = {radius:.3e} m violates the paraxial guard "
                f"R_c >= {_MIN_RADIUS_PER_WAIST:g} w0 = {_MIN_RADIUS_PER_WAIST * w0:.3e} m "
                f"(doc 03 §6)"
            )
            raise ValueError(msg)
        spot = _spot_radius_m(w0, source.wavelength_m, reflector.gap_m)
        if spot > _MAX_SPOT_PER_RADIUS * radius:
            msg = (
                f"spot w(A) = {spot:.3e} m exceeds R_c/3 = "
                f"{_MAX_SPOT_PER_RADIUS * radius:.3e} m (doc 03 §6)"
            )
            raise ValueError(msg)
    elif shape == "wedge":
        angle = reflector.wedge_angle_rad
        if angle is None:  # defensive: the ReflectorConfig validator enforces this
            msg = "reflector.shape 'wedge' requires wedge_angle_rad (doc 03 §c)"
            raise ValueError(msg)
        if abs(angle) > _MAX_WEDGE_ANGLE_RAD:
            msg = (
                f"|wedge_angle_rad| = {abs(angle):.3e} rad exceeds the paraxial range "
                f"{_MAX_WEDGE_ANGLE_RAD:g} rad (doc 03 §6)"
            )
            raise ValueError(msg)
    # plane: R_c -> inf, only the gap guard (already enforced by gap_m > 0).
