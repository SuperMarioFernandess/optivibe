"""Editable subsystem-composition forms (task S7-mod §1/§2).

Replaces the old "pick a variant A/B/C/D" combo with an **editable
composition**: one form per physical subsystem (source / fiber / cantilever /
reflector / detector), each with a preset selector (from
:class:`~optivibe.core.config.presets.PresetStore`) and labelled, unit-carrying
override fields, plus the system-level scalars and a reflector *shape* selector
with dynamic per-shape parameters. The A/B/C/D variants remain as **starting
compositions** that seed the forms.

Thin shell (09 §9): this widget only collects values into a *payload* mapping;
:func:`optivibe.gui.controllers.system_builder.build_system_config` validates it
into a frozen :class:`~optivibe.core.config.subsystems.SystemConfig`, and the
worker resolves that into the flat variant off the UI thread (SW-06). No physics
here. Every field carries a tooltip with its physical meaning and a knowledge-base
reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from optivibe.core.config.loader import default_config_dir
from optivibe.core.config.presets import (
    PresetStore,
    load_system_file,
    save_system_config,
)
from optivibe.core.config.subsystems import SubsystemRef
from optivibe.core.logging import get_logger
from optivibe.gui.controllers.system_builder import (
    build_system_config,
    subsystem_defaults,
)

logger = get_logger(__name__)

__all__ = ["SystemBuilderPanel"]

_STARTING = ("A", "B", "C", "D")
_REFLECTOR_SHAPES = ("cylinder", "sphere", "plane", "wedge")

# Per-subsystem editable override fields, as
# (key, label, unit, tooltip). Floats are entered in SI via a line edit (so a
# value like ``1.55e-6`` is unambiguous); the unit is shown in the label.
_FieldSpec = tuple[str, str, str, str]

_SOURCE_FIELDS: tuple[_FieldSpec, ...] = (
    ("wavelength_m", "wavelength lambda", "m", "Centre wavelength (doc 03 §1; 1550 nm)"),
    ("power_w", "optical power P", "W", "Power delivered to the fiber (doc 07 §2)"),
    ("rin_db_hz", "RIN", "dB/Hz", "Relative intensity noise (doc 07 §1.2)"),
)
_FIBER_FIELDS: tuple[_FieldSpec, ...] = (
    ("mode_field_radius_m", "mode-field radius w0", "m", "Gaussian mode radius (doc 03 §1)"),
    ("fresnel_R1", "endface reflectivity R1", "-", "Fresnel reflectivity (doc 04 §4)"),
    ("clad_diameter_m", "cladding diameter D", "m", "Outer diameter (doc 01 §4.1; informational)"),
)
_REFLECTOR_FIELDS: tuple[_FieldSpec, ...] = (
    ("metallization_rho", "reflectivity rho", "-", "Mirror reflectivity (doc 08 §6; 0.98)"),
    ("gap_m", "air gap A", "m", "Nominal one-way gap (doc 03 §6; 20-40 um)"),
    ("bias_offset_m", "bias Delta x0", "m", "Working-point de-centering (doc 03 §5)"),
)
_DETECTOR_FIELDS: tuple[_FieldSpec, ...] = (
    ("responsivity", "responsivity R", "A/W", "Photodiode responsivity (doc 07 §2)"),
    ("cmrr_db", "CMRR", "dB", "Balanced-channel rejection (doc 07 §1.2)"),
    ("adc_full_scale", "ADC full scale", "out", "AC +/- range in output units (doc 07 §1.4)"),
)


def _line(value: str = "") -> QLineEdit:
    """Build a line edit pre-filled with ``value``."""
    edit = QLineEdit()
    edit.setText(value)
    return edit


def _fmt(value: object) -> str:
    """Format a numeric field value for a line edit (compact, lossless)."""
    if isinstance(value, float):
        return repr(value)
    return str(value)


class _SubsystemForm(QGroupBox):
    """One subsystem: a preset selector plus labelled override line edits.

    Parameters
    ----------
    title : str
        Group-box title.
    subsystem : str
        Subsystem name (``"source"``, ...).
    fields : tuple
        Field specs ``(key, label, unit, tooltip)`` for the override line edits.
    store : PresetStore
        Preset resolver (for the preset list and reseeding).
    """

    def __init__(
        self,
        title: str,
        subsystem: str,
        fields: tuple[_FieldSpec, ...],
        store: PresetStore,
    ) -> None:
        super().__init__(title)
        self._subsystem = subsystem
        self._fields = fields
        self._store = store
        self._edits: dict[str, QLineEdit] = {}

        self._preset = QComboBox()
        self._reload_presets()
        self._preset.currentTextChanged.connect(self._on_preset_changed)

        self._form = QFormLayout(self)
        self._form.addRow("preset", self._preset)
        for key, label, unit, tip in fields:
            edit = _line()
            edit.setToolTip(tip)
            self._edits[key] = edit
            self._form.addRow(f"{label} [{unit}]", edit)

    def _reload_presets(self) -> None:
        """Refresh the preset list (user presets may have appeared)."""
        current = self._preset.currentText()
        self._preset.blockSignals(True)
        self._preset.clear()
        names = sorted(self._store.list_presets(self._subsystem))
        self._preset.addItems(names)
        if current in names:
            self._preset.setCurrentText(current)
        self._preset.blockSignals(False)

    def _on_preset_changed(self, name: str) -> None:
        """Reseed the override fields from the newly chosen bare preset."""
        if name:
            self._reseed(name, {})

    def _reseed(self, preset: str, overrides: dict[str, Any]) -> None:
        """Set the preset and fill the fields from preset + overrides."""
        self._reload_presets()
        self._preset.blockSignals(True)
        self._preset.setCurrentText(preset)
        self._preset.blockSignals(False)
        try:
            values = subsystem_defaults(self._store, self._subsystem, preset)
        except (ValueError, KeyError) as exc:  # pragma: no cover - bad preset on disk
            logger.debug("could not seed %s/%s: %s", self._subsystem, preset, exc)
            return
        values.update(overrides)
        for key, edit in self._edits.items():
            if key in values and values[key] is not None:
                edit.setText(_fmt(values[key]))

    def seed_from_ref(self, ref: SubsystemRef) -> None:
        """Seed the form from a composition's subsystem reference."""
        self._reseed(ref.preset, dict(ref.overrides))

    def preset_name(self) -> str:
        """Return the selected preset name."""
        return self._preset.currentText()

    def overrides(self) -> dict[str, Any]:
        """Collect the override fields into a mapping (SI floats parsed)."""
        out: dict[str, Any] = {}
        for key, edit in self._edits.items():
            text = edit.text().strip()
            if text:
                out[key] = float(text)
        return out

    def ref_payload(self) -> dict[str, Any]:
        """Return the ``{preset, overrides}`` block for this subsystem."""
        return {"preset": self.preset_name(), "overrides": self.overrides()}


class _ReflectorForm(_SubsystemForm):
    """Reflector form with a shape selector and dynamic per-shape parameters.

    Adds the ``shape`` combo (cylinder/sphere/plane/wedge), the curvature radius
    ``R_c`` (cylinder/sphere only) and the wedge angle ``alpha_w`` (wedge only).
    The disabled parameters are sent as ``None`` overrides so a stale value from
    the starting preset cannot leak into the wrong shape (task S7-mod §2).
    """

    def __init__(self, store: PresetStore) -> None:
        super().__init__("Reflector", "reflector", _REFLECTOR_FIELDS, store)
        self._shape = QComboBox()
        self._shape.addItems(_REFLECTOR_SHAPES)
        self._shape.setToolTip("Reflector profile (doc 08 §6; S9-B): selects the optics model")
        self._rc = _line()
        self._rc.setToolTip("Radius of curvature R_c (cylinder/sphere; doc 08 §6)")
        self._alpha = _line()
        self._alpha.setToolTip("Built-in wedge face-tilt alpha_w (wedge only; doc 03 §c)")
        # Insert the shape controls right after the preset row.
        self._form.insertRow(1, "shape", self._shape)
        self._form.insertRow(2, "R_c [m]", self._rc)
        self._form.insertRow(3, "alpha_w [rad]", self._alpha)
        self._shape.currentTextChanged.connect(self._on_shape_changed)
        self._on_shape_changed(self._shape.currentText())

    def _on_shape_changed(self, shape: str) -> None:
        """Enable only the parameters the chosen shape uses."""
        curved = shape in ("cylinder", "sphere")
        self._rc.setEnabled(curved)
        self._alpha.setEnabled(shape == "wedge")

    def _reseed(self, preset: str, overrides: dict[str, Any]) -> None:
        """Reseed base fields then the shape-specific controls."""
        super()._reseed(preset, overrides)
        try:
            values = subsystem_defaults(self._store, self._subsystem, preset)
        except (ValueError, KeyError):  # pragma: no cover - bad preset on disk
            return
        values.update(overrides)
        shape = str(values.get("shape", "cylinder"))
        if hasattr(self, "_shape"):
            self._shape.setCurrentText(shape)
            rc = values.get("curvature_radius_m")
            self._rc.setText(_fmt(rc) if rc is not None else "")
            alpha = values.get("wedge_angle_rad")
            self._alpha.setText(_fmt(alpha) if alpha is not None else "")
            self._on_shape_changed(shape)

    def overrides(self) -> dict[str, Any]:
        """Collect base fields plus shape-conditional curvature / wedge angle."""
        out = super().overrides()
        shape = self._shape.currentText()
        out["shape"] = shape
        if shape in ("cylinder", "sphere"):
            rc = self._rc.text().strip()
            out["curvature_radius_m"] = float(rc) if rc else None
            out["wedge_angle_rad"] = None
        elif shape == "wedge":
            out["curvature_radius_m"] = None
            alpha = self._alpha.text().strip()
            out["wedge_angle_rad"] = float(alpha) if alpha else None
        else:  # plane
            out["curvature_radius_m"] = None
            out["wedge_angle_rad"] = None
        return out


class SystemBuilderPanel(QWidget):
    """Editable composition: subsystem forms + system scalars + save/load.

    Parameters
    ----------
    config_dir : pathlib.Path or None, optional
        Configuration root (presets + variants); defaults to the repository
        ``configs/``.
    parent : QWidget or None, optional
        Parent widget.
    """

    def __init__(self, config_dir: Path | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config_dir = config_dir or default_config_dir()
        self._store = PresetStore(self._config_dir)

        self._starting = QComboBox()
        self._starting.addItems(_STARTING)
        self._starting.setCurrentText("B")
        self._starting.currentTextChanged.connect(self._load_starting)

        # System-level scalars.
        self._name = _line("B")
        self._description = _line("general-purpose wideband")
        self._mode = QComboBox()
        self._mode.addItems(("offresonance", "resonance"))
        self._line_freq = _line()
        self._f_min = _line("1.0")
        self._f_max = _line("10000.0")
        self._full_scale = _line("50.0")
        self._route = QComboBox()
        self._route.addItems(("2", "1"))
        self._eta_bias = _line("0.25")
        self._q_total = _line("2610.0")
        self._target_nea = _line("10.0")
        self._vacuum = QCheckBox("vacuum")
        self._mode.currentTextChanged.connect(self._on_mode_changed)

        # Subsystem forms.
        self._source = _SubsystemForm("Source", "source", _SOURCE_FIELDS, self._store)
        self._fiber = _SubsystemForm("Fiber line", "fiber", _FIBER_FIELDS, self._store)
        self._cantilever = _SubsystemForm(
            "Cantilever",
            "cantilever",
            (("length_m", "length L", "m", "Free length; sets f1 ~ 1/L^2 (doc 02)"),),
            self._store,
        )
        self._reflector = _ReflectorForm(self._store)
        self._detector = _SubsystemForm("Detector", "detector", _DETECTOR_FIELDS, self._store)
        self._balanced = QCheckBox("balanced channel")
        self._reference_arm = QComboBox()
        self._reference_arm.addItems(("matched", "bright"))
        self._detector._form.addRow("balanced", self._balanced)
        self._detector._form.addRow("reference arm", self._reference_arm)
        self._adc_bits = QSpinBox()
        self._adc_bits.setRange(1, 32)
        self._adc_bits.setValue(24)
        self._detector._form.addRow("ADC bits", self._adc_bits)

        self._save_button = QPushButton("Save as...")
        self._load_button = QPushButton("Load...")
        self._save_button.clicked.connect(self._on_save)
        self._load_button.clicked.connect(self._on_load)

        layout = QVBoxLayout(self)
        layout.addWidget(self._system_group())
        layout.addWidget(self._source)
        layout.addWidget(self._fiber)
        layout.addWidget(self._cantilever)
        layout.addWidget(self._reflector)
        layout.addWidget(self._detector)
        io_row = QHBoxLayout()
        io_row.addWidget(self._save_button)
        io_row.addWidget(self._load_button)
        layout.addLayout(io_row)

        self._load_starting("B")

    def _system_group(self) -> QGroupBox:
        group = QGroupBox("System / composition")
        form = QFormLayout(group)
        form.addRow("starting composition", self._starting)
        form.addRow("name", self._name)
        form.addRow("description", self._description)
        form.addRow("mode", self._mode)
        form.addRow("line freq [Hz]", self._line_freq)
        band_row = QHBoxLayout()
        band_row.addWidget(QLabel("f_min"))
        band_row.addWidget(self._f_min)
        band_row.addWidget(QLabel("f_max"))
        band_row.addWidget(self._f_max)
        holder = QWidget()
        holder.setLayout(band_row)
        form.addRow("band [Hz]", holder)
        form.addRow("full scale [g]", self._full_scale)
        form.addRow("route", self._route)
        form.addRow("eta_bias (stub)", self._eta_bias)
        form.addRow("Q total", self._q_total)
        form.addRow("target NEA [ug/rtHz]", self._target_nea)
        form.addRow(self._vacuum)
        return group

    def _on_mode_changed(self, mode: str) -> None:
        """Enable the resonant line frequency only in resonance mode."""
        self._line_freq.setEnabled(mode == "resonance")

    # ------------------------------------------------------------------ #
    # Starting composition / load / save
    # ------------------------------------------------------------------ #
    def _load_starting(self, key: str) -> None:
        """Seed every form from the A/B/C/D starting composition ``key``."""
        path = self._config_dir / "variants" / f"{key}.yaml"
        try:
            system = load_system_file(path)
        except (FileNotFoundError, ValueError) as exc:  # pragma: no cover - missing config
            logger.debug("could not load starting composition %s: %s", key, exc)
            return
        self._apply_system(system)

    def _apply_system(self, system: Any) -> None:
        """Populate all widgets from a :class:`SystemConfig`."""
        self._name.setText(system.name)
        self._description.setText(system.description)
        self._mode.setCurrentText(system.mode)
        self._line_freq.setText(_fmt(system.line_freq_hz) if system.line_freq_hz else "")
        self._f_min.setText(_fmt(system.band.f_min_hz))
        self._f_max.setText(_fmt(system.band.f_max_hz))
        self._full_scale.setText(_fmt(system.full_scale_g))
        self._route.setCurrentText(str(system.route))
        self._eta_bias.setText(_fmt(system.eta_bias))
        self._q_total.setText(_fmt(system.q_total))
        self._target_nea.setText(
            _fmt(system.target_nea_ug_rthz) if system.target_nea_ug_rthz else ""
        )
        self._vacuum.setChecked(system.vacuum)
        self._on_mode_changed(system.mode)
        self._source.seed_from_ref(system.source)
        self._fiber.seed_from_ref(system.fiber)
        self._cantilever.seed_from_ref(system.cantilever)
        self._reflector.seed_from_ref(system.reflector)
        self._detector.seed_from_ref(system.detector)
        det = subsystem_defaults(self._store, "detector", system.detector.preset)
        det.update(system.detector.overrides)
        self._balanced.setChecked(bool(det.get("balanced", True)))
        self._reference_arm.setCurrentText(str(det.get("reference_arm", "matched")))
        self._adc_bits.setValue(int(det.get("adc_bits", 24)))

    def _on_save(self) -> None:  # pragma: no cover - file dialog
        """Save the current composition under ``configs/user/systems``."""
        default = self._config_dir / "user" / "systems" / f"{self._name.text().strip()}.yaml"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save composition", str(default), "YAML (*.yaml)"
        )
        if not path:
            return
        try:
            system = build_system_config(self.system_payload())
            save_system_config(system, Path(path))
        except (ValueError, TypeError) as exc:
            logger.debug("save failed: %s", exc)

    def _on_load(self) -> None:  # pragma: no cover - file dialog
        """Load a saved composition and populate the forms."""
        path, _ = QFileDialog.getOpenFileName(self, "Load composition", "", "YAML (*.yaml)")
        if not path:
            return
        try:
            self._apply_system(load_system_file(Path(path)))
        except (FileNotFoundError, ValueError) as exc:
            logger.debug("load failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Payload
    # ------------------------------------------------------------------ #
    def starting_variant_key(self) -> str:
        """Return the starting composition letter (the scenario variant label)."""
        return self._starting.currentText()

    def _opt_float(self, edit: QLineEdit) -> float | None:
        """Parse an optional float field (blank -> ``None``)."""
        text = edit.text().strip()
        return float(text) if text else None

    def _detector_ref(self) -> dict[str, Any]:
        """Reflector-style ref for the detector, folding in the extra controls."""
        ref = self._detector.ref_payload()
        ref["overrides"]["balanced"] = self._balanced.isChecked()
        ref["overrides"]["reference_arm"] = self._reference_arm.currentText()
        ref["overrides"]["adc_bits"] = self._adc_bits.value()
        return ref

    def system_payload(self) -> dict[str, Any]:
        """Assemble the composition payload from the current form state.

        Returns
        -------
        dict[str, Any]
            A mapping accepted by
            :func:`optivibe.gui.controllers.system_builder.build_system_config`.
        """
        return {
            "name": self._name.text().strip() or "edited",
            "description": self._description.text().strip(),
            "mode": self._mode.currentText(),
            "line_freq_hz": self._opt_float(self._line_freq),
            "band": {
                "f_min_hz": float(self._f_min.text()),
                "f_max_hz": float(self._f_max.text()),
            },
            "full_scale_g": float(self._full_scale.text()),
            "route": int(self._route.currentText()),
            "eta_bias": float(self._eta_bias.text()),
            "q_total": float(self._q_total.text()),
            "target_nea_ug_rthz": self._opt_float(self._target_nea),
            "vacuum": self._vacuum.isChecked(),
            "source": self._source.ref_payload(),
            "fiber": self._fiber.ref_payload(),
            "cantilever": self._cantilever.ref_payload(),
            "reflector": self._reflector.ref_payload(),
            "detector": self._detector_ref(),
        }
