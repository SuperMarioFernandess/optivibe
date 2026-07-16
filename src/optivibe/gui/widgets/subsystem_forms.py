"""Editable subsystem-composition forms (task S7-mod §1/§2; tabs since S-13b).

Replaces the old "pick a variant A/B/C/D" combo with an **editable
composition**: one tab per physical subsystem (source / fiber / cantilever /
reflector / detector), each with a preset selector (from
:class:`~optivibe.core.config.presets.PresetStore`) and labelled, unit-carrying
override fields, plus a "System" tab with the composition-level scalars and a
reflector *shape* selector with dynamic per-shape parameters. The A/B/C/D
variants remain as **starting compositions** that seed the forms.

Since the S-13b polish the panel is a :class:`QTabWidget` (the parameter column
was overloaded as one long stack), every input row carries a faint ``?`` button
opening a short reference note (:func:`optivibe.gui.widgets.ui_helpers.with_help`),
and the measured-data entry points of the S-13 characterization layer live on
their tabs: the OSA spectrum and the RIN trace on *Source*, the tip profile
(R_c) on *Reflector*, the ring-down (Q) on *System*. Each loader accepts the
sidecar YAML **or** its CSV (the same-stem sidecar is found next to it, doc 16
§2a). These loaders *seed the form fields* -- the full measured-twin path with
the provenance artifact remains ``optivibe ingest`` (doc 16 §2a).

Thin shell (09 §9): this widget only collects values into a *payload* mapping;
:func:`optivibe.gui.controllers.system_builder.build_system_config` validates it
into a frozen :class:`~optivibe.core.config.subsystems.SystemConfig`, and the
worker resolves that into the flat variant off the UI thread (SW-06). No physics
here. Every field carries a tooltip and a ``?`` reference with its physical
meaning and a knowledge-base pointer.
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
    QTabWidget,
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
from optivibe.gui.widgets.ui_helpers import with_help

logger = get_logger(__name__)

__all__ = ["SystemBuilderPanel"]

_STARTING = ("A", "B", "C", "D")
_REFLECTOR_SHAPES = ("cylinder", "sphere", "plane", "wedge")

#: File-dialog filter of the S-13 measured-data loaders: either the sidecar
#: YAML or the CSV next to it identifies the artifact (doc 16 §2a).
_ARTIFACT_FILTER = "Characterization artifact (*.yaml *.yml *.csv)"

# Per-subsystem editable override fields, as
# (key, label, unit, tooltip, help). Floats are entered in SI via a line edit
# (so a value like ``1.55e-6`` is unambiguous); the unit is shown in the label;
# the help text opens from the faint ``?`` button next to the field.
_FieldSpec = tuple[str, str, str, str, str]

_PRESET_HELP = (
    "Named building block from configs/presets/<subsystem>/ (user presets under "
    "configs/user/ win over built-ins). Choosing a preset RESEEDS every field "
    "below from its values; anything you then type becomes an explicit override "
    "merged on top of the preset at resolve time. A blank field means 'no "
    "override: the preset (or a derivation) supplies the value'."
)

_SOURCE_FIELDS: tuple[_FieldSpec, ...] = (
    (
        "wavelength_m",
        "wavelength lambda",
        "m",
        "Centre wavelength (doc 03 §1; 1550 nm)",
        "Centre wavelength lambda of the source, metres (1.55e-6 = 1550 nm, the "
        "common platform).\n\nCouples to: the Gaussian beam geometry (Rayleigh "
        "range zR = pi w0^2 / lambda, so the spot size on the mirror w(A) and "
        "the coupling efficiency eta), the fringe period of the parasitic "
        "endface interferometer, and the dlam -> dnu conversion of the "
        "linewidth.\n\nWith a measured spectrum loaded, lambda must lie inside "
        "the table span (a cheap catch of the nm-vs-m unit slip, R-57b); the "
        "spectrum loader seeds it to the measured centroid.",
    ),
    (
        "power_w",
        "optical power P",
        "W",
        "Power delivered to the fiber (doc 07 §2)",
        "Optical power delivered into the fiber, watts (0.016 = 16 mW SLD "
        "default; 20-50 mW typical; 100 mW DFB).\n\nEffect: the photocurrent "
        "scales as I ~ P, so the shot-noise-limited NEA improves as 1/sqrt(P) "
        "while the RIN-limited floor is independent of P (doc 07 §1) -- raising "
        "P helps only until the RIN plateau takes over.",
    ),
    (
        "rin_db_hz",
        "RIN",
        "dB/Hz",
        "Relative intensity noise (doc 07 §1.2); blank for an SLD with a "
        "linewidth = derived ASE floor 2/dnu (M-01); an explicit value replaces "
        "the floor (anti-double-count R-57(v))",
        "Relative intensity noise of the source, dB/Hz. Typical: SLD -120 to "
        "-126 (near its ASE beat floor), low-noise DFB -150 to -155.\n\nBLANK "
        "means 'derive it': for an SLD with a linewidth (or a measured "
        "spectrum) the ASE floor RIN = 2 tau_c = 2/dnu_eff is computed at "
        "resolve time (M-01/R-56). An explicit value REPLACES that floor -- it "
        "is never added to it (anti-double-count, R-57v); a DFB always needs an "
        "explicit value (the ASE relation does not apply to a coherent laser)."
        "\n\nEffect: sets the RIN plateau of the NEA budget; on a balanced "
        "detector it is suppressed by the CMRR.",
    ),
    (
        "linewidth_fwhm_m",
        "linewidth dlam FWHM",
        "m",
        "Spectral FWHM (doc 03 §f'; M-01): drives the derived RIN and the "
        "route-2 wash-out check; forbidden next to a measured spectrum (R-57(a))",
        "Spectral width dlam (FWHM), metres (6e-8 = 60 nm, the route-2 design "
        "anchor at 1550 nm).\n\nCouples to: the coherence length L_c ~ "
        "lambda^2/dlam (60 nm -> L_c ~ 17.7 um) and therefore the ROUTE-2 "
        "WASH-OUT of the parasitic endface fringe: resolve enforces V(A) = "
        "2^-(2A/L_c)^2 < 0.03 at the nominal gap (doc 03 §f'; R-13). Also "
        "yields the derived ASE RIN floor when the RIN field is blank (M-01)."
        "\n\nForbidden next to a measured spectrum -- the table is then the "
        "single source of truth (R-57a); the form clears it automatically.",
    ),
)
#: Source lineshape options (M-10): "(default)" keeps the R-46 effective-scalar
#: behaviour; "measured" is enabled only once a spectrum artifact is loaded.
_LINESHAPES = ("(default)", "gaussian", "lorentzian", "measured")

_LINESHAPE_HELP = (
    "Spectral shape model of the source (M-10):\n\n"
    "(default) -- the R-46 effective-scalar behaviour: Gaussian fringe "
    "visibility and a rectangular noise-equivalent band from the linewidth.\n"
    "gaussian / lorentzian -- analytic shapes; both quadratures (fringe "
    "visibility V(A) and the ASE floor 2 tau_c) follow the chosen shape; "
    "requires the linewidth field.\n"
    "measured -- the loaded OSA table is the single source of truth: V(A), "
    "tau_c and the RIN floor are computed from it and the scalar linewidth is "
    "cleared (R-57a). Enabled only after 'Load measured spectrum...'."
)

_SPECTRUM_LOAD_HELP = (
    "S-13 entry point for the M-15 artifact: pick the sidecar YAML or the CSV "
    "of an OSA trace (lambda, S). The trace is reduced through the standard "
    "optics quadratures (centroid, FWHM, dnu_eff, ASE floor), the wavelength "
    "field is seeded to the measured centroid, lineshape switches to "
    "'measured' and the table travels in the composition overrides.\n\nThis "
    "seeds the FORM; the full measured-twin path with the provenance artifact "
    "is 'optivibe ingest' (doc 16 §2a)."
)

_RIN_LOAD_HELP = (
    "S-13 entry point for the M-16 artifact: a floor-corrected RIN(f) trace "
    "from the PD+TIA bench (shot/dark/Johnson already subtracted). The median "
    "over the band declared in the sidecar seeds the RIN field as an explicit "
    "value -- which REPLACES the derived ASE floor (anti-double-count, R-57v)."
)

_FIBER_FIELDS: tuple[_FieldSpec, ...] = (
    (
        "mode_field_radius_m",
        "mode-field radius w0",
        "m",
        "Gaussian mode radius (doc 03 §1)",
        "Gaussian mode-field radius w0 of the guided mode, metres (SMF-28 at "
        "1550 nm: ~5.2e-6).\n\nCouples to: the beam divergence (zR = pi w0^2 / "
        "lambda), the spot size on the mirror w(A) and thus the coupling "
        "efficiency eta and the displacement sensitivity; the composition-time "
        "geometry guards use it (w(A) <= R_c/3, R_c >= 5 w0, doc 03 §6).",
    ),
    (
        "fresnel_R1",
        "endface reflectivity R1",
        "-",
        "Fresnel reflectivity (doc 04 §4)",
        "Power reflectivity of the fiber endface (bare glass ~0.035; "
        "AR-coated ~1e-4).\n\nThis is the parasitic arm of the endface "
        "interferometer: route 2 washes its fringe out with a broadband source "
        "(V(A) < 0.03), route 1 suppresses it with an AR coating + DFB "
        "(doc 08). Raising R1 raises the DC pedestal and, if the wash-out "
        "fails, an interferometric error term.",
    ),
    (
        "clad_diameter_m",
        "cladding diameter D",
        "m",
        "Outer diameter (doc 01 §4.1; informational)",
        "Outer cladding diameter, metres (1.25e-4 = 125 um standard).\n\n"
        "Informational at composition level: the mechanics reads the fiber "
        "cross-section from the physical constants (doc 01), so this field "
        "documents the part but does not steer the model.",
    ),
)
_REFLECTOR_FIELDS: tuple[_FieldSpec, ...] = (
    (
        "metallization_rho",
        "reflectivity rho",
        "-",
        "Mirror reflectivity (doc 08 §6; 0.98)",
        "Power reflectivity rho of the mirror (0.98 metallized; 0.035 bare "
        "arc-melted glass, the POC prototype R-2).\n\nEffect: scales the "
        "returned optical power and therefore the signal current; the "
        "shot-limited NEA improves as 1/sqrt(rho) while the RIN-limited floor "
        "is unaffected (relative noise).",
    ),
    (
        "gap_m",
        "air gap A",
        "m",
        "Nominal one-way gap (doc 03 §6; 20-40 um)",
        "Nominal one-way air gap A between the fiber endface and the mirror, "
        "metres (design band 20-40 um; POC placeholder 31 um).\n\nCouples to: "
        "the spot size on the mirror w(A) (larger gap -> larger spot -> lower "
        "coupling eta), the geometry guard w(A) <= R_c/3, and the ROUTE-2 "
        "wash-out criterion (the endface fringe must satisfy V(A) < 0.03 at "
        "this gap; broadening the source or enlarging A helps, doc 03 §f').",
    ),
    (
        "bias_offset_m",
        "bias Delta x0",
        "m",
        "Working-point de-centering (doc 03 §5)",
        "Intentional static de-centering Delta x0 of the beam on the mirror, "
        "metres. Sets the working point on the eta(x) curve:\n\n0 -- at the "
        "peak: the linear (1f) response vanishes and the displacement response "
        "is quadratic (2f) -- the POC prototype regime (sleeve centering, "
        "R-4).\nnon-zero -- on the slope: a linear 1f response with the "
        "signed sensitivity s_target; typical bias is a fraction of the spot "
        "size.\n\nIgnored by the flat plane/wedge (no displacement coupling).",
    ),
)
_DETECTOR_FIELDS: tuple[_FieldSpec, ...] = (
    (
        "responsivity",
        "responsivity R",
        "A/W",
        "Photodiode responsivity (doc 07 §2)",
        "Photodiode responsivity R, A/W (~1.0 for InGaAs at 1550 nm).\n\n"
        "Effect: converts optical power to photocurrent; scales the signal and "
        "the shot noise together, so it mainly moves the balance against the "
        "electronics (Johnson/ADC) floors.",
    ),
    (
        "cmrr_db",
        "CMRR",
        "dB",
        "Balanced-channel rejection (doc 07 §1.2)",
        "Common-mode rejection ratio of the balanced pair, dB (typ. 30-50 dB)."
        "\n\nEffect: on a balanced detector the source RIN is common-mode and "
        "is suppressed by the CMRR before it reaches the NEA budget; on a "
        "single-ended detector this field is unused and the full RIN applies.",
    ),
    (
        "adc_full_scale",
        "ADC full scale",
        "out",
        "AC +/- range in output units (doc 07 §1.4)",
        "Full-scale +/- range of the ADC in the detector output units (volts "
        "after the transimpedance).\n\nCouples to: the quantization floor "
        "(together with the ADC bits) and clipping -- the full-scale "
        "acceleration must map inside this range; too generous a range wastes "
        "bits, too tight a range clips at high g.",
    ),
)
_CANTILEVER_FIELDS: tuple[_FieldSpec, ...] = (
    (
        "length_m",
        "length L",
        "m",
        "Free length; sets f1 ~ 1/L^2 (doc 02)",
        "Free cantilever length L from the ferrule exit to the tip, metres "
        "(2-10 mm typical; POC placeholder 4 mm).\n\nThe single strongest "
        "geometric knob: the first eigenfrequency scales as f1 ~ 1/L^2 (doc "
        "02), the tip compliance and thus the mechanical sensitivity grow with "
        "L, and the damping model Q(L) (air + anchor + internal, M-02) follows "
        "it -- the computed Q shown on the System tab updates as you edit L. "
        "Longer L: more sensitivity, lower f1 (narrower usable band), lower "
        "NEA at low f.",
    ),
)

_SHAPE_HELP = (
    "Reflector profile; each shape has a registered optics model (S9-B):\n\n"
    "cylinder -- curved in one axis: the version-1 TARGET-AXIS selector (the "
    "cylinder axis defines the measured axis; cross-axis response is a "
    "metric, doc 00).\n"
    "sphere -- isotropic curvature (the arc-melted POC tip): responds to the "
    "radial displacement, no axis selectivity.\n"
    "wedge -- tilted flat face: an ANGULAR bias working point (alpha_w) "
    "instead of a displacement bias.\n"
    "plane -- flat reference (R_c -> infinity); no displacement coupling, "
    "used for gap-only sensitivity checks.\n\nSwitching the shape "
    "enables/disables the shape parameters below and clears the ones the new "
    "shape ignores."
)

_RC_HELP = (
    "Radius of curvature R_c of the convex mirror, metres (31-62 um presets; "
    "POC placeholder 62.5 um; used by cylinder and sphere only).\n\nCouples "
    "to: the displacement sensitivity of the coupling eta (smaller R_c -> "
    "sharper eta(x) -> higher sensitivity but tighter alignment), and the "
    "paraxial guards R_c >= 5 w0 and w(A) <= R_c/3 (doc 03 §6) -- violating "
    "them fails the composition loudly.\n\nThe 'Load tip profile...' button "
    "below seeds this field from a measured contour (M-17, one azimuth)."
)

_WEDGE_HELP = (
    "Built-in face-tilt angle alpha_w of the wedge, radians (preset 20 mrad; "
    "wedge shape only).\n\nSets an angular bias working point: the returned "
    "beam is deflected by 2 alpha_w, so tip TILT (theta) couples linearly "
    "into eta while pure displacement does not (doc 03 §c)."
)

_PROFILE_LOAD_HELP = (
    "S-13 entry point for the M-17 artifact: a tip-contour trace (x, z) from "
    "the microscope. A Kasa circle fit reduces it to R_c (one azimuth; "
    "astigmatism needs two azimuths and the toroidal optics M-03 -- backlog). "
    "The fit seeds the R_c field; non-circular or collinear contours are "
    "rejected loudly (doc 20 F0-3)."
)

_RINGDOWN_LOAD_HELP = (
    "S-13 entry point for the M-18 artifact: a free-decay record (t, y). The "
    "Hilbert-envelope log decrement yields Q = pi f1 / sigma; the fit seeds "
    "the Q override field (measured Q wins over the Q(L) model, M-02 "
    "semantics) and reports f1 for the cross-check against f1(L). An undamped "
    "tone is rejected loudly."
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

    Every row carries a faint ``?`` button opening the field's reference note
    (what it is, typical values, couplings, effect on the simulation).

    Parameters
    ----------
    title : str
        Group-box title.
    subsystem : str
        Subsystem name (``"source"``, ...).
    fields : tuple
        Field specs ``(key, label, unit, tooltip, help)`` for the override
        line edits.
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
        self._form.addRow("preset", with_help(self._preset, f"{title}: preset", _PRESET_HELP))
        for key, label, unit, tip, help_text in fields:
            edit = _line()
            edit.setToolTip(tip)
            self._edits[key] = edit
            self._form.addRow(f"{label} [{unit}]", with_help(edit, f"{label} [{unit}]", help_text))

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
        """Set the preset and fill the fields from preset + overrides.

        Fields whose value in the (preset + overrides) view is ``None`` or
        absent are **cleared**, not left as they were: a stale number from the
        previous preset would silently become an override of the new one
        (e.g. a linewidth typed for one SLD surviving a switch to a preset
        without one) -- exactly the class of unnoticed edits this panel must
        not produce.
        """
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
            value = values.get(key)
            edit.setText(_fmt(value) if value is not None else "")

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

    def _pick_artifact(self, caption: str) -> Path | None:  # pragma: no cover - dialog
        """Open the shared characterization-artifact file dialog."""
        path, _ = QFileDialog.getOpenFileName(self, caption, "", _ARTIFACT_FILTER)
        return Path(path) if path else None


class _SourceForm(_SubsystemForm):
    """Source form: M-10 lineshape selector + spectrum and RIN-trace loaders.

    Adds the ``lineshape`` combo ("(default)"/gaussian/lorentzian/measured), a
    "Load measured spectrum..." button (the S-13 entry point of the M-15
    artifact) and a "Load RIN trace..." button (M-16 reduce). The ``measured``
    option stays disabled until a spectrum artifact (sidecar YAML + CSV, doc
    16 §2a; either file can be picked) is loaded through
    :func:`optivibe.io.characterization.load_characterization`; the loaded
    table then travels in the overrides exactly like a hand-written one (the
    composition-level M-10 fields), so no new resolve path exists. Per R-57(a)
    selecting ``measured`` forces ``linewidth_fwhm_m`` to ``None`` (the table
    is the single source of truth of its row); a loaded RIN trace seeds the
    RIN field as an explicit value, which *replaces* the derived floor
    (R-57(v)).
    """

    _MEASURED_INDEX = _LINESHAPES.index("measured")

    def __init__(self, store: PresetStore) -> None:
        super().__init__("Source", "source", _SOURCE_FIELDS, store)
        self._lineshape = QComboBox()
        self._lineshape.addItems(_LINESHAPES)
        self._lineshape.setToolTip(
            "Source spectrum shape (M-10): default keeps the R-46 behaviour "
            "(Gaussian visibility, rectangular RIN floor); measured needs a "
            "loaded spectrum artifact (M-15/S-13)"
        )
        self._spectrum_lam: list[float] | None = None
        self._spectrum_psd: list[float] | None = None
        self._spectrum_note = QLabel("no measured spectrum loaded")
        self._spectrum_note.setWordWrap(True)
        self._spectrum_button = QPushButton("Load measured spectrum...")
        self._spectrum_button.setToolTip(
            "Load a characterization artifact (sidecar YAML or its CSV; doc 16 "
            "§2a) and enable lineshape = measured"
        )
        self._spectrum_button.clicked.connect(self._on_load_spectrum)
        self._rin_note = QLabel("")
        self._rin_note.setWordWrap(True)
        self._rin_button = QPushButton("Load RIN trace...")
        self._rin_button.setToolTip(
            "Load a floor-corrected RIN(f) artifact (M-16); the band median "
            "seeds the RIN field as an explicit value (replaces the derived "
            "floor, R-57(v))"
        )
        self._rin_button.clicked.connect(self._on_load_rin)

        self._form.insertRow(
            1, "lineshape", with_help(self._lineshape, "lineshape", _LINESHAPE_HELP)
        )
        self._form.insertRow(
            2,
            "spectrum",
            with_help(
                self._loader_row(self._spectrum_button, self._spectrum_note),
                "Load measured spectrum",
                _SPECTRUM_LOAD_HELP,
            ),
        )
        self._form.addRow(
            "RIN trace",
            with_help(
                self._loader_row(self._rin_button, self._rin_note),
                "Load RIN trace",
                _RIN_LOAD_HELP,
            ),
        )
        self._set_measured_enabled(False)

    @staticmethod
    def _loader_row(button: QPushButton, note: QLabel) -> QWidget:
        """Pack a loader button with its provenance note."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(button)
        row.addWidget(note, stretch=1)
        holder = QWidget()
        holder.setLayout(row)
        return holder

    # ------------------------------------------------------------------ #
    # Spectrum artifact (M-15)
    # ------------------------------------------------------------------ #
    def _set_measured_enabled(self, enabled: bool) -> None:
        """Enable/disable the ``measured`` combo entry (needs a loaded table)."""
        model = self._lineshape.model()
        item = model.item(self._MEASURED_INDEX)  # type: ignore[attr-defined]
        if item is not None:
            item.setEnabled(enabled)

    def _on_load_spectrum(self) -> None:  # pragma: no cover - file dialog
        """Pick a spectrum characterization artifact and load it."""
        path = self._pick_artifact("Load spectrum artifact (sidecar YAML or CSV)")
        if path is not None:
            try:
                self.load_spectrum_artifact(path)
            except (FileNotFoundError, ValueError) as exc:
                logger.debug("spectrum artifact load failed: %s", exc)
                self._spectrum_note.setText(f"load failed: {exc}")

    def load_spectrum_artifact(self, path: Path) -> None:
        """Load a ``kind = "spectrum"`` characterization artifact into the form.

        Seeds the wavelength field to the measured centroid, attaches the
        table, clears the scalar linewidth (R-57(a)) and switches the
        lineshape to ``measured``.

        Parameters
        ----------
        path : pathlib.Path
            Sidecar YAML of the artifact, or its CSV (doc 16 §2a).

        Raises
        ------
        ValueError
            If the artifact is not a spectrum or is malformed.
        FileNotFoundError
            If the sidecar or its data file is missing.
        """
        from optivibe.io.characterization import load_characterization

        result = load_characterization(path)
        if result.spectrum_wavelength_m is None or result.spectrum_psd is None:
            msg = f"{path.name} is a {result.provenance.kind!r} artifact, not a spectrum"
            raise ValueError(msg)
        self._set_spectrum(
            list(result.spectrum_wavelength_m),
            list(result.spectrum_psd),
            note=f"{result.provenance.data_file} ({result.provenance.instrument}, "
            f"{result.provenance.timestamp}; sha {str(result.provenance.sha256)[:8]})",
        )
        centroid = result.get("wavelength_m")
        if centroid is not None:
            self._edits["wavelength_m"].setText(_fmt(centroid.value))
        self._edits["linewidth_fwhm_m"].setText("")  # R-57(a)
        self._lineshape.setCurrentText("measured")

    def _set_spectrum(self, lam: list[float], psd: list[float], *, note: str) -> None:
        """Attach a spectrum table to the form and enable ``measured``."""
        self._spectrum_lam = lam
        self._spectrum_psd = psd
        self._spectrum_note.setText(note)
        self._set_measured_enabled(True)

    def _clear_spectrum(self) -> None:
        """Drop the attached table (preset change / non-measured composition)."""
        self._spectrum_lam = None
        self._spectrum_psd = None
        self._spectrum_note.setText("no measured spectrum loaded")
        self._set_measured_enabled(False)
        if self._lineshape.currentText() == "measured":
            self._lineshape.setCurrentText("(default)")

    # ------------------------------------------------------------------ #
    # RIN trace artifact (M-16 reduce)
    # ------------------------------------------------------------------ #
    def _on_load_rin(self) -> None:  # pragma: no cover - file dialog
        """Pick a RIN-trace characterization artifact and load it."""
        path = self._pick_artifact("Load RIN trace artifact (sidecar YAML or CSV)")
        if path is not None:
            try:
                self.load_rin_artifact(path)
            except (FileNotFoundError, ValueError) as exc:
                logger.debug("RIN artifact load failed: %s", exc)
                self._rin_note.setText(f"load failed: {exc}")

    def load_rin_artifact(self, path: Path) -> None:
        """Load a ``kind = "rin_psd"`` artifact and seed the RIN field.

        The band-median RIN becomes an *explicit* value in the RIN field,
        which replaces the derived ASE floor at resolve time (R-57(v)).

        Parameters
        ----------
        path : pathlib.Path
            Sidecar YAML of the artifact, or its CSV.

        Raises
        ------
        ValueError
            If the artifact does not reduce to a ``rin_db_hz`` parameter.
        FileNotFoundError
            If the sidecar or its data file is missing.
        """
        from optivibe.io.characterization import load_characterization

        result = load_characterization(path)
        rin = result.get("rin_db_hz")
        if rin is None:
            msg = f"{path.name} is a {result.provenance.kind!r} artifact, not a RIN trace"
            raise ValueError(msg)
        self._edits["rin_db_hz"].setText(_fmt(rin.value))
        self._rin_note.setText(
            f"measured RIN {rin.value:.2f} dB/Hz (u = {rin.u:.2g}; "
            f"{result.provenance.data_file}, {result.provenance.instrument}) -- "
            "replaces the derived floor (R-57v)"
        )

    # ------------------------------------------------------------------ #
    # Seeding / payload
    # ------------------------------------------------------------------ #
    def _reseed(self, preset: str, overrides: dict[str, Any]) -> None:
        """Reseed the base fields, then the lineshape + table state."""
        super()._reseed(preset, overrides)
        if not hasattr(self, "_lineshape"):  # base __init__ path
            return
        try:
            values = subsystem_defaults(self._store, self._subsystem, preset)
        except (ValueError, KeyError):  # pragma: no cover - bad preset on disk
            return
        values.update(overrides)
        lam = values.get("spectrum_wavelength_m")
        psd = values.get("spectrum_psd")
        if lam and psd:
            self._set_spectrum(list(lam), list(psd), note="from the composition overrides")
        else:
            self._clear_spectrum()
        shape = values.get("lineshape")
        self._lineshape.setCurrentText(str(shape) if shape else "(default)")
        self._rin_note.setText("")

    def overrides(self) -> dict[str, Any]:
        """Collect the base fields plus the lineshape/table state (M-10 rules)."""
        out = super().overrides()
        shape = self._lineshape.currentText()
        if shape == "measured":
            out["lineshape"] = "measured"
            out["spectrum_wavelength_m"] = self._spectrum_lam
            out["spectrum_psd"] = self._spectrum_psd
            # R-57(a): the table is the single source of truth of its row.
            out["linewidth_fwhm_m"] = None
        elif shape != "(default)":
            out["lineshape"] = shape
        return out


class _ReflectorForm(_SubsystemForm):
    """Reflector form: shape selector, per-shape parameters, profile loader.

    The shape combo enables only the parameters the chosen shape uses
    (curvature for cylinder/sphere, the face angle for the wedge); switching
    shapes emits ``None`` overrides for the parameters the new shape ignores,
    so a stale curvature can never leak into a plane composition. The "Load
    tip profile..." button is the S-13 entry point of the M-17 artifact: a
    Kasa circle fit of the contour seeds the curvature field (one azimuth;
    astigmatism stays with M-03).
    """

    def __init__(self, store: PresetStore) -> None:
        super().__init__("Reflector", "reflector", _REFLECTOR_FIELDS, store)
        self._shape = QComboBox()
        self._shape.addItems(_REFLECTOR_SHAPES)
        self._shape.setToolTip("Reflector profile (S9-B); shapes gate their parameters")
        self._rc = _line()
        self._rc.setToolTip("Radius of curvature R_c (cylinder/sphere; doc 08 §6)")
        self._alpha = _line()
        self._alpha.setToolTip("Wedge face-tilt angle alpha_w (wedge only; doc 03 §c)")
        self._profile_note = QLabel("")
        self._profile_note.setWordWrap(True)
        self._profile_button = QPushButton("Load tip profile (R_c)...")
        self._profile_button.setToolTip(
            "Load a tip-contour artifact (M-17); the circle fit seeds R_c"
        )
        self._profile_button.clicked.connect(self._on_load_profile)

        self._form.insertRow(1, "shape", with_help(self._shape, "shape", _SHAPE_HELP))
        self._form.insertRow(
            2, "curvature R_c [m]", with_help(self._rc, "curvature R_c [m]", _RC_HELP)
        )
        self._form.insertRow(
            3, "wedge angle [rad]", with_help(self._alpha, "wedge angle [rad]", _WEDGE_HELP)
        )
        profile_row = QHBoxLayout()
        profile_row.setContentsMargins(0, 0, 0, 0)
        profile_row.addWidget(self._profile_button)
        profile_row.addWidget(self._profile_note, stretch=1)
        holder = QWidget()
        holder.setLayout(profile_row)
        self._form.addRow("profile", with_help(holder, "Load tip profile", _PROFILE_LOAD_HELP))

        self._shape.currentTextChanged.connect(self._on_shape_changed)
        self._on_shape_changed(self._shape.currentText())

    def _on_shape_changed(self, shape: str) -> None:
        """Enable only the parameters the chosen shape uses."""
        curved = shape in ("cylinder", "sphere")
        self._rc.setEnabled(curved)
        self._profile_button.setEnabled(curved)
        self._alpha.setEnabled(shape == "wedge")

    def _on_load_profile(self) -> None:  # pragma: no cover - file dialog
        """Pick a profile characterization artifact and load it."""
        path = self._pick_artifact("Load tip-profile artifact (sidecar YAML or CSV)")
        if path is not None:
            try:
                self.load_profile_artifact(path)
            except (FileNotFoundError, ValueError) as exc:
                logger.debug("profile artifact load failed: %s", exc)
                self._profile_note.setText(f"load failed: {exc}")

    def load_profile_artifact(self, path: Path) -> None:
        """Load a ``kind = "profile"`` artifact and seed the curvature field.

        Parameters
        ----------
        path : pathlib.Path
            Sidecar YAML of the artifact, or its CSV.

        Raises
        ------
        ValueError
            If the artifact does not reduce to a ``curvature_radius_m``
            parameter (or the contour fails the circle-fit guards).
        FileNotFoundError
            If the sidecar or its data file is missing.
        """
        from optivibe.io.characterization import load_characterization

        result = load_characterization(path)
        r_c = result.get("curvature_radius_m")
        if r_c is None:
            msg = f"{path.name} is a {result.provenance.kind!r} artifact, not a tip profile"
            raise ValueError(msg)
        self._rc.setText(_fmt(r_c.value))
        self._profile_note.setText(
            f"measured R_c {r_c.value * 1e6:.2f} um (u = {(r_c.u or 0.0) * 1e6:.2g} um; "
            f"{result.provenance.data_file}, {result.provenance.instrument}; one azimuth "
            "-- astigmatism = backlog M-03)"
        )

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
            self._profile_note.setText("")

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


_STARTING_HELP = (
    "Which built-in composition seeds every tab: A (compact wideband, "
    "vacuum-optional), B (general-purpose wideband), C (long-throw "
    "sensitivity), D (resonant narrow-line, route 1). Switching RESEEDS all "
    "tabs from that variant -- unsaved edits are replaced. The letter also "
    "names the frozen scenario variant the run is labelled with; your edits "
    "travel separately as the composition payload."
)
_NAME_HELP = (
    "Composition identity used for saved files and run labels. Free text for "
    "user compositions; A-D are reserved for the built-ins."
)
_DESCRIPTION_HELP = "Free-text description shown in reports; no effect on the model."
_MODE_HELP = (
    "Operating regime (doc 08 §6): offresonance -- wideband use well below "
    "f1, flat mechanical response; resonance -- narrowband use on the "
    "resonant line (requires the line frequency below and typically variant "
    "D: route 1, high Q). Affects which DSP calibration applies."
)
_LINE_FREQ_HELP = (
    "Resonant line frequency, Hz -- only read in resonance mode; must sit "
    "near the composition's f1 for the resonant gain to be real."
)
_BAND_HELP = (
    "Assessment band [f_min, f_max], Hz, used by the NEA budget, the DSP "
    "band-limits and the reports. The project spec band is 0.1 Hz - 20 kHz "
    "(doc 00, fixed); a composition may declare a narrower working band. "
    "f_max should stay well below f1 for off-resonance operation."
)
_FULL_SCALE_HELP = (
    "Full-scale acceleration FS, g (spec: 50 g at any band frequency, doc 00 "
    "/ 08 §1.3). Sets the ADC mapping and the clipping checks; behaviour "
    "above FS (margin, nonlinearity) is a study topic, not guaranteed range."
)
_ROUTE_HELP = (
    "Endface-treatment route (doc 08): 2 -- coherent wash-out: a broadband "
    "source (SLD) makes the parasitic endface fringe invisible (V(A) < 0.03 "
    "enforced at resolve when the linewidth or a measured spectrum is "
    "known); 1 -- AR-coated endface + narrow-line DFB (variant D). The route "
    "decides which source/noise inputs are consistent."
)
_ETA_BIAS_HELP = (
    "Optical working-point efficiency eta0 used by the STUB optics only "
    "(S0 path); the physical reflector optics computes its own eta0 from the "
    "geometry (doc 03 §5). Ignored when the Optics stage is 'physical'."
)
_Q_HELP = (
    "Total mechanical quality factor Q of mode 1. Since M-02 this is a "
    "COMPUTED quantity: leave the field BLANK to use the Q(L) damping model "
    "(air + anchor + internal losses at the current cantilever length; "
    "vacuum removes the air channel) shown below. A typed value is an "
    "explicit OVERRIDE -- e.g. a measured ring-down Q (M-18), which wins over "
    "the model by design. Q sets the resonance peak height, the ring-down "
    "time and the Brownian thermal NEA floor."
)
_TARGET_NEA_HELP = (
    "Optional target noise-equivalent acceleration, ug/sqrt(Hz), drawn on "
    "the NEA plots as the design goal; no effect on the model."
)
_VACUUM_HELP = (
    "Operate the variant under vacuum: removes the air-damping channel from "
    "the Q(L) model (higher Q, taller resonance, lower thermal NEA) -- the "
    "A/D packaging option."
)


class SystemBuilderPanel(QTabWidget):
    """Editable composition as tabs: System + one tab per subsystem.

    Tabs: *System* (composition scalars, the computed-Q(L) display with the
    ring-down loader, save/load), *Source*, *Fiber line*, *Cantilever*,
    *Reflector*, *Detector*. The control panel appends its own *Excitation*,
    *Physics layers* and *Reproducibility* pages into this tab widget so the
    whole parameter area is one flat set of tabs.

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
        self._q_total = _line()  # empty = computed by the Q(L) model (R-48)
        self._q_total.setToolTip(
            "Total quality factor of mode 1. Since M-02 this is a COMPUTED "
            "quantity (Q(L) damping model, R-47/R-48): leave blank to use the "
            "model value shown below; a typed value is an explicit override"
        )
        self._q_model = QLabel("Q(L) model: -")
        self._q_model.setToolTip(
            "What a blank Q field resolves to: the Q(L) damping model at the "
            "current cantilever length and vacuum flag (M-02)"
        )
        self._q_note = QLabel("")
        self._q_note.setWordWrap(True)
        self._ringdown_button = QPushButton("Load ring-down (Q)...")
        self._ringdown_button.setToolTip(
            "Load a free-decay artifact (M-18); the log-decrement Q seeds the "
            "override field (measured Q wins over the Q(L) model)"
        )
        self._ringdown_button.clicked.connect(self._on_load_ringdown)
        self._target_nea = _line("10.0")
        self._vacuum = QCheckBox("vacuum")
        self._mode.currentTextChanged.connect(self._on_mode_changed)

        # Subsystem forms.
        self._source = _SourceForm(self._store)
        self._fiber = _SubsystemForm("Fiber line", "fiber", _FIBER_FIELDS, self._store)
        self._cantilever = _SubsystemForm(
            "Cantilever", "cantilever", _CANTILEVER_FIELDS, self._store
        )
        self._reflector = _ReflectorForm(self._store)
        self._detector = _SubsystemForm("Detector", "detector", _DETECTOR_FIELDS, self._store)
        self._balanced = QCheckBox("balanced channel")
        self._reference_arm = QComboBox()
        self._reference_arm.addItems(("matched", "bright"))
        self._detector._form.addRow(
            "balanced",
            with_help(
                self._balanced,
                "balanced",
                "Balanced photodiode pair vs a single-ended detector. Balanced: "
                "the source RIN is common-mode and suppressed by the CMRR; "
                "single-ended (the POC prototype, R-3): the full RIN reaches "
                "the budget and the CMRR field is unused.",
            ),
        )
        self._detector._form.addRow(
            "reference arm",
            with_help(
                self._reference_arm,
                "reference arm",
                "Shot-noise convention of the balanced pair (O-SW-08): "
                "'matched' -- the reference arm carries the same mean power as "
                "the signal arm (shot PSD doubles); 'bright' -- a bright "
                "reference dominates the shot floor. Affects only the noise "
                "bookkeeping, not the signal.",
            ),
        )
        self._adc_bits = QSpinBox()
        self._adc_bits.setRange(1, 32)
        self._adc_bits.setValue(24)
        self._detector._form.addRow(
            "ADC bits",
            with_help(
                self._adc_bits,
                "ADC bits",
                "ADC resolution, bits (24 typical). Together with the ADC full "
                "scale it sets the quantization noise floor; the budget checks "
                "it stays below the analog floors (doc 07 §1.4).",
            ),
        )

        self._save_button = QPushButton("Save as...")
        self._load_button = QPushButton("Load...")
        self._save_button.clicked.connect(self._on_save)
        self._load_button.clicked.connect(self._on_load)

        # Keep the computed-Q(L) display in step with what it depends on:
        # the cantilever choice (preset or length override) and the vacuum
        # flag (the model drops the air channel under vacuum; M-02).
        self._vacuum.toggled.connect(lambda _checked: self.refresh_q_model())
        self._cantilever._preset.currentTextChanged.connect(lambda _name: self.refresh_q_model())
        self._cantilever._edits["length_m"].editingFinished.connect(self.refresh_q_model)

        self.addTab(self._system_page(), "System")
        self.addTab(self._page(self._source), "Source")
        self.addTab(self._page(self._fiber), "Fiber line")
        self.addTab(self._page(self._cantilever), "Cantilever")
        self.addTab(self._page(self._reflector), "Reflector")
        self.addTab(self._page(self._detector), "Detector")
        self.setUsesScrollButtons(True)

        self._load_starting("B")

    @staticmethod
    def _page(widget: QWidget) -> QWidget:
        """Wrap a form in a top-aligned tab page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(widget)
        layout.addStretch(1)
        return page

    def _system_page(self) -> QWidget:
        """Build the *System* tab: composition scalars + Q(L) + save/load."""
        group = QGroupBox("System / composition")
        form = QFormLayout(group)
        form.addRow(
            "starting composition",
            with_help(self._starting, "starting composition", _STARTING_HELP),
        )
        form.addRow("name", with_help(self._name, "name", _NAME_HELP))
        form.addRow("description", with_help(self._description, "description", _DESCRIPTION_HELP))
        form.addRow("mode", with_help(self._mode, "mode", _MODE_HELP))
        form.addRow("line freq [Hz]", with_help(self._line_freq, "line freq [Hz]", _LINE_FREQ_HELP))
        band_row = QHBoxLayout()
        band_row.addWidget(QLabel("f_min"))
        band_row.addWidget(self._f_min)
        band_row.addWidget(QLabel("f_max"))
        band_row.addWidget(self._f_max)
        holder = QWidget()
        holder.setLayout(band_row)
        form.addRow("band [Hz]", with_help(holder, "band [Hz]", _BAND_HELP))
        form.addRow(
            "full scale [g]", with_help(self._full_scale, "full scale [g]", _FULL_SCALE_HELP)
        )
        form.addRow("route", with_help(self._route, "route", _ROUTE_HELP))
        form.addRow("eta_bias (stub)", with_help(self._eta_bias, "eta_bias (stub)", _ETA_BIAS_HELP))
        form.addRow(
            "Q total override (blank = Q(L) model)",
            with_help(self._q_total, "Q total override", _Q_HELP),
        )
        form.addRow("", self._q_model)
        ringdown_row = QHBoxLayout()
        ringdown_row.setContentsMargins(0, 0, 0, 0)
        ringdown_row.addWidget(self._ringdown_button)
        ringdown_row.addWidget(self._q_note, stretch=1)
        rd_holder = QWidget()
        rd_holder.setLayout(ringdown_row)
        form.addRow("ring-down", with_help(rd_holder, "Load ring-down", _RINGDOWN_LOAD_HELP))
        form.addRow(
            "target NEA [ug/rtHz]",
            with_help(self._target_nea, "target NEA [ug/rtHz]", _TARGET_NEA_HELP),
        )
        form.addRow(with_help(self._vacuum, "vacuum", _VACUUM_HELP))

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(group)
        io_row = QHBoxLayout()
        io_row.addWidget(self._save_button)
        io_row.addWidget(self._load_button)
        layout.addLayout(io_row)
        layout.addStretch(1)
        return page

    def _on_mode_changed(self, mode: str) -> None:
        """Enable the resonant line frequency only in resonance mode."""
        self._line_freq.setEnabled(mode == "resonance")

    # ------------------------------------------------------------------ #
    # Measured ring-down (M-18)
    # ------------------------------------------------------------------ #
    def _on_load_ringdown(self) -> None:  # pragma: no cover - file dialog
        """Pick a ring-down characterization artifact and load it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load ring-down artifact (sidecar YAML or CSV)", "", _ARTIFACT_FILTER
        )
        if path:
            try:
                self.load_ringdown_artifact(Path(path))
            except (FileNotFoundError, ValueError) as exc:
                logger.debug("ring-down artifact load failed: %s", exc)
                self._q_note.setText(f"load failed: {exc}")

    def load_ringdown_artifact(self, path: Path) -> None:
        """Load a ``kind = "ringdown"`` artifact and seed the Q override.

        The measured Q lands in the override field (it wins over the Q(L)
        model, M-02 semantics); the measured ``f1`` is shown for the
        cross-check against the model's ``f1(L)``.

        Parameters
        ----------
        path : pathlib.Path
            Sidecar YAML of the artifact, or its CSV.

        Raises
        ------
        ValueError
            If the artifact does not reduce to a ``q_total`` parameter (or
            the record fails the decay guards).
        FileNotFoundError
            If the sidecar or its data file is missing.
        """
        from optivibe.io.characterization import load_characterization

        result = load_characterization(path)
        q_meas = result.get("q_total")
        if q_meas is None:
            msg = f"{path.name} is a {result.provenance.kind!r} artifact, not a ring-down"
            raise ValueError(msg)
        self._q_total.setText(_fmt(q_meas.value))
        f1 = result.get("f1_hz")
        f1_text = f", f1 = {f1.value:.1f} Hz" if f1 is not None else ""
        self._q_note.setText(
            f"measured Q {q_meas.value:.1f} (u = {(q_meas.u or 0.0):.2g}{f1_text}; "
            f"{result.provenance.data_file}, {result.provenance.instrument}) -- "
            "overrides the Q(L) model"
        )
        self.refresh_q_model()

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
        self._q_total.setText(_fmt(system.q_total) if system.q_total else "")
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
        self._q_note.setText("")
        self.refresh_q_model()

    def refresh_q_model(self) -> None:
        """Refresh the computed-``Q(L)`` display next to the override field.

        Delegates the evaluation to the Qt-free controller helper
        :func:`optivibe.gui.controllers.system_builder.model_q_total` (thin
        shell, 09 §9); an invalid cantilever state degrades to a visible
        "n/a", never a crash (10 §7 applies to the *build*, not to a live
        hint label).
        """
        from optivibe.gui.controllers.system_builder import model_q_total

        try:
            q_value = model_q_total(
                self._config_dir,
                self._cantilever.ref_payload(),
                vacuum=self._vacuum.isChecked(),
            )
        except (ValueError, KeyError, FileNotFoundError) as exc:
            logger.debug("Q(L) hint unavailable: %s", exc)
            self._q_model.setText("Q(L) model: n/a (check the cantilever fields)")
            return
        self._q_model.setText(f"Q(L) model: {q_value:.6g} (used when the field above is blank)")

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
            "q_total": self._opt_float(self._q_total),
            "target_nea_ug_rthz": self._opt_float(self._target_nea),
            "vacuum": self._vacuum.isChecked(),
            "source": self._source.ref_payload(),
            "fiber": self._fiber.ref_payload(),
            "cantilever": self._cantilever.ref_payload(),
            "reflector": self._reflector.ref_payload(),
            "detector": self._detector_ref(),
        }
