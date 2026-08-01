"""Control panel widget: one flat tab set of composition, excitation, stages.

Gathers the buyer-facing controls. Since task S7-mod the sensor is described by
an **editable composition** (:class:`~optivibe.gui.widgets.subsystem_forms.SystemBuilderPanel`,
one form per subsystem with presets and overrides) rather than a single A/B/C/D
combo; the A/B/C/D variants survive as *starting compositions*. Since the S-13b
polish the parameter area is a single :class:`QTabWidget`: the composition panel
provides the *System* + subsystem tabs and this widget appends the *Excitation*,
*Physics layers*, *DSP experiment* and *Reproducibility* pages -- the previous
one-column stack was overloaded. Every row carries a faint ``?`` reference note
(:func:`~optivibe.gui.widgets.ui_helpers.with_help`), and the mouse wheel is
guarded app-wide so skimming the panel can never silently edit a combo or spin
box (:func:`~optivibe.gui.widgets.ui_helpers.install_wheel_guard`).

The "Physics layers" toggles select only the **stage implementation** (physical
vs ``stub``): optics ``physical (reflector)`` / ``stub`` (the key stays
``cylinder``; the reflector *shape* is chosen in the composition's Reflector
tab), mechanics, the detector (``photodiode`` / ``stub``), the DSP
(``standard`` / ``stub``) with its sensitivity model and integrator, plus a
seed. The physical *parameters* live in the composition tabs; in particular the
detector's ``balanced`` / reference-arm settings live solely in the Detector tab
(``variant.detector``), so the scenario emits **no** detector override -- one
source of truth (S7-mod cleanup). The inverse chain itself is edited on the *DSP experiment* tab
(:class:`~optivibe.gui.widgets.dsp_controls.DspControls`, task S-22 W-1), which
owns the ``DspOptions`` values and grades them verified / experimental. The two
older selectors on *Physics layers* (sensitivity model and integrator) stay
where users expect them but are **mirrors**, not copies: they read and write the
same options object, so the two tabs can never disagree (owner decision R-1,
2026-08-01; pinned by test). For the same reason the scenario payload no longer
hard-codes ``spectrum_method``/``window``/``sensitivity_freq`` -- those come
from the model, whose defaults are unchanged, so a default run stays
bit-identical. The controls assemble a scenario *payload*
for :func:`optivibe.gui.controllers.scenario_builder.build_scenario_config` and
a composition payload for
:func:`optivibe.gui.controllers.system_builder.build_system_config`. No physics
here: every value flows into the existing config models (09 §9).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from optivibe.core.config.models import DspOptions
from optivibe.gui.i18n import t
from optivibe.gui.widgets.dsp_controls import DspControls
from optivibe.gui.widgets.excitation_builder import ExcitationBuilder
from optivibe.gui.widgets.subsystem_forms import SystemBuilderPanel
from optivibe.gui.widgets.ui_helpers import install_wheel_guard, with_help

__all__ = ["ControlPanel"]

_OPTICS_HELP = (
    "Optics stage implementation: 'physical (reflector)' -- the shape-"
    "dispatching Gaussian-coupling model (the shape itself is chosen on the "
    "Reflector tab); 'stub' -- a linear eta working-point toy (the eta_bias "
    "scalar on the System tab) for plumbing checks. Physical is the default "
    "for any real study."
)
_MECHANICS_HELP = (
    "Mechanics stage implementation: 'modal' -- frequency-domain modal "
    "response of the cantilever (fast, the standard path); 'modal_time' -- "
    "time-domain integration of the same modal model (for shocks/transients); "
    "'stub' -- pass-through for plumbing checks."
)
_DETECTOR_HELP = (
    "Detector stage implementation: 'photodiode' -- the physical photocurrent "
    "model with shot/RIN/Johnson/ADC noise (enables the NEA budget); 'stub' "
    "-- noiseless pass-through (NEA panels show 'not available')."
)
_DSP_HELP = (
    "Inverse-chain (DSP) implementation: 'standard' -- the calibrated "
    "detector-current -> acceleration chain with spectra and metrics; 'stub' "
    "-- a scale-only shortcut for plumbing checks. The sensitivity and "
    "integrator selectors below apply to the standard DSP only."
)
_SENSITIVITY_HELP = (
    "Mirror of the same control on the DSP experiment tab (one value, two "
    "places). How the standard DSP obtains the scalar sensitivity s_target: 'static' "
    "-- the design-point derivative; 'operating_point' -- re-evaluated at the "
    "resolved working point (bias, gap); 'nonlinear_curve' -- inverted "
    "through the full eta(x) curve (handles large drive amplitudes)."
)
_INTEGRATOR_HELP = (
    "Mirror of the same control on the DSP experiment tab (one value, two "
    "places). Acceleration -> velocity/displacement integration: 'frequency' -- "
    "division by (i omega) in the spectrum (fast, exact for stationary "
    "signals); 'time' -- time-domain integration with detrending (better for "
    "transients/shocks); 'leaky' -- the causal scheme of the live mode over a "
    "whole record (for comparing what causality costs)."
)
_SEED_ENABLED_HELP = (
    "Fix the random seed of the noise and random-excitation generators. "
    "Checked: every Run with the same settings is bit-reproducible. "
    "Unchecked: each Run draws fresh noise (for eyeballing run-to-run "
    "spread)."
)
_SEED_HELP = (
    "The seed value used when 'fixed seed' is checked. Any integer; keep it "
    "constant to reproduce a run exactly, change it to draw a different "
    "noise realisation."
)


class ControlPanel(QWidget):
    """Scenario + composition controls assembled into payloads for the worker.

    Parameters
    ----------
    config_dir : pathlib.Path or None, optional
        Configuration root passed to the composition panel.
    parent : QWidget or None, optional
        Parent widget.
    """

    def __init__(self, config_dir: Path | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._system = SystemBuilderPanel(config_dir=config_dir)

        self._excitation = ExcitationBuilder()

        # "Physics layers" select the stage *implementation* (physical vs stub),
        # not physical parameters -- those live in the composition forms above.
        # The optics key stays "cylinder" (registry/ICD key -> the shape-
        # dispatching ReflectorOptics); only its label is friendlier (the shape
        # itself is chosen in the Reflector form).
        self._optics = self._labeled_combo((("physical (reflector)", "cylinder"), ("stub", "stub")))
        self._mechanics = self._combo(("modal", "modal_time", "stub"))
        self._detector = self._combo(("photodiode", "stub"))
        self._dsp = self._combo(("standard", "stub"))
        self._sensitivity = self._combo(("static", "operating_point", "nonlinear_curve"))
        self._integrator = self._combo(("frequency", "time", "leaky"))
        # Owner of the DspOptions values (S-22 W-1); the two combos above mirror it.
        self._experiment = DspControls()
        self._mirroring = False

        self._seed_enabled = QCheckBox(t("fixed seed"))
        self._seed_enabled.setChecked(True)
        self._seed = QSpinBox()
        self._seed.setRange(0, 2_000_000_000)
        self._seed.setValue(7)

        self._dsp.currentTextChanged.connect(self._on_dsp_changed)
        self._experiment.changed.connect(self._mirror_from_experiment)
        self._sensitivity.currentTextChanged.connect(self._mirror_to_experiment)
        self._integrator.currentTextChanged.connect(self._mirror_to_experiment)

        # One flat tab set: the composition panel's tabs + our three pages.
        self._system.addTab(self._excitation_page(), t("Excitation"))
        self._system.addTab(self._stages_page(), t("Physics layers"))
        self._system.addTab(self._experiment_page(), t("DSP experiment"))
        self._system.addTab(self._run_page(), t("Reproducibility"))

        layout = QVBoxLayout(self)
        layout.addWidget(self._system)
        self._on_dsp_changed(self._dsp.currentText())

        # No silent wheel edits anywhere in the app (S-13b §4): the wheel over
        # a combo/spin box scrolls the panel instead of changing the value.
        install_wheel_guard()

    @staticmethod
    def _combo(items: tuple[str, ...]) -> QComboBox:
        """Build a combo box from string items."""
        box = QComboBox()
        box.addItems(items)
        return box

    @staticmethod
    def _labeled_combo(items: tuple[tuple[str, str], ...]) -> QComboBox:
        """Build a combo box of ``(label, data)`` pairs (data is the stage key)."""
        box = QComboBox()
        for label, data in items:
            box.addItem(label, data)
        return box

    def _excitation_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        group = QGroupBox(t("Excitation"))
        inner = QVBoxLayout(group)
        inner.addWidget(self._excitation)
        layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _stages_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        group = QGroupBox(t("Physics layers (stage implementation)"))
        form = QFormLayout(group)
        form.addRow(t("Optics"), with_help(self._optics, "Optics stage", _OPTICS_HELP))
        form.addRow(t("Mechanics"), with_help(self._mechanics, "Mechanics stage", _MECHANICS_HELP))
        form.addRow(t("Detector"), with_help(self._detector, "Detector stage", _DETECTOR_HELP))
        form.addRow(t("DSP"), with_help(self._dsp, "DSP stage", _DSP_HELP))
        form.addRow(
            "Sensitivity", with_help(self._sensitivity, "Sensitivity model", _SENSITIVITY_HELP)
        )
        form.addRow(t("Integrator"), with_help(self._integrator, "Integrator", _INTEGRATOR_HELP))
        layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _experiment_page(self) -> QWidget:
        """Build the DSP-experiment page (the inverse chain, task S-22 W-1)."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self._experiment)
        layout.addStretch(1)
        return page

    def _run_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        group = QGroupBox(t("Reproducibility"))
        form = QFormLayout(group)
        form.addRow(with_help(self._seed_enabled, "fixed seed", _SEED_ENABLED_HELP))
        form.addRow(t("Seed"), with_help(self._seed, "Seed", _SEED_HELP))
        layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _on_dsp_changed(self, key: str) -> None:
        """Enable the chain controls only for the standard DSP (the stub has none)."""
        is_standard = key == "standard"
        self._sensitivity.setEnabled(is_standard)
        self._integrator.setEnabled(is_standard)
        self._experiment.set_chain_enabled(is_standard)

    def _mirror_from_experiment(self) -> None:
        """Push the experiment panel's values into the Physics-layers mirrors."""
        if self._mirroring:
            return
        self._mirroring = True
        try:
            options = self._experiment.dsp_options()
            for combo, value in (
                (self._sensitivity, options.sensitivity_model),
                (self._integrator, options.integrator),
            ):
                if combo.currentText() != value:
                    combo.setCurrentText(value)
        finally:
            self._mirroring = False

    def _mirror_to_experiment(self) -> None:
        """Push a Physics-layers mirror edit back into the experiment panel."""
        if self._mirroring:
            return
        self._mirroring = True
        try:
            options = self._experiment.dsp_options().model_copy(
                update={
                    "sensitivity_model": self._sensitivity.currentText(),
                    "integrator": self._integrator.currentText(),
                }
            )
            self._experiment.set_dsp_options(options)
        finally:
            self._mirroring = False

    @property
    def experiment(self) -> DspControls:
        """The DSP-experiment panel (owner of the chain; exposed for tests)."""
        return self._experiment

    def dsp_options(self) -> DspOptions:
        """Return the inverse-chain options currently selected."""
        return self._experiment.dsp_options()

    @property
    def system(self) -> SystemBuilderPanel:
        """The composition panel (exposed for tests and the physics tab)."""
        return self._system

    def variant_key(self) -> str:
        """Return the starting-composition label (the scenario variant literal)."""
        return self._system.starting_variant_key()

    def system_payload(self) -> dict[str, Any]:
        """Return the editable-composition payload (for ``build_system_config``)."""
        return self._system.system_payload()

    def scenario_payload(self) -> dict[str, Any]:
        """Assemble the scenario payload from the current selections.

        The ``variant`` field carries the starting-composition label (a frozen
        ``Literal`` in :class:`~optivibe.core.config.models.ScenarioConfig`); the
        edited parameters travel separately via :meth:`system_payload` and are
        resolved into the variant on the worker thread (task S7-mod §1). The
        detector's ``balanced`` / ``reference_arm`` settings live solely in the
        Detector composition form (``variant.detector``); the scenario emits no
        detector override, so there is a single source of truth (S7-mod cleanup).

        Returns
        -------
        dict[str, Any]
            A mapping accepted by ``build_scenario_config``.
        """
        excitation = self._excitation.excitation_payload()
        return {
            "name": "gui-run",
            "variant": self.variant_key(),
            "excitation": excitation,
            "stages": {
                "excitation": excitation["kind"],
                "mechanics": self._mechanics.currentText(),
                "optics": str(self._optics.currentData()),
                "detector": self._detector.currentText(),
                "dsp": self._dsp.currentText(),
            },
            "dsp": self._experiment.dsp_payload(),
            "seed": self._seed.value() if self._seed_enabled.isChecked() else None,
        }

    def restore_scenario(self, payload: dict[str, Any]) -> None:
        """Restore stage selections, seed and excitation from a scenario payload.

        The inverse of :meth:`scenario_payload`, used to preserve state across a
        language rebuild (SW-65). Missing fields keep their built defaults.
        """
        stages = payload.get("stages", {})
        if "mechanics" in stages:
            self._mechanics.setCurrentText(str(stages["mechanics"]))
        if "detector" in stages:
            self._detector.setCurrentText(str(stages["detector"]))
        if "dsp" in stages:
            self._dsp.setCurrentText(str(stages["dsp"]))
        if "optics" in stages:
            index = self._optics.findData(str(stages["optics"]))
            if index >= 0:
                self._optics.setCurrentIndex(index)
        dsp = payload.get("dsp")
        if isinstance(dsp, dict):
            self._experiment.load_payload(dsp)
            self._mirror_from_experiment()
        seed = payload.get("seed")
        self._seed_enabled.setChecked(seed is not None)
        if seed is not None:
            self._seed.setValue(int(seed))
        excitation = payload.get("excitation")
        if isinstance(excitation, dict):
            self._excitation.load_payload(excitation)

    def apply_system_payload(self, payload: dict[str, Any]) -> None:
        """Restore the composition tabs from a :meth:`system_payload` mapping.

        Round-trips the payload through ``build_system_config`` (validation) and
        repopulates every subsystem form -- used by the language rebuild (SW-65).
        """
        from optivibe.gui.controllers.system_builder import build_system_config

        self._system._apply_system(build_system_config(payload))
