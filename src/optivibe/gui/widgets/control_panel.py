"""Control panel widget: one flat tab set of composition, excitation, stages.

Gathers the buyer-facing controls. Since task S7-mod the sensor is described by
an **editable composition** (:class:`~optivibe.gui.widgets.subsystem_forms.SystemBuilderPanel`,
one form per subsystem with presets and overrides) rather than a single A/B/C/D
combo; the A/B/C/D variants survive as *starting compositions*. Since the S-13b
polish the parameter area is a single :class:`QTabWidget`: the composition panel
provides the *System* + subsystem tabs and this widget appends the *Excitation*,
*Physics layers* and *Reproducibility* pages -- the previous one-column stack was
overloaded. Every row carries a faint ``?`` reference note
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
source of truth (S7-mod cleanup). The controls assemble a scenario *payload*
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
    "How the standard DSP obtains the scalar sensitivity s_target: 'static' "
    "-- the design-point derivative; 'operating_point' -- re-evaluated at the "
    "resolved working point (bias, gap); 'nonlinear_curve' -- inverted "
    "through the full eta(x) curve (handles large drive amplitudes)."
)
_INTEGRATOR_HELP = (
    "Acceleration -> velocity/displacement integration: 'frequency' -- "
    "division by (i omega) in the spectrum (fast, exact for stationary "
    "signals); 'time' -- time-domain integration with detrending (better for "
    "transients/shocks)."
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
        self._integrator = self._combo(("frequency", "time"))

        self._seed_enabled = QCheckBox("fixed seed")
        self._seed_enabled.setChecked(True)
        self._seed = QSpinBox()
        self._seed.setRange(0, 2_000_000_000)
        self._seed.setValue(7)

        self._dsp.currentTextChanged.connect(self._on_dsp_changed)

        # One flat tab set: the composition panel's tabs + our three pages.
        self._system.addTab(self._excitation_page(), "Excitation")
        self._system.addTab(self._stages_page(), "Physics layers")
        self._system.addTab(self._run_page(), "Reproducibility")

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
        group = QGroupBox("Excitation")
        inner = QVBoxLayout(group)
        inner.addWidget(self._excitation)
        layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _stages_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        group = QGroupBox("Physics layers (stage implementation)")
        form = QFormLayout(group)
        form.addRow("Optics", with_help(self._optics, "Optics stage", _OPTICS_HELP))
        form.addRow("Mechanics", with_help(self._mechanics, "Mechanics stage", _MECHANICS_HELP))
        form.addRow("Detector", with_help(self._detector, "Detector stage", _DETECTOR_HELP))
        form.addRow("DSP", with_help(self._dsp, "DSP stage", _DSP_HELP))
        form.addRow(
            "Sensitivity", with_help(self._sensitivity, "Sensitivity model", _SENSITIVITY_HELP)
        )
        form.addRow("Integrator", with_help(self._integrator, "Integrator", _INTEGRATOR_HELP))
        layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _run_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        group = QGroupBox("Reproducibility")
        form = QFormLayout(group)
        form.addRow(with_help(self._seed_enabled, "fixed seed", _SEED_ENABLED_HELP))
        form.addRow("Seed", with_help(self._seed, "Seed", _SEED_HELP))
        layout.addWidget(group)
        layout.addStretch(1)
        return page

    def _on_dsp_changed(self, key: str) -> None:
        """Enable sensitivity / integrator controls only for the standard DSP."""
        is_standard = key == "standard"
        self._sensitivity.setEnabled(is_standard)
        self._integrator.setEnabled(is_standard)

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
            "dsp": {
                "integrator": self._integrator.currentText(),
                "spectrum_method": "fft",
                "window": "hann",
                "sensitivity_model": self._sensitivity.currentText(),
                "sensitivity_freq": "plateau",
            },
            "seed": self._seed.value() if self._seed_enabled.isChecked() else None,
        }
