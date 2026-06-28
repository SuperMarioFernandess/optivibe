"""Control panel widget: composition, excitation, stage toggles, seed.

Gathers the buyer-facing controls. Since task S7-mod the sensor is described by
an **editable composition** (:class:`~optivibe.gui.widgets.subsystem_forms.SystemBuilderPanel`,
one form per subsystem with presets and overrides) rather than a single A/B/C/D
combo; the A/B/C/D variants survive as *starting compositions*. The rest is as
before -- build an excitation, flip the physics layers (optics ``cylinder``/
``stub``, detector ``stub``/``photodiode`` with balanced + reference-arm, DSP
``stub``/``standard`` with the sensitivity model and integrator) and a seed --
assembled into a scenario *payload* for
:func:`optivibe.gui.controllers.scenario_builder.build_scenario_config`, plus a
**composition payload** for
:func:`optivibe.gui.controllers.system_builder.build_system_config`. No physics
here: every value flows into the existing config models (09 §9). Detector
overrides are emitted only for the ``photodiode`` stage (the option-less ``stub``
takes no arguments; 14 §5).
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

__all__ = ["ControlPanel"]


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

        self._optics = self._combo(("cylinder", "stub"))
        self._mechanics = self._combo(("modal", "modal_time", "stub"))
        self._detector = self._combo(("photodiode", "stub"))
        self._balanced = QCheckBox("balanced channel")
        self._balanced.setChecked(True)
        self._reference_arm = self._combo(("matched", "bright"))
        self._dsp = self._combo(("standard", "stub"))
        self._sensitivity = self._combo(("static", "operating_point", "nonlinear_curve"))
        self._integrator = self._combo(("frequency", "time"))

        self._seed_enabled = QCheckBox("fixed seed")
        self._seed_enabled.setChecked(True)
        self._seed = QSpinBox()
        self._seed.setRange(0, 2_000_000_000)
        self._seed.setValue(7)

        self._detector.currentTextChanged.connect(self._on_detector_changed)
        self._dsp.currentTextChanged.connect(self._on_dsp_changed)

        layout = QVBoxLayout(self)
        layout.addWidget(self._composition_group())
        layout.addWidget(self._excitation_group())
        layout.addWidget(self._stages_group())
        layout.addWidget(self._run_group())
        layout.addStretch(1)
        self._on_detector_changed(self._detector.currentText())
        self._on_dsp_changed(self._dsp.currentText())

    @staticmethod
    def _combo(items: tuple[str, ...]) -> QComboBox:
        """Build a combo box from string items."""
        box = QComboBox()
        box.addItems(items)
        return box

    def _composition_group(self) -> QGroupBox:
        group = QGroupBox("Sensor composition")
        layout = QVBoxLayout(group)
        layout.addWidget(self._system)
        return group

    def _excitation_group(self) -> QGroupBox:
        group = QGroupBox("Excitation")
        layout = QVBoxLayout(group)
        layout.addWidget(self._excitation)
        return group

    def _stages_group(self) -> QGroupBox:
        group = QGroupBox("Physics layers")
        form = QFormLayout(group)
        form.addRow("Optics", self._optics)
        form.addRow("Mechanics", self._mechanics)
        form.addRow("Detector", self._detector)
        form.addRow("", self._balanced)
        form.addRow("Reference arm", self._reference_arm)
        form.addRow("DSP", self._dsp)
        form.addRow("Sensitivity", self._sensitivity)
        form.addRow("Integrator", self._integrator)
        return group

    def _run_group(self) -> QGroupBox:
        group = QGroupBox("Reproducibility")
        form = QFormLayout(group)
        form.addRow(self._seed_enabled)
        form.addRow("Seed", self._seed)
        return group

    def _on_detector_changed(self, key: str) -> None:
        """Enable balanced / reference-arm controls only for the photodiode."""
        is_pd = key == "photodiode"
        self._balanced.setEnabled(is_pd)
        self._reference_arm.setEnabled(is_pd and self._balanced.isChecked())

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
        resolved into the variant on the worker thread (task S7-mod §1).

        Returns
        -------
        dict[str, Any]
            A mapping accepted by ``build_scenario_config``.
        """
        excitation = self._excitation.excitation_payload()
        detector_key = self._detector.currentText()
        payload: dict[str, Any] = {
            "name": "gui-run",
            "variant": self.variant_key(),
            "excitation": excitation,
            "stages": {
                "excitation": excitation["kind"],
                "mechanics": self._mechanics.currentText(),
                "optics": self._optics.currentText(),
                "detector": detector_key,
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
        # Detector overrides are only valid for the (kw-accepting) photodiode;
        # the option-less stub must construct with no arguments (14 §5).
        if detector_key == "photodiode":
            payload["detector"] = {
                "balanced": self._balanced.isChecked(),
                "reference_arm": self._reference_arm.currentText(),
            }
        return payload
