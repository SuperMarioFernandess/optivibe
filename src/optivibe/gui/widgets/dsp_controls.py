"""Experiment panel: the inverse chain, exposed knob by knob (task S-22 W-1).

Every control here already existed in
:class:`~optivibe.core.config.models.DspOptions` and behind the
``INTEGRATOR_REGISTRY`` / ``SENSITIVITY_REGISTRY`` registries; W-1 invents no
new physics, it *shows* the choices the model has always accepted so they can
be tried, compared and taught (backlog 16, S-22).

Three properties carry the design (coordination decision of 2026-07-29):

**The boundary of the verified is visible.** The default chain -- and only it --
is what the golden set and plans 18/19 cover, so the panel grades itself on
every edit: :func:`~optivibe.analysis.compare.chain_status` computes the
verdict from the values, the badge shows it, and the deviation list spells out
exactly which knobs left the default. Nothing here can *declare* a chain
verified.

**A knob that does nothing says so.** Each row carries an applicability tag --
batch / stream / both -- taken from
:data:`~optivibe.analysis.compare.CHAIN_APPLICABILITY`, not invented locally.
The batch-only ones are batch-only for a reason worth learning: a real-time
chain cannot know the whole record, so a zero-phase spectral integrator, a
record-length rFFT or a Welch segmentation of it simply do not exist there
(``docs/theory/06_dsp_algorithm.md`` §3.6/§7.6). Showing such a control as if
it acted on the live stream would be the panel lying to its user.

**Config-first.** The panel is a *view* of ``DspOptions``: it emits payloads,
never runs anything, and holds no truth of its own. That is what lets the same
experiment run head-less as ``optivibe compare`` -- and what lets the two
mirror combos on the ``Physics layers`` tab stay in step with it (they read and
write the same values rather than keeping a copy).
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from optivibe.analysis.compare import (
    CHAIN_APPLICABILITY,
    DEFAULT_CHAIN,
    chain_deltas,
    chain_status,
)
from optivibe.core.config.models import DspOptions
from optivibe.gui.i18n import t, tr
from optivibe.gui.widgets.ui_helpers import with_help

__all__ = ["DspControls"]

#: Window family offered for the spectral estimators. All of them are plain
#: ``scipy.signal.get_window`` names -- the model field is ``window: str``, so
#: this list adds a choice, not a mechanism (S-22 W-3, window branch).
_WINDOWS: tuple[str, ...] = ("hann", "hamming", "blackman", "nuttall", "flattop", "boxcar")

#: ``0`` in the numeric fields means "not set" (``None`` in the model): the
#: variant band edge for ``f_hp_hz`` / ``f_c_stream`` and the estimator's own
#: choice for the Welch lengths. Spelled out as text so an empty control never
#: reads as a numeric zero.
_AUTO = 0

#: Row order of the panel: field -> (label msgid, inline-help catalog key).
#: The field set must match :data:`~optivibe.analysis.compare.EXPERIMENT_FIELDS`
#: (pinned by ``tests/test_gui_compare.py``), so a new option cannot appear in
#: the spec without appearing on screen.
_ROWS: tuple[tuple[str, str, str], ...] = (
    ("integrator", "Integrator", "dsp.exp.integrator.help"),
    ("spectrum_method", "Spectrum", "dsp.exp.spectrum.help"),
    ("window", "Window", "dsp.exp.window.help"),
    ("welch_nperseg", "Welch nperseg", "dsp.exp.nperseg.help"),
    ("welch_noverlap", "Welch noverlap", "dsp.exp.noverlap.help"),
    ("f_hp_hz", "High-pass f_hp", "dsp.exp.f_hp.help"),
    ("f_c_stream", "Streaming cut-off f_c", "dsp.exp.f_c.help"),
    ("sensitivity_model", "Sensitivity model", "dsp.exp.sens.help"),
    ("sensitivity_freq", "Sensitivity vs f", "dsp.exp.sensf.help"),
    ("peak_interpolation", "Peak interpolation", "dsp.exp.interp.help"),
)


class DspControls(QWidget):
    """Editor of the inverse-chain options with the verified/experimental verdict.

    Parameters
    ----------
    parent : QWidget or None, optional
        Parent widget.
    """

    #: Emitted after any option changed (carries no payload: read the panel).
    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Fields outside EXPERIMENT_FIELDS (calibration -- task S-04;
        # deconvolve_hlat -- driven by sensitivity_freq; iso_machine_class)
        # are not editable here but must survive a round-trip, so the panel
        # keeps the chain it was given as the base of every payload.
        self._base = DEFAULT_CHAIN
        self._loading = False

        self._integrator = self._combo(("frequency", "time", "leaky"))
        self._spectrum_method = self._combo(("fft", "welch"))
        self._window = self._combo(_WINDOWS)
        self._window.setCurrentText(DEFAULT_CHAIN.window)
        self._welch_nperseg = self._int_spin(maximum=1 << 20, step=256)
        self._welch_noverlap = self._int_spin(maximum=1 << 20, step=128)
        self._f_hp_hz = self._float_spin(maximum=1.0e5, decimals=3, step=0.5)
        self._f_c_stream = self._float_spin(maximum=1.0e5, decimals=3, step=0.5)
        self._sensitivity_model = self._combo(("static", "operating_point", "nonlinear_curve"))
        self._sensitivity_freq = self._combo(("plateau", "dynamic"))
        self._peak_interpolation = QCheckBox()
        self._peak_interpolation.setChecked(DEFAULT_CHAIN.peak_interpolation)

        self._status = QLabel()
        self._status.setFrameShape(QFrame.Shape.StyledPanel)
        self._status.setWordWrap(True)
        self._reset = QPushButton(t("Reset to default"))
        self._reset.clicked.connect(self.reset_to_default)

        self._group = QGroupBox(t("Experiment: DSP chain"))
        form = QFormLayout(self._group)
        self._widgets: dict[str, QWidget] = {
            "integrator": self._integrator,
            "spectrum_method": self._spectrum_method,
            "window": self._window,
            "welch_nperseg": self._welch_nperseg,
            "welch_noverlap": self._welch_noverlap,
            "f_hp_hz": self._f_hp_hz,
            "f_c_stream": self._f_c_stream,
            "sensitivity_model": self._sensitivity_model,
            "sensitivity_freq": self._sensitivity_freq,
            "peak_interpolation": self._peak_interpolation,
        }
        self._labels: dict[str, QLabel] = {}
        for field, title, help_key in _ROWS:
            label = QLabel(self._row_label(title, field))
            self._labels[field] = label
            form.addRow(label, with_help(self._widgets[field], t(title), tr(help_key)))

        badge_row = QHBoxLayout()
        badge_row.addWidget(self._status, 1)
        badge_row.addWidget(self._reset)

        layout = QVBoxLayout(self)
        layout.addWidget(self._group)
        layout.addLayout(badge_row)
        layout.addStretch(1)

        for combo in (
            self._integrator,
            self._spectrum_method,
            self._window,
            self._sensitivity_model,
            self._sensitivity_freq,
        ):
            combo.currentTextChanged.connect(lambda _text: self._on_edit())
        for spin in (self._welch_nperseg, self._welch_noverlap):
            spin.valueChanged.connect(lambda _value: self._on_edit())
        for dspin in (self._f_hp_hz, self._f_c_stream):
            dspin.valueChanged.connect(lambda _value: self._on_edit())
        self._peak_interpolation.toggled.connect(lambda _checked: self._on_edit())
        self._refresh_status()

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _combo(items: tuple[str, ...]) -> QComboBox:
        """Build a combo box over registry keys (the config values themselves)."""
        box = QComboBox()
        box.addItems(items)
        return box

    @staticmethod
    def _int_spin(*, maximum: int, step: int) -> QSpinBox:
        """Build an integer field whose zero reads ``auto`` (model ``None``)."""
        spin = QSpinBox()
        spin.setRange(_AUTO, maximum)
        spin.setSingleStep(step)
        spin.setSpecialValueText(t("auto"))
        spin.setValue(_AUTO)
        return spin

    @staticmethod
    def _float_spin(*, maximum: float, decimals: int, step: float) -> QDoubleSpinBox:
        """Build a frequency field whose zero reads ``band edge`` (model ``None``)."""
        spin = QDoubleSpinBox()
        spin.setRange(float(_AUTO), maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setSuffix(" Hz")
        spin.setSpecialValueText(t("band edge"))
        spin.setValue(float(_AUTO))
        return spin

    @staticmethod
    def _row_label(title: str, field: str) -> str:
        """Return the row label with its applicability tag (batch / stream / both)."""
        return f"{t(title)} [{tr('dsp.exp.applies.' + CHAIN_APPLICABILITY[field])}]"

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    def dsp_options(self) -> DspOptions:
        """Return the chain currently described by the panel.

        Returns
        -------
        DspOptions
            Validated options: the exposed fields from the widgets, the rest
            carried over from the chain last loaded (``calibration`` and the
            ISO class are not edited here).
        """
        return self._base.model_copy(
            update={
                "integrator": self._integrator.currentText(),
                "spectrum_method": self._spectrum_method.currentText(),
                "window": self._window.currentText(),
                "welch_nperseg": _none_if_auto(self._welch_nperseg.value()),
                "welch_noverlap": _none_if_auto_overlap(self._welch_noverlap.value()),
                "f_hp_hz": _none_if_zero(self._f_hp_hz.value()),
                "f_c_stream": _none_if_zero(self._f_c_stream.value()),
                "sensitivity_model": self._sensitivity_model.currentText(),
                "sensitivity_freq": self._sensitivity_freq.currentText(),
                "peak_interpolation": self._peak_interpolation.isChecked(),
            }
        )

    def dsp_payload(self) -> dict[str, Any]:
        """Return the ``dsp`` block of a scenario payload.

        Returns
        -------
        dict
            Mapping accepted by ``build_scenario_config``; identical to the
            model default while the panel is untouched, which is what keeps the
            default run bit-identical (the acceptance check of W-1).
        """
        return dict(self.dsp_options().model_dump(mode="json"))

    def set_dsp_options(self, options: DspOptions) -> None:
        """Load a chain into the panel without re-emitting :attr:`changed`.

        Parameters
        ----------
        options : DspOptions
            The chain to display (e.g. restored session state, or a mirror
            control on the ``Physics layers`` tab moving).
        """
        self._base = options
        self._loading = True
        try:
            self._integrator.setCurrentText(options.integrator)
            self._spectrum_method.setCurrentText(options.spectrum_method)
            if self._window.findText(options.window) < 0:
                self._window.addItem(options.window)
            self._window.setCurrentText(options.window)
            self._welch_nperseg.setValue(options.welch_nperseg or _AUTO)
            self._welch_noverlap.setValue(
                _AUTO if options.welch_noverlap is None else options.welch_noverlap
            )
            self._f_hp_hz.setValue(options.f_hp_hz or float(_AUTO))
            self._f_c_stream.setValue(options.f_c_stream or float(_AUTO))
            self._sensitivity_model.setCurrentText(options.sensitivity_model)
            self._sensitivity_freq.setCurrentText(options.sensitivity_freq)
            self._peak_interpolation.setChecked(options.peak_interpolation)
        finally:
            self._loading = False
        self._refresh_status()

    def load_payload(self, payload: dict[str, Any]) -> None:
        """Load a ``dsp`` payload mapping (session restore, SW-65).

        Parameters
        ----------
        payload : dict
            Mapping as produced by :meth:`dsp_payload`; unknown or missing
            fields fall back to the model defaults.
        """
        self.set_dsp_options(DspOptions.model_validate(payload))

    def reset_to_default(self) -> None:
        """Return every knob to the verified default chain (W-1 requirement)."""
        self.set_dsp_options(DEFAULT_CHAIN)
        self.changed.emit()

    def status(self) -> str:
        """Return ``"verified"`` or ``"experimental"`` for the current chain."""
        return chain_status(self.dsp_options())

    def status_text(self) -> str:
        """Return the badge line as displayed (exposed for tests)."""
        return str(self._status.text())

    def integrator_combo(self) -> QComboBox:
        """Return the integrator combo (mirrored on the ``Physics layers`` tab)."""
        return self._integrator

    def sensitivity_combo(self) -> QComboBox:
        """Return the sensitivity-model combo (mirrored on ``Physics layers``)."""
        return self._sensitivity_model

    def set_chain_enabled(self, enabled: bool) -> None:
        """Enable or disable the whole chain editor.

        Parameters
        ----------
        enabled : bool
            ``False`` while the selected DSP stage is not the standard chain
            (the stub has no options to vary) or while a run is in flight.
        """
        self._group.setEnabled(enabled)
        self._reset.setEnabled(enabled)

    def retranslate(self) -> None:
        """Refresh the static text after a language change."""
        self._group.setTitle(t("Experiment: DSP chain"))
        self._reset.setText(t("Reset to default"))
        for field, title, _help in _ROWS:
            self._labels[field].setText(self._row_label(title, field))
        for spin in (self._welch_nperseg, self._welch_noverlap):
            spin.setSpecialValueText(t("auto"))
        for dspin in (self._f_hp_hz, self._f_c_stream):
            dspin.setSpecialValueText(t("band edge"))
        self._refresh_status()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _on_edit(self) -> None:
        """Re-grade the chain and notify listeners (unless we are loading)."""
        if self._loading:
            return
        self._refresh_status()
        self.changed.emit()

    def _refresh_status(self) -> None:
        """Redraw the verdict badge and the deviation list."""
        options = self.dsp_options()
        deltas = chain_deltas(options)
        if not deltas:
            self._status.setText(tr("dsp.exp.verified"))
            return
        listing = "; ".join(delta.as_text() for delta in deltas)
        self._status.setText(tr("dsp.exp.experimental", deviations=listing))


def _none_if_auto(value: int) -> int | None:
    """Map the ``auto`` sentinel of a positive-only field to ``None``."""
    return None if value == _AUTO else value


def _none_if_auto_overlap(value: int) -> int | None:
    """Map the overlap field to ``None`` at ``auto`` (``0`` is a legal overlap).

    ``welch_noverlap`` accepts zero (no overlap), so the sentinel and a real
    value collide here. The panel resolves it the honest way: the sentinel wins
    and a deliberate zero overlap is expressed in the YAML, where it cannot be
    confused with "not set" (config-first).
    """
    return None if value == _AUTO else value


def _none_if_zero(value: float) -> float | None:
    """Map the ``band edge`` sentinel of a frequency field to ``None``."""
    return None if value == float(_AUTO) else value
