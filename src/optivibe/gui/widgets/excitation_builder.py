"""Excitation builder widget: the ``ExcitationSpec`` union (task S7 §2, S-21).

A ``kind`` selector over a stacked form (sine / multitone / sweep / random /
shock / composite, plus CSV / WAV / TDMS / UFF / MAT / HDF5 replay via the
loader registry).
It collects a *payload* mapping that
:func:`optivibe.gui.controllers.scenario_builder.build_excitation_spec`
validates -- the widget holds no signal logic, only the input fields (09 §9).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import yaml
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from optivibe.gui.i18n import t
from optivibe.gui.widgets.ui_helpers import with_help

__all__ = ["ExcitationBuilder"]

_KINDS = (
    "sine",
    "multitone",
    "sweep",
    "random",
    "shock",
    "composite",
    "csv",
    "wav",
    "tdms",
    "uff",
    "mat",
    "hdf5",
)
_GENERATED = {"sine", "multitone", "sweep", "random", "shock", "composite"}

#: Carrier modulators offered on the sine page (doc 11 §2.1.3); "none" keeps the
#: unmodulated payload, which is byte-identical to the pre-S-21 one.
_MODULATIONS = ("none", "am", "fm")

_KIND_HELP = (
    "What drives the sensor along the chosen axis:\n\n"
    "sine / multitone / sweep / random / shock -- GENERATED waveforms on the "
    "sampling grid below (fs, duration).\n"
    "composite -- the SUM of several generated components on that same grid "
    "(levels are not renormalized).\n"
    "csv / wav / tdms / uff / mat / hdf5 -- REPLAY of a recorded acceleration "
    "from a file (the grid comes from the file or the page fields; the "
    "sampling row is hidden).\n\nThe excitation is ground acceleration in g "
    "along one axis; pick the kind first, then fill its page."
)
_AXIS_HELP = (
    "Excitation axis in the sensor frame (doc 00): x -- the TARGET axis of "
    "the version-1 cylinder reflector (full response); y / z -- the cross "
    "axes, used to probe the cross-axis sensitivity metric. For the "
    "isotropic sphere the transverse axes are equivalent."
)
_SAMPLING_HELP = (
    "Grid of the generated waveform: fs -- sample rate, Hz (keep fs >= "
    "2.56 x the highest excited frequency for a clean spectrum); duration -- "
    "record length, s (sets the spectral resolution df = 1/T and how many "
    "periods the metrics average over). Hidden for file replay: the grid "
    "then comes from the file."
)
_ABOUT = {
    "sine": (
        "single tone",
        "One tone: frequency [Hz], amplitude [g] and an optional phase [rad]. "
        "The basic probe of one band point -- dominant-frequency recovery, "
        "2f/1f distortion at a bias~0 working point, RMS checks. Keep the "
        "frequency well below f1 for off-resonance use.\n\n"
        "Modulation (optional, doc 11 §2.1.3) turns the tone into a carrier: "
        "AM adds one sideband pair of amplitude m*a_c/2 at f_c +- f_m; FM adds "
        "the Bessel family a_c*|J_k(beta)| at f_c +- k*f_m with "
        "beta = deviation/f_m. Both are marked in the spectrum by the "
        "expected-peak layer. Depth m is limited to 0..1; a deeper AM is "
        "expressible as a composite of carrier + two sideband tones.",
    ),
    "multitone": (
        "sum of tones",
        "A sum of components, each [frequency Hz, amplitude g] and an "
        "optional per-tone phase [rad]. Add/remove components freely; probes "
        "intermodulation and superposition. The crest factor grows with the "
        "component count -- watch the full-scale clipping.",
    ),
    "sweep": (
        "chirp f0 -> f1",
        "A constant-amplitude chirp from f start to f end [Hz], linear or "
        "log in frequency. The standard way to trace the frequency response "
        "over the band in one run; log spacing spends more time at low "
        "frequencies.",
    ),
    "random": (
        "band-limited noise",
        "Gaussian noise band-limited to [lo, hi] Hz with the given g RMS. "
        "ISO-style broadband excitation; PSD-based metrics apply. The peak "
        "factor is ~3-4x the RMS -- watch the full scale.",
    ),
    "shock": (
        "half-sine pulse",
        "A half-sine shock: peak [g], pulse width [ms], start delay [s]. "
        "For transient/overload studies -- pair it with the modal_time "
        "mechanics and the time integrator (Physics layers tab) for a "
        "faithful transient.",
    ),
    "composite": (
        "sum of components",
        "The sum of several generated components on one sampling grid (doc 11 "
        "§2.1.4). Each component keeps the level it declares -- the sum is NOT "
        "renormalized, so the RMS adds in power and the peak may pass full "
        "scale (a warning, not an error).\n\n"
        "The components tab holds one sub-form per component kind (sine / "
        "multitone / sweep / random / shock), added and removed row by row, "
        "each on the composite's axis or on its own -- the parts are summed per "
        "axis. The YAML tab "
        "shows the same components as a scenario file carries them, for what "
        "the sub-forms do not cover; switching tabs converts between the two. "
        "fs / duration are not per component: the grid is defined once, on the "
        "composite above. File-replay kinds and nested composites are not "
        "admissible components.\n\n"
        "Noise components: component 0 inherits the run seed, later ones get a "
        "deterministic sub-seed from their position; give a component its own "
        "'seed:' to pin its realization irrespective of position.",
    ),
    "csv": (
        "CSV replay",
        "Replay a recorded acceleration column from a CSV: column index "
        "(0-based), sample rate fs [Hz] (CSV stores no grid), units of the "
        "stored values (m/s^2 or g).",
    ),
    "wav": (
        "WAV replay",
        "Replay a WAV channel as acceleration: channel index and the "
        "full-scale mapping [g] (WAV samples are normalized to +/-1, so "
        "full scale sets how many g that is).",
    ),
    "tdms": (
        "NI TDMS replay",
        "Replay an NI TDMS channel: group (blank = first), channel index, "
        "fs [Hz] (0 = take wf_increment from the file), units of the stored "
        "values.",
    ),
    "uff": (
        "UFF/UNV replay",
        "Replay a UFF/UNV dataset-58 record: dataset index, fs [Hz] (0 = "
        "take the abscissa increment from the file), units of the stored "
        "values.",
    ),
    "mat": (
        "MATLAB replay",
        "Replay a variable from a MATLAB .mat file: variable name, column "
        "index for 2-D arrays, fs [Hz] (required -- .mat stores no grid), "
        "units of the stored values.",
    ),
    "hdf5": (
        "HDF5 replay",
        "Replay an HDF5 dataset: dataset path inside the file, column index "
        "for 2-D data, fs [Hz] (0 = take an fs attribute from the file when "
        "present), units of the stored values.",
    ),
}


def _spin(
    minimum: float, maximum: float, value: float, decimals: int = 3, step: float = 1.0
) -> QDoubleSpinBox:
    """Build a configured double spin box."""
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    return box


def _note(label: QLabel) -> QLabel:
    """Style a label as an inline explanatory note (same look as the ``about`` row)."""
    label.setWordWrap(True)
    label.setStyleSheet("color: #808080; font-style: italic;")
    return label


#: ``(minimum, maximum, decimals)`` of the tone-row spin boxes. Shared by the
#: row builder and by :meth:`_MultitoneForm.load_tones`, so "can this widget
#: hold that value exactly?" is answered from one place (S-23).
_TONE_FREQ = (0.1, 1.0e5, 2)
_TONE_AMP = (1.0e-3, 200.0, 3)
_TONE_PHASE = (-3.1416, 3.1416, 3)


def _fits(value: object, spec: tuple[float, float, int]) -> bool:
    """Return whether a config value fits a spin box exactly (range + step).

    Parameters
    ----------
    value : object
        Candidate value from a configuration mapping.
    spec : tuple
        ``(minimum, maximum, decimals)`` of the target spin box.

    Returns
    -------
    bool
        ``True`` when the widget would store the value unchanged. A value it
        would clamp or round is *not* representable: the form declines to load
        it rather than silently rewriting the user's config (S-23).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    minimum, maximum, decimals = spec
    if not minimum <= number <= maximum:
        return False
    return abs(round(number, decimals) - number) <= max(1e-12, abs(number) * 1e-9)


@dataclass
class _ToneRow:
    """Widgets of one multitone component row."""

    holder: QWidget
    freq_spin: QDoubleSpinBox
    amp_spin: QDoubleSpinBox
    phase_spin: QDoubleSpinBox
    phase_label: QLabel


class _MultitoneForm(QWidget):
    """Dynamic multitone editor: add/remove components with optional phase.

    Defaults to two components (task S7-mod §3). The core accepts an arbitrary
    number of tones (:class:`~optivibe.core.config.models.MultitoneSpec` has
    ``min_length=1`` and no upper bound) and an optional per-tone phase, so this
    widget needs no core change. A row collects ``[frequency_hz, amplitude_g]``
    or, when *include phase* is checked, ``[frequency_hz, amplitude_g, phase_rad]``.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[_ToneRow] = []
        self._phase = QCheckBox(t("include per-tone phase"))
        self._phase.toggled.connect(self._on_phase_toggled)
        self._add_button = QPushButton(t("+ component"))
        self._add_button.clicked.connect(lambda: self._add_row(240.0, 0.5, 0.0))

        self._rows_box = QVBoxLayout()
        self._rows_box.setContentsMargins(0, 0, 0, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._phase)
        layout.addLayout(self._rows_box)
        layout.addWidget(self._add_button)

        self._add_row(120.0, 1.0, 0.0)
        self._add_row(240.0, 0.5, 0.0)

    def _add_row(self, freq: float, amp: float, phase: float) -> None:
        """Append a tone row pre-filled with the given values."""
        freq_spin = _spin(_TONE_FREQ[0], _TONE_FREQ[1], freq, decimals=_TONE_FREQ[2], step=10.0)
        amp_spin = _spin(_TONE_AMP[0], _TONE_AMP[1], amp, decimals=_TONE_AMP[2], step=0.1)
        phase_spin = _spin(_TONE_PHASE[0], _TONE_PHASE[1], phase, decimals=_TONE_PHASE[2], step=0.1)
        phase_label = QLabel(t("phase [rad]"))
        remove = QPushButton(t("x"))
        remove.setMaximumWidth(28)

        row_layout = QHBoxLayout()
        row_layout.addWidget(QLabel(t("f [Hz]")))
        row_layout.addWidget(freq_spin)
        row_layout.addWidget(QLabel(t("amp [g]")))
        row_layout.addWidget(amp_spin)
        row_layout.addWidget(phase_label)
        row_layout.addWidget(phase_spin)
        row_layout.addWidget(remove)
        holder = QWidget()
        holder.setLayout(row_layout)

        phase_label.setVisible(self._phase.isChecked())
        phase_spin.setVisible(self._phase.isChecked())

        entry = _ToneRow(holder, freq_spin, amp_spin, phase_spin, phase_label)
        self._rows.append(entry)
        self._rows_box.addWidget(holder)
        remove.clicked.connect(lambda: self._remove_row(entry))

    def _remove_row(self, entry: _ToneRow) -> None:
        """Remove a tone row (keeping at least one component)."""
        if len(self._rows) <= 1 or entry not in self._rows:
            return
        self._rows.remove(entry)
        self._rows_box.removeWidget(entry.holder)
        entry.holder.setParent(None)
        entry.holder.deleteLater()

    def _on_phase_toggled(self, checked: bool) -> None:
        """Show or hide the per-tone phase controls."""
        for entry in self._rows:
            entry.phase_spin.setVisible(checked)
            entry.phase_label.setVisible(checked)

    def count(self) -> int:
        """Return the number of components (exposed for tests)."""
        return len(self._rows)

    def set_tones(self, tones: list[list[float]]) -> None:
        """Replace the component rows from ``[[f, a]]`` / ``[[f, a, phase]]``.

        Used to restore state across a language rebuild (SW-65).
        """
        for entry in list(self._rows):
            self._rows.remove(entry)
            self._rows_box.removeWidget(entry.holder)
            entry.holder.setParent(None)
            entry.holder.deleteLater()
        has_phase = any(len(tone) >= 3 for tone in tones)
        self._phase.setChecked(has_phase)
        for tone in tones or [[240.0, 0.5, 0.0]]:
            freq = float(tone[0])
            amp = float(tone[1])
            phase = float(tone[2]) if len(tone) >= 3 else 0.0
            self._add_row(freq, amp, phase)

    def load_tones(self, tones: object) -> bool:
        """Replace the rows from a *configuration* value, if it is representable.

        Unlike :meth:`set_tones` -- which restores a payload this widget itself
        produced -- this accepts arbitrary parsed YAML and answers whether the
        rows can hold it exactly (S-23). Only the compact sequence form is
        accepted; the mapping form of ``Tone`` stays on the YAML path.

        Parameters
        ----------
        tones : object
            Candidate ``tones`` value of a ``multitone`` component.

        Returns
        -------
        bool
            ``True`` when the rows now hold exactly that value, ``False`` when
            nothing was changed because the value is not representable.
        """
        if not isinstance(tones, (list, tuple)) or not tones:
            return False
        parsed: list[list[float]] = []
        for tone in tones:
            if not isinstance(tone, (list, tuple)) or not 2 <= len(tone) <= 3:
                return False
            specs = (_TONE_FREQ, _TONE_AMP, _TONE_PHASE)
            if not all(_fits(value, spec) for value, spec in zip(tone, specs, strict=False)):
                return False
            parsed.append([float(value) for value in tone])
        self.set_tones(parsed)
        return True

    def tones(self) -> list[list[float]]:
        """Collect the tones as ``[[f, a]]`` or ``[[f, a, phase]]`` lists."""
        include_phase = self._phase.isChecked()
        tones: list[list[float]] = []
        for entry in self._rows:
            tone = [entry.freq_spin.value(), entry.amp_spin.value()]
            if include_phase:
                tone.append(entry.phase_spin.value())
            tones.append(tone)
        return tones


# --------------------------------------------------------------------------- #
# Composite components: one sub-form per component kind (task S-23).
# --------------------------------------------------------------------------- #
#: Kinds admissible as a composite component -- the generated kinds minus the
#: composite itself (file replay and nesting are excluded, doc 11 §2.1.4).
_COMPONENT_KINDS = tuple(kind for kind in _KINDS if kind in _GENERATED and kind != "composite")

#: Excitation axes (doc 00); a composite sums its parts per axis (doc 11 §2.1.4).
_AXES = ("x", "y", "z")

#: Index of the "inherit the composite axis" entry of a component's axis combo.
#: A component that does not name an axis takes the composite's (doc 11 §2.1.4),
#: so the sentinel is the *absence* of the key, not a value of it.
_AXIS_INHERIT = 0

_GRID_NOTE = (
    "Sampling grid of every component: fs = {fs} Hz, duration = {dur} s. The "
    "grid is defined once, above, on the composite -- a component that "
    "disagreed with it would be a loud error (doc 11 §2.1.4), so it is not "
    "editable per component here."
)
_SEED_NOTE = (
    "Seeding (doc 11 §2.1.5): the first component inherits the run seed, later "
    "ones get a deterministic sub-seed from their position -- appending never "
    "moves the existing noise streams. A noise component can pin its own seed "
    "instead, and then keeps its realization wherever it sits."
)
_OVER_MODULATION_NOTE = (
    "Depth m > 1 is rejected when the run starts: the envelope changes sign and "
    "m stops being an amplitude ratio (doc 11 §2.1.3). The same waveform is "
    "expressible exactly as three sine components -- the carrier a_c at f_c "
    "plus two sidebands m*a_c/2 at f_c +- f_m."
)
_YAML_ONLY_NOTE = (
    "These components stay on the YAML path: the sub-forms cannot hold them "
    "exactly (an unknown field, a PSD noise level, a value outside a field's "
    "range or step). They are edited here rather than silently rounded."
)


def _load_spin(spin: QDoubleSpinBox, value: object) -> bool:
    """Set a spin box from a configuration value, if it is representable.

    Parameters
    ----------
    spin : QDoubleSpinBox
        Target widget.
    value : object
        Candidate value from a component mapping.

    Returns
    -------
    bool
        ``True`` when the widget now holds the value exactly; ``False`` (nothing
        set) when it would clamp or round it -- such a component is left to the
        YAML path instead of being silently rewritten (S-23).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not _fits(value, (spin.minimum(), spin.maximum(), spin.decimals())):
        return False
    spin.setValue(float(value))
    return True


class _ComponentForm(QWidget):
    """Kind-specific fields of one composite component (task S-23).

    Subclasses own the fields of a single ``kind`` and answer two questions:
    what payload they hold, and whether they can hold a given mapping *exactly*.
    They carry no signal logic -- the payload is validated by
    ``build_excitation_spec`` like every other page (09 §9).
    """

    #: Component ``kind`` this page edits.
    KIND = ""
    #: Configuration keys the page can hold.
    KEYS: tuple[str, ...] = ()
    #: Keys accepted, and dropped, when they carry their model default (so a
    #: round-tripped ``model_dump`` still opens in the form).
    DEFAULTS: ClassVar[dict[str, object]] = {}

    def payload(self) -> dict[str, Any]:
        """Return the kind-specific fields of the component."""
        raise NotImplementedError

    def load(self, data: Mapping[str, Any]) -> bool:
        """Fill the fields from a component mapping.

        Parameters
        ----------
        data : Mapping[str, Any]
            Component fields without ``kind`` / ``axis`` / the grid.

        Returns
        -------
        bool
            ``True`` when the widgets now hold the mapping exactly.
        """
        raise NotImplementedError

    def _unknown(self, data: Mapping[str, Any]) -> bool:
        """Return whether *data* carries a key this page cannot hold."""
        return any(key not in self.KEYS for key in data)


class _SineComponentForm(_ComponentForm):
    """A ``sine`` component with the opt-in AM/FM carrier (doc 11 §2.1.3).

    The depth spin box deliberately reaches past the admissible ``m <= 1``: a
    config that carries an over-modulation is loaded as written and explained,
    not clamped into a different signal. The authoritative rejection stays in
    ``AmModulation`` (10 §7), which names the three-tone route in its message.
    """

    KIND = "sine"
    KEYS = ("frequency_hz", "amplitude_g", "phase_rad", "modulation")
    _MOD_KEYS = ("kind", "f_m_hz", "depth", "deviation_hz", "phase_rad")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._freq = _spin(0.1, 1.0e5, 1000.0, decimals=4, step=10.0)
        self._amp = _spin(1.0e-6, 200.0, 1.0, decimals=6, step=0.1)
        self._phase = _spin(-6.2832, 6.2832, 0.0, decimals=4, step=0.1)
        self._mod = QComboBox()
        self._mod.addItems(_MODULATIONS)
        self._mod_fm = _spin(1.0e-3, 1.0e5, 37.0, decimals=4, step=1.0)
        self._mod_depth = _spin(0.0, 2.0, 0.4, decimals=4, step=0.05)
        self._mod_dev = _spin(0.0, 1.0e5, 50.0, decimals=4, step=1.0)
        self._mod_phase = _spin(-6.2832, 6.2832, 0.0, decimals=4, step=0.1)
        self._warning = QLabel(t(_OVER_MODULATION_NOTE))
        self._warning.setWordWrap(True)
        self._warning.setStyleSheet("color: #b00020;")
        self._fm_label = QLabel(t("f mod [Hz]"))
        self._depth_label = QLabel(t("depth m"))
        self._dev_label = QLabel(t("deviation [Hz]"))
        self._mod_phase_label = QLabel(t("mod phase [rad]"))

        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow(t("frequency [Hz]"), self._freq)
        form.addRow(t("amplitude [g]"), self._amp)
        form.addRow(t("phase [rad]"), self._phase)
        form.addRow(t("modulation"), self._mod)
        form.addRow(self._fm_label, self._mod_fm)
        form.addRow(self._depth_label, self._mod_depth)
        form.addRow(self._dev_label, self._mod_dev)
        form.addRow(self._mod_phase_label, self._mod_phase)
        form.addRow(self._warning)

        self._mod.currentTextChanged.connect(self._on_mode_changed)
        self._mod_depth.valueChanged.connect(lambda _value: self._update_warning())
        self._on_mode_changed(self._mod.currentText())

    def _on_mode_changed(self, mode: str) -> None:
        """Show only the fields the selected modulator needs (doc 11 §2.1.3)."""
        modulated = mode != "none"
        for label, widget in (
            (self._fm_label, self._mod_fm),
            (self._mod_phase_label, self._mod_phase),
        ):
            label.setVisible(modulated)
            widget.setVisible(modulated)
        self._depth_label.setVisible(mode == "am")
        self._mod_depth.setVisible(mode == "am")
        self._dev_label.setVisible(mode == "fm")
        self._mod_dev.setVisible(mode == "fm")
        self._update_warning()

    def _update_warning(self) -> None:
        """Show the over-modulation note while the depth is out of range."""
        self._warning.setVisible(self._mod.currentText() == "am" and self._mod_depth.value() > 1.0)

    def payload(self) -> dict[str, Any]:
        """Return the sine fields, with ``modulation`` only when opted in."""
        base: dict[str, Any] = {
            "frequency_hz": self._freq.value(),
            "amplitude_g": self._amp.value(),
        }
        if self._phase.value() != 0.0:
            base["phase_rad"] = self._phase.value()
        mode = self._mod.currentText()
        if mode == "none":
            return base
        modulation: dict[str, Any] = {"kind": mode, "f_m_hz": self._mod_fm.value()}
        if mode == "am":
            modulation["depth"] = self._mod_depth.value()
        else:
            modulation["deviation_hz"] = self._mod_dev.value()
        if self._mod_phase.value() != 0.0:
            modulation["phase_rad"] = self._mod_phase.value()
        base["modulation"] = modulation
        return base

    def load(self, data: Mapping[str, Any]) -> bool:
        """Fill the carrier and, when present, its modulator."""
        if self._unknown(data) or not {"frequency_hz", "amplitude_g"} <= set(data):
            return False
        if not _load_spin(self._freq, data["frequency_hz"]):
            return False
        if not _load_spin(self._amp, data["amplitude_g"]):
            return False
        if not _load_spin(self._phase, data.get("phase_rad", 0.0)):
            return False
        modulation = data.get("modulation")
        if modulation is None:
            self._mod.setCurrentText("none")
            self._on_mode_changed("none")
            return True
        if not isinstance(modulation, Mapping):
            return False
        mode = str(modulation.get("kind", ""))
        if mode not in ("am", "fm") or any(key not in self._MOD_KEYS for key in modulation):
            return False
        if "f_m_hz" not in modulation or not _load_spin(self._mod_fm, modulation["f_m_hz"]):
            return False
        if not _load_spin(self._mod_phase, modulation.get("phase_rad", 0.0)):
            return False
        if mode == "am":
            if "depth" not in modulation or not _load_spin(self._mod_depth, modulation["depth"]):
                return False
        elif "deviation_hz" not in modulation or not _load_spin(
            self._mod_dev, modulation["deviation_hz"]
        ):
            return False
        self._mod.setCurrentText(mode)
        self._on_mode_changed(mode)
        return True


class _MultitoneComponentForm(_ComponentForm):
    """A ``multitone`` component: the S7-mod tone editor, reused as a sub-form."""

    KIND = "multitone"
    KEYS = ("tones",)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tones = _MultitoneForm()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tones)

    def payload(self) -> dict[str, Any]:
        """Return the tone list of this component."""
        return {"tones": self._tones.tones()}

    def load(self, data: Mapping[str, Any]) -> bool:
        """Fill the tone rows from a ``tones`` sequence."""
        if self._unknown(data) or "tones" not in data:
            return False
        return self._tones.load_tones(data["tones"])


class _SweepComponentForm(_ComponentForm):
    """A ``sweep`` component: chirp bounds, level and spacing."""

    KIND = "sweep"
    KEYS = ("f_start_hz", "f_end_hz", "amplitude_g", "method")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._f0 = _spin(0.1, 1.0e5, 20.0, decimals=4, step=10.0)
        self._f1 = _spin(0.1, 1.0e5, 2000.0, decimals=4, step=10.0)
        self._amp = _spin(1.0e-6, 200.0, 1.0, decimals=6, step=0.1)
        self._method = QComboBox()
        self._method.addItems(("linear", "log"))
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow(t("f start [Hz]"), self._f0)
        form.addRow(t("f end [Hz]"), self._f1)
        form.addRow(t("amplitude [g]"), self._amp)
        form.addRow(t("method"), self._method)

    def payload(self) -> dict[str, Any]:
        """Return the sweep fields of this component."""
        return {
            "f_start_hz": self._f0.value(),
            "f_end_hz": self._f1.value(),
            "amplitude_g": self._amp.value(),
            "method": self._method.currentText(),
        }

    def load(self, data: Mapping[str, Any]) -> bool:
        """Fill the chirp bounds, level and spacing."""
        if self._unknown(data) or not {"f_start_hz", "f_end_hz", "amplitude_g"} <= set(data):
            return False
        if not _load_spin(self._f0, data["f_start_hz"]):
            return False
        if not _load_spin(self._f1, data["f_end_hz"]):
            return False
        if not _load_spin(self._amp, data["amplitude_g"]):
            return False
        method = str(data.get("method", "linear"))
        if method not in ("linear", "log"):
            return False
        self._method.setCurrentText(method)
        return True


class _RandomComponentForm(_ComponentForm):
    """A ``random`` component: band, RMS level and the optional pinned seed.

    A ``psd_g2_hz`` level is deliberately *not* offered here -- the standalone
    ``random`` page does not offer it either, and adding it on one page only
    would make the two disagree. Such a component stays on the YAML path.
    """

    KIND = "random"
    KEYS = ("band_hz", "g_rms", "seed")
    DEFAULTS: ClassVar[dict[str, object]] = {"shape": "flat", "psd_g2_hz": None}
    #: Upper bound of the seed spin box (``QSpinBox`` is 32-bit); a wider seed
    #: is representable in YAML only.
    _SEED_MAX = 2_147_483_647

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._lo = _spin(0.0, 1.0e5, 20.0, decimals=4, step=10.0)
        self._hi = _spin(0.1, 1.0e5, 2000.0, decimals=4, step=10.0)
        self._grms = _spin(1.0e-6, 200.0, 0.05, decimals=6, step=0.01)
        self._own_seed = QCheckBox(t("own seed"))
        self._seed = QSpinBox()
        self._seed.setRange(0, self._SEED_MAX)
        self._seed.setValue(4242)
        self._seed.setEnabled(False)
        self._own_seed.toggled.connect(self._seed.setEnabled)
        seed_row = QHBoxLayout()
        seed_row.setContentsMargins(0, 0, 0, 0)
        seed_row.addWidget(self._own_seed)
        seed_row.addWidget(self._seed)
        seed_holder = QWidget()
        seed_holder.setLayout(seed_row)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow(t("band lo [Hz]"), self._lo)
        form.addRow(t("band hi [Hz]"), self._hi)
        form.addRow(t("g RMS [g]"), self._grms)
        form.addRow(t("seed"), seed_holder)

    def payload(self) -> dict[str, Any]:
        """Return the band, level and (when pinned) the component seed."""
        base: dict[str, Any] = {
            "band_hz": [self._lo.value(), self._hi.value()],
            "g_rms": self._grms.value(),
        }
        if self._own_seed.isChecked():
            base["seed"] = self._seed.value()
        return base

    def load(self, data: Mapping[str, Any]) -> bool:
        """Fill the band, level and pinned seed."""
        if self._unknown(data):
            return False
        band = data.get("band_hz")
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            return False
        if not _load_spin(self._lo, band[0]) or not _load_spin(self._hi, band[1]):
            return False
        if not _load_spin(self._grms, data.get("g_rms")):
            return False
        seed = data.get("seed")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                return False
            if not 0 <= seed <= self._SEED_MAX:
                return False
            self._seed.setValue(seed)
        self._own_seed.setChecked(seed is not None)
        return True


class _ShockComponentForm(_ComponentForm):
    """A ``shock`` component: half-sine peak, width and pre-delay."""

    KIND = "shock"
    KEYS = ("peak_g", "pulse_ms", "delay_s")
    DEFAULTS: ClassVar[dict[str, object]] = {"shape": "half_sine"}

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._peak = _spin(1.0e-3, 1.0e4, 50.0, decimals=4, step=1.0)
        self._pulse = _spin(0.01, 1000.0, 2.0, decimals=4, step=0.1)
        self._delay = _spin(0.0, 60.0, 0.1, decimals=4, step=0.05)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow(t("peak [g]"), self._peak)
        form.addRow(t("pulse [ms]"), self._pulse)
        form.addRow(t("delay [s]"), self._delay)

    def payload(self) -> dict[str, Any]:
        """Return the pulse fields of this component."""
        return {
            "peak_g": self._peak.value(),
            "pulse_ms": self._pulse.value(),
            "delay_s": self._delay.value(),
        }

    def load(self, data: Mapping[str, Any]) -> bool:
        """Fill peak, width and pre-delay."""
        if self._unknown(data) or not {"peak_g", "pulse_ms"} <= set(data):
            return False
        if not _load_spin(self._peak, data["peak_g"]):
            return False
        if not _load_spin(self._pulse, data["pulse_ms"]):
            return False
        return _load_spin(self._delay, data.get("delay_s", 0.0))


class _ComponentRow(QWidget):
    """One composite component: kind, axis and the sub-form of that kind (S-23).

    The axis lives on the row rather than inside the kind pages because every
    kind has one: a composite sums its parts *per axis* (design decision 5 of
    SW-71), which is what lets one run drive several axes at once.
    """

    def __init__(
        self,
        on_remove: Callable[[_ComponentRow], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._kind = QComboBox()
        self._kind.addItems(_COMPONENT_KINDS)
        self._axis = QComboBox()
        self._axis.addItem(t("inherit"))
        self._axis.addItems(_AXES)
        remove = QPushButton(t("x"))
        remove.setMaximumWidth(28)
        remove.clicked.connect(lambda: on_remove(self))

        self._stack = QStackedWidget()
        self._forms: dict[str, _ComponentForm] = {}
        for factory in (
            _SineComponentForm,
            _MultitoneComponentForm,
            _SweepComponentForm,
            _RandomComponentForm,
            _ShockComponentForm,
        ):
            form = factory()
            self._forms[form.KIND] = form
            self._stack.addWidget(form)
        self._kind.currentTextChanged.connect(self._on_kind_changed)

        header = QHBoxLayout()
        header.addWidget(QLabel(t("component")))
        header.addWidget(self._kind)
        header.addWidget(QLabel(t("axis")))
        header.addWidget(self._axis)
        header.addStretch(1)
        header.addWidget(remove)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.addLayout(header)
        layout.addWidget(self._stack)
        self._on_kind_changed(self._kind.currentText())

    def _on_kind_changed(self, kind: str) -> None:
        """Show the sub-form of the selected component kind."""
        self._stack.setCurrentWidget(self._forms[kind])

    def payload(self) -> dict[str, Any]:
        """Return the component mapping (``kind``, its fields, and ``axis`` if named).

        ``axis`` is emitted only when the row names one: leaving it inherited is
        what keeps the composite's own axis row meaningful (doc 11 §2.1.4).
        """
        kind = self._kind.currentText()
        base: dict[str, Any] = {"kind": kind}
        if self._axis.currentIndex() != _AXIS_INHERIT:
            base["axis"] = self._axis.currentText()
        base.update(self._forms[kind].payload())
        return base

    def load(self, data: Mapping[str, Any], grid: tuple[float, float]) -> bool:
        """Fill the row from a component mapping.

        Parameters
        ----------
        data : Mapping[str, Any]
            One entry of a ``components`` list.
        grid : tuple of float
            The composite's ``(fs_hz, duration_s)``. A component may restate the
            shared grid, but a component that *disagrees* with it is a loud error
            (doc 11 §2.1.4) and is left to the YAML path, where the run reports
            it, rather than being quietly adopted here.

        Returns
        -------
        bool
            ``True`` when the row now holds the mapping exactly.
        """
        kind = data.get("kind")
        if not isinstance(kind, str) or kind not in self._forms:
            return False
        axis = data.get("axis")
        if axis is not None and axis not in _AXES:
            return False
        fields: dict[str, Any] = {}
        for key, value in data.items():
            if key in ("kind", "axis"):
                continue
            if key in ("fs_hz", "duration_s"):
                shared = grid[0] if key == "fs_hz" else grid[1]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return False
                if float(value) != shared:
                    return False
                continue
            fields[key] = value
        form = self._forms[kind]
        fields = {
            key: value
            for key, value in fields.items()
            if not (key in form.DEFAULTS and value == form.DEFAULTS[key])
        }
        if not form.load(fields):
            return False
        self._kind.setCurrentText(kind)
        if axis is None:
            self._axis.setCurrentIndex(_AXIS_INHERIT)
        else:
            self._axis.setCurrentText(str(axis))
        return True


class _CompositeForm(QWidget):
    """Components of a ``composite`` excitation: per-kind sub-forms + YAML (S-23).

    Config-first (doc 13, coordination 2026-07-29) now holds in both directions:
    the *components* tab edits the very mapping a scenario file carries, and the
    *YAML* tab shows that mapping as text. Switching tabs converts, so the text
    path S-21 shipped is kept beside the forms, not replaced by them -- it stays
    the way to express what the sub-forms do not cover.

    A component is loaded into the forms only when they can hold it **exactly**.
    An unknown field, a ``psd_g2_hz`` noise level, a value a spin box would round
    or clamp, or a component grid that contradicts the composite -- each keeps
    the whole list on the YAML tab, with the reason on screen. Rounding a user's
    config on the way through a form would be a silent failure (10 §7); an extra
    tab is not.

    Text that does not parse is passed through unchanged: validation then fails
    loudly in ``build_excitation_spec``, where the main window already reports
    it, instead of raising out of a payload getter that other code paths (the
    language rebuild) call outside a try block.
    """

    #: Components of a freshly-built page: an AM carrier over a noise floor --
    #: the stimulus the S-21 text placeholder carried, now as rows.
    _DEFAULT: tuple[dict[str, Any], ...] = (
        {
            "kind": "sine",
            "frequency_hz": 1000.0,
            "amplitude_g": 1.0,
            "modulation": {"kind": "am", "f_m_hz": 37.0, "depth": 0.4},
        },
        {"kind": "random", "band_hz": [20.0, 2000.0], "g_rms": 0.05},
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[_ComponentRow] = []
        self._grid: tuple[float, float] = (0.0, 0.0)
        self._syncing = False

        self._grid_note = _note(QLabel())
        self._seed_note = _note(QLabel(t(_SEED_NOTE)))
        self._yaml_note = _note(QLabel(t(_YAML_ONLY_NOTE)))
        self._yaml_note.setVisible(False)

        self._rows_box = QVBoxLayout()
        self._rows_box.setContentsMargins(0, 0, 0, 0)
        self._add_button = QPushButton(t("+ component"))
        self._add_button.clicked.connect(lambda: self._add_row())

        form_page = QWidget()
        form_layout = QVBoxLayout(form_page)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.addWidget(self._grid_note)
        form_layout.addWidget(self._seed_note)
        form_layout.addLayout(self._rows_box)
        form_layout.addWidget(self._add_button)

        self._text = QPlainTextEdit()
        self._text.setMinimumHeight(140)
        yaml_page = QWidget()
        yaml_layout = QVBoxLayout(yaml_page)
        yaml_layout.setContentsMargins(0, 0, 0, 0)
        yaml_layout.addWidget(self._yaml_note)
        yaml_layout.addWidget(self._text)

        self._tabs = QTabWidget()
        self._tabs.addTab(form_page, t("components"))
        self._tabs.addTab(yaml_page, "YAML")
        self._tabs.currentChanged.connect(self._on_tab_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tabs)

        for component in self._DEFAULT:
            self._add_row(component)
        self.set_grid(0.0, 0.0)
        self._dump()

    # -- grid ---------------------------------------------------------------
    def set_grid(self, fs_hz: float, duration_s: float) -> None:
        """Adopt the composite's sampling grid -- shown here, never edited here.

        Parameters
        ----------
        fs_hz : float
            Sampling frequency of the composite, Hz.
        duration_s : float
            Record length of the composite, s.
        """
        self._grid = (fs_hz, duration_s)
        self._grid_note.setText(t(_GRID_NOTE, fs=f"{fs_hz:g}", dur=f"{duration_s:g}"))

    # -- rows ---------------------------------------------------------------
    def count(self) -> int:
        """Return the number of component rows (exposed for tests)."""
        return len(self._rows)

    def _add_row(self, component: Mapping[str, Any] | None = None) -> _ComponentRow:
        """Append a component row, optionally pre-filled from a mapping."""
        row = _ComponentRow(self._remove_row)
        if component is not None:
            row.load(component, self._grid)
        self._rows.append(row)
        self._rows_box.addWidget(row)
        return row

    def _remove_row(self, row: _ComponentRow) -> None:
        """Remove a component row (keeping at least one component)."""
        if len(self._rows) <= 1 or row not in self._rows:
            return
        self._rows.remove(row)
        self._rows_box.removeWidget(row)
        row.setParent(None)
        row.deleteLater()

    def _clear_rows(self) -> None:
        """Drop every component row."""
        for row in list(self._rows):
            self._rows.remove(row)
            self._rows_box.removeWidget(row)
            row.setParent(None)
            row.deleteLater()

    def _load_rows(self, components: object) -> bool:
        """Rebuild the rows from a parsed component list, if representable.

        Candidate rows are built aside and committed only when *every* component
        loads: a half-applied list would be a config the user never wrote.
        """
        if not isinstance(components, (list, tuple)) or not components:
            return False
        candidates: list[_ComponentRow] = []
        loaded = True
        for component in components:
            if not isinstance(component, Mapping):
                loaded = False
                break
            row = _ComponentRow(self._remove_row)
            candidates.append(row)
            if not row.load(component, self._grid):
                loaded = False
                break
        if not loaded:
            for row in candidates:
                row.deleteLater()
            return False
        self._clear_rows()
        for row in candidates:
            self._rows.append(row)
            self._rows_box.addWidget(row)
        return True

    # -- the two views ------------------------------------------------------
    def _dump(self) -> None:
        """Write the current rows into the YAML view."""
        payload = [row.payload() for row in self._rows]
        self._text.setPlainText(yaml.safe_dump(payload, sort_keys=False))

    def _show_tab(self, index: int) -> None:
        """Switch tabs without triggering the conversion handler."""
        self._syncing = True
        try:
            self._tabs.setCurrentIndex(index)
        finally:
            self._syncing = False

    def _on_tab_changed(self, index: int) -> None:
        """Convert between the two views when the user switches tabs."""
        if self._syncing:
            return
        if index == 1:
            self._dump()
            self._yaml_note.setVisible(False)
            return
        try:
            parsed = yaml.safe_load(self._text.toPlainText())
        except yaml.YAMLError:
            parsed = None
        if self._load_rows(parsed):
            self._yaml_note.setVisible(False)
            return
        self._yaml_note.setVisible(True)
        self._show_tab(1)

    # -- payload ------------------------------------------------------------
    def components(self) -> Any:
        """Return the component list of the active view (or the raw text)."""
        if self._tabs.currentIndex() == 0:
            return [row.payload() for row in self._rows]
        text = self._text.toPlainText()
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError:
            return text
        return parsed if isinstance(parsed, list) else text

    def set_components(self, components: Any) -> None:
        """Restore the editor from a payload value (list or raw text)."""
        if isinstance(components, str):
            self._text.setPlainText(components)
            self._yaml_note.setVisible(True)
            self._show_tab(1)
            return
        if not components:
            return
        if self._load_rows(components):
            self._dump()
            self._yaml_note.setVisible(False)
            self._show_tab(0)
            return
        self._text.setPlainText(yaml.safe_dump(list(components), sort_keys=False))
        self._yaml_note.setVisible(True)
        self._show_tab(1)


class ExcitationBuilder(QWidget):
    """Collect an excitation payload for the S1 discriminated union.

    Parameters
    ----------
    parent : QWidget or None, optional
        Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._kind = QComboBox()
        self._kind.addItems(_KINDS)
        self._axis = QComboBox()
        self._axis.addItems(("x", "y", "z"))
        self._fs = _spin(100.0, 2.0e6, 5000.0, decimals=1, step=100.0)
        self._duration = _spin(0.01, 60.0, 2.0, decimals=3, step=0.1)
        self._stack = QStackedWidget()
        self._build_pages()
        self._kind.currentTextChanged.connect(self._on_kind_changed)

        common = QFormLayout()
        common.addRow(t("Kind"), with_help(self._kind, "Kind", _KIND_HELP))
        common.addRow(t("Axis"), with_help(self._axis, "Axis", _AXIS_HELP))
        self._grid_row_label = QLabel(t("Sampling"))
        grid_row = QHBoxLayout()
        grid_row.addWidget(QLabel(t("fs [Hz]")))
        grid_row.addWidget(self._fs)
        grid_row.addWidget(QLabel(t("dur [s]")))
        grid_row.addWidget(self._duration)
        self._sampling_holder = self._wrap(grid_row)
        common.addRow(
            self._grid_row_label,
            with_help(self._sampling_holder, "Sampling", _SAMPLING_HELP),
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(common)
        layout.addWidget(self._stack)
        self._on_kind_changed(self._kind.currentText())

    @staticmethod
    def _wrap(inner: QHBoxLayout) -> QWidget:
        """Wrap a layout in a widget (so it can be a form row)."""
        holder = QWidget()
        holder.setLayout(inner)
        return holder

    def _build_pages(self) -> None:
        """Build one input page per excitation kind."""
        # sine (with the optional S-21 carrier modulation)
        self._sine_freq = _spin(0.1, 1.0e5, 200.0, decimals=2, step=10.0)
        self._sine_amp = _spin(1e-3, 200.0, 1.0, decimals=3, step=0.1)
        self._sine_phase = _spin(-6.2832, 6.2832, 0.0, decimals=4, step=0.1)
        self._sine_mod = QComboBox()
        self._sine_mod.addItems(_MODULATIONS)
        self._sine_mod_fm = _spin(1e-3, 1.0e5, 10.0, decimals=3, step=1.0)
        self._sine_mod_depth = _spin(0.0, 1.0, 0.5, decimals=3, step=0.05)
        self._sine_mod_dev = _spin(0.0, 1.0e5, 50.0, decimals=3, step=1.0)
        self._sine_mod.currentTextChanged.connect(self._on_modulation_changed)
        self._stack.addWidget(
            self._form(
                [
                    ("frequency [Hz]", self._sine_freq),
                    ("amplitude [g]", self._sine_amp),
                    ("phase [rad]", self._sine_phase),
                    ("modulation", self._sine_mod),
                    ("f mod [Hz]", self._sine_mod_fm),
                    ("depth m", self._sine_mod_depth),
                    ("deviation [Hz]", self._sine_mod_dev),
                ],
                about="sine",
            )
        )
        self._on_modulation_changed(self._sine_mod.currentText())

        # multitone (dynamic components: default 2, add/remove, optional phase)
        self._multitone = _MultitoneForm()
        self._stack.addWidget(self._form([("components", self._multitone)], about="multitone"))

        # sweep / chirp
        self._sweep_f0 = _spin(0.1, 1.0e5, 20.0, decimals=2, step=10.0)
        self._sweep_f1 = _spin(0.1, 1.0e5, 2000.0, decimals=2, step=10.0)
        self._sweep_amp = _spin(1e-3, 200.0, 1.0, decimals=3, step=0.1)
        self._sweep_method = QComboBox()
        self._sweep_method.addItems(("linear", "log"))
        self._stack.addWidget(
            self._form(
                [
                    ("f start [Hz]", self._sweep_f0),
                    ("f end [Hz]", self._sweep_f1),
                    ("amplitude [g]", self._sweep_amp),
                    ("method", self._sweep_method),
                ],
                about="sweep",
            )
        )

        # random
        self._rand_lo = _spin(0.0, 1.0e5, 10.0, decimals=2, step=10.0)
        self._rand_hi = _spin(0.1, 1.0e5, 2000.0, decimals=2, step=10.0)
        self._rand_grms = _spin(1e-3, 200.0, 1.0, decimals=3, step=0.1)
        self._stack.addWidget(
            self._form(
                [
                    ("band lo [Hz]", self._rand_lo),
                    ("band hi [Hz]", self._rand_hi),
                    ("g RMS [g]", self._rand_grms),
                ],
                about="random",
            )
        )

        # shock
        self._shock_peak = _spin(1e-3, 1.0e4, 50.0, decimals=2, step=1.0)
        self._shock_pulse = _spin(0.01, 1000.0, 2.0, decimals=3, step=0.1)
        self._shock_delay = _spin(0.0, 60.0, 0.1, decimals=3, step=0.05)
        self._stack.addWidget(
            self._form(
                [
                    ("peak [g]", self._shock_peak),
                    ("pulse [ms]", self._shock_pulse),
                    ("delay [s]", self._shock_delay),
                ],
                about="shock",
            )
        )

        # composite (config-first: per-kind sub-forms beside the YAML view)
        self._composite = _CompositeForm()
        self._composite.set_grid(self._fs.value(), self._duration.value())
        self._fs.valueChanged.connect(self._on_grid_changed)
        self._duration.valueChanged.connect(self._on_grid_changed)
        self._stack.addWidget(self._form([("components", self._composite)], about="composite"))

        # csv
        self._csv_path = QLineEdit()
        self._csv_browse = QPushButton(t("Browse..."))
        self._csv_browse.clicked.connect(lambda: self._browse(self._csv_path, "CSV (*.csv)"))
        self._csv_column = QSpinBox()
        self._csv_column.setRange(0, 64)
        self._csv_column.setValue(1)
        self._csv_fs = _spin(0.1, 2.0e6, 5000.0, decimals=1, step=100.0)
        self._csv_units = QComboBox()
        self._csv_units.addItems(("m/s^2", "g"))
        path_row = QHBoxLayout()
        path_row.addWidget(self._csv_path, stretch=1)
        path_row.addWidget(self._csv_browse)
        self._stack.addWidget(
            self._form(
                [
                    ("path", self._wrap(path_row)),
                    ("column", self._csv_column),
                    ("fs [Hz]", self._csv_fs),
                    ("units", self._csv_units),
                ],
                about="csv",
            )
        )

        # wav
        self._wav_path = QLineEdit()
        self._wav_browse = QPushButton(t("Browse..."))
        self._wav_browse.clicked.connect(lambda: self._browse(self._wav_path, "WAV (*.wav)"))
        self._wav_channel = QSpinBox()
        self._wav_channel.setRange(0, 32)
        self._wav_fs_g = _spin(1e-3, 1.0e4, 10.0, decimals=3, step=1.0)
        wav_row = QHBoxLayout()
        wav_row.addWidget(self._wav_path, stretch=1)
        wav_row.addWidget(self._wav_browse)
        self._stack.addWidget(
            self._form(
                [
                    ("path", self._wrap(wav_row)),
                    ("channel", self._wav_channel),
                    ("full scale [g]", self._wav_fs_g),
                ],
                about="wav",
            )
        )

        # tdms (NI TDMS; fs from wf_increment when "fs [Hz]" is 0)
        self._tdms_path = QLineEdit()
        tdms_browse = QPushButton(t("Browse..."))
        tdms_browse.clicked.connect(lambda: self._browse(self._tdms_path, "TDMS (*.tdms)"))
        self._tdms_group = QLineEdit()
        self._tdms_group.setPlaceholderText("(first group)")
        self._tdms_channel = QSpinBox()
        self._tdms_channel.setRange(0, 256)
        self._tdms_fs = _spin(0.0, 2.0e6, 0.0, decimals=1, step=100.0)
        self._tdms_units = QComboBox()
        self._tdms_units.addItems(("m/s^2", "g"))
        tdms_row = QHBoxLayout()
        tdms_row.addWidget(self._tdms_path, stretch=1)
        tdms_row.addWidget(tdms_browse)
        self._stack.addWidget(
            self._form(
                [
                    ("path", self._wrap(tdms_row)),
                    ("group", self._tdms_group),
                    ("channel", self._tdms_channel),
                    ("fs [Hz] (0=file)", self._tdms_fs),
                    ("units", self._tdms_units),
                ],
                about="tdms",
            )
        )

        # uff (UFF/UNV dataset-58; fs from abscissa_inc when "fs [Hz]" is 0)
        self._uff_path = QLineEdit()
        uff_browse = QPushButton(t("Browse..."))
        uff_browse.clicked.connect(lambda: self._browse(self._uff_path, "UFF (*.uff *.unv)"))
        self._uff_index = QSpinBox()
        self._uff_index.setRange(0, 4096)
        self._uff_fs = _spin(0.0, 2.0e6, 0.0, decimals=1, step=100.0)
        self._uff_units = QComboBox()
        self._uff_units.addItems(("m/s^2", "g"))
        uff_row = QHBoxLayout()
        uff_row.addWidget(self._uff_path, stretch=1)
        uff_row.addWidget(uff_browse)
        self._stack.addWidget(
            self._form(
                [
                    ("path", self._wrap(uff_row)),
                    ("dataset index", self._uff_index),
                    ("fs [Hz] (0=file)", self._uff_fs),
                    ("units", self._uff_units),
                ],
                about="uff",
            )
        )

        # mat (MATLAB v4/v5/v7; fs required)
        self._mat_path = QLineEdit()
        mat_browse = QPushButton(t("Browse..."))
        mat_browse.clicked.connect(lambda: self._browse(self._mat_path, "MAT (*.mat)"))
        self._mat_key = QLineEdit()
        self._mat_key.setPlaceholderText("variable name")
        self._mat_column = QSpinBox()
        self._mat_column.setRange(0, 256)
        self._mat_fs = _spin(0.1, 2.0e6, 5000.0, decimals=1, step=100.0)
        self._mat_units = QComboBox()
        self._mat_units.addItems(("m/s^2", "g"))
        mat_row = QHBoxLayout()
        mat_row.addWidget(self._mat_path, stretch=1)
        mat_row.addWidget(mat_browse)
        self._stack.addWidget(
            self._form(
                [
                    ("path", self._wrap(mat_row)),
                    ("data key", self._mat_key),
                    ("column", self._mat_column),
                    ("fs [Hz]", self._mat_fs),
                    ("units", self._mat_units),
                ],
                about="mat",
            )
        )

        # hdf5 (.h5/.hdf5; fs required)
        self._hdf5_path = QLineEdit()
        hdf5_browse = QPushButton(t("Browse..."))
        hdf5_browse.clicked.connect(lambda: self._browse(self._hdf5_path, "HDF5 (*.h5 *.hdf5)"))
        self._hdf5_dataset = QLineEdit()
        self._hdf5_dataset.setPlaceholderText("/accel/x")
        self._hdf5_column = QSpinBox()
        self._hdf5_column.setRange(0, 256)
        self._hdf5_fs = _spin(0.1, 2.0e6, 5000.0, decimals=1, step=100.0)
        self._hdf5_units = QComboBox()
        self._hdf5_units.addItems(("m/s^2", "g"))
        hdf5_row = QHBoxLayout()
        hdf5_row.addWidget(self._hdf5_path, stretch=1)
        hdf5_row.addWidget(hdf5_browse)
        self._stack.addWidget(
            self._form(
                [
                    ("path", self._wrap(hdf5_row)),
                    ("dataset", self._hdf5_dataset),
                    ("column", self._hdf5_column),
                    ("fs [Hz]", self._hdf5_fs),
                    ("units", self._hdf5_units),
                ],
                about="hdf5",
            )
        )

    @staticmethod
    def _form(rows: list[tuple[str, QWidget]], about: str | None = None) -> QWidget:
        """Build a form-layout page from (label, widget) rows.

        Parameters
        ----------
        rows : list
            ``(label, widget)`` pairs.
        about : str or None, optional
            Excitation kind key; when given, a summary row with the page's
            reference note (``?``) is placed on top.
        """
        page = QWidget()
        form = QFormLayout(page)
        if about is not None:
            summary, text = _ABOUT[about]
            note = QLabel(t(summary))
            note.setStyleSheet("color: #808080; font-style: italic;")
            form.addRow(t("about"), with_help(note, f"{about} excitation", text))
        for label, widget in rows:
            form.addRow(t(label), widget)
        return page

    def _browse(self, target: QLineEdit, file_filter: str) -> None:  # pragma: no cover - dialog
        """Open a file dialog and write the chosen path into ``target``."""
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", file_filter)
        if path:
            target.setText(path)

    def _on_grid_changed(self, _value: float) -> None:
        """Mirror the composite sampling grid into the component page (doc 11 §2.1.4)."""
        self._composite.set_grid(self._fs.value(), self._duration.value())

    def _on_modulation_changed(self, mode: str) -> None:
        """Show only the fields the selected carrier modulator needs."""
        self._sine_mod_fm.setEnabled(mode != "none")
        self._sine_mod_depth.setEnabled(mode == "am")
        self._sine_mod_dev.setEnabled(mode == "fm")

    def _on_kind_changed(self, kind: str) -> None:
        """Switch the stacked page and toggle the sampling row visibility."""
        self._stack.setCurrentIndex(_KINDS.index(kind))
        generated = kind in _GENERATED
        self._grid_row_label.setVisible(generated)
        self._sampling_holder.setVisible(generated)

    def excitation_payload(self) -> dict[str, Any]:
        """Return the excitation payload for the current kind.

        Returns
        -------
        dict[str, Any]
            A mapping accepted by ``build_excitation_spec`` (and hence by
            ``ScenarioConfig``), including the ``kind`` discriminator.
        """
        kind = self._kind.currentText()
        axis = self._axis.currentText()
        base: dict[str, Any] = {"kind": kind, "axis": axis}
        if kind in _GENERATED:
            base["fs_hz"] = self._fs.value()
            base["duration_s"] = self._duration.value()
        if kind == "sine":
            base["frequency_hz"] = self._sine_freq.value()
            base["amplitude_g"] = self._sine_amp.value()
            if self._sine_phase.value() != 0.0:
                base["phase_rad"] = self._sine_phase.value()
            mode = self._sine_mod.currentText()
            if mode == "am":
                base["modulation"] = {
                    "kind": "am",
                    "f_m_hz": self._sine_mod_fm.value(),
                    "depth": self._sine_mod_depth.value(),
                }
            elif mode == "fm":
                base["modulation"] = {
                    "kind": "fm",
                    "f_m_hz": self._sine_mod_fm.value(),
                    "deviation_hz": self._sine_mod_dev.value(),
                }
        elif kind == "composite":
            base["components"] = self._composite.components()
        elif kind == "multitone":
            base["tones"] = self._multitone.tones()
        elif kind == "sweep":
            base["f_start_hz"] = self._sweep_f0.value()
            base["f_end_hz"] = self._sweep_f1.value()
            base["amplitude_g"] = self._sweep_amp.value()
            base["method"] = self._sweep_method.currentText()
        elif kind == "random":
            base["band_hz"] = [self._rand_lo.value(), self._rand_hi.value()]
            base["g_rms"] = self._rand_grms.value()
        elif kind == "shock":
            base["peak_g"] = self._shock_peak.value()
            base["pulse_ms"] = self._shock_pulse.value()
            base["delay_s"] = self._shock_delay.value()
        elif kind == "csv":
            base["path"] = self._csv_path.text().strip()
            base["column"] = self._csv_column.value()
            base["fs_hz"] = self._csv_fs.value()
            base["units"] = self._csv_units.currentText()
        elif kind == "wav":
            base["path"] = self._wav_path.text().strip()
            base["channel"] = self._wav_channel.value()
            base["full_scale_g"] = self._wav_fs_g.value()
        elif kind == "tdms":
            base["path"] = self._tdms_path.text().strip()
            group = self._tdms_group.text().strip()
            if group:
                base["group"] = group
            base["channel"] = self._tdms_channel.value()
            if self._tdms_fs.value() > 0.0:
                base["fs_hz"] = self._tdms_fs.value()
            base["units"] = self._tdms_units.currentText()
        elif kind == "uff":
            base["path"] = self._uff_path.text().strip()
            base["dataset_index"] = self._uff_index.value()
            if self._uff_fs.value() > 0.0:
                base["fs_hz"] = self._uff_fs.value()
            base["units"] = self._uff_units.currentText()
        elif kind == "mat":
            base["path"] = self._mat_path.text().strip()
            base["data_key"] = self._mat_key.text().strip()
            base["column"] = self._mat_column.value()
            base["fs_hz"] = self._mat_fs.value()
            base["units"] = self._mat_units.currentText()
        elif kind == "hdf5":
            base["path"] = self._hdf5_path.text().strip()
            base["dataset"] = self._hdf5_dataset.text().strip()
            base["column"] = self._hdf5_column.value()
            base["fs_hz"] = self._hdf5_fs.value()
            base["units"] = self._hdf5_units.currentText()
        return base

    def load_payload(self, payload: dict[str, Any]) -> None:
        """Restore the widgets from an :meth:`excitation_payload` mapping.

        Used to preserve the excitation across a language rebuild (SW-65).
        Unknown or missing fields are left at their freshly-built defaults.
        """
        kind = str(payload.get("kind", self._kind.currentText()))
        if kind in _KINDS:
            self._kind.setCurrentText(kind)
        axis = str(payload.get("axis", self._axis.currentText()))
        if axis in ("x", "y", "z"):
            self._axis.setCurrentText(axis)
        if "fs_hz" in payload:
            self._fs.setValue(float(payload["fs_hz"]))
        if "duration_s" in payload:
            self._duration.setValue(float(payload["duration_s"]))

        def _set(spin: Any, key: str) -> None:
            if key in payload:
                spin.setValue(float(payload[key]))

        if kind == "sine":
            _set(self._sine_freq, "frequency_hz")
            _set(self._sine_amp, "amplitude_g")
            _set(self._sine_phase, "phase_rad")
            modulation = payload.get("modulation") or {}
            mode = str(modulation.get("kind", "none"))
            if mode in _MODULATIONS:
                self._sine_mod.setCurrentText(mode)
            if "f_m_hz" in modulation:
                self._sine_mod_fm.setValue(float(modulation["f_m_hz"]))
            if "depth" in modulation:
                self._sine_mod_depth.setValue(float(modulation["depth"]))
            if "deviation_hz" in modulation:
                self._sine_mod_dev.setValue(float(modulation["deviation_hz"]))
            self._on_modulation_changed(self._sine_mod.currentText())
        elif kind == "composite":
            self._composite.set_components(payload.get("components", []))
        elif kind == "multitone":
            self._multitone.set_tones([list(map(float, tone)) for tone in payload.get("tones", [])])
        elif kind == "sweep":
            _set(self._sweep_f0, "f_start_hz")
            _set(self._sweep_f1, "f_end_hz")
            _set(self._sweep_amp, "amplitude_g")
            if "method" in payload:
                self._sweep_method.setCurrentText(str(payload["method"]))
        elif kind == "random":
            band = payload.get("band_hz") or [None, None]
            if band[0] is not None:
                self._rand_lo.setValue(float(band[0]))
            if band[1] is not None:
                self._rand_hi.setValue(float(band[1]))
            _set(self._rand_grms, "g_rms")
        elif kind == "shock":
            _set(self._shock_peak, "peak_g")
            _set(self._shock_pulse, "pulse_ms")
            _set(self._shock_delay, "delay_s")
        elif kind == "csv":
            self._csv_path.setText(str(payload.get("path", "")))
            _set(self._csv_column, "column")
            _set(self._csv_fs, "fs_hz")
            if "units" in payload:
                self._csv_units.setCurrentText(str(payload["units"]))
        elif kind == "wav":
            self._wav_path.setText(str(payload.get("path", "")))
            _set(self._wav_channel, "channel")
            _set(self._wav_fs_g, "full_scale_g")
        elif kind == "tdms":
            self._tdms_path.setText(str(payload.get("path", "")))
            self._tdms_group.setText(str(payload.get("group", "")))
            _set(self._tdms_channel, "channel")
            _set(self._tdms_fs, "fs_hz")
            if "units" in payload:
                self._tdms_units.setCurrentText(str(payload["units"]))
        elif kind == "uff":
            self._uff_path.setText(str(payload.get("path", "")))
            _set(self._uff_index, "dataset_index")
            _set(self._uff_fs, "fs_hz")
            if "units" in payload:
                self._uff_units.setCurrentText(str(payload["units"]))
        elif kind == "mat":
            self._mat_path.setText(str(payload.get("path", "")))
            self._mat_key.setText(str(payload.get("data_key", "")))
            _set(self._mat_column, "column")
            _set(self._mat_fs, "fs_hz")
            if "units" in payload:
                self._mat_units.setCurrentText(str(payload["units"]))
        elif kind == "hdf5":
            self._hdf5_path.setText(str(payload.get("path", "")))
            self._hdf5_dataset.setText(str(payload.get("dataset", "")))
            _set(self._hdf5_column, "column")
            _set(self._hdf5_fs, "fs_hz")
            if "units" in payload:
                self._hdf5_units.setCurrentText(str(payload["units"]))
