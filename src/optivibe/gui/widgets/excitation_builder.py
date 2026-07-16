"""Excitation builder widget: the S1 ``ExcitationSpec`` union (task S7 §2).

A ``kind`` selector over a stacked form (sine / multitone / sweep / random /
shock, plus CSV / WAV / TDMS / UFF / MAT / HDF5 replay via the loader registry).
It collects a *payload* mapping that
:func:`optivibe.gui.controllers.scenario_builder.build_excitation_spec`
validates -- the widget holds no signal logic, only the input fields (09 §9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
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
    "csv",
    "wav",
    "tdms",
    "uff",
    "mat",
    "hdf5",
)
_GENERATED = {"sine", "multitone", "sweep", "random", "shock"}

_KIND_HELP = (
    "What drives the sensor along the chosen axis:\n\n"
    "sine / multitone / sweep / random / shock -- GENERATED waveforms on the "
    "sampling grid below (fs, duration).\n"
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
        "One tone: frequency [Hz] and amplitude [g]. The basic probe of one "
        "band point -- dominant-frequency recovery, 2f/1f distortion at a "
        "bias~0 working point, RMS checks. Keep the frequency well below f1 "
        "for off-resonance use.",
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
        self._phase = QCheckBox("include per-tone phase")
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
        freq_spin = _spin(0.1, 1.0e5, freq, decimals=2, step=10.0)
        amp_spin = _spin(1e-3, 200.0, amp, decimals=3, step=0.1)
        phase_spin = _spin(-3.1416, 3.1416, phase, decimals=3, step=0.1)
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
        # sine
        self._sine_freq = _spin(0.1, 1.0e5, 200.0, decimals=2, step=10.0)
        self._sine_amp = _spin(1e-3, 200.0, 1.0, decimals=3, step=0.1)
        self._stack.addWidget(
            self._form(
                [("frequency [Hz]", self._sine_freq), ("amplitude [g]", self._sine_amp)],
                about="sine",
            )
        )

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
