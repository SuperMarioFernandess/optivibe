"""Excitation builder widget: the S1 ``ExcitationSpec`` union (task S7 §2).

A ``kind`` selector over a stacked form (sine / multitone / sweep / random /
shock, plus CSV / WAV / TDMS / UFF / MAT / HDF5 replay via the loader registry).
It collects a *payload* mapping that
:func:`optivibe.gui.controllers.scenario_builder.build_excitation_spec`
validates -- the widget holds no signal logic, only the input fields (09 §9).
"""

from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
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
        common.addRow("Kind", self._kind)
        common.addRow("Axis", self._axis)
        self._grid_row_label = QLabel("Sampling")
        grid_row = QHBoxLayout()
        grid_row.addWidget(QLabel("fs [Hz]"))
        grid_row.addWidget(self._fs)
        grid_row.addWidget(QLabel("dur [s]"))
        grid_row.addWidget(self._duration)
        self._sampling_holder = self._wrap(grid_row)
        common.addRow(self._grid_row_label, self._sampling_holder)

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
            self._form([("frequency [Hz]", self._sine_freq), ("amplitude [g]", self._sine_amp)])
        )

        # multitone (three optional tone rows)
        self._tone_freqs = [
            _spin(0.1, 1.0e5, f, decimals=2, step=10.0) for f in (120.0, 240.0, 360.0)
        ]
        self._tone_amps = [_spin(0.0, 200.0, a, decimals=3, step=0.1) for a in (1.0, 0.5, 0.0)]
        rows = []
        for i, (f, a) in enumerate(zip(self._tone_freqs, self._tone_amps, strict=True), start=1):
            row = QHBoxLayout()
            row.addWidget(QLabel("f [Hz]"))
            row.addWidget(f)
            row.addWidget(QLabel("amp [g]"))
            row.addWidget(a)
            rows.append((f"tone {i}", self._wrap(row)))
        self._stack.addWidget(self._form(rows))

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
                ]
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
                ]
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
                ]
            )
        )

        # csv
        self._csv_path = QLineEdit()
        self._csv_browse = QPushButton("Browse...")
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
                ]
            )
        )

        # wav
        self._wav_path = QLineEdit()
        self._wav_browse = QPushButton("Browse...")
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
                ]
            )
        )

        # tdms (NI TDMS; fs from wf_increment when "fs [Hz]" is 0)
        self._tdms_path = QLineEdit()
        tdms_browse = QPushButton("Browse...")
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
                ]
            )
        )

        # uff (UFF/UNV dataset-58; fs from abscissa_inc when "fs [Hz]" is 0)
        self._uff_path = QLineEdit()
        uff_browse = QPushButton("Browse...")
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
                ]
            )
        )

        # mat (MATLAB v4/v5/v7; fs required)
        self._mat_path = QLineEdit()
        mat_browse = QPushButton("Browse...")
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
                ]
            )
        )

        # hdf5 (.h5/.hdf5; fs required)
        self._hdf5_path = QLineEdit()
        hdf5_browse = QPushButton("Browse...")
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
                ]
            )
        )

    @staticmethod
    def _form(rows: list[tuple[str, QWidget]]) -> QWidget:
        """Build a form-layout page from (label, widget) rows."""
        page = QWidget()
        form = QFormLayout(page)
        for label, widget in rows:
            form.addRow(label, widget)
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
            tones = [
                [f.value(), a.value()]
                for f, a in zip(self._tone_freqs, self._tone_amps, strict=True)
                if a.value() > 0.0
            ]
            base["tones"] = tones or [[self._tone_freqs[0].value(), 1.0]]
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
