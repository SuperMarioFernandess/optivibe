"""Comparison tab: two DSP chains over one input, side by side (task S-22 W-2).

The tab answers the question the experiment panel provokes: *and what does that
actually change in the numbers?* It runs two chains over one common input and
shows the answer twice -- as an **overlay** (the recovered acceleration trace
and the recovered spectrum, one curve per chain) and as a **table of
differences** in the metric vocabulary of 17 §1 (RMS a/v/x, the leading
dominant, the second-harmonic ratio, the band RMS velocity behind the ISO
grade), each row carrying the relative difference against the reference chain.

Three things this widget does *not* do, on purpose:

* **It computes nothing.** The comparison is a
  :class:`~optivibe.gui.workers.jobs.CompareJob` running off the UI thread
  (invariant SW-06); the numbers arrive ready-made in a
  :class:`~optivibe.analysis.compare.ComparisonResult`, and this view only
  draws them.
* **It opens no input of its own.** The common input is a scenario the panel
  assembled or a ``kind: analyze`` spec -- the same two routes the live
  oscilloscope offers, resolved by the same code (one data path, so a
  difference in the numbers cannot come from the reader).
* **It does not decide what is trustworthy.** Each chain carries the verdict
  computed from its own options; the reference chain is chain A (the one edited
  on the *DSP experiment* tab), chain B is the local variation, and both
  verdicts are on screen while their numbers are.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from optivibe.analysis.compare import ComparisonResult
from optivibe.core.config.models import DspOptions
from optivibe.core.types import FloatArray
from optivibe.gui.i18n import t, tr
from optivibe.gui.widgets.dsp_controls import DspControls
from optivibe.gui.widgets.ui_helpers import tab_header

__all__ = ["ComparePanel"]

#: Source keys (payload values, not user-visible text) -- same vocabulary the
#: live controls use, because it is the same pair of sources.
SOURCE_SCENARIO = "scenario"
SOURCE_RECORD = "record"

#: One pen per chain, in chain order (two chains today, room for a third).
_CHAIN_COLORS: tuple[str, ...] = ("#1f77b4", "#d62728", "#2ca02c")

_MAX_POINTS = 4000


def _decimate(*arrays: FloatArray, n_max: int = _MAX_POINTS) -> list[FloatArray]:
    """Stride-decimate parallel arrays to at most ``n_max`` points."""
    if not arrays:
        return []
    stride = max(1, arrays[0].size // n_max)
    return [np.asarray(a, dtype=np.float64)[::stride] for a in arrays]


class ComparePanel(QWidget):
    """Source selector, chain-B editor, overlay plots and the difference table.

    Parameters
    ----------
    parent : QWidget or None, optional
        Parent widget.
    """

    #: Emitted when the user asks to run the comparison.
    compare_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spec_path: str | None = None
        self._result: ComparisonResult | None = None

        self._source = QComboBox()
        self._source.addItem(t("synthetic scenario"), SOURCE_SCENARIO)
        self._source.addItem(t("recorded capture"), SOURCE_RECORD)
        self._source.currentIndexChanged.connect(lambda _index: self._sync_source())
        self._browse = QPushButton(t("Spec..."))
        self._browse.clicked.connect(self._on_browse_clicked)
        self._run = QPushButton(t("Compare chains"))
        self._run.clicked.connect(lambda: self.compare_requested.emit())

        self._chain_b = DspControls()
        self._status = QLabel()
        self._status.setFrameShape(QFrame.Shape.StyledPanel)
        self._status.setWordWrap(True)

        self._plots = pg.GraphicsLayoutWidget()
        self._p_accel = pg.PlotItem(title=t("Recovered acceleration"))
        self._p_accel.addLegend(offset=(-10, 5))
        self._p_accel.setLabel("left", t("a"), units="m/s^2")
        self._p_accel.setLabel("bottom", t("time"), units="s")
        self._p_spec = pg.PlotItem(title=t("Recovered spectrum"))
        self._p_spec.addLegend(offset=(-10, 5))
        self._p_spec.setLabel("bottom", t("frequency"), units="Hz")
        self._p_spec.setLabel("left", t("amplitude"))
        self._p_spec.setLogMode(x=False, y=True)
        self._plots.addItem(self._p_accel, row=0, col=0)
        self._plots.addItem(self._p_spec, row=1, col=0)

        self._table = QTableWidget(0, 1)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        controls = QHBoxLayout()
        controls.addWidget(QLabel(t("source")))
        controls.addWidget(self._source)
        controls.addWidget(self._browse)
        controls.addWidget(self._run)
        controls.addStretch(1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel(tr("compare.chain_b")))
        left_layout.addWidget(self._chain_b)
        left_layout.addStretch(1)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self._plots)
        right.addWidget(self._table)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)

        body = QSplitter(Qt.Orientation.Horizontal)
        body.addWidget(left)
        body.addWidget(right)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 3)

        self._header = tab_header(t("Compare DSP chains"), tr("compare.tab.help"))
        layout = QVBoxLayout(self)
        layout.addWidget(self._header)
        layout.addLayout(controls)
        layout.addWidget(self._status)
        layout.addWidget(body, 1)
        self.reset_status()
        self._sync_source()

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    def source_kind(self) -> str:
        """Return the selected source key (:data:`SOURCE_SCENARIO`/``_RECORD``)."""
        return str(self._source.currentData())

    def spec_path(self) -> str | None:
        """Return the chosen analyze-spec path (record source), if any."""
        return self._spec_path

    def chain_b_options(self) -> DspOptions:
        """Return the chain the local editor describes (chain B)."""
        return self._chain_b.dsp_options()

    def set_chain_b_options(self, options: DspOptions) -> None:
        """Seed chain B (used to start it from the currently edited chain).

        Parameters
        ----------
        options : DspOptions
            Options to display.
        """
        self._chain_b.set_dsp_options(options)

    def result(self) -> ComparisonResult | None:
        """Return the last comparison shown (``None`` before the first run)."""
        return self._result

    def set_busy(self, busy: bool) -> None:
        """Disable the trigger while a comparison is in flight.

        Parameters
        ----------
        busy : bool
            Whether a comparison job is running.
        """
        self._run.setEnabled(not busy)

    def reset_status(self, message: str | None = None) -> None:
        """Reset the status strip to its idle text (or to ``message``)."""
        self._status.setText(message if message is not None else tr("compare.idle"))

    def status_text(self) -> str:
        """Return the status strip text (exposed for tests)."""
        return str(self._status.text())

    def show_result(self, result: ComparisonResult) -> None:
        """Draw one comparison: the overlays, the table and the verdicts.

        Parameters
        ----------
        result : ComparisonResult
            Outcome of a :class:`~optivibe.gui.workers.jobs.CompareJob`. Every
            number is read off the result; nothing is computed here.
        """
        self._result = result
        self._draw_overlay(result)
        self._fill_table(result)
        verdicts = " | ".join(
            tr(
                "compare.chain_verdict",
                name=outcome.name,
                status=outcome.status,
                deviations=(
                    "; ".join(delta.as_text() for delta in outcome.deltas)
                    or tr("compare.no_deviation")
                ),
            )
            for outcome in result.chains
        )
        self._status.setText(
            tr("compare.done", input=result.input_label, status=result.status) + " | " + verdicts
        )

    def retranslate(self) -> None:
        """Refresh the static text after a language change."""
        self._run.setText(t("Compare chains"))
        self._source.setItemText(0, t("synthetic scenario"))
        self._source.setItemText(1, t("recorded capture"))
        self._chain_b.retranslate()
        self._sync_source()
        if self._result is None:
            self.reset_status()
        else:
            self.show_result(self._result)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _draw_overlay(self, result: ComparisonResult) -> None:
        """Redraw both overlay panels, one curve per chain."""
        self._p_accel.clear()
        self._p_spec.clear()
        for index, outcome in enumerate(result.chains):
            color = _CHAIN_COLORS[index % len(_CHAIN_COLORS)]
            pen = pg.mkPen(color, width=1)
            name = f"{outcome.name} [{outcome.status}]"
            accel = outcome.result.a
            time = np.arange(accel.size, dtype=np.float64) / result.fs
            time_d, accel_d = _decimate(time, accel)
            self._p_accel.plot(time_d, accel_d, pen=pen, name=name)
            spectrum = outcome.result.spectrum
            if spectrum is not None:
                freq_d, values_d = _decimate(spectrum.freq, spectrum.values)
                self._p_spec.plot(freq_d, np.abs(values_d), pen=pen, name=name)

    def _fill_table(self, result: ComparisonResult) -> None:
        """Fill the difference table: one row per metric, one block per chain."""
        chains = result.chains
        headers = [t("metric"), t("unit")]
        for index, outcome in enumerate(chains):
            headers.append(outcome.name)
            if index:
                headers.append(tr("compare.rel_to", name=chains[0].name))
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(len(result.rows))
        for row_index, row in enumerate(result.rows):
            cells: list[str] = [row.key, row.unit]
            for index, value in enumerate(row.values):
                cells.append("--" if value is None else f"{value:.6g}")
                if index:
                    rel = row.rel_diff[index]
                    cells.append("--" if rel is None else f"{rel * 100.0:+.3f} %")
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self._table.setItem(row_index, column, item)

    def _sync_source(self) -> None:
        """Enable the spec picker for the record source and show the choice."""
        is_record = self.source_kind() == SOURCE_RECORD
        self._browse.setEnabled(is_record)
        if is_record and self._spec_path is not None:
            self._browse.setText(self._spec_path.rsplit("/", 1)[-1])
            self._browse.setToolTip(self._spec_path)
        elif is_record:
            self._browse.setText(tr("live.stream.no_spec"))
            self._browse.setToolTip("")
        else:
            self._browse.setText(t("Spec..."))
            self._browse.setToolTip("")

    def set_spec_path(self, path: str | None) -> None:
        """Set the analyze-spec path used by the record source.

        Parameters
        ----------
        path : str or None
            Path to a ``kind: analyze`` YAML, or ``None`` to clear it.
        """
        self._spec_path = path
        self._sync_source()

    def _on_browse_clicked(self) -> None:  # pragma: no cover - file dialog
        """Pick the analyze spec that describes the record to compare on."""
        path, _filter = QFileDialog.getOpenFileName(
            self, t("Open analyze spec"), "", t("YAML files (*.yaml *.yml)")
        )
        if path:
            self.set_spec_path(path)
