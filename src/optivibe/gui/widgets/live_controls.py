"""Controls and provenance strip of the real-time oscilloscope (task O-SW-03).

Deliberately a **separate widget** from :mod:`optivibe.gui.widgets.live_view`:
the Live tab is the next stop of the DSP comparison bench (backlog S-22), and a
bar that owns its own state and speaks to the outside through two signals is an
extension point, whereas the same rows inlined into the view would be one more
knot to untie.

Two halves:

**Controls** -- the minimal set decision 13 §4.5 allows: start/stop, source
(synthetic scenario or a recorded capture), update rate, plus the replay pace
and the looping switch the two sources need. This is not an instrument panel:
nothing here changes the model, which is edited on the left and, for the
synthetic source, read straight from the control panel.

**Provenance** -- not decoration and not optional. ``warmed`` says whether the
causal filters have settled (before that the numbers are *not final*), the drop
counter says whether the source lost samples, and both are on screen at all
times. An accelerated replay shows the drop counter as "n/a" rather than ``0``:
without an obligation to a clock the counter is undefined, and a reassuring zero
would be its own small lie (theory-06 §5.7). The window length is shown too, so
nobody mistakes half a second of trace for the whole record.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from optivibe.gui.i18n import t, tr
from optivibe.gui.widgets.ui_helpers import with_help
from optivibe.gui.workers.stream import Speed, StreamFrame

__all__ = ["LiveControls"]

#: Source keys (payload values; not user-visible text).
SOURCE_SCENARIO = "scenario"
SOURCE_RECORD = "record"

#: Replay pace keys, in combo order, with their English labels.
_SPEED_LABELS: tuple[tuple[Speed, str], ...] = (
    ("realtime", "real time"),
    ("max", "as fast as possible"),
)


class LiveControls(QWidget):
    """Start/stop bar, source selector and provenance strip of the live mode.

    Parameters
    ----------
    parent : QWidget or None, optional
        Parent widget.
    """

    #: Emitted when the user asks to start streaming.
    start_requested = Signal()
    #: Emitted when the user asks to stop streaming.
    stop_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spec_path: Path | None = None
        self._running = False

        self._start = QPushButton(t("Start live"))
        self._start.clicked.connect(self._on_start_clicked)

        self._source = QComboBox()
        self._source.addItem(t("synthetic scenario"), SOURCE_SCENARIO)
        self._source.addItem(t("recorded capture"), SOURCE_RECORD)
        self._source.currentIndexChanged.connect(lambda _index: self._sync_source())

        self._browse = QPushButton(t("Spec..."))
        self._browse.clicked.connect(self._on_browse_clicked)

        self._rate = QDoubleSpinBox()
        self._rate.setRange(0.5, 30.0)
        self._rate.setDecimals(1)
        self._rate.setSingleStep(1.0)
        self._rate.setValue(10.0)
        self._rate.setSuffix(" /s")

        self._speed = QComboBox()
        for key, label in _SPEED_LABELS:
            self._speed.addItem(t(label), key)

        self._loop = QComboBox()
        self._loop.addItem(t("play once"), False)
        self._loop.addItem(t("loop"), True)

        self._provenance = QLabel()
        self._provenance.setFrameShape(QFrame.Shape.StyledPanel)
        self._provenance.setWordWrap(True)
        self.reset_provenance()

        self._rate_label = QLabel(t("rate"))
        self._source_label = QLabel(t("source"))
        row = QHBoxLayout()
        row.setContentsMargins(2, 2, 2, 2)
        row.addWidget(self._start)
        row.addWidget(self._source_label)
        row.addWidget(self._source)
        row.addWidget(self._browse)
        row.addWidget(self._rate_label)
        row.addWidget(with_help(self._rate, t("Update rate"), tr("live.stream.rate.help")))
        row.addWidget(self._speed)
        row.addWidget(self._loop)
        row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(self._provenance)
        self._sync_source()

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    def source_kind(self) -> str:
        """Return the selected source key (:data:`SOURCE_SCENARIO`/``_RECORD``)."""
        return str(self._source.currentData())

    def rate_hz(self) -> float:
        """Return the requested frame rate, Hz (the reference profile is 10/s)."""
        return float(self._rate.value())

    def speed(self) -> Speed:
        """Return the replay pace (``"realtime"`` or ``"max"``)."""
        key = str(self._speed.currentData())
        return "max" if key == "max" else "realtime"

    def loop_enabled(self) -> bool:
        """Return whether the record restarts when it ends."""
        return bool(self._loop.currentData())

    def spec_path(self) -> Path | None:
        """Return the chosen analyze-spec path (record source), if any."""
        return self._spec_path

    def set_spec_path(self, path: Path | None) -> None:
        """Set the analyze-spec path used by the record source.

        Parameters
        ----------
        path : pathlib.Path or None
            Path to a ``kind: analyze`` YAML, or ``None`` to clear it.
        """
        self._spec_path = path
        self._sync_source()

    def set_running(self, running: bool) -> None:
        """Switch the bar between the idle and the live layout.

        Parameters
        ----------
        running : bool
            Whether a stream is live. While it is, the settings are frozen:
            changing them mid-stream would silently detach the numbers on
            screen from the labels next to them.
        """
        self._running = running
        self._start.setText(t("Stop live") if running else t("Start live"))
        self._source.setEnabled(not running)
        self._speed.setEnabled(not running)
        self._loop.setEnabled(not running)
        self._rate.setEnabled(not running)
        self._sync_source()

    def reset_provenance(self, message: str | None = None) -> None:
        """Clear the provenance strip back to its idle text.

        Parameters
        ----------
        message : str or None, optional
            Text to show instead of the default idle line (e.g. an error).
        """
        self._provenance.setText(message if message is not None else tr("live.stream.idle"))

    def provenance_text(self) -> str:
        """Return the provenance line currently displayed (exposed for tests)."""
        return str(self._provenance.text())

    def show_frame(self, frame: StreamFrame) -> None:
        """Render the provenance of one live frame.

        Parameters
        ----------
        frame : StreamFrame
            The frame whose provenance to display. Nothing is computed here:
            every field is carried ready-made by the worker (09 §9).
        """
        text = tr(
            "live.stream.provenance",
            source=frame.source_label,
            warmed=tr("live.stream.warmed") if frame.warmed else tr("live.stream.warming"),
            dropped=(
                tr("live.stream.dropped_na")
                if frame.dropped_samples is None
                else str(frame.dropped_samples)
            ),
            elapsed=f"{frame.elapsed_s:.1f}",
            window=f"{frame.window_s * 1.0e3:.0f}",
        )
        if frame.loops:
            text = f"{text} | {tr('live.stream.loops', n=frame.loops)}"
        if frame.seam:
            text = f"{text} | {tr('live.stream.seam')}"
        self._provenance.setText(text)

    def retranslate(self) -> None:
        """Refresh the static text after a language change."""
        self._start.setText(t("Stop live") if self._running else t("Start live"))
        self._source_label.setText(t("source"))
        self._rate_label.setText(t("rate"))
        self._source.setItemText(0, t("synthetic scenario"))
        self._source.setItemText(1, t("recorded capture"))
        for index, (_key, label) in enumerate(_SPEED_LABELS):
            self._speed.setItemText(index, t(label))
        self._loop.setItemText(0, t("play once"))
        self._loop.setItemText(1, t("loop"))
        self._sync_source()
        if not self._running:
            self.reset_provenance()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _sync_source(self) -> None:
        """Enable the spec picker for the record source and show the choice."""
        is_record = self._source.currentData() == SOURCE_RECORD
        self._browse.setEnabled(is_record and not self._running)
        if is_record and self._spec_path is not None:
            self._browse.setText(self._spec_path.name)
            self._browse.setToolTip(str(self._spec_path))
        elif is_record:
            self._browse.setText(tr("live.stream.no_spec"))
            self._browse.setToolTip("")
        else:
            self._browse.setText(t("Spec..."))
            self._browse.setToolTip("")

    def _on_start_clicked(self) -> None:
        """Route the button to the start or the stop request."""
        if self._running:
            self.stop_requested.emit()
        else:
            self.start_requested.emit()

    def _on_browse_clicked(self) -> None:  # pragma: no cover - file dialog
        """Pick the analyze spec that describes the record to replay."""
        path, _filter = QFileDialog.getOpenFileName(
            self, t("Open analyze spec"), "", t("YAML files (*.yaml *.yml)")
        )
        if path:
            self.set_spec_path(Path(path))
