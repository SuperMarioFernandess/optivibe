"""Shared GUI helpers: inline help buttons and the mouse-wheel guard.

Two small, purely presentational utilities used across the parameter tabs
(no physics here, 09 §9):

``with_help``
    Wraps an input widget together with a faint ``?`` button; clicking it opens
    a short reference note (what the parameter is, its typical values, what it
    couples to and how it affects the simulation). The notes are plain text
    written next to the field specs, so the reference lives with the code that
    defines the field.

``install_wheel_guard``
    Application-level event filter that stops the mouse wheel from silently
    changing combo boxes and spin boxes: skimming the parameter column with
    the wheel must scroll the panel, never edit a value (an accidental,
    unnoticed edit is a reproducibility hazard). The wheel event is re-sent to
    the enclosing scroll-area viewport, so page scrolling over these widgets
    keeps working; keyboard and mouse-click editing are unaffected.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QHBoxLayout,
    QMessageBox,
    QScrollArea,
    QToolButton,
    QWidget,
)

__all__ = ["install_wheel_guard", "with_help"]


def _help_button(title: str, text: str, parent: QWidget) -> QToolButton:
    """Build the faint ``?`` button that opens a reference note."""
    button = QToolButton(parent)
    button.setText("?")
    button.setAutoRaise(True)
    button.setCursor(Qt.CursorShape.WhatsThisCursor)
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    button.setFixedSize(18, 18)
    button.setStyleSheet(
        "QToolButton {color: #9a9a9a; border: none; font-weight: bold;}"
        "QToolButton:hover {color: #303030;}"
    )
    button.setToolTip("What is this parameter?")
    button.clicked.connect(lambda: QMessageBox.information(parent, title, text))
    return button


def with_help(widget: QWidget, title: str, text: str) -> QWidget:
    """Wrap an input widget with a faint inline ``?`` help button.

    Parameters
    ----------
    widget : QWidget
        The input control (line edit, combo, spin box, sub-layout holder).
    title : str
        Window title of the reference note (usually the row label).
    text : str
        The reference note: what the parameter is, typical values, couplings,
        effect on the simulation.

    Returns
    -------
    QWidget
        A container holding ``widget`` and the help button, suitable as a
        ``QFormLayout`` row field.
    """
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    row.addWidget(widget, stretch=1)
    row.addWidget(_help_button(title, text, holder))
    return holder


class _WheelGuard(QObject):
    """Blocks wheel edits on combos/spin boxes app-wide; keeps page scrolling."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Consume wheel events aimed at value widgets; forward to the scroller."""
        if event.type() != QEvent.Type.Wheel:
            return False
        if not isinstance(obj, (QComboBox, QAbstractSpinBox)):
            return False
        area = _enclosing_scroll_area(obj)
        if area is not None:
            QApplication.sendEvent(area.viewport(), event)
        return True


def _enclosing_scroll_area(widget: QWidget) -> QScrollArea | None:
    """Walk up the parents to the nearest scroll area (if any)."""
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QScrollArea):
            return parent
        parent = parent.parentWidget()
    return None


_GUARD: _WheelGuard | None = None


def install_wheel_guard() -> None:
    """Install the wheel guard once on the running application.

    Idempotent: repeated calls (one per panel instance) keep a single filter.
    Value widgets keep keyboard and click editing; the wheel over them scrolls
    the surrounding panel instead of editing the value.
    """
    global _GUARD
    app = QApplication.instance()
    if app is None or _GUARD is not None:  # pragma: no cover - app always exists in GUI
        return
    _GUARD = _WheelGuard(app)
    app.installEventFilter(_GUARD)
