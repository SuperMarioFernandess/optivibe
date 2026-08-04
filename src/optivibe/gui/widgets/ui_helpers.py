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
    QAbstractButton,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QTabWidget,
    QToolButton,
    QWidget,
)

from optivibe.gui.i18n import resolve, retranslate_text, t

__all__ = ["install_wheel_guard", "retranslate_tree", "tab_header", "with_help"]


def _help_button(title: str, text: str, parent: QWidget) -> QToolButton:
    """Build the faint ``?`` button that opens a reference note.

    ``title``/``text`` are msgids -- either a catalog key or the English
    source; they are translated *at click time* via
    :func:`~optivibe.gui.i18n.resolve`, so the popup always follows the current
    language with no retranslation bookkeeping. Callers must pass the msgid,
    never an already-translated string: a translated string cannot be looked up
    again, which both hides real gaps and blinds the S-26 guard.
    """
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
    button.setToolTip(t("What is this parameter?"))
    button.clicked.connect(lambda: QMessageBox.information(parent, resolve(title), resolve(text)))
    return button


def with_help(widget: QWidget, title: str, text: str) -> QWidget:
    """Wrap an input widget with a faint inline ``?`` help button.

    Parameters
    ----------
    widget : QWidget
        The input control (line edit, combo, spin box, sub-layout holder).
    title : str
        Msgid (catalog key or English source) of the note's window title,
        usually the row label. Not a pre-translated string -- see
        :func:`_help_button`.
    text : str
        Msgid of the reference note: what the parameter is, typical values,
        couplings, effect on the simulation.

    Returns
    -------
    QWidget
        A container holding ``widget`` and the help button, suitable as a
        ``QFormLayout`` row field.
    """
    resolved_title = resolve(title)
    resolved_text = resolve(text)
    # Populate the widget's What's-This so the Shift+F1 "?" mode surfaces the
    # same help as the inline button (otherwise the mode has nothing to show).
    widget.setWhatsThis(f"{resolved_title}\n\n{resolved_text}")
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    row.addWidget(widget, stretch=1)
    # Pass the English msgids so the popup translates live at click time.
    row.addWidget(_help_button(title, text, holder))
    holder.setWhatsThis(f"{resolved_title}\n\n{resolved_text}")
    return holder


class _TabHeader(QWidget):
    """A tab description strip: a bold title plus a ``?`` note about the tab.

    Stores the English msgids so :meth:`retranslate` can refresh the title after
    a language switch; the ``?`` note translates at click time.
    """

    def __init__(self, title: str, about: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._label = QLabel(resolve(title))
        self._label.setStyleSheet("font-weight: bold;")
        row = QHBoxLayout(self)
        row.setContentsMargins(2, 2, 2, 2)
        row.addWidget(self._label)
        row.addWidget(_help_button(title, about, self))
        row.addStretch(1)

    def retranslate(self) -> None:
        """Refresh the title label in the current language."""
        self._label.setText(resolve(self._title))


def tab_header(title: str, about: str) -> _TabHeader:
    """Build a tab-description header (bold title + ``?`` about-this-tab note)."""
    return _TabHeader(title, about)


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


def retranslate_tree(root: QWidget) -> None:
    """Re-render every localized string under ``root`` in the active language.

    A panel that translates only at build time is half localized: the control
    the user is looking at keeps the language it was born in. The hand-written
    ``retranslate`` methods cover the controls somebody remembered, which is the
    same failure mode as a hand-kept string list (13, ``SW-78``) -- the S-26
    guard found several dozen row captions, tool-tips and What's-This notes
    that a switch never reached.

    This walk is generic instead: every rendered string is looked back up in
    the catalog (:func:`~optivibe.gui.i18n.retranslate_text`) and re-emitted, so
    a control added tomorrow follows the language without a new line here.
    Values, paths and configuration tokens are not in the catalog and pass
    through untouched.

    Combo *entries* are deliberately left alone: on several controls the shown
    text **is** the payload value (``currentText()`` feeds the scenario), so a
    generic rewrite there would change the run, not the wording. Combos that do
    carry prose relabel themselves in their panel's ``retranslate`` -- and if
    one forgets, the S-26 switch guard says so instead of this walk silently
    corrupting a configuration.
    """
    for widget in [root, *root.findChildren(QWidget)]:
        if isinstance(widget, QLabel):
            widget.setText(retranslate_text(widget.text()))
        if isinstance(widget, QAbstractButton):
            widget.setText(retranslate_text(widget.text()))
        if isinstance(widget, QGroupBox):
            widget.setTitle(retranslate_text(widget.title()))
        if isinstance(widget, QLineEdit):
            widget.setPlaceholderText(retranslate_text(widget.placeholderText()))
        if isinstance(widget, QTabWidget):
            for index in range(widget.count()):
                widget.setTabText(index, retranslate_text(widget.tabText(index)))
        widget.setToolTip(retranslate_text(widget.toolTip()))
        widget.setWhatsThis(retranslate_text(widget.whatsThis()))
