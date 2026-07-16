"""Light / dark application themes (Batch 1, SW-66).

A tiny, dependency-free theming layer: the Fusion style plus a hand-set
:class:`~PySide6.QtGui.QPalette`. ``light`` restores the platform default
palette; ``dark`` applies a neutral dark palette. Applied process-wide to the
:class:`~PySide6.QtWidgets.QApplication` and persisted via ``QSettings``.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

__all__ = ["THEMES", "apply_theme"]

#: Selectable theme names (order defines the preferences combo).
THEMES: tuple[str, ...] = ("light", "dark")


def _dark_palette() -> QPalette:
    """Build a neutral dark palette for the Fusion style."""
    p = QPalette()
    window = QColor(0x2B, 0x2B, 0x2B)
    base = QColor(0x23, 0x23, 0x23)
    alt = QColor(0x31, 0x31, 0x31)
    text = QColor(0xE6, 0xE6, 0xE6)
    disabled = QColor(0x7F, 0x7F, 0x7F)
    highlight = QColor(0x2A, 0x6B, 0xB8)
    p.setColor(QPalette.ColorRole.Window, window)
    p.setColor(QPalette.ColorRole.WindowText, text)
    p.setColor(QPalette.ColorRole.Base, base)
    p.setColor(QPalette.ColorRole.AlternateBase, alt)
    p.setColor(QPalette.ColorRole.ToolTipBase, window)
    p.setColor(QPalette.ColorRole.ToolTipText, text)
    p.setColor(QPalette.ColorRole.Text, text)
    p.setColor(QPalette.ColorRole.Button, window)
    p.setColor(QPalette.ColorRole.ButtonText, text)
    p.setColor(QPalette.ColorRole.BrightText, QColor(0xFF, 0x55, 0x55))
    p.setColor(QPalette.ColorRole.Link, QColor(0x5A, 0x9B, 0xE6))
    p.setColor(QPalette.ColorRole.Highlight, highlight)
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(0xFF, 0xFF, 0xFF))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        p.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    return p


def apply_theme(app: QApplication, name: str) -> None:
    """Apply a theme by name to the application (unknown names -> light).

    Parameters
    ----------
    app : QApplication
        The running application instance.
    name : str
        A theme name from :data:`THEMES`.
    """
    app.setStyle("Fusion")
    if name == "dark":
        app.setPalette(_dark_palette())
    else:
        app.setPalette(app.style().standardPalette())
