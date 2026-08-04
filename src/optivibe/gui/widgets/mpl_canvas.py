"""Embed a pure :mod:`optivibe.viz` matplotlib figure in the Qt UI (task S7 §4).

``viz/`` returns Qt-free :class:`matplotlib.figure.Figure` objects (SW-09); this
widget hosts one on a :class:`~matplotlib.backends.backend_qtagg.FigureCanvasQTAgg`
inside the report / sweep / Monte-Carlo tabs, with a navigation toolbar. Swapping
the displayed figure replaces the canvas (matplotlib binds one figure per
canvas), so each new analysis result renders cleanly. The figure is built by
``viz``; this widget never plots -- it only embeds (architecture 09 §9).
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from optivibe.gui.i18n import current_language, t, translate_optional

__all__ = ["MplFigureView", "translate_figure"]


def translate_figure(figure: Figure) -> Figure:
    """Translate a figure's static axis/title/legend text into the active language.

    ``viz/`` is Qt-free and language-agnostic (SW-09): it always emits English
    labels. Rather than teach the core about languages, the GUI post-translates
    the figure here via the same English-msgid catalog -- a no-op in English.
    Pure-LaTeX/maths labels and data-bearing (formatted) titles are not in the
    catalog and pass through unchanged, which is why the lookup is the
    non-recording :func:`~optivibe.gui.i18n.translate_optional`: here a miss is
    normal by construction, unlike a widget string, where it is the ``SW-76``
    defect (13, ``SW-78``).
    """
    if current_language() == "en":
        return figure
    for ax in figure.get_axes():
        ax.set_xlabel(translate_optional(ax.get_xlabel()))
        ax.set_ylabel(translate_optional(ax.get_ylabel()))
        title = ax.get_title()
        if title:
            ax.set_title(translate_optional(title))
        legend = ax.get_legend()
        if legend is not None:
            for entry in legend.get_texts():
                entry.set_text(translate_optional(entry.get_text()))
    suptitle = getattr(figure, "_suptitle", None)
    if suptitle is not None:
        suptitle.set_text(translate_optional(suptitle.get_text()))
    return figure


class MplFigureView(QWidget):
    """A container that embeds (and hot-swaps) a matplotlib figure.

    Parameters
    ----------
    placeholder : str, optional
        Text shown before any figure is set.
    parent : QWidget or None, optional
        Parent widget.
    """

    def __init__(self, placeholder: str = "No figure yet.", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._placeholder_text = placeholder
        self._placeholder = QLabel(t(placeholder))
        self._placeholder.setWordWrap(True)
        self._layout.addWidget(self._placeholder)
        self._canvas: FigureCanvasQTAgg | None = None
        self._toolbar: NavigationToolbar2QT | None = None
        self._figure: Figure | None = None

    def retranslate(self) -> None:
        """Refresh the placeholder text for the active language."""
        self._placeholder.setText(t(self._placeholder_text))

    @property
    def figure(self) -> Figure | None:
        """The currently displayed figure (``None`` before the first set)."""
        return self._figure

    def set_figure(self, figure: Figure) -> None:
        """Display ``figure``, replacing any previous canvas.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            A figure produced by :mod:`optivibe.viz` (Qt-free).
        """
        translate_figure(figure)
        self._clear()
        self._placeholder.hide()
        canvas = FigureCanvasQTAgg(figure)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        toolbar = NavigationToolbar2QT(canvas, self)
        self._layout.addWidget(toolbar)
        self._layout.addWidget(canvas, stretch=1)
        canvas.draw_idle()
        self._canvas = canvas
        self._toolbar = toolbar
        self._figure = figure

    def _clear(self) -> None:
        """Remove the current canvas/toolbar (if any)."""
        for widget in (self._toolbar, self._canvas):
            if widget is not None:
                self._layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
        self._canvas = None
        self._toolbar = None
        self._figure = None
