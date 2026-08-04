"""The Preferences dialog: language, theme and start-up options (Batch 1, SW-66).

A thin, model-free dialog. It reads the current values, emits the chosen
language and theme *live* (so the window can re-language / re-theme without a
restart) and returns the full selection for the caller to persist via
``QSettings``. It owns no policy -- the window decides what to do with the
choices.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from optivibe.gui.i18n import LANGUAGE_LABELS, LANGUAGES, t, tr
from optivibe.gui.theme import THEMES

__all__ = ["PreferencesDialog"]

_THEME_LABEL_KEY = {"light": "Light", "dark": "Dark"}


class PreferencesDialog(QDialog):
    """Choose language, theme and start-up options.

    Signals
    -------
    language_previewed(str)
        Emitted with a language code when the language selection changes, so the
        window can apply it immediately.
    theme_previewed(str)
        Emitted with a theme name when the theme selection changes.
    """

    language_previewed = Signal(str)
    theme_previewed = Signal(str)

    def __init__(
        self,
        *,
        language: str,
        theme: str,
        restore_geometry: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("Preferences"))
        self._start_language = language

        self._language = QComboBox()
        for code in LANGUAGES:
            self._language.addItem(LANGUAGE_LABELS[code], code)
        self._language.setCurrentIndex(max(0, LANGUAGES.index(language)))
        self._language.currentIndexChanged.connect(self._on_language)

        self._theme = QComboBox()
        for name in THEMES:
            self._theme.addItem(t(_THEME_LABEL_KEY[name]), name)
        self._theme.setCurrentIndex(max(0, THEMES.index(theme)))
        self._theme.currentIndexChanged.connect(self._on_theme)

        self._restore = QCheckBox(t("Restore window size on start"))
        self._restore.setChecked(restore_geometry)

        form = QFormLayout()
        form.addRow(t("Language"), self._language)
        form.addRow(t("Theme"), self._theme)
        form.addRow(self._restore)

        note = QLabel(t("Language and theme apply immediately; other options apply on restart."))
        note.setWordWrap(True)
        note.setStyleSheet("color: #808080; font-style: italic;")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        # Qt labels the standard buttons from its own (unloaded) translations,
        # so without this the two most visible controls of the dialog stay
        # English in a Russian session (S-26).
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr("prefs.ok"))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr("prefs.cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self._on_reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)

    def _on_language(self) -> None:
        self.language_previewed.emit(str(self._language.currentData()))

    def _on_theme(self) -> None:
        self.theme_previewed.emit(str(self._theme.currentData()))

    def _on_reject(self) -> None:
        """Revert a live language preview before closing on Cancel."""
        if str(self._language.currentData()) != self._start_language:
            self.language_previewed.emit(self._start_language)
        self.reject()

    def selection(self) -> dict[str, object]:
        """Return the chosen ``language``, ``theme`` and ``restore_geometry``."""
        return {
            "language": str(self._language.currentData()),
            "theme": str(self._theme.currentData()),
            "restore_geometry": self._restore.isChecked(),
        }
