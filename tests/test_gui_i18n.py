"""GUI localization + Batch-1 interface features (i18n, menu, theme, check, log).

Head-less via ``QT_QPA_PLATFORM=offscreen`` (conftest). English is the default
and its catalog values are byte-identical to the former inline strings, so the
frozen S7 tests are unaffected; here we exercise the Russian path and the new
Preferences / menu / theme / composition-check / log-dock features.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QKeySequence

from optivibe.gui.i18n import (
    CATALOG,
    LANGUAGES,
    set_language,
    t,
    tr,
)
from optivibe.gui.main_window import MainWindow
from optivibe.gui.theme import THEMES
from optivibe.gui.widgets.preferences_dialog import PreferencesDialog


@pytest.fixture(autouse=True)
def _reset_i18n() -> object:
    """Keep English the default for other test files; clean persisted keys."""
    set_language("en")
    yield
    set_language("en")
    settings = QSettings("OptiVibe", "OptiVibe")
    settings.remove("language")
    settings.remove("geometry")
    settings.sync()


# --------------------------------------------------------------------------- #
# Catalog + translation primitives
# --------------------------------------------------------------------------- #
def test_catalog_entries_are_complete() -> None:
    """Every catalog entry carries a non-empty English and Russian value."""
    for key, entry in CATALOG.items():
        assert set(entry) >= set(LANGUAGES), key
        assert entry["en"].strip(), key
        assert entry["ru"].strip(), key


def test_t_is_english_passthrough_then_russian() -> None:
    """``t`` returns the source in English and its translation in Russian."""
    assert t("Run") == "Run"
    assert t("does-not-exist") == "does-not-exist"
    set_language("ru")
    assert t("Run") == "Запуск"
    assert t("System") == "Система"


def test_t_translates_a_template_then_formats() -> None:
    """A note template is translated first, then filled with values."""
    set_language("ru")
    out = tr("status.exported", n=2, directory="/tmp")
    assert "2" in out and "/tmp" in out
    assert out != "Exported 2 file(s) to /tmp."


# --------------------------------------------------------------------------- #
# Live language switch
# --------------------------------------------------------------------------- #
def test_language_switch_relabels_the_chrome(qtbot) -> None:
    """Switching to Russian re-labels tabs, buttons, menu and panel tabs."""
    window = MainWindow()
    qtbot.addWidget(window)
    set_language("ru")
    assert window._tabs.tabText(0) == "Онлайн"
    assert window._run_button.text() == "Запуск"
    assert window._check_button.text() == "Проверить композицию"
    assert window.control_panel.system.tabText(0) == "Система"
    assert next(a.text() for a in window.menuBar().actions()) == "&Файл"
    set_language("en")
    assert window._tabs.tabText(0) == "Live"
    assert window.control_panel.system.tabText(0) == "System"


def test_language_switch_preserves_scenario_state(qtbot) -> None:
    """The composition and excitation survive a language rebuild round-trip."""
    window = MainWindow()
    qtbot.addWidget(window)
    before = window.control_panel.scenario_payload()
    name_before = window.control_panel.system_payload()["name"]
    set_language("ru")
    after = window.control_panel.scenario_payload()
    assert after["excitation"]["kind"] == before["excitation"]["kind"]
    assert after["stages"] == before["stages"]
    assert window.control_panel.system_payload()["name"] == name_before


# --------------------------------------------------------------------------- #
# Menu / shortcuts
# --------------------------------------------------------------------------- #
def test_menu_bar_exposes_actions_with_shortcuts(qtbot) -> None:
    """The Run menu carries Run (Ctrl+R) and Report (Ctrl+Shift+R)."""
    window = MainWindow()
    qtbot.addWidget(window)
    run = window._menu_actions["menu.run_action"]
    report = window._menu_actions["menu.report_action"]
    assert run.shortcut() == QKeySequence("Ctrl+R")
    assert report.shortcut() == QKeySequence("Ctrl+Shift+R")


# --------------------------------------------------------------------------- #
# Composition check
# --------------------------------------------------------------------------- #
def test_composition_check_passes_for_the_default_variant(qtbot, config_dir: Path) -> None:
    """The default composition resolves and the check reports a pass."""
    window = MainWindow(config_dir=config_dir)
    qtbot.addWidget(window)
    captured: dict[str, object] = {}
    window._show_check = lambda ok, detail: captured.update(ok=ok, detail=detail)  # type: ignore[method-assign]
    window._on_check()
    assert captured["ok"] is True


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
def test_theme_can_be_applied(qtbot) -> None:
    """Applying each theme is accepted and remembered."""
    window = MainWindow()
    qtbot.addWidget(window)
    for name in THEMES:
        window._apply_theme(name)
        assert window._theme == name


# --------------------------------------------------------------------------- #
# Log dock
# --------------------------------------------------------------------------- #
def test_log_dock_toggles_and_mirrors_the_logger(qtbot) -> None:
    """Toggling shows the dock, and logger records land in the view."""
    window = MainWindow()
    qtbot.addWidget(window)
    assert window._log_dock.isHidden()
    window._toggle_log()
    assert not window._log_dock.isHidden()
    assert window._log_action.isChecked()
    logging.getLogger("optivibe.test_i18n").warning("marker-xyzzy")
    assert "marker-xyzzy" in window._log_view.toPlainText()


# --------------------------------------------------------------------------- #
# Preferences dialog
# --------------------------------------------------------------------------- #
def test_preferences_dialog_previews_and_reports_selection(qtbot) -> None:
    """Changing the language combo previews it and is reported in selection()."""
    dialog = PreferencesDialog(language="en", theme="light", restore_geometry=True)
    qtbot.addWidget(dialog)
    previews: list[str] = []
    dialog.language_previewed.connect(previews.append)
    ru_index = LANGUAGES.index("ru")
    dialog._language.setCurrentIndex(ru_index)
    assert previews and previews[-1] == "ru"
    assert dialog.selection()["language"] == "ru"
    assert dialog.selection()["theme"] in THEMES
