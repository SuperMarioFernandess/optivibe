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


# --------------------------------------------------------------------------- #
# What's-This help text
# --------------------------------------------------------------------------- #
def test_with_help_populates_whats_this(qtbot) -> None:
    """with_help sets the widget's What's-This so Shift+F1 mode has content."""
    from PySide6.QtWidgets import QLineEdit

    from optivibe.gui.widgets.ui_helpers import with_help

    edit = QLineEdit()
    holder = with_help(edit, "Variant", "A reference note.")
    qtbot.addWidget(holder)
    assert edit.whatsThis().strip()
    assert "reference note" in edit.whatsThis()


# --------------------------------------------------------------------------- #
# Matplotlib figure translation
# --------------------------------------------------------------------------- #
def test_translate_figure_localizes_axis_labels() -> None:
    """translate_figure maps static English axis labels to Russian, no-op in EN."""
    from matplotlib.figure import Figure

    from optivibe.gui.widgets.mpl_canvas import translate_figure

    set_language("ru")
    fig = Figure()
    ax = fig.add_subplot()
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("amplitude")
    translate_figure(fig)
    assert ax.get_xlabel() == "частота [Гц]"
    assert ax.get_ylabel() == "амплитуда"

    set_language("en")
    fig2 = Figure()
    ax2 = fig2.add_subplot()
    ax2.set_xlabel("frequency [Hz]")
    translate_figure(fig2)
    assert ax2.get_xlabel() == "frequency [Hz]"


# --------------------------------------------------------------------------- #
# Secondary tabs: about header + live retranslation
# --------------------------------------------------------------------------- #
def test_secondary_tabs_have_about_headers(qtbot) -> None:
    """Live / Report / Sweeps / Monte-Carlo / Physics each carry a tab header."""
    window = MainWindow()
    qtbot.addWidget(window)
    for panel in (window._live, window._report, window._sweep, window._monte, window._physics):
        assert panel._header is not None


def test_secondary_tabs_translate_on_language_switch(qtbot) -> None:
    """Switching to Russian relabels the secondary-tab controls in place."""
    window = MainWindow()
    qtbot.addWidget(window)
    set_language("ru")
    assert window._sweep._controls.title() == "Развёртка"
    assert window._sweep._run.text() == "Запустить развёртку"
    assert window._monte._run.text() == "Запустить Монте-Карло"
    assert window._report._budget_label.text() == "Бюджет ошибок"
    assert window._physics._refresh_button.text() == "Обновить из композиции"
    assert window._live._cantilever_check.text() == "консоль"
    set_language("en")
    assert window._sweep._controls.title() == "Sweep"
    assert window._live._cantilever_check.text() == "cantilever"


# --------------------------------------------------------------------------- #
# Citation convention: theory blocks are never cited as a bare "doc NN"
# --------------------------------------------------------------------------- #
#: Bare-number citations of the DSP theory block. Number 06 of the knowledge
#: base is the *physics decision log*, so "doc 06" / "док 06" sends the reader
#: to the wrong document; the block is ``docs/theory/06_dsp_algorithm.md`` and
#: is cited as ``theory-06 §X`` or by path (rule of 2026-07-31, doc 13). The
#: ambiguity already cost one incident (the block overwrote the decision log in
#: the knowledge base), and in the catalog it reaches the user's screen.
_AMBIGUOUS_THEORY_CITATIONS = ("doc 06", "док 06", "doc-06", "док-06")


def test_catalog_cites_the_dsp_theory_block_unambiguously() -> None:
    """No catalog string may cite the theory block as a bare ``doc 06``.

    Scans the whole catalog rather than the single string that was wrong, so a
    new tooltip cannot reintroduce the ambiguity in either locale.
    """
    offenders = [
        (key, lang)
        for key, entry in CATALOG.items()
        for lang in LANGUAGES
        for bad in _AMBIGUOUS_THEORY_CITATIONS
        if bad in entry[lang].lower()
    ]
    assert not offenders, f"cite the block as 'theory-06 §X' or by path: {offenders}"


def test_streaming_cutoff_help_points_at_the_theory_block() -> None:
    """The ``f_c_stream`` help cites ``theory-06 §9.3-2`` in both locales."""
    entry = CATALOG["dsp.exp.f_c.help"]
    for lang in LANGUAGES:
        assert "theory-06 §9.3-2" in entry[lang]


# --------------------------------------------------------------------------- #
# Catalog coverage: a page must not half-translate
# --------------------------------------------------------------------------- #
#: Texts that are the same in both locales by nature: the inline help glyph and
#: a format name. Everything else on a page belongs in the catalog.
_LOCALE_NEUTRAL = ("?", "YAML")


def test_catalog_covers_the_excitation_builder_strings(qtbot) -> None:
    """Every visible string of the excitation panel resolves through the catalog.

    Guards the invariant of SW-65 -- an English source in the code is a msgid,
    so a string added without its catalog entry degrades a Russian session to
    English silently. That is exactly what happened between S7-mod and S-21: the
    modulation rows, the composite page and the per-tone phase box reached the
    screen untranslated. Scanning the built widget (rather than a hand-kept
    list) is what keeps a new field from repeating it.
    """
    from PySide6.QtWidgets import QAbstractButton, QLabel

    from optivibe.gui.widgets.excitation_builder import (
        _ABOUT,
        _AXIS_HELP,
        _GRID_NOTE,
        _KIND_HELP,
        _SAMPLING_HELP,
        ExcitationBuilder,
    )

    sources = {_KIND_HELP, _AXIS_HELP, _SAMPLING_HELP, _GRID_NOTE}
    for summary, about in _ABOUT.values():
        sources.update((summary, about))

    builder = ExcitationBuilder()
    qtbot.addWidget(builder)
    filled_templates = {builder._composite._grid_note.text()}
    for kind in ("composite", "sine", "multitone", "sweep", "random", "shock"):
        builder._kind.setCurrentText(kind)
    for widget in (*builder.findChildren(QLabel), *builder.findChildren(QAbstractButton)):
        text = widget.text()
        if text.strip() and text not in filled_templates:
            sources.add(text)

    index = {entry["en"] for entry in CATALOG.values()}
    missing = sorted(text for text in sources if text not in index and text not in _LOCALE_NEUTRAL)
    assert not missing, f"add a catalog entry (both locales) for: {missing}"
