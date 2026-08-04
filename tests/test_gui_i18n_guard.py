"""Localization as a checkable invariant over the *whole* GUI (task S-26).

Why a guard at all (13, ``SW-76``). :func:`~optivibe.gui.i18n.t` uses the
English source string as its msgid, so a string added to a widget without its
catalog entry does not raise, does not fail ``mypy`` and does not fail a test --
it silently keeps its English text in a Russian session. The gates stay green
while the user reads a half-translated page. That is a *class* of defect, and it
already reached the screen twice (``SW-71`` -> S-23). ``S-23`` closed one page
with a scanning test; this module closes the class.

Three complementary mechanisms, because no single one sees everything:

``test_no_gui_string_bypasses_the_catalog``
    Records every :func:`t` / :func:`tr` lookup made while the GUI is built and
    exercised, and fails on any that the catalog does not resolve. This is the
    ``SW-76`` class exactly -- *a string in the code with no catalog entry* --
    and it sees *inside* composed captions (``f"{t(label)} [{unit}]"``), which a
    text-comparison test cannot.

``test_every_visible_string_is_localized``
    Builds the tree twice (English, Russian) and compares position by position.
    A string that comes out **identical** in both locales either never went
    through the catalog at all (a bare ``QLabel("speed")``) or is legitimately
    locale-neutral; the latter must be named in :data:`_LOCALE_NEUTRAL`. This
    catches what the recorder cannot: text that never calls ``t``/``tr``.

``test_language_switch_retranslates_the_built_tree``
    Compares a tree built in English and *switched* to Russian against a tree
    built in Russian. Generalizes ``test_secondary_tabs_translate_on_language_
    switch`` from a hand-picked handful of controls to every string on screen.

Neither list of strings is hand-kept: both scans walk the object tree, so a
field added tomorrow is scanned tomorrow. The only hand-kept list is the
*exemption* set, which is the point -- adding to it is a visible decision.

**The probes are themselves probed.** A guard that finds nothing is
indistinguishable from a guard that cannot find anything; this project has paid
for that three times (``mypy`` without the gui extra, the S7 probe comparing
``id(QThread)``, and the "covered but never asserted" branch of S-25, 18 §7c).
Every mechanism above therefore has a companion test that feeds it a
deliberately broken widget and asserts it fails.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QEvent, QSettings
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMenu,
    QTabWidget,
    QWidget,
)

from optivibe.core.config.presets import PresetStore
from optivibe.gui.i18n import (
    CATALOG,
    LANGUAGE_LABELS,
    LANGUAGES,
    record_translation_misses,
    set_language,
    t,
)
from optivibe.gui.main_window import MainWindow
from optivibe.gui.widgets.preferences_dialog import PreferencesDialog

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def _clean_language_state() -> Iterator[None]:
    """Pin English and clear the persisted language **before** and after.

    :class:`MainWindow` applies ``QSettings("language")`` before it builds a
    single widget, so a value left behind by another test (or by a developer
    who once switched the app to Russian) silently builds the tree in a
    language this module did not ask for. The first draft of this guard passed
    for exactly that reason: both "locales" were English.
    """
    settings = QSettings("OptiVibe", "OptiVibe")
    settings.remove("language")
    settings.remove("geometry")
    settings.sync()
    set_language("en")
    yield
    set_language("en")
    settings.remove("language")
    settings.remove("geometry")
    settings.sync()


# --------------------------------------------------------------------------- #
# Exemptions: strings that are the same in English and Russian *by nature*
# --------------------------------------------------------------------------- #
#: Configuration vocabulary. These reach the screen as combo entries, tolerance
#: check-boxes and sweep parameters, but they are the very tokens the user
#: writes in a scenario / composition YAML (11 §2, 09 §5): translating them
#: would make the form disagree with the file it edits. Several are also read
#: back from ``currentText()`` to build the payload, so a translation would
#: change behaviour, not just wording.
_CONFIG_TOKENS = frozenset(
    {
        # excitation kinds (11 §2)
        "sine",
        "multitone",
        "sweep",
        "random",
        "shock",
        "composite",
        "csv",
        "wav",
        "tdms",
        "uff",
        "mat",
        "hdf5",
        # axes and units of the excitation grid (00, 01 §2)
        "x",
        "y",
        "z",
        "m/s^2",
        "g",
        # sweep method / modulation kind / spectrum-estimator options
        "linear",
        "log",
        "none",
        "am",
        "fm",
        "fft",
        "welch",
        # window family (theory-06 §4)
        "hann",
        "hamming",
        "blackman",
        "nuttall",
        "flattop",
        "boxcar",
        # inverse-chain options (05, theory-06)
        "frequency",
        "time",
        "leaky",
        "static",
        "operating_point",
        "nonlinear_curve",
        "plateau",
        "dynamic",
        "measured",
        # stage implementations (09 §5 registry keys)
        "modal",
        "modal_time",
        "photodiode",
        "standard",
        "stub",
        # reflector shapes and detector conventions (03, 07 §1.2)
        "cylinder",
        "sphere",
        "plane",
        "wedge",
        "matched",
        "bright",
        # source lineshape sentinel: read back verbatim by the payload (M-10)
        "(default)",
        "gaussian",
        "lorentzian",
        # composition mode / endface route (08 §6, 08)
        "offresonance",
        "resonance",
        "1",
        "2",
        # starting compositions and sweep modes
        "A",
        "B",
        "C",
        "D",
        "design",
        "response",
        # swept / tolerance parameter names (identical to the config fields)
        "length_m",
        "power_w",
        "full_scale_g",
        "q_total",
        "gap_m",
        "radius_of_curvature_m",
        "bias_offset_m",
        "epsilon_x",
        "amplitude_g",
        "frequency_hz",
    }
)

#: Symbols, acronyms and format names. Symbols come from 01 §2 and are the same
#: in every language by convention; acronyms are used untranslated in Russian
#: engineering text; ``YAML`` is a format name and ``?`` is the inline-help
#: glyph. The bracketed unit is part of the caption but never translated
#: (10 §6: SI symbols are not localized).
_SYMBOLS = frozenset(
    {
        "?",
        "YAML",
        "f_min",
        "f_max",
        "DSP",
        "NEA(f)",
        "CMRR [dB]",
        "RIN [dB/Hz]",
        # an example HDF5 dataset path shown as a placeholder
        "/accel/x",
        # the language selector names each language in that language, on purpose
        *LANGUAGE_LABELS.values(),
    }
)


def _preset_names() -> frozenset[str]:
    """Preset identifiers offered by the preset combos.

    Derived from the store rather than listed, so adding ``configs/presets/...``
    does not break this guard: a preset name is a file stem the user references
    from a composition YAML, not prose.
    """
    store = PresetStore(Path(__file__).resolve().parent.parent / "configs")
    names: set[str] = set()
    for subsystem in ("source", "fiber", "cantilever", "reflector", "detector"):
        names.update(store.list_presets(subsystem))
    return frozenset(names)


def _locale_neutral() -> frozenset[str]:
    """The full exemption set (config vocabulary + symbols + preset names)."""
    return _CONFIG_TOKENS | _SYMBOLS | _preset_names()


#: Catalog entries whose Russian value equals the English one. Without this
#: check a translation could be "done" by copying the English text, and the
#: locale scan below would see two different-looking trees while the user still
#: reads English. Each key here is an acronym, a symbol or a pure format
#: template with no translatable words.
_IDENTICAL_BY_NATURE = frozenset(
    {
        "source.rin.label",  # RIN
        "detector.cmrr.label",  # CMRR
        "stage.dsp.label",  # DSP
        "live.panel.nea",  # NEA(f)
        "plot.axis.a",  # symbol a, 01 §2
        "plot.axis.v",  # symbol v, 01 §2
        "exc.row.remove",  # the "x" remove glyph
        "system.yaml.filter",  # format name + glob
        "status.progress",  # "... {message}"
        "source.spectrum.note",  # "{file} ({instrument}, {timestamp}; sha {sha})"
        "compare.chain_verdict",  # "{name} [{status}]: {deviations}"
        "dsp.exp.row.nperseg",  # Welch nperseg -- proper name + config field
        "dsp.exp.row.noverlap",  # Welch noverlap -- ditto
    }
)


# --------------------------------------------------------------------------- #
# The scan
# --------------------------------------------------------------------------- #
def _is_third_party_chrome(widget: QWidget) -> bool:
    """Is this widget part of a toolbar whose text OptiVibe does not own?

    The matplotlib navigation toolbar (``Home`` / ``Pan`` / ``Zoom`` / ...) is
    embedded verbatim by :class:`~optivibe.gui.widgets.mpl_canvas.MplFigureView`
    and labels itself from matplotlib, not from :data:`CATALOG`. Relabelling
    somebody else's toolbar is a decision, not a translation, so it is skipped
    *structurally* -- by ownership, not by listing its captions -- and recorded
    in the backlog instead of being silently patched here.
    """
    node: QWidget | None = widget
    while node is not None:
        if type(node).__name__.startswith("NavigationToolbar"):
            return True
        node = node.parentWidget()
    return False


def visible_strings(root: QWidget) -> list[tuple[str, str]]:
    """Collect every user-visible string under ``root``.

    Walks the Qt object tree and reads each surface a user can actually read:
    labels, button/check-box captions, group titles, combo entries, line-edit
    placeholders, tab captions, tool-tips, What's-This notes, window titles and
    the menu bar. Positional and deterministic, so two builds of the same tree
    line up entry by entry.

    Returns
    -------
    list of (str, str)
        ``(where, text)`` pairs in tree order. ``where`` names the surface and
        the widget class, so a failure points at something findable.
    """
    out: list[tuple[str, str]] = []
    for widget in [root, *root.findChildren(QWidget)]:
        if _is_third_party_chrome(widget):
            continue
        name = type(widget).__name__
        if isinstance(widget, QLabel):
            out.append((f"label/{name}", widget.text()))
        if isinstance(widget, QAbstractButton):
            out.append((f"button/{name}", widget.text()))
        if isinstance(widget, QGroupBox):
            out.append((f"group-title/{name}", widget.title()))
        if isinstance(widget, QComboBox):
            for index in range(widget.count()):
                out.append((f"combo-item[{index}]/{name}", widget.itemText(index)))
        if isinstance(widget, QLineEdit):
            out.append((f"placeholder/{name}", widget.placeholderText()))
        if isinstance(widget, QTabWidget):
            for index in range(widget.count()):
                out.append((f"tab[{index}]/{name}", widget.tabText(index)))
        out.append((f"tooltip/{name}", widget.toolTip()))
        out.append((f"whats-this/{name}", widget.whatsThis()))
        out.append((f"window-title/{name}", widget.windowTitle()))
    for menu in root.findChildren(QMenu):
        out.append(("menu-title/QMenu", menu.title()))
        for action in menu.actions():
            out.append(("menu-action/QAction", action.text()))
    return out


def untranslated(
    english: list[tuple[str, str]],
    russian: list[tuple[str, str]],
    exempt: frozenset[str],
) -> list[str]:
    """Report strings that came out identical in both locales.

    Pure function over two scans, so it can be exercised on a deliberately
    broken widget without building the application (see the self-checks).

    Raises
    ------
    AssertionError
        If the two scans do not line up -- the trees must be built the same way
        for a positional comparison to mean anything.
    """
    assert len(english) == len(russian), (
        f"the two builds produced different trees ({len(english)} vs "
        f"{len(russian)} strings); the scan cannot be compared positionally"
    )
    offenders: list[str] = []
    seen: set[str] = set()
    for (where_en, text_en), (where_ru, text_ru) in zip(english, russian, strict=True):
        assert where_en == where_ru, f"tree order diverged: {where_en} vs {where_ru}"
        if not text_en.strip() or text_en != text_ru or text_en in exempt:
            continue
        if text_en in seen:
            continue
        seen.add(text_en)
        offenders.append(f"{where_en}: {text_en!r}")
    return offenders


def _exercise(config_dir: Path) -> list[QWidget]:
    """Build the GUI and walk it through every state that shows new strings.

    A widget only exposes text once it exists, so the scan is only as wide as
    the states it visits: every outer tab, every parameter tab, every excitation
    kind (including the composite sub-forms), every reflector shape, both sweep
    modes, both live/compare sources, and the Preferences dialog.
    """
    window = MainWindow(config_dir=config_dir)
    panel = window.control_panel
    system = panel.system

    for index in range(system.count()):
        system.setCurrentIndex(index)
    for index in range(window._tabs.count()):
        window._tabs.setCurrentIndex(index)

    builder = panel._excitation
    for index in range(builder._kind.count()):
        builder._kind.setCurrentIndex(index)
    composite = builder._composite
    row = composite._add_row()
    for index in range(row._kind.count()):
        row._kind.setCurrentIndex(index)

    for index in range(system._reflector._shape.count()):
        system._reflector._shape.setCurrentIndex(index)
    for index in range(system._mode.count()):
        system._mode.setCurrentIndex(index)
    for index in range(window._sweep._mode.count()):
        window._sweep._mode.setCurrentIndex(index)
    for index in range(window._live.controls._source.count()):
        window._live.controls._source.setCurrentIndex(index)
    for index in range(window._compare._source.count()):
        window._compare._source.setCurrentIndex(index)

    dialog = PreferencesDialog(language="en", theme="light", restore_geometry=True)
    return [window, dialog]


def _scan(config_dir: Path, qtbot: Any, language: str) -> list[tuple[str, str]]:
    """Build and exercise the GUI in ``language`` and return its visible text."""
    set_language(language)
    strings: list[tuple[str, str]] = []
    for widget in _exercise(config_dir):
        qtbot.addWidget(widget)
        strings.extend(visible_strings(widget))
    return strings


# --------------------------------------------------------------------------- #
# 1. No string in the code bypasses the catalog
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("language", LANGUAGES)
def test_no_gui_string_bypasses_the_catalog(qtbot, config_dir: Path, language: str) -> None:
    """Every ``t``/``tr`` lookup made while building the GUI resolves.

    A miss here is the ``SW-76`` defect itself: the call site believes it is
    localized, the lookup falls through to the English source, and nothing else
    in the pipeline notices. Checked in both locales because ``tr`` can miss a
    key in either, while ``t`` degrades only in Russian.
    """
    with record_translation_misses() as misses:
        widgets = _exercise(config_dir)
    for widget in widgets:
        qtbot.addWidget(widget)
    assert not sorted(set(misses)), (
        "these lookups have no catalog entry, so the string stays English in a "
        f"Russian session -- add it to CATALOG (both locales): {sorted(set(misses))}"
    )


# --------------------------------------------------------------------------- #
# 2. Every visible string differs between the locales (or is exempt)
# --------------------------------------------------------------------------- #
def test_every_visible_string_is_localized(qtbot, config_dir: Path) -> None:
    """No user-visible string comes out identical in English and Russian.

    Catches the half of the class the miss recorder cannot see: text that never
    reaches ``t``/``tr`` at all -- a bare ``QLabel("speed")``, a literal
    ``setToolTip(...)``, a placeholder. Legitimately neutral text (config
    vocabulary, symbols, preset names) is named in the exemption set, which is
    deliberately narrow: it lists tokens, not categories.
    """
    english = _scan(config_dir, qtbot, "en")
    russian = _scan(config_dir, qtbot, "ru")
    offenders = untranslated(english, russian, _locale_neutral())
    assert not offenders, (
        "these strings are identical in both locales; translate them, or add "
        f"them to the exemption set with a reason: {offenders}"
    )


# --------------------------------------------------------------------------- #
# 3. A language switch reaches the widgets that already exist
# --------------------------------------------------------------------------- #
def test_language_switch_retranslates_the_built_tree(qtbot, config_dir: Path) -> None:
    """Switching the language relabels the existing tree, not just new widgets.

    Translating at build time is only half the contract: a control built in
    English and never refreshed keeps its English text forever. The reference
    is a tree *built* in Russian -- anything the switched tree still shows in
    English is a missing ``retranslate`` (the precedent is
    ``test_secondary_tabs_translate_on_language_switch``, generalized here from
    six controls to the whole window).
    """
    # The window is compared in its *default* state: a language switch rebuilds
    # the control panel from a payload round-trip (SW-65), so a tree walked
    # through every excitation kind and sub-form first would legitimately come
    # back in a different shape and the positional diff would mean nothing.
    set_language("ru")
    fresh = MainWindow(config_dir=config_dir)
    qtbot.addWidget(fresh)
    reference = visible_strings(fresh)

    set_language("en")
    window = MainWindow(config_dir=config_dir)
    qtbot.addWidget(window)
    set_language("ru")
    # ``_relanguage`` retires the old control panel and physics tab with
    # ``deleteLater``; without an event-loop turn they are still children of the
    # window and the walk would see both the old and the new tree.
    QApplication.processEvents()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()
    switched = visible_strings(window)

    assert len(switched) == len(reference), (
        "the switched tree has a different shape than a freshly built one; "
        "compare the rebuild in MainWindow._relanguage"
    )
    stale = sorted(
        {
            f"{where}: {after!r} (built in Russian: {expected!r})"
            for (where, after), (_where, expected) in zip(switched, reference, strict=True)
            if after != expected
        }
    )
    assert not stale, f"these strings did not follow the language switch: {stale}"


# --------------------------------------------------------------------------- #
# 4. The catalog itself is not "translated" by copying English
# --------------------------------------------------------------------------- #
def test_catalog_entries_are_not_english_in_both_locales() -> None:
    """A Russian value equal to the English one must be justified.

    Without this the other guards can be satisfied by copying: the tree scan
    would see the same string on both sides only for entries listed here, and
    the miss recorder resolves happily against a duplicated value.
    """
    copied = sorted(
        key
        for key, entry in CATALOG.items()
        if entry["en"] == entry["ru"] and key not in _IDENTICAL_BY_NATURE
    )
    assert not copied, (
        "these catalog entries carry the English text as their Russian value; "
        f"translate them or list them as identical by nature: {copied}"
    )


def test_catalog_has_no_duplicate_english_sources() -> None:
    """Two entries must not claim the same English msgid with different Russian.

    ``t`` indexes the catalog *by its English value*, so a duplicated source
    silently makes one of the two translations unreachable. This is not
    hypothetical: it happened while S-26 was being written -- a new key reused
    ``source.rin.tip`` and dropped the field tool-tip out of the index, which
    the miss recorder caught.
    """
    by_source: dict[str, set[str]] = {}
    for entry in CATALOG.values():
        by_source.setdefault(entry["en"], set()).add(entry["ru"])
    clashes = sorted(source for source, values in by_source.items() if len(values) > 1)
    assert not clashes, f"one English source maps to several Russian texts: {clashes}"


# --------------------------------------------------------------------------- #
# 5. Self-checks: the probes are shown to fail on a broken case
# --------------------------------------------------------------------------- #
def test_the_miss_recorder_reports_a_string_without_a_catalog_entry() -> None:
    """Feed the recorder a string that is not in the catalog; it must report it.

    Without this, ``test_no_gui_string_bypasses_the_catalog`` passing would be
    indistinguishable from the recorder never recording anything (18 §7c).
    """
    with record_translation_misses() as misses:
        t("Run")  # a real msgid: must not be recorded
        t("no such string, on purpose (S-26 self-check)")
    assert misses == ["t: no such string, on purpose (S-26 self-check)"]


def test_the_locale_scan_reports_an_untranslated_widget(qtbot) -> None:
    """Scan a widget carrying one translated and one bare label.

    Proves the comparison actually distinguishes the two, and that the
    exemption set can silence the bare one -- i.e. both branches work.
    """
    from PySide6.QtWidgets import QVBoxLayout

    def build() -> QWidget:
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.addWidget(QLabel(t("Run")))
        layout.addWidget(QLabel("deliberately untranslated"))
        return holder

    set_language("en")
    english_widget = build()
    qtbot.addWidget(english_widget)
    english = visible_strings(english_widget)
    set_language("ru")
    russian_widget = build()
    qtbot.addWidget(russian_widget)
    russian = visible_strings(russian_widget)
    set_language("en")

    offenders = untranslated(english, russian, frozenset())
    assert offenders == ["label/QLabel: 'deliberately untranslated'"]
    assert not untranslated(english, russian, frozenset({"deliberately untranslated"}))


def test_the_switch_guard_reports_a_widget_that_does_not_retranslate(qtbot) -> None:
    """A label built in English and left alone must be reported as stale.

    The retranslation check is a positional diff, so it can only work if a
    stale string really shows up as a difference; here it is made to.
    """
    set_language("en")
    stale_label = QLabel(t("Run"))
    qtbot.addWidget(stale_label)
    set_language("ru")
    fresh_label = QLabel(t("Run"))
    qtbot.addWidget(fresh_label)

    switched = visible_strings(stale_label)
    reference = visible_strings(fresh_label)
    set_language("en")

    differences = [
        (where, after, expected)
        for (where, after), (_where, expected) in zip(switched, reference, strict=True)
        if after != expected
    ]
    assert differences == [("label/QLabel", "Run", "Запуск")]
