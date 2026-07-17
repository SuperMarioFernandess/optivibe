"""In-app localization: a bilingual string catalog with live language switching.

A lightweight, in-repo alternative to Qt's ``.ts``/``.qm`` toolchain (SW-65).
The GUI text lives in one bilingual :data:`CATALOG` keyed by stable string IDs;
widgets call :func:`tr` at *build* time, and a language change re-runs the panel
build (state is preserved through the existing payload round-trip), so every
``tr`` re-evaluates in the new language. English is the default and its catalog
values are byte-identical to the previously inline strings, so the frozen tests
and golden are unaffected; Russian is opt-in and persisted via ``QSettings``.

Why not ``QTranslator``: ``tr()`` resolves at widget-construction time, the app
builds its widgets once, and the long help texts are module-level constants
outside any ``QObject`` -- native retranslation would need a hand-written
``retranslateUi`` pass plus a ``lupdate``/``lrelease`` build step and a bundled
``.qm``. The catalog + rebuild approach fits the existing centralized-constants
structure and leaves CI/packaging untouched (13 SW-65).
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

__all__ = [
    "LANGUAGES",
    "LANGUAGE_LABELS",
    "available_languages",
    "current_language",
    "language_bus",
    "set_language",
    "t",
    "tr",
]

#: Supported language codes (English first: the default and the test baseline).
LANGUAGES: tuple[str, ...] = ("en", "ru")

#: Human-readable, self-named language labels for the selector.
LANGUAGE_LABELS: dict[str, str] = {"en": "English", "ru": "Русский"}

_current: str = "en"


class _LanguageBus(QObject):
    """App-wide signal carrier: emits ``changed(code)`` on a language switch."""

    changed = Signal(str)


_BUS = _LanguageBus()


def language_bus() -> _LanguageBus:
    """Return the process-wide language-change signal bus."""
    return _BUS


def available_languages() -> tuple[str, ...]:
    """Return the supported language codes."""
    return LANGUAGES


def current_language() -> str:
    """Return the active language code."""
    return _current


def set_language(code: str) -> None:
    """Switch the active language and notify listeners.

    Parameters
    ----------
    code : str
        A code from :data:`LANGUAGES`. Unknown codes are ignored (defensive);
        a no-op switch (same code) does not emit, to avoid needless rebuilds.
    """
    global _current
    if code not in LANGUAGES or code == _current:
        return
    _current = code
    _BUS.changed.emit(code)


_EN_INDEX: dict[str, str] | None = None


def _en_index() -> dict[str, str]:
    """Build (once) the English-source -> Russian index used by :func:`t`.

    Lets call sites keep their English literal as the msgid (gettext style):
    no key churn at the hundreds of existing ``with_help``/label sites. Later
    duplicates overwrite earlier ones, which is harmless because a repeated
    English source maps to the same Russian text.
    """
    global _EN_INDEX
    if _EN_INDEX is None:
        _EN_INDEX = {entry["en"]: entry["ru"] for entry in CATALOG.values()}
    return _EN_INDEX


def t(text: str, /, **fmt: object) -> str:
    """Translate by English source string (msgid = the English text itself).

    Returns the English text unchanged in English mode (byte-identical to the
    former inline literals, so the frozen tests are unaffected) and its Russian
    counterpart in Russian mode; an unknown source degrades to itself. The text
    may carry ``{placeholders}`` -- the *template* is translated, then filled,
    so provenance notes localize without per-value keys.

    Parameters
    ----------
    text : str
        The English source string (possibly a ``str.format`` template).
    **fmt : object
        Optional format fields.

    Returns
    -------
    str
        The localized (and, if ``fmt`` given, formatted) string.
    """
    out = _en_index().get(text, text) if _current == "ru" else text
    return out.format(**fmt) if fmt else out


def tr(key: str, /, **fmt: object) -> str:
    """Translate a catalog ``key`` into the active language.

    Missing keys fall back to the English entry, then to the key itself, so a
    forgotten translation degrades to readable English rather than crashing.

    Parameters
    ----------
    key : str
        Catalog key (e.g. ``"source.wavelength.help"``).
    **fmt : object
        Optional ``str.format`` fields for keys that carry ``{placeholders}``
        (used by the provenance notes that mix text with measured values).

    Returns
    -------
    str
        The localized string.
    """
    entry = CATALOG.get(key)
    if entry is None:
        return key
    text = entry.get(_current) or entry.get("en") or key
    return text.format(**fmt) if fmt else text


def _e(en: str, ru: str) -> dict[str, str]:
    """Build a catalog entry (English + Russian)."""
    return {"en": en, "ru": ru}


# --------------------------------------------------------------------------- #
# The bilingual catalog. English values MUST stay byte-identical to the strings
# they replaced (the frozen GUI tests assert several of them verbatim).
# --------------------------------------------------------------------------- #
_PHYSICS_NOTES_EN = """\
<h3>Reference models for the current composition</h3>
<p><b>Mechanics (docs 02 / 05).</b> The fiber cantilever is a clamped-free beam.
The first bending mode sets f1 ~ 1/L^2; the lateral transfer is
H_lat(f) = H_lat^QS * D(f) with single-mode amplification |D(f1)| = Q. Shorter L
raises f1 and widens the flat band but lowers the quasi-static compliance
(sensitivity) -- the core design trade shown by the f1(L) and |H_lat(f)| curves.</p>
<p><b>Reflector coupling (doc 03).</b> eta(dx) is the Gaussian overlap between the
returning beam and the fiber mode. The static de-centering Delta x0 sets the
working point eta0 on the slope; cylinder/sphere are curved (finite R_c), the
plane is flat (no displacement coupling) and the wedge adds an angular bias.</p>
<p><b>Detector reference arm (doc 07 §1.2).</b> "matched" balances the bright and
reference arms (common-mode RIN rejection limited by CMRR); "bright" leaves the
reference arm dark (no RIN cancellation, higher shot floor). This is the open
question O-SW-08.</p>
<p><b>Inverse / DSP (docs 05 / 11).</b> The standard inverse de-rotates D(f),
applies the calibrated optical sensitivity and integrates to v and x. The
sensitivity model -- "static" (plateau slope), "operating_point" (local slope at
eta0) or "nonlinear_curve" (full eta(dx) inversion) -- trades bias against
robustness. The integrator runs in "frequency" (omega-domain) or "time" form.</p>
<p><b>NEA(f) (docs 07 / 08).</b> The noise-equivalent acceleration density with
its shot / RIN / Johnson plateaus is a measured budget; press
<i>Compute NEA(f)</i> to run it through the worker (a Report run). The sensor
<b>family</b> sweep is on the <i>Sweeps</i> tab.</p>
"""

_PHYSICS_NOTES_RU = """\
<h3>Справочные модели текущей композиции</h3>
<p><b>Механика (док 02 / 05).</b> Волоконная консоль — балка «заделка-свободный
конец». Первая изгибная мода задаёт f1 ~ 1/L²; боковая передача
H_lat(f) = H_lat^QS · D(f) с одномодовым усилением |D(f1)| = Q. Короче L —
выше f1 и шире плоская полоса, но ниже квазистатическая податливость
(чувствительность) — ключевой проектный компромисс, показываемый кривыми f1(L)
и |H_lat(f)|.</p>
<p><b>Связь отражателя (док 03).</b> η(dx) — гауссово перекрытие возвращённого
пучка и моды волокна. Статическая расцентровка Δx0 задаёт рабочую точку η0 на
склоне; cylinder/sphere кривые (конечный R_c), plane плоский (нет связи по
смещению), wedge добавляет угловое смещение.</p>
<p><b>Опорное плечо детектора (док 07 §1.2).</b> «matched» балансирует яркое и
опорное плечи (синфазное подавление RIN ограничено CMRR); «bright» оставляет
опорное плечо тёмным (нет подавления RIN, выше дробовой пол). Это открытый
вопрос O-SW-08.</p>
<p><b>Обратная цепочка / DSP (док 05 / 11).</b> Стандартная инверсия де-вращает
D(f), применяет калиброванную оптическую чувствительность и интегрирует в v и x.
Модель чувствительности — «static» (наклон плато), «operating_point» (локальный
наклон в η0) или «nonlinear_curve» (полная инверсия η(dx)) — балансирует
смещение против робастности. Интегратор работает в форме «frequency» (ω-область)
или «time».</p>
<p><b>NEA(f) (док 07 / 08).</b> Плотность шумо-эквивалентного ускорения с плато
дробового / RIN / джонсоновского — измеренный бюджет; нажмите
<i>Вычислить NEA(f)</i>, чтобы прогнать через воркер (прогон «Отчёт»). Развёртка
<b>семейства</b> датчиков — на вкладке <i>Развёртки</i>.</p>
"""


CATALOG: dict[str, dict[str, str]] = {
    # ------------------------------------------------------------------ app
    "menu.export_dir": _e("Export to directory", "Экспорт в каталог"),
    "help.tooltip": _e("What is this parameter?", "Что это за параметр?"),
    "app.title": _e(
        "OptiVibe - fiber-optic vibration sensor digital twin",
        "OptiVibe — цифровой двойник волоконно-оптического датчика вибрации",
    ),
    # ------------------------------------------------------------------ outer tabs
    "tab.live": _e("Live", "Онлайн"),
    "tab.report": _e("Report", "Отчёт"),
    "tab.sweeps": _e("Sweeps", "Развёртки"),
    "tab.montecarlo": _e("Monte-Carlo", "Монте-Карло"),
    "tab.physics": _e("Physics", "Физмодели"),
    # ------------------------------------------------------------------ actions
    "action.run": _e("Run", "Запуск"),
    "action.report": _e("Report", "Отчёт"),
    "action.cancel": _e("Cancel", "Отмена"),
    "action.export": _e("Export...", "Экспорт..."),
    "action.check": _e("Check composition", "Проверить композицию"),
    # ------------------------------------------------------------------ status
    "status.ready": _e(
        "Ready. Pick a variant and excitation, then Run.",
        "Готово. Выберите композицию и возбуждение, затем «Запуск».",
    ),
    "status.running": _e("Running {label} ...", "Выполняется {label} ..."),
    "status.could_not_start": _e("Could not start: {exc}", "Не удалось запустить: {exc}"),
    "status.progress": _e("... {message}", "... {message}"),
    "status.invalid_sweep": _e("Invalid sweep: {exc}", "Некорректная развёртка: {exc}"),
    "status.invalid_mc": _e("Invalid Monte-Carlo: {exc}", "Некорректный Монте-Карло: {exc}"),
    "status.invalid_scenario": _e("Invalid scenario: {exc}", "Некорректный сценарий: {exc}"),
    "status.invalid_composition": _e(
        "Invalid composition: {exc}", "Некорректная композиция: {exc}"
    ),
    "status.failed": _e("Failed: {message}", "Ошибка: {message}"),
    "status.cancelled": _e("Cancelled.", "Отменено."),
    "status.unrecognised": _e(
        "Finished with an unrecognised result.",
        "Завершено с нераспознанным результатом.",
    ),
    "status.nothing_export": _e("Nothing to export yet.", "Пока нечего экспортировать."),
    "status.exported": _e(
        "Exported {n} file(s) to {directory}.",
        "Экспортировано файлов: {n} в {directory}.",
    ),
    "status.report_ready": _e(
        "Report ready: amplitude ratio {ratio:.4f}, recovery rel err {rel:.2e}.",
        "Отчёт готов: отношение амплитуд {ratio:.4f}, отн. ошибка восстановления {rel:.2e}.",
    ),
    "status.sweep_done": _e(
        "Sweep '{name}' ({mode}) over {parameter}: {n} points.",
        "Развёртка «{name}» ({mode}) по {parameter}: точек {n}.",
    ),
    "status.mc_done": _e(
        "Monte-Carlo '{name}': {n} draws.",
        "Монте-Карло «{name}»: реализаций {n}.",
    ),
    "status.run_done": _e(
        "Done: variant {name}, {n} samples, dominant {dominant} Hz.",
        "Готово: композиция {name}, отсчётов {n}, доминанта {dominant} Гц.",
    ),
    # ------------------------------------------------------------------ menu
    "menu.file": _e("&File", "&Файл"),
    "menu.run": _e("&Run", "&Прогон"),
    "menu.view": _e("&View", "&Вид"),
    "menu.help": _e("&Help", "&Справка"),
    "menu.open_composition": _e("Open composition...", "Открыть композицию..."),
    "menu.save_composition": _e("Save composition as...", "Сохранить композицию как..."),
    "menu.export": _e("Export results...", "Экспорт результатов..."),
    "menu.quit": _e("Quit", "Выход"),
    "menu.run_action": _e("Run", "Запуск"),
    "menu.report_action": _e("Report", "Отчёт"),
    "menu.cancel_action": _e("Cancel", "Отмена"),
    "menu.check_action": _e("Check composition", "Проверить композицию"),
    "menu.preferences": _e("Preferences...", "Настройки..."),
    "menu.toggle_log": _e("Show log panel", "Показать панель журнала"),
    "menu.whats_this": _e("What's This? mode", "Режим «Что это?»"),
    "menu.manual": _e("Open manual", "Открыть руководство"),
    "manual.not_found": _e(
        "Documentation folder not found at:\n{path}",
        "Каталог документации не найден по пути:\n{path}",
    ),
    "menu.about": _e("About OptiVibe", "О программе OptiVibe"),
    "about.text": _e(
        "OptiVibe -- digital twin of a fiber-optic vibration sensor.\n"
        "Desktop shell over a Qt-free core; all physics lives in the core.",
        "OptiVibe — цифровой двойник волоконно-оптического датчика вибрации.\n"
        "Десктоп-оболочка над Qt-независимым ядром; вся физика — в ядре.",
    ),
    # ------------------------------------------------------------------ preferences
    "prefs.title": _e("Preferences", "Настройки"),
    "prefs.language": _e("Language", "Язык"),
    "prefs.theme": _e("Theme", "Тема"),
    "prefs.theme.light": _e("Light", "Светлая"),
    "prefs.theme.dark": _e("Dark", "Тёмная"),
    "prefs.restore_geometry": _e(
        "Restore window size on start", "Восстанавливать размер окна при старте"
    ),
    "prefs.ok": _e("OK", "ОК"),
    "prefs.cancel": _e("Cancel", "Отмена"),
    "prefs.note": _e(
        "Language and theme apply immediately; other options apply on restart.",
        "Язык и тема применяются сразу; прочие параметры — при перезапуске.",
    ),
    # ------------------------------------------------------------------ log dock
    "log.title": _e("Log", "Журнал"),
    "log.clear": _e("Clear", "Очистить"),
    # ------------------------------------------------------------------ check dialog
    "check.title": _e("Composition check", "Проверка композиции"),
    "check.ok": _e(
        "Composition resolves; all geometry/optics guards pass.",
        "Композиция разрешается; все гварды геометрии/оптики пройдены.",
    ),
    "check.fail": _e(
        "Composition rejected:\n{reason}",
        "Композиция отклонена:\n{reason}",
    ),
    "check.resolved": _e(
        "Resolved variant '{name}'.\nf1 = {f1} Hz, Q = {q}.",
        "Разрешена композиция «{name}».\nf1 = {f1} Гц, Q = {q}.",
    ),
    # ------------------------------------------------------------------ common form
    "form.preset": _e("preset", "пресет"),
    "preset.help": _e(
        "Named building block from configs/presets/<subsystem>/ (user presets under "
        "configs/user/ win over built-ins). Choosing a preset RESEEDS every field "
        "below from its values; anything you then type becomes an explicit override "
        "merged on top of the preset at resolve time. A blank field means 'no "
        "override: the preset (or a derivation) supplies the value'.",
        "Именованный строительный блок из configs/presets/<подсистема>/ "
        "(пользовательские пресеты в configs/user/ имеют приоритет над встроенными). "
        "Выбор пресета ЗАНОВО заполняет все поля ниже его значениями; всё, что вы "
        "затем впишете, становится явным override, накладываемым поверх пресета при "
        "разрешении. Пустое поле означает «без override: значение даёт пресет (или "
        "вывод)».",
    ),
    # ------------------------------------------------------------------ tabs (panel)
    "panel.tab.system": _e("System", "Система"),
    "panel.tab.source": _e("Source", "Источник"),
    "panel.tab.fiber": _e("Fiber line", "Волокно"),
    "panel.tab.cantilever": _e("Cantilever", "Консоль"),
    "panel.tab.reflector": _e("Reflector", "Отражатель"),
    "panel.tab.detector": _e("Detector", "Детектор"),
    "panel.tab.excitation": _e("Excitation", "Возбуждение"),
    "panel.tab.physics": _e("Physics layers", "Физмодели"),
    "panel.tab.repro": _e("Reproducibility", "Воспроизводимость"),
    # groups
    "group.source": _e("Source", "Источник"),
    "group.fiber": _e("Fiber line", "Волокно"),
    "group.cantilever": _e("Cantilever", "Консоль"),
    "group.reflector": _e("Reflector", "Отражатель"),
    "group.detector": _e("Detector", "Детектор"),
    "group.system": _e("System / composition", "Система / композиция"),
    "group.excitation": _e("Excitation", "Возбуждение"),
    "group.stages": _e("Physics layers (stage implementation)", "Физмодели (реализация стадий)"),
    "group.repro": _e("Reproducibility", "Воспроизводимость"),
    # ------------------------------------------------------------------ source fields
    "source.wavelength.label": _e("wavelength lambda", "длина волны λ"),
    "source.wavelength.tip": _e(
        "Centre wavelength (doc 03 §1; 1550 nm)",
        "Центральная длина волны (док 03 §1; 1550 нм)",
    ),
    "source.wavelength.help": _e(
        "Centre wavelength lambda of the source, metres (1.55e-6 = 1550 nm, the "
        "common platform).\n\nCouples to: the Gaussian beam geometry (Rayleigh "
        "range zR = pi w0^2 / lambda, so the spot size on the mirror w(A) and "
        "the coupling efficiency eta), the fringe period of the parasitic "
        "endface interferometer, and the dlam -> dnu conversion of the "
        "linewidth.\n\nWith a measured spectrum loaded, lambda must lie inside "
        "the table span (a cheap catch of the nm-vs-m unit slip, R-57b); the "
        "spectrum loader seeds it to the measured centroid.",
        "Центральная длина волны λ источника, метры (1.55e-6 = 1550 нм — "
        "базовая платформа).\n\nСвязь: геометрия гауссова пучка (рэлеевская длина "
        "zR = π w0² / λ, а значит размер пятна на зеркале w(A) и эффективность "
        "ввода η), период полос паразитного интерферометра торца и пересчёт "
        "Δλ → Δν ширины линии.\n\nПри загруженном измеренном спектре λ обязана "
        "лежать внутри диапазона таблицы (дешёвая ловушка путаницы нм-vs-м, "
        "R-57б); загрузчик спектра сеет λ по измеренному центроиду.",
    ),
    "source.power.label": _e("optical power P", "оптическая мощность P"),
    "source.power.tip": _e(
        "Power delivered to the fiber (doc 07 §2)",
        "Мощность, вводимая в волокно (док 07 §2)",
    ),
    "source.power.help": _e(
        "Optical power delivered into the fiber, watts (0.016 = 16 mW SLD "
        "default; 20-50 mW typical; 100 mW DFB).\n\nEffect: the photocurrent "
        "scales as I ~ P, so the shot-noise-limited NEA improves as 1/sqrt(P) "
        "while the RIN-limited floor is independent of P (doc 07 §1) -- raising "
        "P helps only until the RIN plateau takes over.",
        "Оптическая мощность, вводимая в волокно, ватты (0.016 = 16 мВт — "
        "умолчание SLD; типично 20–50 мВт; DFB 100 мВт).\n\nЭффект: фототок "
        "I ~ P, поэтому NEA в пределе дробового шума улучшается как 1/√P, а "
        "RIN-ограниченный пол от P не зависит (док 07 §1) — рост P помогает лишь "
        "до выхода на RIN-плато.",
    ),
    "source.rin.label": _e("RIN", "RIN"),
    "source.rin.tip": _e(
        "Relative intensity noise (doc 07 §1.2); blank for an SLD with a "
        "linewidth = derived ASE floor 2/dnu (M-01); an explicit value replaces "
        "the floor (anti-double-count R-57(v))",
        "Относительный шум интенсивности (док 07 §1.2); пусто для SLD с шириной "
        "линии = выведенный ASE-пол 2/Δν (M-01); явное значение замещает пол "
        "(анти-двойной-счёт R-57(в))",
    ),
    "source.rin.help": _e(
        "Relative intensity noise of the source, dB/Hz. Typical: SLD -120 to "
        "-126 (near its ASE beat floor), low-noise DFB -150 to -155.\n\nBLANK "
        "means 'derive it': for an SLD with a linewidth (or a measured "
        "spectrum) the ASE floor RIN = 2 tau_c = 2/dnu_eff is computed at "
        "resolve time (M-01/R-56). An explicit value REPLACES that floor -- it "
        "is never added to it (anti-double-count, R-57v); a DFB always needs an "
        "explicit value (the ASE relation does not apply to a coherent laser)."
        "\n\nEffect: sets the RIN plateau of the NEA budget; on a balanced "
        "detector it is suppressed by the CMRR.",
        "Относительный шум интенсивности источника, дБ/Гц. Типично: SLD от −120 "
        "до −126 (вблизи ASE-пола биений), малошумящий DFB от −150 до −155."
        "\n\nПУСТО означает «вывести»: для SLD с шириной линии (или измеренным "
        "спектром) ASE-пол RIN = 2 τ_c = 2/Δν_eff вычисляется при разрешении "
        "(M-01/R-56). Явное значение ЗАМЕЩАЕТ этот пол — оно к нему не "
        "прибавляется (анти-двойной-счёт, R-57в); для DFB всегда нужно явное "
        "значение (соотношение ASE к когерентному лазеру неприменимо).\n\n"
        "Эффект: задаёт RIN-плато бюджета NEA; на балансном детекторе подавляется "
        "по CMRR.",
    ),
    "source.linewidth.label": _e("linewidth dlam FWHM", "ширина линии Δλ FWHM"),
    "source.linewidth.tip": _e(
        "Spectral FWHM (doc 03 §f'; M-01): drives the derived RIN and the "
        "route-2 wash-out check; forbidden next to a measured spectrum (R-57(a))",
        "Спектральная ширина FWHM (док 03 §f'; M-01): задаёт выведенный RIN и "
        "проверку смыва по маршруту 2; запрещена рядом с измеренным спектром "
        "(R-57(а))",
    ),
    "source.linewidth.help": _e(
        "Spectral width dlam (FWHM), metres (6e-8 = 60 nm, the route-2 design "
        "anchor at 1550 nm).\n\nCouples to: the coherence length L_c ~ "
        "lambda^2/dlam (60 nm -> L_c ~ 17.7 um) and therefore the ROUTE-2 "
        "WASH-OUT of the parasitic endface fringe: resolve enforces V(A) = "
        "2^-(2A/L_c)^2 < 0.03 at the nominal gap (doc 03 §f'; R-13). Also "
        "yields the derived ASE RIN floor when the RIN field is blank (M-01)."
        "\n\nForbidden next to a measured spectrum -- the table is then the "
        "single source of truth (R-57a); the form clears it automatically.",
        "Спектральная ширина Δλ (FWHM), метры (6e-8 = 60 нм — проектный якорь "
        "маршрута 2 при 1550 нм).\n\nСвязь: длина когерентности L_c ~ λ²/Δλ "
        "(60 нм → L_c ~ 17.7 мкм) и, следовательно, СМЫВ ПО МАРШРУТУ 2 паразитной "
        "полосы торца: разрешение требует V(A) = 2^−(2A/L_c)² < 0.03 при "
        "номинальном зазоре (док 03 §f'; R-13). Также даёт выведенный ASE-пол "
        "RIN, когда поле RIN пусто (M-01).\n\nЗапрещена рядом с измеренным "
        "спектром — тогда единственный источник истины таблица (R-57а); форма "
        "очищает поле автоматически.",
    ),
    # source lineshape + loaders
    "source.lineshape.label": _e("lineshape", "форма линии"),
    "source.lineshape.tip": _e(
        "Source spectrum shape (M-10): default keeps the R-46 behaviour "
        "(Gaussian visibility, rectangular RIN floor); measured needs a "
        "loaded spectrum artifact (M-15/S-13)",
        "Форма спектра источника (M-10): default сохраняет поведение R-46 "
        "(гауссова видность, прямоугольный RIN-пол); measured требует "
        "загруженного артефакта спектра (M-15/S-13)",
    ),
    "source.lineshape.help": _e(
        "Spectral shape model of the source (M-10):\n\n"
        "(default) -- the R-46 effective-scalar behaviour: Gaussian fringe "
        "visibility and a rectangular noise-equivalent band from the linewidth.\n"
        "gaussian / lorentzian -- analytic shapes; both quadratures (fringe "
        "visibility V(A) and the ASE floor 2 tau_c) follow the chosen shape; "
        "requires the linewidth field.\n"
        "measured -- the loaded OSA table is the single source of truth: V(A), "
        "tau_c and the RIN floor are computed from it and the scalar linewidth is "
        "cleared (R-57a). Enabled only after 'Load measured spectrum...'.",
        "Модель формы спектра источника (M-10):\n\n"
        "(default) — эффективно-скалярное поведение R-46: гауссова видность полос "
        "и прямоугольная шумо-эквивалентная полоса из ширины линии.\n"
        "gaussian / lorentzian — аналитические формы; обе квадратуры (видность "
        "V(A) и ASE-пол 2 τ_c) следуют выбранной форме; требуется поле ширины "
        "линии.\n"
        "measured — загруженная таблица OSA есть единственный источник истины: "
        "V(A), τ_c и RIN-пол вычисляются из неё, а скалярная ширина линии "
        "очищается (R-57а). Доступно только после «Загрузить измеренный спектр...».",
    ),
    "source.spectrum.row": _e("spectrum", "спектр"),
    "source.spectrum.button": _e("Load measured spectrum...", "Загрузить измеренный спектр..."),
    "source.spectrum.button.tip": _e(
        "Load a characterization artifact (sidecar YAML or its CSV; doc 16 "
        "§2a) and enable lineshape = measured",
        "Загрузить характеризационный артефакт (сайдкар YAML или его CSV; док 16 "
        "§2a) и включить форму линии = measured",
    ),
    "source.spectrum.help": _e(
        "S-13 entry point for the M-15 artifact: pick the sidecar YAML or the CSV "
        "of an OSA trace (lambda, S). The trace is reduced through the standard "
        "optics quadratures (centroid, FWHM, dnu_eff, ASE floor), the wavelength "
        "field is seeded to the measured centroid, lineshape switches to "
        "'measured' and the table travels in the composition overrides.\n\nThis "
        "seeds the FORM; the full measured-twin path with the provenance artifact "
        "is 'optivibe ingest' (doc 16 §2a).",
        "Точка входа S-13 для артефакта M-15: выберите сайдкар YAML или CSV "
        "трассы OSA (λ, S). Трасса сводится штатными оптическими квадратурами "
        "(центроид, FWHM, Δν_eff, ASE-пол), поле длины волны сеется по "
        "измеренному центроиду, форма линии переключается в «measured», а таблица "
        "едет в overrides композиции.\n\nЭто сеет ФОРМУ; полный путь измеренного "
        "двойника с артефактом провенанса — «optivibe ingest» (док 16 §2a).",
    ),
    "source.spectrum.none": _e("no measured spectrum loaded", "измеренный спектр не загружен"),
    "source.spectrum.from_overrides": _e(
        "from the composition overrides", "из overrides композиции"
    ),
    "source.rin_trace.row": _e("RIN trace", "трасса RIN"),
    "source.rin_trace.button": _e("Load RIN trace...", "Загрузить трассу RIN..."),
    "source.rin_trace.button.tip": _e(
        "Load a floor-corrected RIN(f) artifact (M-16); the band median "
        "seeds the RIN field as an explicit value (replaces the derived "
        "floor, R-57(v))",
        "Загрузить артефакт RIN(f) с вычтенным полом (M-16); медиана по полосе "
        "сеет поле RIN явным значением (замещает выведенный пол, R-57(в))",
    ),
    "source.rin_trace.help": _e(
        "S-13 entry point for the M-16 artifact: a floor-corrected RIN(f) trace "
        "from the PD+TIA bench (shot/dark/Johnson already subtracted). The median "
        "over the band declared in the sidecar seeds the RIN field as an explicit "
        "value -- which REPLACES the derived ASE floor (anti-double-count, R-57v).",
        "Точка входа S-13 для артефакта M-16: трасса RIN(f) с вычтенным полом со "
        "стенда PD+TIA (дробовой/тёмновой/джонсоновский уже вычтены). Медиана по "
        "полосе, объявленной в сайдкаре, сеет поле RIN явным значением — которое "
        "ЗАМЕЩАЕТ выведенный ASE-пол (анти-двойной-счёт, R-57в).",
    ),
    # note: mixes text and measured values (format placeholders).
    "source.rin_trace.note": _e(
        "measured RIN {value:.2f} dB/Hz (u = {u:.2g}; {file}, {instrument}) -- "
        "replaces the derived floor (R-57v)",
        "измеренный RIN {value:.2f} дБ/Гц (u = {u:.2g}; {file}, {instrument}) — "
        "замещает выведенный пол (R-57в)",
    ),
    "source.spectrum.note": _e(
        "{file} ({instrument}, {timestamp}; sha {sha})",
        "{file} ({instrument}, {timestamp}; sha {sha})",
    ),
    # ------------------------------------------------------------------ fiber
    "fiber.w0.label": _e("mode-field radius w0", "радиус модового поля w0"),
    "fiber.w0.tip": _e("Gaussian mode radius (doc 03 §1)", "Гауссов модовый радиус (док 03 §1)"),
    "fiber.w0.help": _e(
        "Gaussian mode-field radius w0 of the guided mode, metres (SMF-28 at "
        "1550 nm: ~5.2e-6).\n\nCouples to: the beam divergence (zR = pi w0^2 / "
        "lambda), the spot size on the mirror w(A) and thus the coupling "
        "efficiency eta and the displacement sensitivity; the composition-time "
        "geometry guards use it (w(A) <= R_c/3, R_c >= 5 w0, doc 03 §6).",
        "Гауссов радиус модового поля w0 направляемой моды, метры (SMF-28 при "
        "1550 нм: ~5.2e-6).\n\nСвязь: расходимость пучка (zR = π w0² / λ), размер "
        "пятна на зеркале w(A) и, значит, эффективность ввода η и чувствительность "
        "к смещению; геометрические гварды при сборке используют его "
        "(w(A) ≤ R_c/3, R_c ≥ 5 w0, док 03 §6).",
    ),
    "fiber.R1.label": _e("endface reflectivity R1", "отражение торца R1"),
    "fiber.R1.tip": _e("Fresnel reflectivity (doc 04 §4)", "Френелевское отражение (док 04 §4)"),
    "fiber.R1.help": _e(
        "Power reflectivity of the fiber endface (bare glass ~0.035; "
        "AR-coated ~1e-4).\n\nThis is the parasitic arm of the endface "
        "interferometer: route 2 washes its fringe out with a broadband source "
        "(V(A) < 0.03), route 1 suppresses it with an AR coating + DFB "
        "(doc 08). Raising R1 raises the DC pedestal and, if the wash-out "
        "fails, an interferometric error term.",
        "Отражение по мощности торца волокна (голое стекло ~0.035; с "
        "просветлением ~1e-4).\n\nЭто паразитное плечо интерферометра торца: "
        "маршрут 2 смывает его полосу широкополосным источником (V(A) < 0.03), "
        "маршрут 1 подавляет её просветлением + DFB (док 08). Рост R1 поднимает "
        "постоянный пьедестал и, если смыв не удался, интерферометрический член "
        "ошибки.",
    ),
    "fiber.clad.label": _e("cladding diameter D", "диаметр оболочки D"),
    "fiber.clad.tip": _e(
        "Outer diameter (doc 01 §4.1; informational)",
        "Внешний диаметр (док 01 §4.1; информационно)",
    ),
    "fiber.clad.help": _e(
        "Outer cladding diameter, metres (1.25e-4 = 125 um standard).\n\n"
        "Informational at composition level: the mechanics reads the fiber "
        "cross-section from the physical constants (doc 01), so this field "
        "documents the part but does not steer the model.",
        "Внешний диаметр оболочки, метры (1.25e-4 = 125 мкм стандарт).\n\n"
        "Информационно на уровне композиции: механика берёт сечение волокна из "
        "физических констант (док 01), поэтому поле документирует деталь, но не "
        "управляет моделью.",
    ),
    # ------------------------------------------------------------------ reflector
    "reflector.rho.label": _e("reflectivity rho", "отражение ρ"),
    "reflector.rho.tip": _e(
        "Mirror reflectivity (doc 08 §6; 0.98)", "Отражение зеркала (док 08 §6; 0.98)"
    ),
    "reflector.rho.help": _e(
        "Power reflectivity rho of the mirror (0.98 metallized; 0.035 bare "
        "arc-melted glass, the POC prototype R-2).\n\nEffect: scales the "
        "returned optical power and therefore the signal current; the "
        "shot-limited NEA improves as 1/sqrt(rho) while the RIN-limited floor "
        "is unaffected (relative noise).",
        "Отражение по мощности ρ зеркала (0.98 металлизация; 0.035 голое "
        "оплавленное стекло — прототип POC R-2).\n\nЭффект: масштабирует "
        "возвращённую оптическую мощность и, значит, сигнальный ток; NEA в "
        "пределе дробового шума улучшается как 1/√ρ, а RIN-ограниченный пол не "
        "меняется (относительный шум).",
    ),
    "reflector.gap.label": _e("air gap A", "воздушный зазор A"),
    "reflector.gap.tip": _e(
        "Nominal one-way gap (doc 03 §6; 20-40 um)",
        "Номинальный односторонний зазор (док 03 §6; 20–40 мкм)",
    ),
    "reflector.gap.help": _e(
        "Nominal one-way air gap A between the fiber endface and the mirror, "
        "metres (design band 20-40 um; POC placeholder 31 um).\n\nCouples to: "
        "the spot size on the mirror w(A) (larger gap -> larger spot -> lower "
        "coupling eta), the geometry guard w(A) <= R_c/3, and the ROUTE-2 "
        "wash-out criterion (the endface fringe must satisfy V(A) < 0.03 at "
        "this gap; broadening the source or enlarging A helps, doc 03 §f').",
        "Номинальный односторонний воздушный зазор A между торцом волокна и "
        "зеркалом, метры (проектная полоса 20–40 мкм; заглушка POC 31 мкм).\n\n"
        "Связь: размер пятна на зеркале w(A) (больше зазор → больше пятно → ниже "
        "η), геометрический гвард w(A) ≤ R_c/3 и критерий СМЫВА ПО МАРШРУТУ 2 "
        "(полоса торца обязана давать V(A) < 0.03 при этом зазоре; помогает "
        "уширение источника или увеличение A, док 03 §f').",
    ),
    "reflector.bias.label": _e("bias Delta x0", "смещение Δx0"),
    "reflector.bias.tip": _e(
        "Working-point de-centering (doc 03 §5)",
        "Рабочая расцентровка (док 03 §5)",
    ),
    "reflector.bias.help": _e(
        "Intentional static de-centering Delta x0 of the beam on the mirror, "
        "metres. Sets the working point on the eta(x) curve:\n\n0 -- at the "
        "peak: the linear (1f) response vanishes and the displacement response "
        "is quadratic (2f) -- the POC prototype regime (sleeve centering, "
        "R-4).\nnon-zero -- on the slope: a linear 1f response with the "
        "signed sensitivity s_target; typical bias is a fraction of the spot "
        "size.\n\nIgnored by the flat plane/wedge (no displacement coupling).",
        "Намеренная статическая расцентровка Δx0 пучка на зеркале, метры. Задаёт "
        "рабочую точку на кривой η(x):\n\n0 — на вершине: линейный (1f) отклик "
        "исчезает, отклик по смещению квадратичен (2f) — режим прототипа POC "
        "(центровка втулкой, R-4).\nненулевое — на склоне: линейный отклик 1f со "
        "знаковой чувствительностью s_target; типичное смещение — доля размера "
        "пятна.\n\nИгнорируется плоскостью/клином (нет связи по смещению).",
    ),
    "reflector.shape.label": _e("shape", "форма"),
    "reflector.shape.tip": _e(
        "Reflector profile (S9-B); shapes gate their parameters",
        "Профиль отражателя (S9-B); формы включают свои параметры",
    ),
    "reflector.shape.help": _e(
        "Reflector profile; each shape has a registered optics model (S9-B):\n\n"
        "cylinder -- curved in one axis: the version-1 TARGET-AXIS selector (the "
        "cylinder axis defines the measured axis; cross-axis response is a "
        "metric, doc 00).\n"
        "sphere -- isotropic curvature (the arc-melted POC tip): responds to the "
        "radial displacement, no axis selectivity.\n"
        "wedge -- tilted flat face: an ANGULAR bias working point (alpha_w) "
        "instead of a displacement bias.\n"
        "plane -- flat reference (R_c -> infinity); no displacement coupling, "
        "used for gap-only sensitivity checks.\n\nSwitching the shape "
        "enables/disables the shape parameters below and clears the ones the new "
        "shape ignores.",
        "Профиль отражателя; у каждой формы своя зарегистрированная модель оптики "
        "(S9-B):\n\ncylinder — кривизна по одной оси: селектор ЦЕЛЕВОЙ ОСИ версии 1 "
        "(ось цилиндра задаёт измеряемую ось; перекрёстный отклик — метрика, "
        "док 00).\nsphere — изотропная кривизна (оплавленный кончик POC): "
        "отвечает на радиальное смещение, без селективности по осям.\nwedge — "
        "наклонная плоская грань: УГЛОВАЯ рабочая точка (α_w) вместо смещения.\n"
        "plane — плоский эталон (R_c → ∞); без связи по смещению, для проверок "
        "чувствительности только по зазору.\n\nСмена формы включает/выключает "
        "параметры ниже и очищает те, что новая форма игнорирует.",
    ),
    "reflector.rc.label": _e("curvature R_c [m]", "радиус кривизны R_c [м]"),
    "reflector.rc.tip": _e(
        "Radius of curvature R_c (cylinder/sphere; doc 08 §6)",
        "Радиус кривизны R_c (cylinder/sphere; док 08 §6)",
    ),
    "reflector.rc.help": _e(
        "Radius of curvature R_c of the convex mirror, metres (31-62 um presets; "
        "POC placeholder 62.5 um; used by cylinder and sphere only).\n\nCouples "
        "to: the displacement sensitivity of the coupling eta (smaller R_c -> "
        "sharper eta(x) -> higher sensitivity but tighter alignment), and the "
        "paraxial guards R_c >= 5 w0 and w(A) <= R_c/3 (doc 03 §6) -- violating "
        "them fails the composition loudly.\n\nThe 'Load tip profile...' button "
        "below seeds this field from a measured contour (M-17, one azimuth).",
        "Радиус кривизны R_c выпуклого зеркала, метры (пресеты 31–62 мкм; "
        "заглушка POC 62.5 мкм; используется только cylinder и sphere).\n\nСвязь: "
        "чувствительность η к смещению (меньше R_c → круче η(x) → выше "
        "чувствительность, но жёстче юстировка) и параксиальные гварды R_c ≥ 5 w0 "
        "и w(A) ≤ R_c/3 (док 03 §6) — их нарушение громко валит композицию.\n\n"
        "Кнопка «Загрузить профиль кончика...» ниже сеет это поле по измеренному "
        "контуру (M-17, один азимут).",
    ),
    "reflector.wedge.label": _e("wedge angle [rad]", "угол клина [рад]"),
    "reflector.wedge.tip": _e(
        "Wedge face-tilt angle alpha_w (wedge only; doc 03 §c)",
        "Угол наклона грани клина α_w (только wedge; док 03 §c)",
    ),
    "reflector.wedge.help": _e(
        "Built-in face-tilt angle alpha_w of the wedge, radians (preset 20 mrad; "
        "wedge shape only).\n\nSets an angular bias working point: the returned "
        "beam is deflected by 2 alpha_w, so tip TILT (theta) couples linearly "
        "into eta while pure displacement does not (doc 03 §c).",
        "Встроенный угол наклона грани α_w клина, радианы (пресет 20 мрад; только "
        "форма wedge).\n\nЗадаёт угловую рабочую точку: возвращённый пучок "
        "отклоняется на 2 α_w, поэтому НАКЛОН кончика (θ) линейно связывается с η, "
        "а чистое смещение — нет (док 03 §c).",
    ),
    "reflector.profile.row": _e("profile", "профиль"),
    "reflector.profile.button": _e(
        "Load tip profile (R_c)...", "Загрузить профиль кончика (R_c)..."
    ),
    "reflector.profile.button.tip": _e(
        "Load a tip-contour artifact (M-17); the circle fit seeds R_c",
        "Загрузить артефакт контура кончика (M-17); аппроксимация окружностью сеет R_c",
    ),
    "reflector.profile.help": _e(
        "S-13 entry point for the M-17 artifact: a tip-contour trace (x, z) from "
        "the microscope. A Kasa circle fit reduces it to R_c (one azimuth; "
        "astigmatism needs two azimuths and the toroidal optics M-03 -- backlog). "
        "The fit seeds the R_c field; non-circular or collinear contours are "
        "rejected loudly (doc 20 F0-3).",
        "Точка входа S-13 для артефакта M-17: трасса контура кончика (x, z) с "
        "микроскопа. Аппроксимация окружностью по Косе сводит её к R_c (один "
        "азимут; астигматизм требует двух азимутов и тороидальной оптики M-03 — "
        "бэклог). Аппроксимация сеет поле R_c; неокружные или коллинеарные "
        "контуры отклоняются громко (док 20 F0-3).",
    ),
    "reflector.profile.note": _e(
        "measured R_c {rc:.2f} um (u = {u:.2g} um; {file}, {instrument}; one azimuth "
        "-- astigmatism = backlog M-03)",
        "измеренный R_c {rc:.2f} мкм (u = {u:.2g} мкм; {file}, {instrument}; один "
        "азимут — астигматизм = бэклог M-03)",
    ),
    # ------------------------------------------------------------------ cantilever
    "cantilever.length.label": _e("length L", "длина L"),
    "cantilever.length.tip": _e(
        "Free length; sets f1 ~ 1/L^2 (doc 02)",
        "Свободная длина; задаёт f1 ~ 1/L² (док 02)",
    ),
    "cantilever.length.help": _e(
        "Free cantilever length L from the ferrule exit to the tip, metres "
        "(2-10 mm typical; POC placeholder 4 mm).\n\nThe single strongest "
        "geometric knob: the first eigenfrequency scales as f1 ~ 1/L^2 (doc "
        "02), the tip compliance and thus the mechanical sensitivity grow with "
        "L, and the damping model Q(L) (air + anchor + internal, M-02) follows "
        "it -- the computed Q shown on the System tab updates as you edit L. "
        "Longer L: more sensitivity, lower f1 (narrower usable band), lower "
        "NEA at low f.",
        "Свободная длина консоли L от выхода феррулы до кончика, метры (типично "
        "2–10 мм; заглушка POC 4 мм).\n\nСамый сильный геометрический рычаг: "
        "первая собственная частота f1 ~ 1/L² (док 02), податливость кончика и "
        "значит механическая чувствительность растут с L, а модель затухания "
        "Q(L) (воздух + заделка + внутренние, M-02) следует за ней — вычисленный "
        "Q на вкладке «Система» обновляется при правке L. Длиннее L: выше "
        "чувствительность, ниже f1 (уже рабочая полоса), ниже NEA на низких f.",
    ),
    # ------------------------------------------------------------------ detector
    "detector.resp.label": _e("responsivity R", "чувствительность R"),
    "detector.resp.tip": _e(
        "Photodiode responsivity (doc 07 §2)", "Чувствительность фотодиода (док 07 §2)"
    ),
    "detector.resp.help": _e(
        "Photodiode responsivity R, A/W (~1.0 for InGaAs at 1550 nm).\n\n"
        "Effect: converts optical power to photocurrent; scales the signal and "
        "the shot noise together, so it mainly moves the balance against the "
        "electronics (Johnson/ADC) floors.",
        "Чувствительность фотодиода R, А/Вт (~1.0 для InGaAs при 1550 нм).\n\n"
        "Эффект: переводит оптическую мощность в фототок; масштабирует сигнал и "
        "дробовой шум вместе, поэтому в основном сдвигает баланс против полов "
        "электроники (джонсоновский/АЦП).",
    ),
    "detector.cmrr.label": _e("CMRR", "CMRR"),
    "detector.cmrr.tip": _e(
        "Balanced-channel rejection (doc 07 §1.2)",
        "Подавление балансного канала (док 07 §1.2)",
    ),
    "detector.cmrr.help": _e(
        "Common-mode rejection ratio of the balanced pair, dB (typ. 30-50 dB)."
        "\n\nEffect: on a balanced detector the source RIN is common-mode and "
        "is suppressed by the CMRR before it reaches the NEA budget; on a "
        "single-ended detector this field is unused and the full RIN applies.",
        "Коэффициент подавления синфазного сигнала балансной пары, дБ (типично "
        "30–50 дБ).\n\nЭффект: на балансном детекторе RIN источника синфазен и "
        "подавляется по CMRR до попадания в бюджет NEA; на несимметричном "
        "детекторе поле не используется и действует полный RIN.",
    ),
    "detector.adcfs.label": _e("ADC full scale", "полная шкала АЦП"),
    "detector.adcfs.tip": _e(
        "AC +/- range in output units (doc 07 §1.4)",
        "AC ± диапазон в единицах выхода (док 07 §1.4)",
    ),
    "detector.adcfs.help": _e(
        "Full-scale +/- range of the ADC in the detector output units (volts "
        "after the transimpedance).\n\nCouples to: the quantization floor "
        "(together with the ADC bits) and clipping -- the full-scale "
        "acceleration must map inside this range; too generous a range wastes "
        "bits, too tight a range clips at high g.",
        "Полный ± диапазон АЦП в единицах выхода детектора (вольты после "
        "трансимпеданса).\n\nСвязь: пол квантования (вместе с разрядностью АЦП) и "
        "клиппинг — полношкальное ускорение обязано укладываться в этот "
        "диапазон; слишком широкий диапазон тратит биты, слишком узкий — "
        "клиппует на больших g.",
    ),
    "detector.balanced.label": _e("balanced", "балансный"),
    "detector.balanced.checkbox": _e("balanced channel", "балансный канал"),
    "detector.balanced.help": _e(
        "Balanced photodiode pair vs a single-ended detector. Balanced: "
        "the source RIN is common-mode and suppressed by the CMRR; "
        "single-ended (the POC prototype, R-3): the full RIN reaches "
        "the budget and the CMRR field is unused.",
        "Балансная пара фотодиодов против несимметричного детектора. Балансный: "
        "RIN источника синфазен и подавляется по CMRR; несимметричный (прототип "
        "POC, R-3): полный RIN попадает в бюджет, поле CMRR не используется.",
    ),
    "detector.refarm.label": _e("reference arm", "опорное плечо"),
    "detector.refarm.help": _e(
        "Shot-noise convention of the balanced pair (O-SW-08): "
        "'matched' -- the reference arm carries the same mean power as "
        "the signal arm (shot PSD doubles); 'bright' -- a bright "
        "reference dominates the shot floor. Affects only the noise "
        "bookkeeping, not the signal.",
        "Соглашение о дробовом шуме балансной пары (O-SW-08): 'matched' — опорное "
        "плечо несёт ту же среднюю мощность, что сигнальное (дробовой PSD "
        "удваивается); 'bright' — яркое опорное доминирует над дробовым полом. "
        "Влияет только на учёт шума, не на сигнал.",
    ),
    "detector.adcbits.label": _e("ADC bits", "разрядность АЦП"),
    "detector.adcbits.help": _e(
        "ADC resolution, bits (24 typical). Together with the ADC full "
        "scale it sets the quantization noise floor; the budget checks "
        "it stays below the analog floors (doc 07 §1.4).",
        "Разрядность АЦП, биты (типично 24). Вместе с полной шкалой АЦП задаёт "
        "пол шума квантования; бюджет проверяет, что он ниже аналоговых полов "
        "(док 07 §1.4).",
    ),
    # ------------------------------------------------------------------ system scalars
    "system.starting.label": _e("starting composition", "стартовая композиция"),
    "system.starting.help": _e(
        "Which built-in composition seeds every tab: A (compact wideband, "
        "vacuum-optional), B (general-purpose wideband), C (long-throw "
        "sensitivity), D (resonant narrow-line, route 1). Switching RESEEDS all "
        "tabs from that variant -- unsaved edits are replaced. The letter also "
        "names the frozen scenario variant the run is labelled with; your edits "
        "travel separately as the composition payload.",
        "Какая встроенная композиция сеет все вкладки: A (компактная "
        "широкополосная, опц. вакуум), B (универсальная широкополосная), C "
        "(длинноходовая чувствительность), D (резонансная узкополосная, маршрут "
        "1). Переключение ЗАНОВО заполняет все вкладки из этого варианта — "
        "несохранённые правки заменяются. Буква также именует замороженный вариант "
        "сценария, которым помечается прогон; ваши правки едут отдельно как payload "
        "композиции.",
    ),
    "system.name.label": _e("name", "имя"),
    "system.name.help": _e(
        "Composition identity used for saved files and run labels. Free text for "
        "user compositions; A-D are reserved for the built-ins.",
        "Идентичность композиции для сохраняемых файлов и меток прогонов. "
        "Свободный текст для пользовательских композиций; A–D зарезервированы за "
        "встроенными.",
    ),
    "system.description.label": _e("description", "описание"),
    "system.description.help": _e(
        "Free-text description shown in reports; no effect on the model.",
        "Свободное описание, показываемое в отчётах; на модель не влияет.",
    ),
    "system.mode.label": _e("mode", "режим"),
    "system.mode.help": _e(
        "Operating regime (doc 08 §6): offresonance -- wideband use well below "
        "f1, flat mechanical response; resonance -- narrowband use on the "
        "resonant line (requires the line frequency below and typically variant "
        "D: route 1, high Q). Affects which DSP calibration applies.",
        "Режим работы (док 08 §6): offresonance — широкополосно значительно ниже "
        "f1, плоский механический отклик; resonance — узкополосно на резонансной "
        "линии (требует частоту линии ниже и, как правило, вариант D: маршрут 1, "
        "высокий Q). Определяет применяемую калибровку DSP.",
    ),
    "system.line_freq.label": _e("line freq [Hz]", "частота линии [Гц]"),
    "system.line_freq.help": _e(
        "Resonant line frequency, Hz -- only read in resonance mode; must sit "
        "near the composition's f1 for the resonant gain to be real.",
        "Резонансная частота линии, Гц — читается только в режиме resonance; "
        "должна быть вблизи f1 композиции, чтобы резонансный выигрыш был реален.",
    ),
    "system.band.label": _e("band [Hz]", "полоса [Гц]"),
    "system.band.help": _e(
        "Assessment band [f_min, f_max], Hz, used by the NEA budget, the DSP "
        "band-limits and the reports. The project spec band is 0.1 Hz - 20 kHz "
        "(doc 00, fixed); a composition may declare a narrower working band. "
        "f_max should stay well below f1 for off-resonance operation.",
        "Полоса оценки [f_min, f_max], Гц, используется бюджетом NEA, полосовыми "
        "ограничениями DSP и отчётами. Спецификационная полоса проекта — 0.1 Гц – "
        "20 кГц (док 00, зафиксирована); композиция может объявить более узкую "
        "рабочую полосу. f_max должна оставаться значительно ниже f1 для "
        "внерезонансной работы.",
    ),
    "system.full_scale.label": _e("full scale [g]", "полная шкала [g]"),
    "system.full_scale.help": _e(
        "Full-scale acceleration FS, g (spec: 50 g at any band frequency, doc 00 "
        "/ 08 §1.3). Sets the ADC mapping and the clipping checks; behaviour "
        "above FS (margin, nonlinearity) is a study topic, not guaranteed range.",
        "Полношкальное ускорение FS, g (спец: 50 g на любой частоте полосы, док 00 "
        "/ 08 §1.3). Задаёт отображение АЦП и проверки клиппинга; поведение выше "
        "FS (запас, нелинейность) — предмет исследования, не гарантированный "
        "диапазон.",
    ),
    "system.route.label": _e("route", "маршрут"),
    "system.route.help": _e(
        "Endface-treatment route (doc 08): 2 -- coherent wash-out: a broadband "
        "source (SLD) makes the parasitic endface fringe invisible (V(A) < 0.03 "
        "enforced at resolve when the linewidth or a measured spectrum is "
        "known); 1 -- AR-coated endface + narrow-line DFB (variant D). The route "
        "decides which source/noise inputs are consistent.",
        "Маршрут обработки торца (док 08): 2 — когерентный смыв: широкополосный "
        "источник (SLD) делает паразитную полосу торца невидимой (V(A) < 0.03 "
        "требуется при разрешении, когда известна ширина линии или измеренный "
        "спектр); 1 — просветлённый торец + узколинейный DFB (вариант D). Маршрут "
        "решает, какие входы источника/шума согласованы.",
    ),
    "system.eta_bias.label": _e("eta_bias (stub)", "eta_bias (заглушка)"),
    "system.eta_bias.help": _e(
        "Optical working-point efficiency eta0 used by the STUB optics only "
        "(S0 path); the physical reflector optics computes its own eta0 from the "
        "geometry (doc 03 §5). Ignored when the Optics stage is 'physical'.",
        "Эффективность рабочей точки η0, используемая только ЗАГЛУШКОЙ оптики "
        "(путь S0); физическая оптика отражателя вычисляет собственный η0 из "
        "геометрии (док 03 §5). Игнорируется, когда стадия оптики «physical».",
    ),
    "system.q.label": _e(
        "Q total override (blank = Q(L) model)", "Q override (пусто = модель Q(L))"
    ),
    "system.q.help": _e(
        "Total mechanical quality factor Q of mode 1. Since M-02 this is a "
        "COMPUTED quantity: leave the field BLANK to use the Q(L) damping model "
        "(air + anchor + internal losses at the current cantilever length; "
        "vacuum removes the air channel) shown below. A typed value is an "
        "explicit OVERRIDE -- e.g. a measured ring-down Q (M-18), which wins over "
        "the model by design. Q sets the resonance peak height, the ring-down "
        "time and the Brownian thermal NEA floor.",
        "Полная механическая добротность Q моды 1. С M-02 это ВЫЧИСЛЯЕМАЯ "
        "величина: оставьте поле ПУСТЫМ, чтобы использовать модель затухания Q(L) "
        "(воздух + заделка + внутренние потери при текущей длине консоли; вакуум "
        "убирает воздушный канал), показанную ниже. Введённое значение — явный "
        "OVERRIDE — например, измеренный Q из ring-down (M-18), который по замыслу "
        "побеждает модель. Q задаёт высоту резонансного пика, время ring-down и "
        "тепловой броуновский пол NEA.",
    ),
    "system.q_model.initial": _e("Q(L) model: -", "модель Q(L): -"),
    "system.q_model.na": _e(
        "Q(L) model: n/a (check the cantilever fields)",
        "модель Q(L): н/д (проверьте поля консоли)",
    ),
    "system.q_model.value": _e(
        "Q(L) model: {q:.6g} (used when the field above is blank)",
        "модель Q(L): {q:.6g} (используется, когда поле выше пусто)",
    ),
    "system.q_total.tip": _e(
        "Total quality factor of mode 1. Since M-02 this is a COMPUTED "
        "quantity (Q(L) damping model, R-47/R-48): leave blank to use the "
        "model value shown below; a typed value is an explicit override",
        "Полная добротность моды 1. С M-02 это ВЫЧИСЛЯЕМАЯ величина (модель "
        "затухания Q(L), R-47/R-48): оставьте пустым для модельного значения "
        "ниже; введённое значение — явный override",
    ),
    "system.q_model.tip": _e(
        "What a blank Q field resolves to: the Q(L) damping model at the "
        "current cantilever length and vacuum flag (M-02)",
        "Во что разрешается пустое поле Q: модель затухания Q(L) при текущей "
        "длине консоли и флаге вакуума (M-02)",
    ),
    "system.ringdown.row": _e("ring-down", "ring-down"),
    "system.ringdown.button": _e("Load ring-down (Q)...", "Загрузить ring-down (Q)..."),
    "system.ringdown.button.tip": _e(
        "Load a free-decay artifact (M-18); the log-decrement Q seeds the "
        "override field (measured Q wins over the Q(L) model)",
        "Загрузить артефакт свободного затухания (M-18); Q по логдекременту сеет "
        "поле override (измеренный Q побеждает модель Q(L))",
    ),
    "system.ringdown.help": _e(
        "S-13 entry point for the M-18 artifact: a free-decay record (t, y). The "
        "Hilbert-envelope log decrement yields Q = pi f1 / sigma; the fit seeds "
        "the Q override field (measured Q wins over the Q(L) model, M-02 "
        "semantics) and reports f1 for the cross-check against f1(L). An undamped "
        "tone is rejected loudly.",
        "Точка входа S-13 для артефакта M-18: запись свободного затухания (t, y). "
        "Логдекремент огибающей Гильберта даёт Q = π f1 / σ; аппроксимация сеет "
        "поле override Q (измеренный Q побеждает модель Q(L), семантика M-02) и "
        "сообщает f1 для сверки с f1(L). Незатухающий тон отклоняется громко.",
    ),
    "system.ringdown.note": _e(
        "measured Q {q:.1f} (u = {u:.2g}{f1}; {file}, {instrument}) -- overrides the Q(L) model",
        "измеренный Q {q:.1f} (u = {u:.2g}{f1}; {file}, {instrument}) — переопределяет модель Q(L)",
    ),
    "system.ringdown.f1": _e(", f1 = {f1:.1f} Hz", ", f1 = {f1:.1f} Гц"),
    "system.target_nea.label": _e("target NEA [ug/rtHz]", "целевой NEA [мкg/√Гц]"),
    "system.target_nea.help": _e(
        "Optional target noise-equivalent acceleration, ug/sqrt(Hz), drawn on "
        "the NEA plots as the design goal; no effect on the model.",
        "Опциональное целевое шумо-эквивалентное ускорение, мкg/√Гц, рисуется на "
        "графиках NEA как проектная цель; на модель не влияет.",
    ),
    "system.vacuum.label": _e("vacuum", "вакуум"),
    "system.vacuum.help": _e(
        "Operate the variant under vacuum: removes the air-damping channel from "
        "the Q(L) model (higher Q, taller resonance, lower thermal NEA) -- the "
        "A/D packaging option.",
        "Эксплуатировать вариант в вакууме: убирает канал воздушного затухания из "
        "модели Q(L) (выше Q, выше резонанс, ниже тепловой NEA) — опция корпуса "
        "A/D.",
    ),
    "system.save": _e("Save as...", "Сохранить как..."),
    "system.load": _e("Load...", "Загрузить..."),
    # ------------------------------------------------------------------ stages
    "stage.optics.label": _e("Optics", "Оптика"),
    "stage.optics.help": _e(
        "Optics stage implementation: 'physical (reflector)' -- the shape-"
        "dispatching Gaussian-coupling model (the shape itself is chosen on the "
        "Reflector tab); 'stub' -- a linear eta working-point toy (the eta_bias "
        "scalar on the System tab) for plumbing checks. Physical is the default "
        "for any real study.",
        "Реализация стадии оптики: «physical (reflector)» — модель гауссова ввода "
        "с диспетчеризацией по форме (сама форма выбирается на вкладке "
        "«Отражатель»); «stub» — линейная игрушка рабочей точки η (скаляр "
        "eta_bias на вкладке «Система») для проверок обвязки. Physical — умолчание "
        "для любого реального исследования.",
    ),
    "stage.mechanics.label": _e("Mechanics", "Механика"),
    "stage.mechanics.help": _e(
        "Mechanics stage implementation: 'modal' -- frequency-domain modal "
        "response of the cantilever (fast, the standard path); 'modal_time' -- "
        "time-domain integration of the same modal model (for shocks/transients); "
        "'stub' -- pass-through for plumbing checks.",
        "Реализация стадии механики: «modal» — частотный модальный отклик консоли "
        "(быстро, штатный путь); «modal_time» — временнáя интеграция той же "
        "модальной модели (для ударов/переходных); «stub» — сквозной проброс для "
        "проверок обвязки.",
    ),
    "stage.detector.label": _e("Detector", "Детектор"),
    "stage.detector.help": _e(
        "Detector stage implementation: 'photodiode' -- the physical photocurrent "
        "model with shot/RIN/Johnson/ADC noise (enables the NEA budget); 'stub' "
        "-- noiseless pass-through (NEA panels show 'not available').",
        "Реализация стадии детектора: «photodiode» — физическая модель фототока с "
        "дробовым/RIN/джонсоновским/АЦП шумом (включает бюджет NEA); «stub» — "
        "бесшумный проброс (панели NEA показывают «недоступно»).",
    ),
    "stage.dsp.label": _e("DSP", "DSP"),
    "stage.dsp.help": _e(
        "Inverse-chain (DSP) implementation: 'standard' -- the calibrated "
        "detector-current -> acceleration chain with spectra and metrics; 'stub' "
        "-- a scale-only shortcut for plumbing checks. The sensitivity and "
        "integrator selectors below apply to the standard DSP only.",
        "Реализация обратной цепочки (DSP): «standard» — калиброванная цепочка "
        "ток детектора → ускорение со спектрами и метриками; «stub» — только "
        "масштабирование для проверок обвязки. Селекторы чувствительности и "
        "интегратора ниже относятся только к standard DSP.",
    ),
    "stage.sensitivity.label": _e("Sensitivity", "Чувствительность"),
    "stage.sensitivity.help": _e(
        "How the standard DSP obtains the scalar sensitivity s_target: 'static' "
        "-- the design-point derivative; 'operating_point' -- re-evaluated at the "
        "resolved working point (bias, gap); 'nonlinear_curve' -- inverted "
        "through the full eta(x) curve (handles large drive amplitudes).",
        "Как standard DSP получает скалярную чувствительность s_target: «static» — "
        "производная в проектной точке; «operating_point» — пересчёт в "
        "разрешённой рабочей точке (смещение, зазор); «nonlinear_curve» — "
        "инверсия по полной кривой η(x) (для больших амплитуд возбуждения).",
    ),
    "stage.integrator.label": _e("Integrator", "Интегратор"),
    "stage.integrator.help": _e(
        "Acceleration -> velocity/displacement integration: 'frequency' -- "
        "division by (i omega) in the spectrum (fast, exact for stationary "
        "signals); 'time' -- time-domain integration with detrending (better for "
        "transients/shocks).",
        "Интегрирование ускорение → скорость/смещение: «frequency» — деление на "
        "(iω) в спектре (быстро, точно для стационарных сигналов); «time» — "
        "временнáя интеграция с детрендингом (лучше для переходных/ударов).",
    ),
    "repro.seed_enabled.label": _e("fixed seed", "фикс. seed"),
    "repro.seed_enabled.help": _e(
        "Fix the random seed of the noise and random-excitation generators. "
        "Checked: every Run with the same settings is bit-reproducible. "
        "Unchecked: each Run draws fresh noise (for eyeballing run-to-run "
        "spread).",
        "Зафиксировать seed генераторов шума и случайного возбуждения. Отмечено: "
        "каждый прогон при тех же настройках бит-воспроизводим. Снято: каждый "
        "прогон берёт свежий шум (чтобы оценить разброс между прогонами).",
    ),
    "repro.seed.label": _e("Seed", "Seed"),
    "repro.seed.help": _e(
        "The seed value used when 'fixed seed' is checked. Any integer; keep it "
        "constant to reproduce a run exactly, change it to draw a different "
        "noise realisation.",
        "Значение seed, используемое при отмеченном «фикс. seed». Любое целое; "
        "держите постоянным для точного воспроизведения прогона, меняйте для "
        "другой реализации шума.",
    ),
    # ------------------------------------------------------------------ excitation
    "exc.kind.label": _e("Kind", "Тип"),
    "exc.kind.help": _e(
        "What drives the sensor along the chosen axis:\n\n"
        "sine / multitone / sweep / random / shock -- GENERATED waveforms on the "
        "sampling grid below (fs, duration).\n"
        "csv / wav / tdms / uff / mat / hdf5 -- REPLAY of a recorded acceleration "
        "from a file (the grid comes from the file or the page fields; the "
        "sampling row is hidden).\n\nThe excitation is ground acceleration in g "
        "along one axis; pick the kind first, then fill its page.",
        "Что возбуждает датчик вдоль выбранной оси:\n\n"
        "sine / multitone / sweep / random / shock — ГЕНЕРИРУЕМЫЕ сигналы на сетке "
        "дискретизации ниже (fs, длительность).\n"
        "csv / wav / tdms / uff / mat / hdf5 — ВОСПРОИЗВЕДЕНИЕ записанного "
        "ускорения из файла (сетка берётся из файла или полей страницы; строка "
        "дискретизации скрыта).\n\nВозбуждение — ускорение основания в g вдоль "
        "одной оси; сначала выберите тип, затем заполните его страницу.",
    ),
    "exc.axis.label": _e("Axis", "Ось"),
    "exc.axis.help": _e(
        "Excitation axis in the sensor frame (doc 00): x -- the TARGET axis of "
        "the version-1 cylinder reflector (full response); y / z -- the cross "
        "axes, used to probe the cross-axis sensitivity metric. For the "
        "isotropic sphere the transverse axes are equivalent.",
        "Ось возбуждения в системе датчика (док 00): x — ЦЕЛЕВАЯ ось цилиндра "
        "версии 1 (полный отклик); y / z — перекрёстные оси, для метрики "
        "перекрёстной чувствительности. Для изотропной сферы поперечные оси "
        "эквивалентны.",
    ),
    "exc.sampling.label": _e("Sampling", "Дискретизация"),
    "exc.sampling.help": _e(
        "Grid of the generated waveform: fs -- sample rate, Hz (keep fs >= "
        "2.56 x the highest excited frequency for a clean spectrum); duration -- "
        "record length, s (sets the spectral resolution df = 1/T and how many "
        "periods the metrics average over). Hidden for file replay: the grid "
        "then comes from the file.",
        "Сетка генерируемого сигнала: fs — частота дискретизации, Гц (держите "
        "fs ≥ 2.56 × наибольшей возбуждаемой частоты для чистого спектра); "
        "длительность — длина записи, с (задаёт спектральное разрешение df = 1/T "
        "и число периодов усреднения метрик). Скрыта при воспроизведении файла: "
        "сетка тогда из файла.",
    ),
    "exc.about.label": _e("about", "о типе"),
    "exc.fs.label": _e("fs [Hz]", "fs [Гц]"),
    "exc.dur.label": _e("dur [s]", "длит. [с]"),
    # excitation per-kind summaries + descriptions
    "exc.sine.summary": _e("single tone", "один тон"),
    "exc.sine.about": _e(
        "One tone: frequency [Hz] and amplitude [g]. The basic probe of one "
        "band point -- dominant-frequency recovery, 2f/1f distortion at a "
        "bias~0 working point, RMS checks. Keep the frequency well below f1 "
        "for off-resonance use.",
        "Один тон: частота [Гц] и амплитуда [g]. Базовая проба одной точки полосы "
        "— восстановление доминанты, искажение 2f/1f в рабочей точке bias~0, "
        "проверки RMS. Держите частоту значительно ниже f1 для внерезонансной "
        "работы.",
    ),
    "exc.multitone.summary": _e("sum of tones", "сумма тонов"),
    "exc.multitone.about": _e(
        "A sum of components, each [frequency Hz, amplitude g] and an "
        "optional per-tone phase [rad]. Add/remove components freely; probes "
        "intermodulation and superposition. The crest factor grows with the "
        "component count -- watch the full-scale clipping.",
        "Сумма компонент, каждая [частота Гц, амплитуда g] и опц. фаза на тон "
        "[рад]. Добавляйте/удаляйте компоненты свободно; проба интермодуляции и "
        "суперпозиции. Пик-фактор растёт с числом компонент — следите за "
        "клиппингом по полной шкале.",
    ),
    "exc.sweep.summary": _e("chirp f0 -> f1", "чирп f0 → f1"),
    "exc.sweep.about": _e(
        "A constant-amplitude chirp from f start to f end [Hz], linear or "
        "log in frequency. The standard way to trace the frequency response "
        "over the band in one run; log spacing spends more time at low "
        "frequencies.",
        "Чирп постоянной амплитуды от f нач до f кон [Гц], линейный или "
        "логарифмический по частоте. Штатный способ снять частотную "
        "характеристику по полосе за один прогон; лог-шаг проводит больше "
        "времени на низких частотах.",
    ),
    "exc.random.summary": _e("band-limited noise", "полосовой шум"),
    "exc.random.about": _e(
        "Gaussian noise band-limited to [lo, hi] Hz with the given g RMS. "
        "ISO-style broadband excitation; PSD-based metrics apply. The peak "
        "factor is ~3-4x the RMS -- watch the full scale.",
        "Гауссов шум, ограниченный полосой [низ, верх] Гц с заданным СКЗ в g. "
        "Широкополосное возбуждение в стиле ISO; применяются метрики на основе "
        "PSD. Пик-фактор ~3–4× СКЗ — следите за полной шкалой.",
    ),
    "exc.shock.summary": _e("half-sine pulse", "полусинусный импульс"),
    "exc.shock.about": _e(
        "A half-sine shock: peak [g], pulse width [ms], start delay [s]. "
        "For transient/overload studies -- pair it with the modal_time "
        "mechanics and the time integrator (Physics layers tab) for a "
        "faithful transient.",
        "Полусинусный удар: пик [g], ширина импульса [мс], задержка старта [с]. "
        "Для исследований переходных/перегрузок — сочетайте с механикой "
        "modal_time и временным интегратором (вкладка «Физмодели») для верного "
        "переходного процесса.",
    ),
    "exc.csv.summary": _e("CSV replay", "воспроизв. CSV"),
    "exc.csv.about": _e(
        "Replay a recorded acceleration column from a CSV: column index "
        "(0-based), sample rate fs [Hz] (CSV stores no grid), units of the "
        "stored values (m/s^2 or g).",
        "Воспроизвести записанный столбец ускорения из CSV: индекс столбца (с 0), "
        "частота дискретизации fs [Гц] (CSV не хранит сетку), единицы "
        "хранимых значений (м/с² или g).",
    ),
    "exc.wav.summary": _e("WAV replay", "воспроизв. WAV"),
    "exc.wav.about": _e(
        "Replay a WAV channel as acceleration: channel index and the "
        "full-scale mapping [g] (WAV samples are normalized to +/-1, so "
        "full scale sets how many g that is).",
        "Воспроизвести канал WAV как ускорение: индекс канала и отображение "
        "полной шкалы [g] (отсчёты WAV нормированы к ±1, поэтому полная шкала "
        "задаёт, сколько это g).",
    ),
    "exc.tdms.summary": _e("NI TDMS replay", "воспроизв. NI TDMS"),
    "exc.tdms.about": _e(
        "Replay an NI TDMS channel: group (blank = first), channel index, "
        "fs [Hz] (0 = take wf_increment from the file), units of the stored "
        "values.",
        "Воспроизвести канал NI TDMS: группа (пусто = первая), индекс канала, "
        "fs [Гц] (0 = взять wf_increment из файла), единицы хранимых значений.",
    ),
    "exc.uff.summary": _e("UFF/UNV replay", "воспроизв. UFF/UNV"),
    "exc.uff.about": _e(
        "Replay a UFF/UNV dataset-58 record: dataset index, fs [Hz] (0 = "
        "take the abscissa increment from the file), units of the stored "
        "values.",
        "Воспроизвести запись UFF/UNV dataset-58: индекс набора, fs [Гц] (0 = "
        "взять шаг абсциссы из файла), единицы хранимых значений.",
    ),
    "exc.mat.summary": _e("MATLAB replay", "воспроизв. MATLAB"),
    "exc.mat.about": _e(
        "Replay a variable from a MATLAB .mat file: variable name, column "
        "index for 2-D arrays, fs [Hz] (required -- .mat stores no grid), "
        "units of the stored values.",
        "Воспроизвести переменную из файла MATLAB .mat: имя переменной, индекс "
        "столбца для 2-D массивов, fs [Гц] (обязательно — .mat не хранит сетку), "
        "единицы хранимых значений.",
    ),
    # excitation page row labels (short, unit-bearing)
    "exc.row.Sampling": _e("Sampling", "Дискретизация"),
    "exc.row.components": _e("components", "компоненты"),
    "exc.row.method": _e("method", "метод"),
    "exc.row.frequency": _e("frequency [Hz]", "частота [Гц]"),
    "exc.row.amplitude": _e("amplitude [g]", "амплитуда [g]"),
    "exc.row.f": _e("f [Hz]", "f [Гц]"),
    "exc.row.amp": _e("amp [g]", "амп [g]"),
    "exc.row.phase": _e("phase [rad]", "фаза [рад]"),
    "exc.row.fstart": _e("f start [Hz]", "f нач [Гц]"),
    "exc.row.fend": _e("f end [Hz]", "f кон [Гц]"),
    "exc.row.bandlo": _e("band lo [Hz]", "полоса низ [Гц]"),
    "exc.row.bandhi": _e("band hi [Hz]", "полоса верх [Гц]"),
    "exc.row.grms": _e("g RMS [g]", "СКЗ [g]"),
    "exc.row.peak": _e("peak [g]", "пик [g]"),
    "exc.row.pulse": _e("pulse [ms]", "импульс [мс]"),
    "exc.row.delay": _e("delay [s]", "задержка [с]"),
    "exc.row.fs": _e("fs [Hz]", "fs [Гц]"),
    "exc.row.dur": _e("dur [s]", "длит. [с]"),
    "exc.row.fullscale": _e("full scale [g]", "полная шкала [g]"),
    "exc.row.column": _e("column", "столбец"),
    "exc.row.channel": _e("channel", "канал"),
    "exc.row.units": _e("units", "единицы"),
    "exc.row.group": _e("group", "группа"),
    "exc.row.path": _e("path", "путь"),
    "exc.row.dataset": _e("dataset", "набор"),
    "exc.row.dataset_index": _e("dataset index", "индекс набора"),
    "exc.row.data_key": _e("data key", "ключ данных"),
    "exc.row.add_component": _e("+ component", "+ компонента"),
    "exc.row.remove": _e("x", "x"),
    "exc.row.browse": _e("Browse...", "Обзор..."),
    "exc.row.about": _e("about", "о типе"),
    "exc.hdf5.summary": _e("HDF5 replay", "воспроизв. HDF5"),
    "exc.hdf5.about": _e(
        "Replay an HDF5 dataset: dataset path inside the file, column index "
        "for 2-D data, fs [Hz] (0 = take an fs attribute from the file when "
        "present), units of the stored values.",
        "Воспроизвести набор HDF5: путь набора внутри файла, индекс столбца для "
        "2-D данных, fs [Гц] (0 = взять атрибут fs из файла, если есть), единицы "
        "хранимых значений.",
    ),
    # ================================================================== #
    # Matplotlib axis / title / legend labels (static; translated via t()
    # in translate_figure). Pure-LaTeX and data-formatted labels are omitted.
    # ================================================================== #
    "ax.frequency_hz": _e("frequency [Hz]", "частота [Гц]"),
    "ax.time_s": _e("time [s]", "время [с]"),
    "ax.amplitude": _e("amplitude", "амплитуда"),
    "ax.count": _e("count", "количество"),
    "ax.distribution": _e("distribution", "распределение"),
    "ax.a_ms2": _e("a [m/s^2]", "a [м/с²]"),
    "ax.v_ms": _e("v [m/s]", "v [м/с]"),
    "ax.x_m": _e("x [m]", "x [м]"),
    "ax.true": _e("true", "истинное"),
    "ax.recovered": _e("recovered", "восстановленное"),
    "ax.true_a": _e("true a", "истинное a"),
    "ax.recovered_a": _e("recovered a", "восстановленное a"),
    "ax.total": _e("total", "суммарно"),
    "ax.plateau_analytic": _e("plateau (analytic)", "плато (аналитика)"),
    "ax.spectrogram": _e("spectrogram", "спектрограмма"),
    "ax.acc_true_rec": _e(
        "acceleration: true vs recovered", "ускорение: истинное vs восстановленное"
    ),
    "ax.recovered_kin": _e("recovered kinematics", "восстановленная кинематика"),
    "ax.truth_vs_recovery": _e(
        "truth vs recovery (target axis)", "истина vs восстановление (целевая ось)"
    ),
    "ax.residual": _e("residual [m/s$^2$]", "невязка [m/s$^2$]"),
    "ax.cantilever_length_mm": _e("cantilever length L [mm]", "длина консоли L [мм]"),
    "ax.first_mode_vs_length": _e("First bending mode vs length", "Первая изгибная мода от длины"),
    "ax.lateral_transfer": _e(
        "Lateral transfer function (current cantilever)",
        "Боковая передаточная функция (текущая консоль)",
    ),
    "ax.spec_limit_50g": _e("50 g (spec limit)", "50 g (предел спец.)"),
    "ax.nea_contribution": _e("NEA contribution [ug/sqrt(Hz)]", "вклад в NEA [мкg/√Гц]"),
    "ax.nea_ug": _e("NEA [ug/sqrt(Hz)]", "NEA [мкg/√Гц]"),
    "ax.noise_accel": _e("noise-equivalent acceleration", "шумо-эквивалентное ускорение"),
    "ax.shot": _e("shot", "дробовой"),
    "ax.rin": _e("rin", "RIN"),
    "ax.johnson": _e("johnson", "джонсоновский"),
    "ax.thermal": _e("thermal", "тепловой"),
    "ax.samples": _e("samples", "отсчёты"),
    # ================================================================== #
    # Live tab
    # ================================================================== #
    "live.about": _e(
        "Live view of the last run: the cantilever bend animation over "
        "PyQtGraph panels for input-vs-recovered acceleration, the detector "
        "signal, recovered velocity/displacement, the amplitude spectrum and "
        "the NEA(f) density. The check-row shows/hides each panel (session "
        "only); no controls here change the model -- edit on the left and Run.",
        "Онлайн-вид последнего прогона: анимация изгиба консоли над панелями "
        "PyQtGraph для ускорения вход-vs-восстановленное, сигнала детектора, "
        "восстановленных скорости/смещения, амплитудного спектра и плотности "
        "NEA(f). Ряд флажков показывает/скрывает панели (только сессия); контролы "
        "здесь модель не меняют — правьте слева и жмите «Запуск».",
    ),
    "live.cantilever": _e("cantilever", "консоль"),
    "live.panel.accel": _e("acceleration", "ускорение"),
    "live.panel.det": _e("detector", "детектор"),
    "live.panel.vel": _e("velocity", "скорость"),
    "live.panel.disp": _e("displacement", "смещение"),
    "live.panel.spec": _e("spectrum", "спектр"),
    "live.panel.nea": _e("NEA(f)", "NEA(f)"),
    "live.title.accel": _e(
        "Acceleration: input vs recovered", "Ускорение: вход vs восстановленное"
    ),
    "live.title.det": _e("Detector signal", "Сигнал детектора"),
    "live.title.vel": _e("Recovered velocity", "Восстановленная скорость"),
    "live.title.disp": _e("Recovered displacement", "Восстановленное смещение"),
    "live.title.spec": _e("Recovered amplitude spectrum", "Восстановленный амплитудный спектр"),
    "live.title.nea_prompt": _e(
        "NEA(f) - run Report for the budget", "NEA(f) — запустите «Отчёт» для бюджета"
    ),
    "live.title.nea_na": _e(
        "NEA(f) - not available (use the photodiode detector)",
        "NEA(f) — недоступно (нужен фотодиодный детектор)",
    ),
    "live.title.nea_ok": _e(
        "NEA(f) with shot / RIN / Johnson / thermal plateaus",
        "NEA(f) с плато дробового / RIN / джонсоновского / теплового",
    ),
    "live.axis.input": _e("input", "вход"),
    # ================================================================== #
    # Report tab
    # ================================================================== #
    "report.about": _e(
        "Display-only report of the last Run: the truth-vs-recovery a/v/x "
        "figure, the NEA(f) budget (needs the photodiode detector) and the "
        "recovered-acceleration spectrogram, plus the error-budget summary "
        "(amplitude ratio, recovery error). Press Report on the left to build it.",
        "Отчёт (только просмотр) последнего прогона: график истина-vs-"
        "восстановление a/v/x, бюджет NEA(f) (нужен фотодиодный детектор) и "
        "спектрограмма восстановленного ускорения, плюс сводка бюджета ошибок "
        "(отношение амплитуд, ошибка восстановления). Жмите «Отчёт» слева.",
    ),
    "report.tab.truth": _e("Truth vs recovery", "Истина vs восстановление"),
    "report.tab.nea": _e("NEA budget", "Бюджет NEA"),
    "report.tab.spectrogram": _e("Spectrogram", "Спектрограмма"),
    "report.error_budget": _e("Error budget", "Бюджет ошибок"),
    "report.ph.truth": _e(
        "Run 'Report' to build the truth-vs-recovery figure.",
        "Запустите «Отчёт», чтобы построить график истина-vs-восстановление.",
    ),
    "report.ph.nea": _e(
        "NEA budget (needs the photodiode detector).",
        "Бюджет NEA (нужен фотодиодный детектор).",
    ),
    "report.ph.spectrogram": _e(
        "Recovered-acceleration spectrogram.",
        "Спектрограмма восстановленного ускорения.",
    ),
    # ================================================================== #
    # Sweeps tab
    # ================================================================== #
    "sweep.about": _e(
        "Sweep one parameter across a grid and plot the resulting NEA / "
        "response, to see a trend rather than a single point. Pick the variant, "
        "the mode (design geometry vs excitation response), the parameter, the "
        "start/stop/count grid and linear/log spacing, then Run sweep. The heavy "
        "run happens off the UI thread.",
        "Развернуть один параметр по сетке и построить NEA / отклик, чтобы "
        "увидеть тренд, а не одну точку. Выберите вариант, режим (геометрия "
        "design vs отклик response), параметр, сетку старт/стоп/число и линейный/"
        "лог шаг, затем «Запустить развёртку». Тяжёлый прогон идёт вне UI-потока.",
    ),
    "sweep.group": _e("Sweep", "Развёртка"),
    "sweep.variant.label": _e("Variant", "Вариант"),
    "sweep.variant.help": _e(
        "Which built-in composition (A-D) is the baseline for the sweep; every "
        "grid point starts from it and overrides the swept parameter.",
        "Какая встроенная композиция (A–D) — базовая для развёртки; каждая точка "
        "сетки стартует из неё и переопределяет разворачиваемый параметр.",
    ),
    "sweep.mode.label": _e("Mode", "Режим"),
    "sweep.mode.help": _e(
        "'design' sweeps a geometry/design parameter (length, R_c, power, bias, "
        "full-scale) and reports the NEA trend; 'response' sweeps an excitation "
        "parameter (amplitude, frequency) and reports the recovery response. The "
        "parameter list and default grid follow the mode.",
        "«design» разворачивает геометрический/проектный параметр (длина, R_c, "
        "мощность, смещение, полная шкала) и показывает тренд NEA; «response» "
        "разворачивает параметр возбуждения (амплитуда, частота) и показывает "
        "отклик восстановления. Список параметров и сетка по умолчанию зависят от "
        "режима.",
    ),
    "sweep.parameter.label": _e("Parameter", "Параметр"),
    "sweep.parameter.help": _e(
        "The parameter to sweep across the grid; the options depend on the mode "
        "(design geometry vs excitation response).",
        "Параметр для развёртки по сетке; варианты зависят от режима (геометрия "
        "design vs отклик response).",
    ),
    "sweep.grid.label": _e("Grid", "Сетка"),
    "sweep.grid.help": _e(
        "The grid of values: start and stop bounds (in the parameter's SI unit), "
        "num points, and log spacing (denser at small values). Log needs a "
        "positive start.",
        "Сетка значений: границы start и stop (в СИ-единице параметра), число "
        "точек num и лог-шаг (плотнее при малых значениях). Лог требует "
        "положительного start.",
    ),
    "sweep.start": _e("start", "старт"),
    "sweep.stop": _e("stop", "стоп"),
    "sweep.num": _e("num", "число"),
    "sweep.log": _e("log spacing", "лог-шаг"),
    "sweep.run": _e("Run sweep", "Запустить развёртку"),
    "sweep.ph": _e(
        "Run a sweep to see NEA / response vs the parameter.",
        "Запустите развёртку, чтобы увидеть NEA / отклик от параметра.",
    ),
    # ================================================================== #
    # Monte-Carlo tab
    # ================================================================== #
    "mc.about": _e(
        "Tolerance Monte-Carlo: draw the tolerance parameters from their "
        "distributions many times and plot the spread of the output metric, to "
        "see robustness rather than a nominal value. Pick the variant, the draw "
        "count, whether to estimate cross-axis (slower) and which tolerances to "
        "include, then Run. The heavy run happens off the UI thread.",
        "Монте-Карло по допускам: многократно разыгрывает параметры допусков из "
        "их распределений и строит разброс выходной метрики — чтобы увидеть "
        "робастность, а не номинал. Выберите вариант, число реализаций, оценивать "
        "ли перекрёстную ось (медленнее) и какие допуски включить, затем "
        "«Запуск». Тяжёлый прогон идёт вне UI-потока.",
    ),
    "mc.group": _e("Monte-Carlo", "Монте-Карло"),
    "mc.variant.label": _e("Variant", "Вариант"),
    "mc.variant.help": _e(
        "Which built-in composition (A-D) is the baseline; each draw perturbs it "
        "by the selected tolerances.",
        "Какая встроенная композиция (A–D) — базовая; каждая реализация возмущает "
        "её выбранными допусками.",
    ),
    "mc.draws.label": _e("Draws", "Реализации"),
    "mc.draws.help": _e(
        "Number of Monte-Carlo draws. More draws tighten the estimated spread "
        "but cost linearly more compute.",
        "Число реализаций Монте-Карло. Больше реализаций — точнее оценка "
        "разброса, но линейно дороже по вычислениям.",
    ),
    "mc.cross_axis": _e("estimate cross-axis (slower)", "оценивать перекрёстную ось (медленнее)"),
    "mc.cross_axis.help": _e(
        "Also estimate the cross-axis response for each draw (drives the "
        "cross-axis sensitivity metric, doc 00). Roughly triples the per-draw "
        "cost (extra off-axis excitations).",
        "Дополнительно оценивать перекрёстный отклик для каждой реализации "
        "(метрика перекрёстной чувствительности, док 00). Примерно утраивает "
        "стоимость реализации (доп. внеосевые возбуждения).",
    ),
    "mc.tolerances.label": _e("Tolerances", "Допуски"),
    "mc.tolerances.help": _e(
        "Which parameters are drawn from their tolerance distributions each "
        "run: q_total (lognormal, 30%), R_c (normal, 5%), gap (normal, 5 um), "
        "bias (normal, 0.1 um), epsilon_x lateral offset (normal, 0.1 um). "
        "Uncheck to hold a parameter fixed at nominal.",
        "Какие параметры разыгрываются из распределений допусков в каждом "
        "прогоне: q_total (логнормальное, 30%), R_c (нормальное, 5%), зазор "
        "(нормальное, 5 мкм), смещение (нормальное, 0.1 мкм), боковой сдвиг "
        "epsilon_x (нормальное, 0.1 мкм). Снимите флажок, чтобы держать параметр "
        "на номинале.",
    ),
    "mc.run": _e("Run Monte-Carlo", "Запустить Монте-Карло"),
    "mc.ph": _e(
        "Run a Monte-Carlo to see the metric distribution.",
        "Запустите Монте-Карло, чтобы увидеть распределение метрики.",
    ),
    # ================================================================== #
    # Physics tab
    # ================================================================== #
    "physics.about": _e(
        "Reference design curves for the current composition, so you can see "
        "where your edits land before running: f1(L), the lateral transfer "
        "|H_lat(f)| and the coupling eta(dx) are light and rebuilt by Refresh; "
        "the measured NEA(f) budget is heavy and runs via Compute NEA(f) (a "
        "Report off the UI thread). The Reference notes tab explains the models.",
        "Справочные проектные кривые текущей композиции — чтобы видеть, куда "
        "попадают правки до прогона: f1(L), боковая передача |H_lat(f)| и связь "
        "η(dx) лёгкие и пересобираются кнопкой «Обновить»; измеренный бюджет "
        "NEA(f) тяжёлый и считается по «Вычислить NEA(f)» (Отчёт вне UI-потока). "
        "Вкладка «Справочные заметки» поясняет модели.",
    ),
    "physics.tab.curves": _e("Design curves", "Проектные кривые"),
    "physics.tab.notes": _e("Reference notes", "Справочные заметки"),
    "physics.refresh": _e("Refresh from composition", "Обновить из композиции"),
    "physics.nea": _e("Compute NEA(f)", "Вычислить NEA(f)"),
    "physics.ph.f1": _e("Press Refresh to build f1(L).", "Нажмите «Обновить» для f1(L)."),
    "physics.ph.hlat": _e(
        "Press Refresh to build |H_lat(f)|.", "Нажмите «Обновить» для |H_lat(f)|."
    ),
    "physics.ph.eta": _e("Press Refresh to build eta(dx).", "Нажмите «Обновить» для η(dx)."),
    "physics.ph.nea": _e(
        "Press Compute NEA(f) to run the budget.",
        "Нажмите «Вычислить NEA(f)» для бюджета.",
    ),
    # Reference notes (HTML). English kept byte-identical to the module block.
    "physics.notes.html": _e(
        _PHYSICS_NOTES_EN,
        _PHYSICS_NOTES_RU,
    ),
}
