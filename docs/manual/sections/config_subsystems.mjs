// docs/manual/sections/config_subsystems.mjs
// NEW (S-DOC-2, свёртка S9-A). Раздел §3.4 «Композиция подсистем»: редактируемые
// формы подсистем вместо выбора «вариант A/B/C/D», пресеты и Save/Load.
// Источник истины по UI/коду: src/optivibe/core/config/subsystems.py,
// core/config/presets.py, gui/widgets/subsystem_forms.py,
// gui/controllers/system_builder.py. Физика/числа — база 01/03/07/08.
import {
  R, c, P, para, H, xr, flink, ext, links, sep, code, cap, bullets, nums, img, tbl, fml,
  AlignmentType, Paragraph, TextRun, PageBreak,
  REPO, FIG, INK, MUT, ACC, ACCD, LINE, CW,
} from "../shared.mjs";

export function section() {
  const K = [];
  K.push(H(2, "3.4 Композиция подсистем (конструктор системы)", "s_cfg_subsys"));
  K.push(para([R("Датчик описывается не выбором готового «варианта», а "), R("редактируемой композицией", { bold: true }), R(": по одной форме на физическую подсистему — источник, волокно, консоль, отражатель, детектор — каждая с выбором пресета и подписанными полями-переопределениями в единицах СИ. Варианты A/B/C/D остаются "), R("стартовыми композициями", { bold: true }), R(", которые засеивают формы (форма "), c("SystemBuilderPanel"), R(", "), flink("src/optivibe/gui/widgets/subsystem_forms.py", "gui/widgets/subsystem_forms.py"), R("). Формы лишь собирают «полезную нагрузку», а её валидирует в неизменяемый "), c("SystemConfig"), R(" сборщик "), flink("src/optivibe/gui/controllers/system_builder.py", "gui/controllers/system_builder.py"), R(" — физики в GUI нет (09 §9).")]));

  // ---- five-block: preset selector ----
  K.push(H(3, "Селектор пресета подсистемы", "s_cfg_preset"));
  K.push(...nums([
    [R("Что это в UI. ", { bold: true }), R("Выпадающий список пресетов подсистемы ("), c("QComboBox"), R(" в "), c("_SubsystemForm"), R("). Смена пресета пере-засеивает поля-переопределения значениями выбранного «строительного блока» ("), c("_on_preset_changed"), R(" → "), c("subsystem_defaults"), R(").")],
    [R("Что это физически / зачем. ", { bold: true }), R("Пресет — это именованный физический блок: источник SLD/DFB, волокно SMF-28, кремнёвая консоль, цилиндрическое зеркало R_c, балансный 24-битный детектор. Соответствует реальной комплектации прибора, а не «режиму программы» (база: 01 §4, 03 §1, 07–08).")],
    [R("Как устроено внутри. ", { bold: true }), R("Список читается из "), c("PresetStore.list_presets(subsystem)"), R(" ("), flink("src/optivibe/core/config/presets.py", "core/config/presets.py"), R("): два яруса — встроенные "), c("configs/presets/<sub>/*.yaml"), R(" и пользовательские "), c("configs/user/presets/<sub>/*.yaml"), R(" (пользовательский перекрывает встроенный при совпадении имени). Значения блока даёт "), c("PresetStore.build_<sub>(SubsystemRef)"), R(".")],
    [R("Что меняет (цепочка следствий). ", { bold: true }), R("Выбранный блок → поля формы → при сборке поля стекаются в плоский "), c("VariantConfig"), R(" (см. ниже), который читают стадии. Напр. пресет консоли с "), c("length_m"), R(" задаёт "), c("L → f₁ → полоса"), R(" и "), c("L → H_QS → NEA"), R(".")],
    [R("Связи и типичные ошибки. ", { bold: true }), R("Опечатка в имени поля-переопределения не «теряется тихо»: блоки подсистем — pydantic с "), c("extra=\"forbid\""), R(", поэтому неизвестный ключ падает при сборке (10 §7). См. "), xr("§5.3 Контракты", "s_arch_contracts"), R(".")],
  ], "n1"));

  // ---- subsystem → model → preset dir table ----
  K.push(H(3, "Подсистемы, их модели и каталоги пресетов", "s_cfg_models"));
  K.push(tbl(
    ["Подсистема", "Модель (pydantic)", "Ключевые поля → плоский VariantConfig", "Пресеты"],
    [
      ["источник", "SourceConfig", "source_kind, wavelength_m, power_w, rin_db_hz (вычисляется, если опущен), linewidth_fwhm_m, lineshape, spectrum_wavelength_m/spectrum_psd", "configs/presets/source/"],
      ["волокно", "FiberConfig", "mode_field_radius_m→optics, fresnel_R1→endface_reflectivity", "configs/presets/fiber/"],
      ["консоль", "CantileverConfig", "length_m→length_m; material (информац., 01 §4)", "configs/presets/cantilever/"],
      ["отражатель", "ReflectorConfig", "shape, curvature_radius_m, metallization_rho, gap_m, bias_offset_m, wedge_angle_rad", "configs/presets/reflector/"],
      ["детектор", "DetectorConfig", "responsivity→responsivity_a_w; balanced/adc_*→detector.*", "configs/presets/detector/"],
    ],
    [1300, 1700, 4060, 2300]
  ));
  K.push(para([R("Системные скаляры, не принадлежащие одной подсистеме ("), c("name"), R(", "), c("band"), R(", "), c("mode"), R(", "), c("full_scale_g"), R(", "), c("route"), R(", "), c("q_total"), R(", "), c("vacuum"), R(", "), c("eta_bias"), R("), живут на самом "), c("SystemConfig"), R(". Замечание по реализации: "), c("material"), R(" консоли и "), c("clad_diameter_m"), R(" в v1 информационные — механика берёт их из глобальных констант "), c("configs/constants.yaml"), R(" (01 §4); проводка из подсистемы в путь констант — отложенный цикл (14 §8).")]));

  // ---- resolve seam ----
  K.push(H(3, "Резолвинг композиции в плоский вариант", "s_cfg_resolve"));
  K.push(para([R("Две прослойки специально. Редактируемый слой (формы/пресеты) и "), R("разрешённый", { italics: true }), R(" слой "), c("VariantConfig"), R(", который читают стадии, оставлены без изменений байт-в-байт. "), c("SystemConfig.resolve(store)"), R(" пере-уплощает блоки обратно в "), c("VariantConfig"), R(" — поэтому стадии не правятся, а композиция A/B/C/D разрешается "), R("бит-идентично", { bold: true }), R(" прежним плоским вариантам (это и проверяется golden-тестами).")]));
  K.push(...code([
    "# composed form (S9-A): configs/variants/B.yaml — реальный пример",
    "name: B",
    "band: { f_min_hz: 1.0, f_max_hz: 10000.0 }",
    "full_scale_g: 50.0",
    "q_total: null              # ВЫЧИСЛЯЕТСЯ моделью Q(L) → 2606.49 (R-48)",
    "cantilever:",
    "  preset: silica",
    "  overrides: { length_m: 2.0e-3 }   # L = 2.0 мм",
    "reflector:",
    "  preset: cyl_rc31        # R_c = 31 мкм, Δx0 = 0.992 мкм",
    "detector:",
    "  preset: balanced_24bit",
    "",
    "# в коде:  variant = SystemConfig(...).resolve(PresetStore(config_dir))",
  ]));
  K.push(H(3, "Вычисляемые поля: q_total и rin_db_hz (что задавать, а что нет)", "s_cfg_computed"));
  K.push(para([R("Два поля, которые раньше были числами из таблицы, теперь "), R("вычисляются при резолвинге", { bold: true }), R(" — и это меняет то, как заполняется композиция.")]));
  K.push(...nums([
    [c("q_total"), R(" — "), R("оставьте пустым (null)", { bold: true }), R(". Добротность считает модель Q(L) (воздух + заделка + внутренние потери; "), flink("src/optivibe/mechanics/damping.py", "mechanics/damping.py"), R("): для L = 2.0 мм получается 2606.49. Все поставляемые варианты A/B/C/D идут с "), c("q_total: null"), R(". Явно заданное число остаётся приоритетным "), R("переопределением (override)", { bold: true }), R(" — используйте его только тогда, когда Q измерена на стенде (ring-down).")],
    [c("rin_db_hz"), R(" — "), R("либо задайте явно, либо дайте модели вывести его", { bold: true }), R(". Если поле опущено, а источник — SLD, RIN выводится из ширины линии "), c("linewidth_fwhm_m"), R(" (Δλ) и формы линии "), c("lineshape"), R(" как пол ASE-биения, либо из табличного спектра ("), c("lineshape: measured"), R("). Поставляемые пресеты SLD/DFB задают RIN явно (−126 / −155 дБ/Гц) — это тоже override. Для DFB вывод RIN из Δλ "), R("запрещён", { bold: true }), R(" (соотношение справедливо только для теплового/ASE-света): без явного "), c("rin_db_hz"), R(" композиция не соберётся.")],
  ], "n2"));
  K.push(para([R("Правила пар «форма линии ↔ вход» (валидатор откажет громко): аналитическая форма ("), c("gaussian"), R("/"), c("lorentzian"), R(") "), R("требует", { bold: true }), R(" "), c("linewidth_fwhm_m"), R("; режим "), c("measured"), R(" "), R("требует", { bold: true }), R(" обе колонки таблицы ("), c("spectrum_wavelength_m"), R(", "), c("spectrum_psd"), R(") и "), R("запрещает", { bold: true }), R(" скалярную Δλ (единственный источник истины — таблица); центральная λ должна лежать внутри диапазона таблицы. Таблица спектра задаётся пока "), R("инлайн в YAML", { italics: true }), R(" — загрузка CSV-трассы OSA с провенансом и неопределённостями отложена (бэклог S-13/M-15).")]));
  K.push(para([R("Ворота когерентного смыва. "), R("Если задана Δλ и выбран маршрут 2, при резолвинге проверяется условие V(A) < 0.03 при номинальном зазоре. Не прошло — "), R("громкая ошибка", { bold: true }), R(" (интенсивностная модель сигнала неприменима, см. §4.3-бис), а не тихий откат. Для лоренцевой формы линии критерий строже на 12.5 % (A ≥ 1.2647·L_c против 1.1246·L_c).")]));
  K.push(para([R("Контракт не расширился: Δλ, "), c("lineshape"), R(" и таблица спектра — "), R("композиционные", { italics: true }), R(" поля; в разрешённый "), c("VariantConfig"), R(" они не попадают (туда идёт уже скалярный "), c("rin_db_hz"), R("). Вычисляемые числа контракта ("), c("q_total"), R(" и выведенный "), c("rin_db_hz"), R(") округляются до 6 значащих цифр — иначе последний бит был бы платформозависим (R-51).")]));

  K.push(para([R("При резолвинге проверяются межподсистемные геометрические ограничения (рабочее время дублируется на время конфигурации): для curved-форм "), c("R_c ≥ 5·w0"), R(" и пятно "), c("w(A) ≤ R_c/3"), R("; для клина "), c("|α_w| ≤ 0.15 рад"), R(" ("), c("_check_composition_geometry"), R(", зеркалит "), flink("src/optivibe/optics/cylinder.py", "optics/cylinder.py"), R("). Нарушение — громкая ошибка, прогон помечается как неуспешный (без тихого отката).")]));

  // ---- five-block: Save/Load ----
  K.push(H(3, "Сохранение и загрузка композиции", "s_cfg_saveload"));
  K.push(...nums([
    [R("Что это в UI. ", { bold: true }), R("Кнопки Save/Load в панели композиции ("), c("QFileDialog"), R("): сохранить текущую композицию в файл и загрузить её обратно.")],
    [R("Что это физически / зачем. ", { bold: true }), R("Воспроизводимая запись «как собран датчик» — чтобы вернуться к конфигурации стенда или передать её коллеге; аналог паспорта прибора.")],
    [R("Как устроено внутри. ", { bold: true }), R("Файлы — "), c("configs/user/systems/<name>.yaml"), R("; запись/чтение через "), c("save_system_config"), R(" / "), c("load_system_file"), R(" ("), flink("src/optivibe/core/config/presets.py", "core/config/presets.py"), R("), оба валидируют через "), c("SystemConfig"), R(".")],
    [R("Что меняет (цепочка следствий). ", { bold: true }), R("Загруженная композиция засеивает все формы; дальнейший Run строит из неё сценарий и резолвит на рабочем потоке (SW-06).")],
    [R("Связи и типичные ошибки. ", { bold: true }), R("Имя композиции — любое непустое (для A/B/C/D это «A»…«D»). Неизвестный пресет/неверный ключ/непрошедшая геометрия → ошибка при загрузке, а не молчаливый дефолт. Выбор формы отражателя и её физика — "), xr("§4.7", "s_phys_reflectors"), R(".")],
  ], "n1"));

  K.push(links([
    xr("Формы подсистем → §3.1", "s_ui_gui"), sep(),
    xr("Семейство отражателей → §4.7", "s_phys_reflectors"), sep(),
    xr("Контракты/реестр → §5.3/§5.4", "s_arch_contracts"), sep(),
    flink("src/optivibe/core/config/subsystems.py"), sep(),
    flink("configs/presets/"),
  ]));
  return K;
}
