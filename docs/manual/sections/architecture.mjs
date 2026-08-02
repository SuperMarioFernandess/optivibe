// docs/manual/sections/architecture.mjs
// AUTO-SLICED (S-DOC-2) from manual_build_doc.js lines 267..345; content verbatim.
// Owns its bookmarks/links. Edit this module only; build.mjs orders it via manifest.mjs.
import {
  R, c, P, para, H, xr, flink, ext, links, sep, code, cap, bullets, nums, img, tbl, fml,
  AlignmentType, Paragraph, TextRun, PageBreak,
  REPO, FIG, INK, MUT, ACC, ACCD, LINE, CW,
} from "../shared.mjs";

export function section() {
  const K = [];
  K.push(H(1,"5. Архитектура ПО","ch5"));
  K.push(H(2,"5.1 Принципы","s_arch_principles"));
  K.push(...bullets([
    [R("Чистое ядро. ",{bold:true}), R("core и физические модули не знают об оболочке, форматах данных и менеджере; зависимости направлены внутрь. GUI/CLI/IO — адаптеры на краю.")],
    [R("Порты и адаптеры + реестр. ",{bold:true}), R("Каждый этап = типизированная стадия с контрактом вход/выход; сменные реализации регистрируются и выбираются конфигом.")],
    [R("Конфиг-ориентированность. ",{bold:true}), R("Параметры, варианты и сценарии — в YAML, валидируются pydantic. Числа берутся из базы знаний; тест сверяет конфиги с реперами.")],
    [R("Контракты данных — хребет. ",{bold:true}), R("Стадии общаются только через типизированные модели (внутренний ICD, зеркало документа 04).")],
    [R("Тесты против физики. ",{bold:true}), R("Golden-тесты на аналитические реперы + property-тесты на инварианты — первоклассная часть архитектуры.")],
  ]));

  K.push(H(2,"5.2 Дерево пакетов","s_arch_tree"));
  K.push(...code([
    "src/optivibe/",
    "  core/{types,stages,registry,units,logging}.py   # контракты, протоколы, реестр, СИ",
    "  core/config/{models,loader}.py                  # pydantic-модели и загрузка YAML",
    "  excitation/        # генераторы 3D a(t) + replay (S1)",
    "  mechanics/         # модальная модель, H_lat(f), решатели (S2)",
    "  optics/            # гауссов пучок, η(Δx), цилиндр, матрица ∂S/∂q (S3)",
    "  detector/          # фототок, шумы, опорный канал, АЦП (S4)",
    "  dsp/               # калибровка, a→v→x, спектры, ISO, NEA, чувствительность (S5/S6)",
    "  pipeline/          # оркестратор forward+inverse",
    "  analysis/          # truth-vs-recovery, NEA-бюджет, развёртки, Монте-Карло (S6)",
    "  viz/               # ЧИСТЫЕ фигуры matplotlib/plotly (без Qt)",
    "  io/loaders.py      # CSV/WAV/TDMS/UFF/MAT/HDF5 → Excitation (S1/S8)",
    "  io/records.py      # записи фототока → analyze (S-02)",
    "  io/{characterization,ingest}.py  # измеренные параметры + провенанс (S-13)",
    "  dsp/streaming.py   # причинный потоковый слой реального времени (S-03)",
    "  analysis/{expected_peaks,compare}.py  # ожидаемые пики (S-16/17), стенд ЦОС (S-22)",
    "  excitation/composite.py  # композит и АМ/ЧМ-модуляция (S-21)",
    "  cli/               # optivibe run/report/sweep/analyze/compare/ingest",
    "  gui/               # десктоп-приложение PySide6/PyQtGraph (S7)",
    "  gui/workers/stream.py    # Qt-free источники и цикл живого режима (O-SW-03)",
    "configs/{constants.yaml, variants/*, scenarios/*}   # зеркала 01/08",
    "examples/  tests/  docs/  packaging/                # сценарии, тесты, доки, сборка",
  ]));
  K.push(para([R("Полное дерево с пометками по этапам — в базе знаний (документ 14 §2). Ссылки на пакеты: "), flink("src/optivibe/core/"), R(", "), flink("src/optivibe/dsp/"), R(", "), flink("src/optivibe/gui/"), R(".")]));

  K.push(H(2,"5.3 Контракты данных (внутренний ICD)","s_arch_contracts"));
  K.push(para([R("Все контракты — иммутабельные "), c("frozen dataclass"), R(" с однократной валидацией формы/единиц; массивы — 1-D float64 в СИ ("), flink("src/optivibe/core/types.py","core/types.py"), R("). Это зеркало межсистемного интерфейса (документ 04).")]));
  K.push(tbl(
    ["Контракт","Ключевые поля","Смысл"],
    [
      ["Excitation","a_x, a_y, a_z [м/с²], fs, seed, meta","вход системы: 3D ускорение"],
      ["TipState","dx, dy, dz [м], theta_x, theta_y [рад], fs","вектор состояния торца (ICD)"],
      ["OpticalResponse","eta, eta_x?, eta_y?, bias, fs","эффективность обратного ввода"],
      ["DetectorOutput","samples, fs, dc_level, units∈{A,V}, noise","отсчёты фотодетектора + шум"],
      ["VibrationResult","a, v, x, fs, dominant_freqs_hz, rms, cross_residual, spectrum?, iso?","выход обратной задачи"],
      ["Spectrum","freq, values, kind∈{amplitude,psd}, window, method","спектральное представление"],
    ],
    [2050,4350,2960]
  ));
  K.push(para([R("Поле "), c("DetectorOutput.noise"), R(" несёт всё для DSP: "), c("i_dc_a"), R(", "), c("psd_shot/rin/johnson/total_a2_hz"), R(", "), c("sigma_i_a"), R(", "), c("balanced"), R(", "), c("reference_arm"), R(", "), c("adc_lsb"), R(". Производительность: pydantic — для конфигов/метаданных, frozen-dataclass — для контрактов с массивами.")]));

  K.push(H(2,"5.4 Реестр и расширения","s_arch_registry"));
  K.push(para([R("Реестр ("), flink("src/optivibe/core/registry.py","core/registry.py"), R(") отображает ключ из конфига в реализацию стадии. Добавить отражатель, источник, метод DSP или загрузчик — значит зарегистрировать новый адаптер, не трогая ядро.")]));
  K.push(tbl(
    ["Реестр","Зарегистрированные ключи","Умолчание"],
    [
      ["EXCITATION_REGISTRY","sine, multitone, sweep, random, shock, composite, csv, wav, tdms, uff, mat, hdf5","sine"],
      ["MECHANICS_REGISTRY","modal, modal_time, stub","modal"],
      ["OPTICS_REGISTRY","cylinder, reflector, stub","cylinder"],
      ["REFLECTOR_MODEL_REGISTRY","cylinder, sphere, plane, wedge (за ключом reflector; §4.7)","cylinder"],
      ["DETECTOR_REGISTRY","stub, photodiode","stub"],
      ["DSP_REGISTRY","stub, standard","stub"],
      ["INTEGRATOR_REGISTRY","frequency, time, leaky","frequency"],
      ["SENSITIVITY_REGISTRY","static, operating_point, nonlinear_curve","static"],
      ["LOADER_REGISTRY","csv, wav, tdms, uff, mat, hdf5 (replay-вход)","—"],
      ["RECORD_REGISTRY","csv, tdms, hdf5 (записи фототока для analyze)","—"],
      ["CHARACTERIZATION_REGISTRY","scalar, spectrum, rin_psd, ringdown, profile (§3.5)","—"],
      ["PEAK_PREDICTOR_REGISTRY","mode, harmonic, intermod, sideband (§6.8); mains, alias, f_mount объявлены и пусты","—"],
    ],
    [2650,5210,1500]
  ));
  K.push(para([R("Реестров стало больше не случайно: каждый новый слой — ввод измерений, ожидаемые пики, альтернативы цепочки ЦОС — добавлялся "), R("регистрацией", { bold: true }), R(", а не правкой ядра. Именно поэтому контракты стадий и эталон вариантов A/B/C/D не менялись ни в одной из последних шести волн.")]));

  K.push(H(2,"5.5 Конфигурация и сценарии","s_arch_config"));
  K.push(para([R("Три уровня YAML: "), c("configs/constants.yaml"), R(" (физконстанты, зеркало документа 01), "), c("configs/variants/{A,B,C,D,proto_poc}.yaml"), R(" (параметры вариантов, зеркало 08; включают блоки optics и detector; "), c("proto_poc"), R(" — конфигурация прототипа с плейсхолдерами до фазы 0), и сценарий прогона. Сценарий задаёт "), c("{variant, excitation, stages, detector, dsp, seed}"), R("; "), c("DspOptions"), R(" включает "), c("integrator"), R(", "), c("spectrum_method"), R(", "), c("window"), R(", "), c("welch_nperseg"), R("/"), c("welch_noverlap"), R(", "), c("f_hp_hz"), R(", "), c("f_c_stream"), R(", "), c("calibration∈{ideal,bench}"), R(", "), c("sensitivity_model"), R(", "), c("sensitivity_freq"), R(", "), c("deconvolve_hlat"), R(", "), c("peak_interpolation"), R(", "), c("iso_machine_class"), R(" (все с дефолтами = v1; отклонение от них помечает прогон экспериментальным — §3.7). Спецификации аналитики ("), c("SweepSpec"), R("/"), c("MonteCarloSpec"), R(") — отдельные модели в "), flink("src/optivibe/analysis/spec.py","analysis/spec.py"), R(".")]));

  K.push(H(2,"5.6 Конвейер: forward + inverse","s_arch_pipeline"));
  K.push(para([R("Оркестратор ("), flink("src/optivibe/pipeline/orchestrator.py","pipeline/orchestrator.py"), R(") компонует стадии по контрактам:")]));
  K.push(...code([
    "forward:  Excitation → [mechanics] → TipState → [optics] → OpticalResponse",
    "                     → [detector]  → DetectorOutput",
    "inverse:  DetectorOutput → [dsp] → VibrationResult",
  ]));
  K.push(para([R("Фасад "), c("run_scenario(path, config_dir)"), R(" грузит сценарий, выбирает реализации стадий из реестров, прогоняет цепочку и возвращает результаты. Субсид детектора детерминированно выводится из сида сценария (воспроизводимость).")]));
  K.push(links([ xr("Алгоритмы по стадиям → §6","ch6"), sep(), xr("Реализация контрактов и реестра → §7.1–7.2","s_impl_types"), sep(), flink("src/optivibe/pipeline/"), sep(), R("база знаний: документ 09",{size:18,color:MUT}) ]));


  return K;
}
