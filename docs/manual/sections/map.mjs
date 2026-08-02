// docs/manual/sections/map.mjs
// AUTO-SLICED (S-DOC-2) from manual_build_doc.js lines 695..722; content verbatim.
// Owns its bookmarks/links. Edit this module only; build.mjs orders it via manifest.mjs.
import {
  R, c, P, para, H, xr, flink, ext, links, sep, code, cap, bullets, nums, img, tbl, fml,
  AlignmentType, Paragraph, TextRun, PageBreak,
  REPO, FIG, INK, MUT, ACC, ACCD, LINE, CW,
} from "../shared.mjs";

export function section() {
  const K = [];
  K.push(H(1,"10. Карта связей (физика ↔ алгоритм ↔ код ↔ файл)","ch10"));
  K.push(para("Сводная таблица переходов между уровнями описания. Ячейки «Физика/Алгоритм/Код» — внутренние ссылки на разделы документа; «Файл» — ссылки на репозиторий. База знаний: физика — документы 00–08, ПО — документы 09–14. Таблица сверена по дереву репозитория целиком: каждая ссылка «Файл» указывает на существующий путь."));
  const mapRow=(name, phys, algo, codeS, file, fileLabel)=>[
    name,
    phys?[xr(phys[0],phys[1])]:[R("—",{size:18,color:MUT})],
    algo?[xr(algo[0],algo[1])]:[R("—",{size:18,color:MUT})],
    codeS?[xr(codeS[0],codeS[1])]:[R("—",{size:18,color:MUT})],
    [flink(file, fileLabel)],
  ];
  K.push(tbl(
    ["Подсистема","Физика","Алгоритм","Код","Файл"],
    [
      mapRow("Возбуждение", null, ["§6.1","s_algo_exc"], ["§7.4","s_impl_exc"], "src/optivibe/excitation/tonal.py","excitation/tonal.py"),
      mapRow("Композит и АМ/ЧМ", null, ["§6.1","s_algo_composite"], ["§7.13","s_impl_composite"], "src/optivibe/excitation/composite.py","excitation/composite.py"),
      mapRow("Механика", ["§4.2","s_phys_mech"], ["§6.2","s_algo_mech"], ["§7.5","s_impl_mech"], "src/optivibe/mechanics/cantilever.py","mechanics/cantilever.py"),
      mapRow("Оптика", ["§4.3","s_phys_opt"], ["§6.3","s_algo_opt"], ["§7.6","s_impl_opt"], "src/optivibe/optics/cylinder.py","optics/cylinder.py"),
      mapRow("Детектор/шумы", ["§4.4","s_phys_det"], ["§6.4","s_algo_det"], ["§7.7","s_impl_det"], "src/optivibe/detector/photodiode.py","detector/photodiode.py"),
      mapRow("DSP / обратная", ["§4.5","s_phys_e2e"], ["§6.5","s_algo_dsp"], ["§7.8","s_impl_dsp"], "src/optivibe/dsp/calibration.py","dsp/calibration.py"),
      mapRow("Чувствительность", ["§4.5","s_phys_e2e"], ["§6.6","s_algo_sens"], ["§7.8","s_impl_dsp"], "src/optivibe/dsp/sensitivity.py","dsp/sensitivity.py"),
      mapRow("NEA (шум→вход)", ["§4.4","s_phys_det"], ["§6.5","s_algo_dsp"], ["§7.9","s_impl_nea"], "src/optivibe/dsp/nea.py","dsp/nea.py"),
      mapRow("Аналитика", ["§4.6","s_phys_family"], ["§6.7","s_algo_analysis"], null, "src/optivibe/analysis/","analysis/"),
      mapRow("Контракты (ICD)", null, ["§5.3","s_arch_contracts"], ["§7.1","s_impl_types"], "src/optivibe/core/types.py","core/types.py"),
      mapRow("Реестр", null, ["§5.4","s_arch_registry"], ["§7.2","s_impl_registry"], "src/optivibe/core/registry.py","core/registry.py"),
      mapRow("Конвейер", ["§4.5","s_phys_e2e"], ["§5.6","s_arch_pipeline"], null, "src/optivibe/pipeline/orchestrator.py","pipeline/orchestrator.py"),
      mapRow("GUI", null, ["§3.1","s_ui_gui"], ["§7.10","s_impl_gui"], "src/optivibe/gui/","gui/"),
      mapRow("Композиция/пресеты", null, ["§3.4","s_cfg_subsys"], null, "src/optivibe/core/config/subsystems.py","core/config/subsystems.py"),
      mapRow("Отражатели (семейство)", ["§4.7","s_phys_reflectors"], null, null, "src/optivibe/optics/reflector.py","optics/reflector.py"),
      mapRow("Демпфирование Q(L)", ["§4.2","s_phys_mech"], ["§6.2","s_algo_mech"], ["§7.5","s_impl_mech"], "src/optivibe/mechanics/damping.py","mechanics/damping.py"),
      mapRow("Тепловой пол NEA_th", ["§4.4","s_phys_det"], ["§6.5","s_algo_dsp"], ["§7.9","s_impl_nea"], "src/optivibe/mechanics/thermal.py","mechanics/thermal.py"),
      mapRow("Спектр источника (Δλ, V, RIN)", ["§4.3-бис","s_phys_src"], ["§6.3","s_algo_opt"], ["§7.6","s_impl_opt"], "src/optivibe/optics/source.py","optics/source.py"),
      mapRow("Импорт записей (replay)", null, ["§3.3","s_ui_data"], null, "src/optivibe/io/loaders.py","io/loaders.py"),
      mapRow("Записи выхода прибора", null, ["§3.3","s_ui_data"], null, "src/optivibe/io/records.py","io/records.py"),
      mapRow("Ввод измеренных параметров", null, ["§3.5","s_ingest"], ["§7.14","s_impl_ingest"], "src/optivibe/io/ingest.py","io/ingest.py"),
      mapRow("Артефакты характеризации", null, ["§3.5","s_ingest_kinds"], ["§7.14","s_impl_ingest"], "src/optivibe/io/characterization.py","io/characterization.py"),
      mapRow("Потоковый слой (реальное время)", null, ["§6.5-бис","s_algo_stream"], ["§7.11","s_impl_stream"], "src/optivibe/dsp/streaming.py","dsp/streaming.py"),
      mapRow("Живой режим (GUI)", null, ["§3.6","s_live"], ["§7.15","s_impl_compare"], "src/optivibe/gui/workers/stream.py","gui/workers/stream.py"),
      mapRow("Ожидаемые пики", null, ["§6.8","s_algo_peaks"], ["§7.12","s_impl_peaks"], "src/optivibe/analysis/expected_peaks.py","analysis/expected_peaks.py"),
      mapRow("Стенд сравнения ЦОС", null, ["§3.7","s_cmp"], ["§7.15","s_impl_compare"], "src/optivibe/analysis/compare.py","analysis/compare.py"),
      mapRow("Локализация интерфейса", null, ["§3.1","s_ui_prefs"], null, "src/optivibe/gui/i18n.py","gui/i18n.py"),
      mapRow("CLI", null, ["§3.2","s_ui_cli"], null, "src/optivibe/cli/main.py","cli/main.py"),
    ],
    [1750,1430,1430,1430,3320]
  ));

  return K;
}
