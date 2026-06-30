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
  K.push(para("Сводная таблица переходов между уровнями описания. Ячейки «Физика/Алгоритм/Код» — внутренние ссылки на разделы документа; «Файл» — ссылки на репозиторий. База знаний: физика — документы 00–08, ПО — документы 09–14."));
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
      mapRow("Возбуждение", null, ["§6.1","s_algo_exc"], ["§7.4","s_impl_exc"], "src/optivibe/excitation/generators.py","excitation/generators.py"),
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
    ],
    [1750,1430,1430,1430,3320]
  ));

  return K;
}
