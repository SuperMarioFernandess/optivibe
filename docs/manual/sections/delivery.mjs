// docs/manual/sections/delivery.mjs
// AUTO-SLICED (S-DOC-2) from manual_build_doc.js lines 682..693; content verbatim.
// Owns its bookmarks/links. Edit this module only; build.mjs orders it via manifest.mjs.
import {
  R, c, P, para, H, xr, flink, ext, links, sep, code, cap, bullets, nums, img, tbl, fml,
  AlignmentType, Paragraph, TextRun, PageBreak,
  REPO, FIG, INK, MUT, ACC, ACCD, LINE, CW,
} from "../shared.mjs";

export function section() {
  const K = [];
  K.push(H(1,"9. Поставка и установка для пользователя","ch9"));
  K.push(para([R("Десктоп-приложение упаковывается в автономный бандл (PyInstaller, one-dir, Windows-first): "), flink("packaging/optivibe-gui.spec","packaging/optivibe-gui.spec"), R(" + лаунчер "), flink("packaging/optivibe_gui_launch.py","packaging/optivibe_gui_launch.py"), R(". Бандл включает "), c("configs/"), R(" и "), c("examples/"), R(".")]));
  K.push(...code([
    "uv sync --extra packaging",
    "pyinstaller packaging/optivibe-gui.spec --noconfirm   # → dist/OptiVibe/",
    "#  запуск: dist/OptiVibe/OptiVibe(.exe)  → откроется GUI",
  ]));
  K.push(para([R("CI ("), flink(".github/workflows/ci.yml",".github/workflows/ci.yml"), R(") содержит "), R("два", { bold: true }), R(" джоба: "), c("quality"), R(" (Linux — "), c("uv sync --locked"), R(" + ruff/format/mypy/pytest с покрытием, плюс два headless-прогона сценариев) и "), c("package-windows"), R(" (Windows — не-GUI тесты и сборка бандла PyInstaller как артефакта). GUI-смоук собранного бандла автоматизировать на headless-раннере нельзя, поэтому он остаётся задокументированным ручным шагом (см. "), flink("docs/packaging.md"), R(").")]));
  K.push(H(2,"9.1 Документация собирается локально","s_deliv_docs"));
  K.push(para([R("Джоб "), c("docs"), R(" и публикация на GitHub Pages "), R("отменены", { bold: true }), R(", а не «временно отключены»: программно включить Pages из CI нельзя (токен рабочего процесса не создаёт сайт), а токен с админскими правами в CI — недопустимый риск; плюс нежелательная публичная экспозиция внутренних документов. Поэтому обе ветки документации собираются "), R("на машине разработчика", { bold: true }), R(", а собранные файлы "), R("не версионируются", { bold: true }), R(" — в репозитории живёт только исходник:")]));
  K.push(...code([
    "# руководство (этот документ): модульный ESM-исходник → .docx",
    "cd docs/manual && npm ci",
    "python3 figures/build_figures.py     # рисунки (matplotlib, формулы из базы)",
    "node build.mjs                       # сборка → перенумерация закладок → валидация",
    "",
    "# справочно-теоретический слой docs/theory/ → офлайновый HTML",
    "python docs/theory/tools/fetch_mathjax.py   # однократно: вендорит MathJax (пин версии+sha512)",
    "python docs/theory/tools/build_theory.py    # блоки → docs/theory/_build/*.html",
  ]));
  K.push(para([R("Порядок для теории существен: без вендоренного MathJax сборка "), R("отказывает с инструкцией", { bold: true }), R(", а не молча выдаёт документ без формул. Не версионируются: "), c(".docx"), R(", "), c("node_modules/"), R(", "), c("figures/*.png"), R(", "), c("docs/theory/_build/"), R(" и вендоренный MathJax — все они в "), c(".gitignore"), R(". Валидация "), c(".docx"), R(" встроена в "), c("build.mjs"), R(": один запуск даёт валидный файл (проверяются архив, XML и уникальность идентификаторов закладок).")]));
  K.push(links([ xr("Тестирование → §8","ch8"), sep(), flink("packaging/"), sep(), flink("docs/manual/"), sep(), flink("docs/theory/"), sep(), R("база знаний: документ 14 §1, 15 §6/§9.5",{size:18,color:MUT}) ]));


  return K;
}
