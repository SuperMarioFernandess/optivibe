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
  K.push(para([R("CI ("), flink(".github/workflows/ci.yml",".github/workflows/ci.yml"), R(") содержит три джоба: "), c("quality"), R(" ("), c("uv sync --locked"), R(" + ruff/mypy/pytest), "), c("package-windows"), R(" (сборка артефакта-бандла) и "), c("docs"), R(" ("), c("mkdocs build --strict"), R(" + публикация на GitHub Pages). Документация-сайт содержит руководство пользователя, импорт данных, карту «физика → модуль» и страницу упаковки. GUI-смоук собранного бандла — ручной (см. "), flink("docs/packaging.md"), R(").")]));
  K.push(links([ xr("Тестирование → §8","ch8"), sep(), flink("packaging/"), sep(), flink("docs/user-guide.md"), sep(), R("база знаний: документ 14 §1",{size:18,color:MUT}) ]));


  return K;
}
