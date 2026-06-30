// docs/manual/sections/01_quickstart.mjs
// AUTO-SLICED (S-DOC-2) from manual_build_doc.js lines 93..122; content verbatim.
// Owns its bookmarks/links. Edit this module only; build.mjs orders it via manifest.mjs.
import {
  R, c, P, para, H, xr, flink, ext, links, sep, code, cap, bullets, nums, img, tbl, fml,
  AlignmentType, Paragraph, TextRun, PageBreak,
  REPO, FIG, INK, MUT, ACC, ACCD, LINE, CW,
} from "../shared.mjs";

export function section() {
  const K = [];
  K.push(H(1,"2. Быстрый старт","ch2"));
  K.push(para([R("Требуется Python ≥ 3.11 и менеджер окружения "), c("uv"), R(" (или pip). Зависимости разнесены по extras: "), c("dev"), R(", "), c("gui"), R(", "), c("docs"), R(", "), c("io-formats"), R(" (форматы приборов), "), c("packaging"), R(".")]));
  K.push(H(3,"Установка","s_qs_install"));
  K.push(...code([
    "# клонировать репозиторий",
    "git clone https://github.com/SuperMarioFernandess/optivibe && cd optivibe",
    "",
    "# установить с нужными extras (воспроизводимо из uv.lock)",
    "uv sync --locked --extra dev --extra gui --extra io-formats",
    "#  или через pip:",
    "pip install -e \".[dev,gui,io-formats]\"",
  ]));
  K.push(H(3,"Первый прогон (CLI)","s_qs_run"));
  K.push(para([R("Сценарий — это YAML-файл, описывающий воспроизводимый прогон. Запуск приёмочного сценария "), c("hello.yaml"), R(":")]));
  K.push(...code([
    "optivibe run examples/hello.yaml",
    "#  → dominant 120.000 Hz, exit 0",
    "",
    "# восстановление вибрации (полный тракт: детектор + DSP):",
    "optivibe run examples/recover_sine.yaml      # 1g@200 Гц восстановлен в пределах NEA",
    "optivibe report examples/recover_sine.yaml   # бюджет «истина vs восстановление» + NEA",
    "optivibe sweep examples/nea_vs_L.yaml --out out/L   # развёртка NEA(L)",
  ]));
  K.push(H(3,"Запуск десктоп-приложения","s_qs_gui"));
  K.push(...code([ "optivibe-gui" ]));
  K.push(para([R("Откроется окно с панелью управления и вкладками Live / Report / Sweeps / Monte-Carlo. Подробно — глава "), xr("3 «Взаимодействие с ПО»","ch3"), R(".")]));
  K.push(links([ xr("Архитектура и сценарии → §5","ch5"), sep(), xr("Команды-ворота и тесты → §8","ch8"), sep(), flink("examples/hello.yaml"), sep(), flink("README.md") ]));


  return K;
}
