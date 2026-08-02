// docs/manual/sections/ui_cli.mjs
// AUTO-SLICED (S-DOC-2) from manual_build_doc.js lines 165..192; content verbatim.
// Owns its bookmarks/links. Edit this module only; build.mjs orders it via manifest.mjs.
import {
  R, c, P, para, H, xr, flink, ext, links, sep, code, cap, bullets, nums, img, tbl, fml,
  AlignmentType, Paragraph, TextRun, PageBreak,
  REPO, FIG, INK, MUT, ACC, ACCD, LINE, CW,
} from "../shared.mjs";

export function section() {
  const K = [];
  K.push(H(2,"3.2 Командная строка (CLI)","s_ui_cli"));
  K.push(para([R("CLI ("), flink("src/optivibe/cli/main.py","cli/main.py"), R(") даёт шесть подкоманд — весь функционал доступен без GUI (headless-паритет):")]));
  K.push(...code([
    "optivibe run     <scenario.yaml>             # прямой+обратный прогон, доминанты/метрики",
    "optivibe report  <scenario.yaml> [--figures] # бюджет «истина vs восстановление», NEA, ожидаемые пики",
    "optivibe sweep   <spec.yaml> [--out DIR]     # развёртки / Монте-Карло (npz + фигуры)",
    "optivibe analyze <spec.yaml> [--record F]    # анализ записи прибора: фототок → метрики, минуя forward",
    "optivibe compare <spec.yaml> [--provenance F]# сравнение цепочек ЦОС на одном входе (§3.7)",
    "optivibe ingest  <sidecar.yaml>... [--dry-run]  # измерения → композиция двойника (§3.5)",
  ]));
  K.push(para([R("Коды выхода: "), c("0"), R(" — успех, "), c("2"), R(" — ошибка входа (нечитаемый файл, конфликт значений, сработавший гвард композиции). Три последние подкоманды раскрыты в своих разделах: "), xr("§3.3 Импорт данных","s_ui_data"), R(", "), xr("§3.5 Ввод измеренных параметров","s_ingest"), R(", "), xr("§3.7 Стенд сравнения","s_cmp"), R(".")]));
  K.push(para("Пример сценария (упрощённо): вариант B, синус 1g на 200 Гц, физический детектор и калиброванный DSP — восстановление вибрации."));
  K.push(...code([
    "# examples/recover_sine.yaml",
    "variant: B",
    "seed: 12345",
    "excitation:",
    "  kind: sine",
    "  axis: x",
    "  fs_hz: 50000",
    "  duration_s: 0.5",
    "  frequency_hz: 200.0",
    "  amplitude_g: 1.0",
    "stages:                 # выбор реализаций стадий из реестров",
    "  optics: cylinder",
    "  detector: photodiode",
    "  dsp: standard",
    "detector: { balanced: true, reference_arm: matched }",
    "dsp: { integrator: frequency, calibration: ideal }",
  ]));
  K.push(para([R("Поля и допустимые значения определяются pydantic-моделями конфигурации ("), flink("src/optivibe/core/config/models.py","core/config/models.py"), R("); неизвестные/конфликтующие значения дают понятную ошибку. Структура сценария — глава "), xr("5.5","s_arch_config"), R(".")]));

  return K;
}
