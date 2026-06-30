// docs/manual/sections/testing.mjs
// AUTO-SLICED (S-DOC-2) from manual_build_doc.js lines 645..680; content verbatim.
// Owns its bookmarks/links. Edit this module only; build.mjs orders it via manifest.mjs.
import {
  R, c, P, para, H, xr, flink, ext, links, sep, code, cap, bullets, nums, img, tbl, fml,
  AlignmentType, Paragraph, TextRun, PageBreak,
  REPO, FIG, INK, MUT, ACC, ACCD, LINE, CW,
} from "../shared.mjs";

export function section() {
  const K = [];
  K.push(H(1,"8. Запуск и тестирование","ch8"));
  K.push(para("Раздел для освоения и обучения других: как проверить, что сборка исправна, и как устроена стратегия тестов, доказывающая согласованность кода с физикой."));
  K.push(H(2,"8.1 Команды-ворота (DoD)","s_test_gates"));
  K.push(...code([
    "uv sync --locked --extra dev --extra gui --extra io-formats",
    "ruff check . && ruff format --check .          # линт + формат",
    "mypy                                            # строгая типизация (74 файла)",
    "QT_QPA_PLATFORM=offscreen pytest --cov=optivibe --cov-fail-under=85",
    "#  → 315 passed, покрытие ядра 91 % (gui/ исключён из порога)",
    "optivibe run examples/hello.yaml                # dominant 120.000 Hz, exit 0",
  ]));
  K.push(H(2,"8.2 Приёмочные сценарии","s_test_accept"));
  K.push(para("Набор сценариев демонстрирует и проверяет работу подсистем. Ожидаемые доминирующие частоты — стабильные реперы регресса."));
  K.push(tbl(
    ["Сценарий","Что демонстрирует","Ожидаемо"],
    [
      ["hello","базовый прогон (stub-оптика и stub-детектор)","120.000 Гц"],
      ["multitone / sweep / random / shock","генераторы возбуждения","120 / 14.5 / 597.5 / 24995 Гц"],
      ["replay_csv / replay_wav / replay_tdms","импорт реальных данных","50 / 440 / 50 Гц"],
      ["resonance_sweep","резонанс варианта D","5005 Гц (= f₁ D)"],
      ["cross_axis","перекрёстная ось y (квадратичный канал)","240 Гц (= 2f)"],
      ["linearity_ramp","нелинейность у границы линейности","≈ f₁ B"],
      ["noise_floor / sine_with_noise","детектор: пол шума и сигнал над ним","σ_i ≈ 2.3 нА / 200 Гц"],
      ["recover_sine","полное восстановление (1g восстановлен)","200 Гц, в пределах NEA"],
      ["vibration_severity","виброоценка по ISO","50 Гц, зона C"],
    ],
    [2350,4710,2300]
  ));
  K.push(H(2,"8.3 Стратегия тестов","s_test_strategy"));
  K.push(...bullets([
    [R("golden против аналитики: ",{bold:true}), R("сверка с реперами базы — f₁ ≈ 100/L², эффективный наклон оптики, NEA-полка и мастер-закон NEA ∝ L⁻⁴, s_target бит-в-бит для умолчаний v1.")],
    [R("property-тесты (hypothesis): ",{bold:true}), R("Парсеваль для спектра, round-trip интегрирования a→x→a, линейность калибровки, детерминизм по сиду.")],
    [R("регресс и смоук: ",{bold:true}), R("импорт всех модулей, неизменность доминант и умолчаний; GUI-смоук (pytest-qt, offscreen) с доказательством расчёта вне UI-потока.")],
  ]));
  K.push(links([ xr("Команды и приёмки → §3.2","s_ui_cli"), sep(), flink("tests/"), sep(), R("база знаний: документ 10 §10, 11 §7",{size:18,color:MUT}) ]));

  return K;
}
