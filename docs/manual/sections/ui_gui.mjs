// docs/manual/sections/ui_gui.mjs
// AUTO-SLICED (S-DOC-2) from manual_build_doc.js lines 124..164; content verbatim.
// Owns its bookmarks/links. Edit this module only; build.mjs orders it via manifest.mjs.
import {
  R, c, P, para, H, xr, flink, ext, links, sep, code, cap, bullets, nums, img, tbl, fml,
  AlignmentType, Paragraph, TextRun, PageBreak,
  REPO, FIG, INK, MUT, ACC, ACCD, LINE, CW,
} from "../shared.mjs";

export function section() {
  const K = [];
  K.push(H(1,"3. Взаимодействие с ПО","ch3"));
  K.push(para([R("ПО доступно двумя путями с общим ядром: десктоп-приложение (PySide6 + PyQtGraph) для интерактивной работы и показа, и командная строка (CLI) для воспроизводимых прогонов и автоматизации. Оба используют один и тот же конвейер "), c("pipeline"), R(" и аналитику "), c("analysis"), R(" — поэтому результат идентичен (headless-паритет).")]));

  K.push(H(2,"3.1 Десктоп-приложение (GUI)","s_ui_gui"));
  K.push(para([R("Запуск — командой "), c("optivibe-gui"), R(" (точка входа "), flink("src/optivibe/gui/app.py","gui/app.py"), R("). GUI — тонкая оболочка над ядром: внутри нет физики и DSP, только сбор конфигурации, запуск расчёта в рабочем потоке и отображение результатов.")]));
  K.push(H(3,"Главное окно: панель управления и вкладки","s_ui_window"));
  K.push(para([R("Окно ("), flink("src/optivibe/gui/main_window.py","gui/main_window.py"), R(") состоит из левой панели управления и области вкладок с кнопками действий внизу.")]));
  K.push(tbl(
    ["Элемент окна","Назначение"],
    [
      ["Панель управления (слева)","Композиция датчика в формах подсистем (SystemBuilderPanel; источник/волокно/консоль/отражатель/детектор), конструктор возбуждения, переключатели стадий, сид. A/B/C/D — стартовые композиции, которые далее правятся; из форм собирается SystemConfig → сценарий. Подробно — §3.4."],
      ["Вкладка Live","Живые графики PyQtGraph: анимация изгиба консоли, вход a(t) против восстановления, сигнал детектора, скорость/перемещение, спектр, NEA(f) с разложением."],
      ["Вкладка Report","Встроенные фигуры matplotlib: бюджет «истина vs восстановление» (a/v/x), спектр, спектрограмма, NEA(f) с компонентами."],
      ["Вкладка Sweeps","Карты развёрток: NEA и чувствительность по {L, R_c, P, bias, FS, вариант}; отклик/THD по амплитуде 0.1g→>50g."],
      ["Вкладка Monte-Carlo","Гистограммы и боксплоты статистики NEA и перекрёстной чувствительности по разбросам допусков."],
      ["Кнопка Run","Запускает сценарий (forward+inverse) и обновляет вкладку Live."],
      ["Кнопка Report","Считает бюджет ошибки и NEA, рисует фигуры на вкладках Report."],
      ["Кнопка Cancel","Кооперативно отменяет текущий расчёт (поздний результат отбрасывается)."],
      ["Кнопка Export","Сохраняет результаты (npz) и фигуры (PNG/PDF)."],
    ],
    [2700,6660]
  ));
  K.push(H(3,"Конструктор возбуждения и переключатели стадий","s_ui_controls"));
  K.push(para([R("Конструктор возбуждения ("), flink("src/optivibe/gui/widgets/excitation_builder.py","gui/widgets/excitation_builder.py"), R(") задаёт тип входного сигнала (синус, мультитон, свип, случайный по PSD, удар) или импорт файла (CSV/WAV/TDMS/UFF/MAT/HDF5). Переключатели стадий ("), flink("src/optivibe/gui/widgets/control_panel.py","gui/widgets/control_panel.py"), R(") наглядно показывают слои физики — их можно включать по отдельности:")]));
  K.push(tbl(
    ["Стадия","Ключи (выбор)","Умолчание","Что меняет"],
    [
      ["Механика","modal / modal_time / stub","modal","Частотный или временной решатель H_lat(f); stub — тождество для регресса."],
      ["Оптика","cylinder / stub","cylinder","Физическая модель η(Δx) с цилиндром или линейный стаб."],
      ["Детектор","stub / photodiode","stub","photodiode добавляет фототок, шумы (дробовой/RIN/Джонсон), опорный канал, АЦП."],
      ["DSP","stub / standard","stub","standard — калиброванное восстановление a→v→x, спектры, ISO, NEA."],
      ["Чувствительность","static / operating_point / nonlinear_curve","static","Модель s_target: статическая, по рабочей точке, нелинейная (для >50g)."],
      ["Интегратор","frequency / time","frequency","a→v→x: спектральное 1/(jω) или временное с детрендом."],
    ],
    [1750,2750,1360,3500]
  ));
  K.push(para([R("Замечание: детектор и DSP по умолчанию — "), c("stub"), R(" (идеальные), чтобы реалистичный шум не «прятал» демонстрационные эффекты; для восстановления вибрации включите "), c("photodiode"), R(" и "), c("standard"), R(" (как в "), flink("examples/recover_sine.yaml"), R(").")]));
  K.push(H(3,"Модель потоков: расчёт вне UI","s_ui_threads"));
  K.push(para([R("Тяжёлые прогоны выполняются в рабочем потоке, поэтому окно не «зависает». Qt-free задачи ("), flink("src/optivibe/gui/workers/jobs.py","gui/workers/jobs.py"), R(": "), c("ScenarioJob"), R(", "), c("ReportJob"), R(", "), c("SweepJob"), R(", "), c("MonteCarloJob"), R(") вызывают только "), c("pipeline"), R("/"), c("analysis"), R(" и исполняются общим "), c("JobWorker(QObject)"), R(" на "), c("QThread"), R(" под управлением "), c("JobController"), R(" ("), flink("src/optivibe/gui/controllers/job_controller.py","gui/controllers/job_controller.py"), R("). Прогресс/результат/ошибка приходят сигналами; отмена кооперативная. Инвариант «расчёт идёт вне UI-потока» закреплён тестом "), c("test_computation_runs_off_the_gui_thread"), R(".")]));
  K.push(links([ xr("Реализация воркера → §7.10","s_impl_gui"), sep(), xr("Архитектура GUI → §5","ch5"), sep(), flink("src/optivibe/gui/"), sep(), R("база знаний: документ 09 §9",{size:18,color:MUT}) ]));

  return K;
}
