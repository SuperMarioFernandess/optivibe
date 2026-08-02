// docs/manual/manifest.mjs
// Order of section modules -> chapters of OptiVibe_Руководство.docx (doc 15 §3/§6).
// build.mjs imports each in this order, calls its section(), and concatenates the
// blocks after the cover/TOC front matter. Edit ORDER here to move a chapter; edit
// the module file to change its content. The two S8–S9 additions sit next to the
// chapter they extend (composition under ch3 GUI; reflector family under ch4),
// so chapter numbering stays stable and edits stay local (DoD: one module at a time).
// Разделы 3.5–3.7 (S-13, O-SW-03, S-22) добавлены В КОНЕЦ главы 3 намеренно: вставка
// «по смыслу» (ingest рядом с импортом данных) сдвинула бы номера 3.4+ и все ссылки
// на них — стабильность номеров дороже тематической близости, связь даётся ссылками.
export const ORDER = [
  "00_intro.mjs",          // 1. Введение
  "01_quickstart.mjs",     // 2. Быстрый старт
  "ui_gui.mjs",            // 3. Взаимодействие с ПО — лид + 3.1 GUI
  "ui_cli.mjs",            //                         3.2 CLI
  "ui_data.mjs",           //                         3.3 Импорт данных (S8 расширён)
  "config_subsystems.mjs", //                         3.4 Композиция подсистем (S9-A, NEW)
  "ui_ingest.mjs",         //                         3.5 Ввод измеренных параметров (S-13, NEW)
  "ui_live.mjs",           //                         3.6 Живой режим / осциллограф (O-SW-03, NEW)
  "ui_compare.mjs",        //                         3.7 Стенд сравнения цепочки ЦОС (S-22, NEW)
  "physmodel.mjs",         // 4. Физическая модель
  "reflectors.mjs",        //                         4.7 Семейство отражателей (S9-B, NEW)
  "architecture.mjs",      // 5. Архитектура
  "algorithms.mjs",        // 6. Алгоритмы
  "implementation.mjs",    // 7. Реализация в коде
  "testing.mjs",           // 8. Запуск и тестирование
  "delivery.mjs",          // 9. Поставка
  "map.mjs",               // 10. Карта связей
  "glossary.mjs",          // 11. Глоссарий
  "backlog.mjs",           // 12. Открытые вопросы
];
