// docs/manual/sections/backlog.mjs
// AUTO-SLICED (S-DOC-2) from manual_build_doc.js lines 748..761; content verbatim.
// Owns its bookmarks/links. Edit this module only; build.mjs orders it via manifest.mjs.
import {
  R, c, P, para, H, xr, flink, ext, links, sep, code, cap, bullets, nums, img, tbl, fml,
  AlignmentType, Paragraph, TextRun, PageBreak,
  REPO, FIG, INK, MUT, ACC, ACCD, LINE, CW,
} from "../shared.mjs";

export function section() {
  const K = [];
  K.push(H(1,"12. Открытые вопросы и направления расширения","ch12"));
  K.push(para("Эти пункты не влияют на работу v1, но полезны для понимания границ модели и планов (база знаний: документы 13 §открытые вопросы, 14 §8)."));
  K.push(...bullets([
    [R("Анализатор реального выхода прибора: ",{bold:true}), R("обработка настоящих захватов фотодетектора напрямую (минуя forward) — превращает софт в анализатор реального датчика.")],
    [R("Реал-тайм «осциллограф»: ",{bold:true}), R("потоковый режим сверх воспроизведения батча.")],
    [R("Полная 3D/вектор-инверсия и live-калибровка по стенду: ",{bold:true}), R("швы заложены (оси A/D сигнатуры чувствительности).")],
    [R("Вычислимая модель Q(L), формы отражателя sphere/plane/wedge, parquet-вывод: ",{bold:true}), R("точки расширения за реестрами.")],
    [R("Ускорение (numba/torch): ",{bold:true}), R("по результатам профилирования, за интерфейсом реализации.")],
  ]));
  K.push(para([R("Дорожная карта по чатам разработки (S0–S9) и журнал решений — документы 12 и 13 базы знаний. Текущий слепок репозитория — документ 14.")]));
  K.push(links([ xr("Семейство и аналитика → §4.6 / §6.7","s_phys_family"), sep(), flink("docs/"), sep(), ext("репозиторий проекта","https://github.com/SuperMarioFernandess/optivibe") ]));


  return K;
}
