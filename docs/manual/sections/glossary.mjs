// docs/manual/sections/glossary.mjs
// AUTO-SLICED (S-DOC-2) from manual_build_doc.js lines 724..746; content verbatim.
// Owns its bookmarks/links. Edit this module only; build.mjs orders it via manifest.mjs.
import {
  R, c, P, para, H, xr, flink, ext, links, sep, code, cap, bullets, nums, img, tbl, fml,
  AlignmentType, Paragraph, TextRun, PageBreak,
  REPO, FIG, INK, MUT, ACC, ACCD, LINE, CW,
} from "../shared.mjs";

export function section() {
  const K = [];
  K.push(H(1,"11. Глоссарий величин и обозначений","ch11"));
  K.push(tbl(
    ["Обозначение","Смысл","Единицы / значение"],
    [
      ["L","длина консоли (волокна)","мм; A=5, B=2, C=1.41"],
      ["f₁","первая собственная частота","≈ 100/L² кГц"],
      ["H_lat(f)","поперечная передаточная (ускорение→смещение торца)","нм/g; H^QS≈0.0384·L⁴"],
      ["D(f)","резонансный множитель осциллятора","1/[1−(f/f₁)²+i(f/f₁)/Q]"],
      ["Q","добротность (демпфирование)","безразм.; пик ≈ 2600"],
      ["η, η₀","эффективность ввода и её рабочая точка","безразм.; η₀≈0.25"],
      ["Δx₀ (bias)","смещение рабочей точки","мкм; ≈2 (A) / 0.99 (B/C/D)"],
      ["R_c","радиус кривизны цилиндра","мкм; 62.5 (A) / 31 (B/C/D)"],
      ["R_c","радиус кривизны (cylinder/sphere)","мкм; None для plane/wedge"],
      ["α_w (wedge_angle)","встроенный угол клина (wedge)","рад; |α_w|≤0.15; возврат 2·α_w"],
      ["ρ (metallization_rho)","отражательная способность зеркала","безразм.; ≈0.98"],
      ["shape","форма отражателя","cylinder/sphere/plane/wedge (реестр)"],
      ["пресет / композиция","блок подсистемы / SystemConfig","configs/presets/<sub>, configs/user"],
      ["∂η/∂Δx","оптический наклон (эффективный)","≈ −0.16/мкм"],
      ["S, I_DC, I_AC","сигнал/постоянный/переменный фототок","А; S=ℛP(R₁+ρη)"],
      ["s_target","сквозная чувствительность","А/(м/с²)"],
      ["NEA","шумовой пол, приведённый ко входу","мкg/√Гц"],
      ["FS","полная шкала (проектный вход)","g"],
      ["(Δx,Δy,Δz,θx,θy)","вектор состояния торца (ICD)","м, рад"],
    ],
    [2150,4350,2860]
  ));
  K.push(para([R("Ключи реестров и умолчания — "), xr("§5.4","s_arch_registry"), R("; структура пакетов — "), xr("§5.2","s_arch_tree"), R("; варианты A/B/C/D — "), xr("§4.6","s_phys_family"), R(". Полные обозначения и константы — база знаний, документ 01.")]));

  return K;
}
