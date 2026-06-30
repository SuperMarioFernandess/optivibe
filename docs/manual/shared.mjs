// docs/manual/shared.mjs
// Encapsulates docx-js for the modular manual (doc 15 §6): low-level block
// helpers (H/P/para/code/tbl/xr/flink/ext/links/sep/cap/bullets/nums/img),
// the paragraph/character styles, the numbering configs, the cover + TOC
// front-matter, and `buildDocument(children)` that wraps the assembled block
// list into a paged Document and serialises it. Helpers are byte-faithful to
// the S-DOC-1 monolith (`manual_build_doc.js`); only `require` -> ESM `import`
// and the figure path (now repo-relative `figures/`) changed, so the assembled
// output is the same document. Each `sections/*.mjs` module imports the helpers
// it needs and exports `section() -> Block[]`; `build.mjs` concatenates them in
// `manifest.mjs` order and calls `buildDocument`.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, LevelFormat, ExternalHyperlink, InternalHyperlink, Bookmark,
  TableOfContents, HeadingLevel, BorderStyle, WidthType, ShadingType, PageBreak,
  Footer, PageNumber, TabStopType, TabStopPosition, UnderlineType,
} from "docx";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------- constants (monolith parity) ----------
export const REPO = "https://github.com/SuperMarioFernandess/optivibe/blob/main/";
// Figures live next to the build now (was an absolute scratch dir in S-DOC-1).
export const FIG = path.join(__dirname, "figures") + path.sep;
export const INK = "1B2631", MUT = "5D6D7E", ACC = "2E5496", ACCD = "1F3864", LINE = "BFC9D4";
export const CW = 9360; // content width US Letter, 1" margins

// ---------- inline / block helpers (verbatim behaviour) ----------
export const R = (text, o = {}) => new TextRun({ text, size: o.size || 21, bold: !!o.bold,
  italics: !!o.italics, color: o.color || INK, font: o.font || "Calibri", break: o.break });
export const c = (text) => new TextRun({ text, font: "Consolas", size: 18, color: "7D3C98" });
export const P = (children, o = {}) => new Paragraph({
  spacing: { before: o.before ?? 20, after: o.after ?? 120, line: 276, lineRule: "auto" },
  alignment: o.align, indent: o.indent,
  children: Array.isArray(children) ? children : [children] });
export const para = (text, o = {}) => P([R(text, o)], o);
export function H(level, text, bm) {
  const h = level === 1 ? HeadingLevel.HEADING_1 : level === 2 ? HeadingLevel.HEADING_2
    : HeadingLevel.HEADING_3;
  return new Paragraph({ heading: h,
    children: [new Bookmark({ id: bm, children: [new TextRun(text)] })] });
}
export const xr = (label, anchor) => new InternalHyperlink({ anchor,
  children: [new TextRun({ text: label, style: "Hyperlink", size: 21 })] });
export const flink = (p, label) => new ExternalHyperlink({ link: REPO + p,
  children: [new TextRun({ text: label || p, style: "Hyperlink", font: "Consolas", size: 18 })] });
export const ext = (label, url) => new ExternalHyperlink({ link: url,
  children: [new TextRun({ text: label, style: "Hyperlink", size: 21 })] });
export function links(parts) {
  return new Paragraph({ spacing: { before: 40, after: 170 }, indent: { left: 120 },
    children: [new TextRun({ text: "↳ связи: ", size: 18, bold: true, color: ACC }), ...parts] });
}
export const sep = () => new TextRun({ text: "   ·   ", size: 18, color: MUT });
export function code(lines) {
  return lines.map((ln) => new Paragraph({ shading: { type: ShadingType.CLEAR, fill: "F4F5F7" },
    spacing: { before: 0, after: 0, line: 248, lineRule: "auto" }, indent: { left: 160 },
    children: [new TextRun({ text: ln === "" ? " " : ln, font: "Consolas", size: 17, color: INK })] }));
}
export const cap = (text) => new Paragraph({ spacing: { before: 50, after: 150 },
  children: [new TextRun({ text, italics: true, size: 18, color: MUT })] });
// Centred formula line. In the S-DOC-1 monolith this was a local `const fml`
// defined in the ch3 block but used by the ch4 physics formulas; once chapters
// became separate modules it had to move into shared so both can import it.
export const fml = (t) => new Paragraph({ alignment: AlignmentType.CENTER,
  spacing: { before: 80, after: 120 },
  children: [new TextRun({ text: t, italics: true, size: 24, color: ACCD, font: "Calibri" })] });
export function bullets(items, o = {}) {
  return items.map((it) => new Paragraph({ numbering: { reference: "bul", level: 0 },
    spacing: { before: 10, after: 60, line: 268, lineRule: "auto" },
    children: Array.isArray(it) ? it : [R(it, o)] }));
}
export function nums(items, ref) {
  return items.map((it) => new Paragraph({ numbering: { reference: ref, level: 0 },
    spacing: { before: 10, after: 60, line: 268, lineRule: "auto" },
    children: Array.isArray(it) ? it : [R(it)] }));
}
export function img(file, w, h, caption) {
  const out = [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 30 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(FIG + file),
      transformation: { width: w, height: h },
      altText: { title: caption || "figure", description: caption || "figure", name: caption || "figure" } })] })];
  if (caption) out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 170 },
    children: [new TextRun({ text: caption, italics: true, size: 17, color: MUT })] }));
  return out;
}
export function tbl(header, rows, widths) {
  const tw = widths.reduce((a, b) => a + b, 0);
  const bd = { style: BorderStyle.SINGLE, size: 1, color: LINE };
  const bds = { top: bd, bottom: bd, left: bd, right: bd };
  const mk = (content, w, head, ri) => {
    const runs = Array.isArray(content) ? content
      : [new TextRun({ text: String(content), bold: !!head, color: head ? "FFFFFF" : INK, size: 18 })];
    return new TableCell({ borders: bds, width: { size: w, type: WidthType.DXA }, verticalAlign: "center",
      shading: { type: ShadingType.CLEAR, fill: head ? ACC : (ri % 2 ? "F4F7FB" : "FFFFFF") },
      margins: { top: 55, bottom: 55, left: 110, right: 110 },
      children: [new Paragraph({ spacing: { after: 0, line: 252, lineRule: "auto" }, children: runs })] });
  };
  const head = new TableRow({ tableHeader: true, children: header.map((h, i) => mk(h, widths[i], true, 0)) });
  const body = rows.map((r, ri) => new TableRow({ children: r.map((cc, i) => mk(cc, widths[i], false, ri)) }));
  return new Table({ width: { size: tw, type: WidthType.DXA }, columnWidths: widths, rows: [head, ...body] });
}

// ---------- front matter (cover + TOC), was "part A" of the monolith ----------
export function frontMatter() {
  const K = [];
  K.push(new Paragraph({ spacing: { before: 1600, after: 60 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "OptiVibe", bold: true, size: 72, color: ACCD, font: "Calibri" })] }));
  K.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
    children: [new TextRun({ text: "Цифровой двойник волоконно-оптического датчика вибрации", size: 30, color: ACC, font: "Calibri" })] }));
  K.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
    children: [new TextRun({ text: "Руководство по продукту: физическая модель, алгоритмы, реализация и работа с ПО", size: 24, italics: true, color: MUT })] }));
  K.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 30 },
    children: [new TextRun({ text: "Версия документа 1.1  ·  ПО v1 (этапы S0–S9)  ·  30 июня 2026", size: 20, color: INK })] }));
  K.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 30 },
    children: [new TextRun({ text: "Аудитория: пользователи, инженеры по вибрации, оптике и механике, новые члены команды", size: 20, color: MUT })] }));
  K.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 30 },
    children: [new TextRun({ text: "Репозиторий: ", size: 20, color: MUT }),
      new ExternalHyperlink({ link: "https://github.com/SuperMarioFernandess/optivibe",
        children: [new TextRun({ text: "github.com/SuperMarioFernandess/optivibe", style: "Hyperlink", size: 20 })] })] }));
  K.push(new Paragraph({ children: [new PageBreak()] }));
  K.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Содержание")] }));
  K.push(new Paragraph({ spacing: { after: 120 },
    children: [new TextRun({ text: "После открытия в Word обновите оглавление: правый клик по нему → «Обновить поле» → «Обновить целиком».", italics: true, size: 18, color: MUT })] }));
  K.push(new TableOfContents("Содержание", { hyperlink: true, headingStyleRange: "1-3" }));
  K.push(new Paragraph({ children: [new PageBreak()] }));
  return K;
}

// ---------- document assembly (was the "СБОРКА ДОКУМЕНТА" tail) ----------
export function buildDocument(children) {
  return new Document({
    creator: "OptiVibe",
    title: "OptiVibe — Руководство по продукту",
    description: "Руководство: физическая модель, алгоритмы, реализация, работа с ПО",
    styles: {
      default: { document: { run: { font: "Calibri", size: 21, color: INK } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 32, bold: true, font: "Calibri", color: ACCD },
          paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0, keepNext: true } },
        { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 26, bold: true, font: "Calibri", color: ACC },
          paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1, keepNext: true } },
        { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 23, bold: true, font: "Calibri", color: "34679E" },
          paragraph: { spacing: { before: 180, after: 90 }, outlineLevel: 2, keepNext: true } },
      ],
      characterStyles: [
        { id: "Hyperlink", name: "Hyperlink", basedOn: "DefaultParagraphFont",
          run: { color: "0563C1", underline: { type: UnderlineType.SINGLE, color: "0563C1" } } },
      ],
    },
    numbering: {
      config: [
        { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
          alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 560, hanging: 280 } } } }] },
        { reference: "n1", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
          alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 560, hanging: 280 } } } }] },
      ],
    },
    sections: [{
      properties: { page: { size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
      footers: {
        default: new Footer({ children: [new Paragraph({
          tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
          border: { top: { style: BorderStyle.SINGLE, size: 4, color: "D5DBE0", space: 6 } },
          children: [
            new TextRun({ text: "OptiVibe — Руководство по продукту", size: 16, color: MUT }),
            new TextRun({ text: "\tстр. ", size: 16, color: MUT }),
            new TextRun({ children: [PageNumber.CURRENT], size: 16, color: MUT }),
            new TextRun({ text: " / ", size: 16, color: MUT }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 16, color: MUT }),
          ] })] }),
      },
      children,
    }],
  });
}

export { Packer };
// Raw docx primitives re-exported so sliced bodies that used them directly in
// the monolith keep working without importing 'docx' themselves.
export { AlignmentType, Paragraph, TextRun, PageBreak };
