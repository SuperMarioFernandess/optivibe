// docs/manual/build.mjs
// Build entry point (doc 15 §6): `node docs/manual/build.mjs [out.docx]`.
// Generates figures if missing, imports each section module in manifest order,
// concatenates their blocks after the cover/TOC front matter, and serialises the
// Document. The heavy docx-js details live in shared.mjs; modules carry content.
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { frontMatter, buildDocument, Packer, FIG } from "./shared.mjs";
import { ORDER } from "./manifest.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT = process.argv[2] || path.join(__dirname, "OptiVibe_Руководство.docx");

// 1) Ensure figures exist (build_figures.py writes into figures/).
const needFigs = !fs.existsSync(path.join(FIG, "mode.png"));
if (needFigs) {
  const py = spawnSync("python3", [path.join(__dirname, "figures", "build_figures.py")],
    { stdio: "inherit" });
  if (py.status !== 0) { console.error("figure generation failed"); process.exit(1); }
}

// 2) Assemble: front matter + each module's section() in manifest order.
const children = [...frontMatter()];
let count = children.length;
for (const file of ORDER) {
  const mod = await import(path.join(__dirname, "sections", file));
  if (typeof mod.section !== "function") {
    console.error(`section() missing in ${file}`); process.exit(1);
  }
  const blocks = mod.section();
  children.push(...blocks);
  console.log(`+ ${file.padEnd(24)} ${blocks.length} blocks  (total ${(count += blocks.length)})`);
}

// 3) Serialise.
const doc = buildDocument(children);
const buf = await Packer.toBuffer(doc);
fs.writeFileSync(OUT, buf);
console.log(`written: ${buf.length} bytes -> ${OUT}`);

// 4) Post-process + validate (doc 15 §6): docx-js can emit duplicate numeric
// bookmark ids (the TOC collides with the first bookmark); validate.py --fix
// renumbers start/end pairs in place and re-checks. A non-zero exit fails build.
const vp = spawnSync("python3", [path.join(__dirname, "validate.py"), OUT, "--fix"],
  { stdio: "inherit" });
if (vp.status !== 0) { console.error("validation failed"); process.exit(1); }
