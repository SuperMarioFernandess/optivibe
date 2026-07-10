// docs/manual/build.mjs
// Build entry point (doc 15 §6): `node docs/manual/build.mjs [out.docx]`.
// Generates figures if missing, imports each section module in manifest order,
// concatenates their blocks after the cover/TOC front matter, serialises the
// Document, then post-processes + validates it.
//
// Cross-platform notes:
//  * dynamic import() of an absolute path must be a file:// URL on Windows
//    (a bare "D:\..." path is read as URL scheme "d:") -> pathToFileURL().
//  * the real Python may be `py -3` (Windows launcher). Plain python/python3 on
//    Windows are often Microsoft Store *alias stubs* that print "Python" and
//    exit 0 without running anything -- so we probe each candidate and only keep
//    one that actually reports Python 3. `py -3` also overrides the validate.py
//    shebang (#!/usr/bin/env python3), which the bare launcher fails to resolve.
import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";
import { frontMatter, buildDocument, Packer, FIG } from "./shared.mjs";
import { ORDER } from "./manifest.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const OUT = process.argv[2] || path.join(__dirname, "OptiVibe_Руководство.docx");

// Resolve a Python command that really runs Python 3 (as [bin, ...prefixArgs]).
function resolvePython() {
  const candidates = process.platform === "win32"
    ? [["py", "-3"], ["python"], ["python3"], ["py"]]
    : [["python3"], ["python"]];
  for (const c of candidates) {
    const probe = spawnSync(c[0], [...c.slice(1), "-c", "import sys;print(sys.version_info[0])"],
      { encoding: "utf8" });
    if (!probe.error && probe.status === 0 && String(probe.stdout).trim() === "3") return c;
  }
  return null;
}

const PY = resolvePython();
if (!PY) {
  console.error("No working Python 3 found. On Windows try `py -3 --version`; if python/python3 " +
    "print only 'Python', disable the Microsoft Store aliases or install python.org (Add to PATH).");
  process.exit(1);
}
const runPython = (args) => spawnSync(PY[0], [...PY.slice(1), ...args], { stdio: "inherit" });

// 1) Ensure figures exist (build_figures.py writes into figures/).
if (!fs.existsSync(path.join(FIG, "mode.png"))) {
  const py = runPython([path.join(__dirname, "figures", "build_figures.py")]);
  if (py.status !== 0) { console.error("figure generation failed"); process.exit(1); }
}

// 2) Assemble: front matter + each module's section() in manifest order.
const children = [...frontMatter()];
let count = children.length;
for (const file of ORDER) {
  const url = pathToFileURL(path.join(__dirname, "sections", file)).href; // Windows-safe
  const mod = await import(url);
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
const vp = runPython([path.join(__dirname, "validate.py"), OUT, "--fix"]);
if (vp.status !== 0) { console.error("validation failed"); process.exit(1); }
