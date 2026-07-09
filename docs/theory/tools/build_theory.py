"""Optional local build of the ``docs/theory/`` layer to self-contained HTML.

The theory sources are Markdown + LaTeX (``$...$`` / ``$$...$$``). This script
renders each block to a **single, self-contained** HTML file that opens offline
in a browser, with **MathJax embedded** (no CDN) — per the ``docs/theory``
addendum to standard 15 and O-SW-04 (revised SW-48: local build, no GitHub
Pages). We use MathJax (not KaTeX): it is more complete on the multi-line
derivation constructs used in the blocks.

Rendering is delegated to ``pandoc`` (the same toolchain family the manual uses
for its ``.docx``). To stay CDN-free the MathJax runtime must be *vendored*
locally under ``tools/mathjax/`` as ``tex-svg.js`` (fetched once by
``tools/fetch_mathjax.py``); ``pandoc --embed-resources --standalone
--mathjax=<local>`` then inlines it into the output HTML. The **SVG** bundle is
used (not ``tex-mml-chtml.js``): the CHTML renderer loads fonts dynamically at
runtime and is therefore *not* self-contained offline, while the SVG renderer
needs no external resources at all.

Outputs are **local artefacts** and are not versioned (mirrors the O-SW-09 rule
for the manual ``.docx``; add ``docs/theory/_build/`` to ``.gitignore``).

Usage
-----
    python docs/theory/tools/fetch_mathjax.py           # once: vendor MathJax
    python docs/theory/tools/build_theory.py            # build all blocks
    python docs/theory/tools/build_theory.py 01_sensing_element.md

This script only *reads* Markdown and *writes* HTML; it never edits the
repository sources (documentation does not change code, 18 §5/§6).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

THEORY_DIR = Path(__file__).resolve().parent.parent
BUILD_DIR = THEORY_DIR / "_build"
MATHJAX = THEORY_DIR / "tools" / "mathjax" / "tex-svg.js"


def _markdown_sources(names: list[str]) -> list[Path]:
    """Resolve the Markdown blocks to build.

    Parameters
    ----------
    names : list of str
        Explicit file names; when empty, every ``*.md`` in the theory root
        (excluding ``README.md``) is built.

    Returns
    -------
    list of pathlib.Path
        Existing Markdown source paths.
    """
    if names:
        return [THEORY_DIR / name for name in names]
    return sorted(p for p in THEORY_DIR.glob("*.md") if p.name != "README.md")


def _pandoc_command(source: Path, target: Path) -> list[str]:
    """Build the pandoc argument vector for one block.

    Parameters
    ----------
    source : pathlib.Path
        Markdown input.
    target : pathlib.Path
        HTML output.

    Returns
    -------
    list of str
        The command line for :func:`subprocess.run`.
    """
    cmd = [
        "pandoc",
        str(source),
        "--from=markdown+tex_math_dollars",
        "--to=html5",
        "--standalone",
        "--embed-resources",
        "--metadata=lang:ru",
        f"--metadata=title:{source.stem}",
        "--output",
        str(target),
    ]
    # Vendored MathJax keeps the HTML offline/self-contained (no CDN). A bare
    # ``--mathjax`` fallback is deliberately NOT used: combined with
    # ``--embed-resources`` pandoc tries to inline MathJax from a system path
    # that usually does not exist and the build fails opaquely (root cause of
    # the 2026-07 breakage). ``build`` refuses early with a clear message
    # instead; run ``fetch_mathjax.py`` once to vendor the runtime.
    cmd.append(f"--mathjax={MATHJAX}")
    return cmd


def build(names: list[str]) -> int:
    """Render the requested blocks to ``_build/*.html``.

    Parameters
    ----------
    names : list of str
        Block file names, or empty for all blocks.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    if shutil.which("pandoc") is None:
        print("pandoc not found: install pandoc to build the HTML view", file=sys.stderr)
        return 1
    if not MATHJAX.is_file():
        print(
            f"vendored MathJax not found at {MATHJAX}: run "
            "`python docs/theory/tools/fetch_mathjax.py` once (fetches the "
            "mathjax npm package and vendors es5/tex-svg.js), then re-run "
            "this build",
            file=sys.stderr,
        )
        return 1
    BUILD_DIR.mkdir(exist_ok=True)
    for source in _markdown_sources(names):
        if not source.is_file():
            print(f"skip: {source} not found", file=sys.stderr)
            continue
        target = BUILD_DIR / f"{source.stem}.html"
        subprocess.run(_pandoc_command(source, target), check=True)
        print(f"built {target.relative_to(THEORY_DIR)}")
    return 0


def main() -> int:
    """Command-line entry point.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Build the docs/theory HTML view.")
    parser.add_argument("blocks", nargs="*", help="block file names (default: all)")
    args = parser.parse_args()
    return build(args.blocks)


if __name__ == "__main__":
    raise SystemExit(main())
