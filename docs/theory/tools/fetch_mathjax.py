"""One-off vendoring of the MathJax runtime for the ``docs/theory`` HTML build.

Downloads the ``mathjax`` npm package tarball (pinned version) from the npm
registry and extracts ``es5/tex-svg.js`` into ``tools/mathjax/``. The vendored
file is a *fetched dependency*, not a source: it stays untracked
(``.gitignore``) — mirroring the O-SW-09 rule that built/fetched artefacts are
local. Uses only the standard library (``urllib``, ``tarfile``); the npm
registry is the same toolchain family the manual (``docs/manual``) already
relies on.

Why ``tex-svg.js`` and not ``tex-mml-chtml.js``: the CHTML renderer loads font
resources dynamically at runtime, so an "embedded" CHTML bundle is still not
self-contained offline; the SVG renderer needs no external resources.

Usage
-----
    python docs/theory/tools/fetch_mathjax.py
"""

from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

MATHJAX_VERSION = "3.2.2"
TARBALL_URL = f"https://registry.npmjs.org/mathjax/-/mathjax-{MATHJAX_VERSION}.tgz"
# sha512 of the published tarball, from the npm registry metadata
# (``npm view mathjax@3.2.2 dist.integrity``); pins the artefact.
TARBALL_SHA512 = (
    "06df9249553c7811b6ef30a155ec0e89c62ced7b1db78d2a9b8f94a47c97ee4d"
    "3f3bd36588f73ec7bee4d7f144b0fb2328f64626fb5164cd6f3be81e5b43a11b"
)
MEMBER = "package/es5/tex-svg.js"
TARGET = Path(__file__).resolve().parent / "mathjax" / "tex-svg.js"


def fetch() -> int:
    """Download the pinned MathJax tarball and vendor ``tex-svg.js``.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    print(f"fetching {TARBALL_URL} ...")
    with urllib.request.urlopen(TARBALL_URL) as response:
        blob = response.read()
    digest = hashlib.sha512(blob).hexdigest()
    if digest != TARBALL_SHA512:
        print(
            "integrity check failed: tarball sha512 does not match the pinned "
            f"value for mathjax@{MATHJAX_VERSION}; refusing to vendor",
            file=sys.stderr,
        )
        return 1
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        member = archive.getmember(MEMBER)
        extracted = archive.extractfile(member)
        if extracted is None:
            print(f"{MEMBER} not found in tarball", file=sys.stderr)
            return 1
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_bytes(extracted.read())
    print(f"vendored {TARGET} ({TARGET.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(fetch())
