#!/usr/bin/env python3
"""Validate (and optionally repair) OptiVibe_Руководство.docx (doc 15 §6).

Checks, in order:
  1. the file is a readable ZIP (OOXML container) and survives ``testzip``;
  2. the mandatory parts exist (``[Content_Types].xml``, ``word/document.xml``);
  3. every XML part is well-formed;
  4. bookmark integrity in ``word/document.xml``: each ``w:bookmarkStart`` has a
     unique ``w:id`` and a matching ``w:bookmarkEnd`` (and vice-versa). Duplicate
     numeric ids are the failure doc 15 §6 calls out (Word mis-links the TOC /
     internal hyperlinks when two bookmarks share an id).

With ``--fix`` the script renumbers bookmark ids so every start/end pair gets a
fresh, unique number (a stack pairs starts with the next end), repacks the
container in place, and re-validates. Exit code 0 = valid, 1 = invalid.

Usage:
    python docs/manual/validate.py OptiVibe_Руководство.docx [--fix]
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from xml.etree import ElementTree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DOC_PART = "word/document.xml"
REQUIRED = ("[Content_Types].xml", DOC_PART)


def _read_parts(path: str) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise ValueError(f"corrupt ZIP entry: {bad}")
        return {n: zf.read(n) for n in zf.namelist()}


def _check_bookmarks(doc_xml: bytes) -> list[str]:
    """Return a list of bookmark problems (empty list = OK)."""
    errs: list[str] = []
    root = ET.fromstring(doc_xml)
    starts: list[str] = []
    ends: list[str] = []
    names: dict[str, str] = {}
    for el in root.iter():
        if el.tag == f"{{{W}}}bookmarkStart":
            bid = el.get(f"{{{W}}}id")
            name = el.get(f"{{{W}}}name")
            starts.append(bid)
            if bid is not None:
                names.setdefault(bid, name or "")
        elif el.tag == f"{{{W}}}bookmarkEnd":
            ends.append(el.get(f"{{{W}}}id"))

    dup = sorted({i for i in starts if starts.count(i) > 1})
    if dup:
        errs.append(f"duplicate bookmarkStart ids: {dup}")
    s, e = set(starts), set(ends)
    if s - e:
        errs.append(f"bookmarkStart ids without matching End: {sorted(s - e)}")
    if e - s:
        errs.append(f"bookmarkEnd ids without matching Start: {sorted(e - s)}")
    return errs


def validate(path: str) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    try:
        parts = _read_parts(path)
    except (zipfile.BadZipFile, ValueError) as exc:
        return False, [f"container: {exc}"]

    for req in REQUIRED:
        if req not in parts:
            msgs.append(f"missing required part: {req}")

    for name, data in parts.items():
        if name.endswith(".xml") or name.endswith(".rels"):
            try:
                ET.fromstring(data)
            except ET.ParseError as exc:
                msgs.append(f"malformed XML in {name}: {exc}")

    if DOC_PART in parts:
        msgs.extend(_check_bookmarks(parts[DOC_PART]))

    return (len(msgs) == 0), msgs


def fix_bookmarks(path: str) -> int:
    """Renumber bookmark ids to a fresh unique pair sequence; repack in place."""
    parts = _read_parts(path)
    xml = parts[DOC_PART].decode("utf-8")

    counter = [0]
    stack: list[int] = []

    def _start(m: re.Match[str]) -> str:
        counter[0] += 1
        stack.append(counter[0])
        return re.sub(r'w:id="\d+"', f'w:id="{counter[0]}"', m.group(0))

    def _end(m: re.Match[str]) -> str:
        bid = stack.pop() if stack else counter[0]
        return re.sub(r'w:id="\d+"', f'w:id="{bid}"', m.group(0))

    xml = re.sub(r"<w:bookmarkStart\b[^>]*/?>", _start, xml)
    xml = re.sub(r"<w:bookmarkEnd\b[^>]*/?>", _end, xml)
    parts[DOC_PART] = xml.encode("utf-8")

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in parts.items():
            zf.writestr(name, data)
    return counter[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate OptiVibe manual .docx")
    ap.add_argument("path")
    ap.add_argument("--fix", action="store_true", help="renumber duplicate bookmark ids in place")
    args = ap.parse_args()

    ok, msgs = validate(args.path)
    if ok:
        print(f"VALID: {args.path}")
        return 0

    print(f"INVALID: {args.path}")
    for m in msgs:
        print(f"  - {m}")

    if args.fix and any("bookmark" in m for m in msgs):
        n = fix_bookmarks(args.path)
        print(f"  fixed: renumbered {n} bookmark pairs; re-validating...")
        ok, msgs = validate(args.path)
        print("VALID" if ok else "STILL INVALID", args.path)
        for m in msgs:
            print(f"  - {m}")
        return 0 if ok else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
