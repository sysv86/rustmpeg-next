#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Refresh the README footer with the current Rust line count and the
self-reported workspace completion from docs/status.json.

The footer lives between <!-- STATS:START --> and <!-- STATS:END -->.
Exit code 0 always; prints whether the file changed.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
STATUS = ROOT / "docs" / "status.json"
CRATES = ROOT / "crates"

START = "<!-- STATS:START -->"
END = "<!-- STATS:END -->"


def rust_lines() -> int:
    total = 0
    for path in sorted(CRATES.rglob("*.rs")):
        try:
            with path.open("rb") as handle:
                total += sum(1 for _ in handle)
        except OSError:
            continue
    return total


def completion() -> int:
    data = json.loads(STATUS.read_text(encoding="utf-8"))
    crates = data.get("crates", [])
    if not crates:
        return 0
    return round(sum(int(c.get("percent", 0)) for c in crates) / len(crates))


def source_stats() -> tuple[int, int]:
    """Return (unsafe occurrences in Rust code, estimated source tokens).

    `unsafe` is counted outside line and block comments; the token count is the
    usual `chars / 4` estimate for BPE tokenizers and is labelled as such.
    """
    unsafe = 0
    chars = 0
    for path in sorted(CRATES.rglob("*.rs")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        chars += len(text)
        code = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        code = re.sub(r"//[^\n]*", "", code)
        unsafe += len(re.findall(r"(?<![\w])unsafe(?![\w])", code))
    return unsafe, round(chars / 4)


def main() -> int:
    lines = rust_lines()
    percent = completion()
    unsafe, tokens = source_stats()
    text = README.read_text(encoding="utf-8")
    replacement = (
        f"{START}Rust lines: {lines:,} · workspace completion: {percent}% · "
        f"unsafe: {unsafe} · source tokens: ~{tokens:,} (chars/4){END}"
    )
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        print("stats markers not found", file=sys.stderr)
        return 1
    updated = pattern.sub(replacement, text)
    if updated == text:
        print("unchanged")
        return 0
    README.write_text(updated, encoding="utf-8")
    print(f"updated: {lines} lines, {percent}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
