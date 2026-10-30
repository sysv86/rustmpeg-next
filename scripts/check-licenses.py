#!/usr/bin/env python3
"""Audit the cargo dependency graph for license compliance.

Allowed: Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, ISC, Zlib, Unlicense,
CC0-1.0, Unicode-3.0, and Apache-2.0 WITH LLVM-exception.

Forbidden anywhere in the graph: GPL, LGPL, AGPL, MPL, SSPL, BUSL, CDLA,
unknown/custom licenses.

Usage: python3 scripts/check-licenses.py [--verbose]
Exit code 0 if compliant, 1 otherwise.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ALLOWED = {
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "zlib",
    "unlicense",
    "cc0-1.0",
    "unicode-3.0",
}
ALLOWED_EXCEPTIONS = {"llvm-exception"}
SPDX_ALLOWED = {
    "Apache-2.0",
    "Apache-2.0 AND BSD-2-Clause",
    "Apache-2.0 AND BSD-3-Clause",
    "Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause",
}
SPDX_PREFIX = "// SPDX-License-Identifier: "
FORBIDDEN = re.compile(r"(?:^|[^a-z])(?:a?gpl|lgpl|mpl|sspl|busl|cdla)", re.I)
ROOT_PACKAGES = {"rustmpeg-core", "rustmpeg-format", "rustmpeg-audio",
                 "rustmpeg-image", "rustmpeg-video", "rustmpeg-vulkan"}

# rustmpeg-video contains VTM-derived (BSD-3-Clause) and dav2d-derived
# (BSD-2-Clause) source, and rustmpeg-image contains OpenEXRCore-derived
# (BSD-3-Clause) source, so their expressions describe the combined package.
# All other workspace crates are original Apache-2.0 code.
ROOT_LICENSE_OVERRIDES = {
    "rustmpeg-video": "Apache-2.0 AND BSD-2-Clause AND BSD-3-Clause",
    "rustmpeg-image": "Apache-2.0 AND BSD-3-Clause",
}


def tokenize(expr: str) -> list[str]:
    expr = expr.replace("/", " OR ")
    return re.findall(r"\(|\)|AND|OR|WITH|[A-Za-z0-9.+\-]+", expr)


class Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse_or(self):
        node = self.parse_and()
        while self.peek() == "OR":
            self.take()
            node = ("or", node, self.parse_and())
        return node

    def parse_and(self):
        node = self.parse_factor()
        while self.peek() == "AND":
            self.take()
            node = ("and", node, self.parse_factor())
        return node

    def parse_factor(self):
        tok = self.peek()
        if tok == "(":
            self.take()
            node = self.parse_or()
            if self.peek() != ")":
                raise ValueError("unbalanced parentheses")
            self.take()
            return node
        if tok is None:
            raise ValueError("unexpected end of expression")
        lic = self.take()
        if self.peek() == "WITH":
            self.take()
            exc = self.take()
            return ("with", lic, exc)
        return ("lic", lic)


def allowed(node) -> bool:
    kind = node[0]
    if kind == "lic":
        return node[1].lower() in ALLOWED
    if kind == "with":
        return node[1].lower() in ALLOWED and node[2].lower() in ALLOWED_EXCEPTIONS
    if kind == "or":
        return allowed(node[1]) or allowed(node[2])
    if kind == "and":
        return allowed(node[1]) and allowed(node[2])
    raise ValueError(f"unknown node {kind}")


def evaluate(expr: str) -> bool:
    return allowed(Parser(tokenize(expr)).parse_or())


def scan_spdx_headers() -> tuple[int, list[str]]:
    """Check every Rust source file for an approved SPDX expression."""
    violations: list[str] = []
    checked = 0
    for path in sorted(Path("crates").rglob("*.rs")):
        checked += 1
        try:
            first = path.read_text(errors="replace").splitlines()[0]
        except IndexError:
            violations.append(f"{path}: empty Rust source file")
            continue
        if not first.startswith(SPDX_PREFIX):
            violations.append(f"{path}: missing SPDX header")
            continue
        expr = first[len(SPDX_PREFIX):].strip()
        if expr not in SPDX_ALLOWED:
            violations.append(f"{path}: unexpected SPDX expression {expr!r}")
        if FORBIDDEN.search(expr):
            violations.append(f"{path}: forbidden license token in {expr!r}")
    return checked, violations


def main() -> int:
    verbose = "--verbose" in sys.argv[1:]
    proc = subprocess.run(
        ["cargo", "metadata", "--format-version", "1"],
        check=True, capture_output=True, text=True,
    )
    metadata = json.loads(proc.stdout)
    violations: list[str] = []
    seen: dict[str, int] = {}

    for pkg in sorted(metadata["packages"], key=lambda p: (p["name"], p["version"])):
        name = pkg["name"]
        version = pkg["version"]
        lic = pkg.get("license") or pkg.get("license_file") or ""
        label = f"{name} {version}"

        if name in ROOT_PACKAGES:
            expected = ROOT_LICENSE_OVERRIDES.get(name, "Apache-2.0")
            if lic != expected:
                violations.append(
                    f"{label}: workspace crate must declare {expected!r}, got {lic!r}"
                )
            continue

        if not lic:
            violations.append(f"{label}: missing license")
            continue

        if FORBIDDEN.search(lic):
            violations.append(f"{label}: forbidden copyleft/data license in {lic!r}")
            continue

        try:
            ok = evaluate(lic)
        except ValueError as exc:
            violations.append(f"{label}: cannot parse {lic!r}: {exc}")
            continue

        if not ok:
            violations.append(f"{label}: no allowed license in {lic!r}")
            continue

        seen[lic] = seen.get(lic, 0) + 1

    checked, spdx_violations = scan_spdx_headers()
    violations.extend(spdx_violations)

    print(f"cargo metadata: {len(metadata['packages'])} packages")
    print(f"rust sources: {checked} SPDX headers checked")
    for lic, count in sorted(seen.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {count:4d}  {lic}")
    if verbose:
        for pkg in sorted(metadata["packages"], key=lambda p: p["name"]):
            print(f"  {pkg['name']} {pkg['version']}: {pkg.get('license')}")

    if violations:
        print("\nLICENSE VIOLATIONS:")
        for item in violations:
            print(f"  - {item}")
        return 1

    print("\nlicense audit: OK (all packages permissive, no copyleft)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
