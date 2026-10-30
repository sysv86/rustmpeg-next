#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render the README badges from the supplied pixel-art designs.

The frames, the moss panel, the label text, and the rust panel are the supplied
designs, copied verbatim into ``docs/badges/templates/``. Only the numeric part
is generated:

* the value glyphs are the white digits of ``digits-pack`` (7x9 cells on a 3 px
  source grid), drawn with their dark outline into a single marked group
  ``<g id="badge-value" data-cell=... data-origin=...>``;
* letters and punctuation use the 5x7 pixel alphabet below, outlined the same
  way;
* when the value does not fit the supplied value panel, the panel is widened by
  whole grid steps. The frame and the label keep their original geometry; only
  coordinates at or right of the panel cap are translated.

Exactly one element carries the number, so the workflow is idempotent:

* ``<g id="badge-value">`` is deleted and regenerated on every run;
* every ``<path>`` with ``fill="#1a0803"`` or ``fill="#ffffff"`` in the body
  loses the subpaths that start at or right of the value panel's left edge;
  those subpaths are the supplied sample digits and their dark outline, and
  dropping them prevents a ghost number behind the generated one.

The templates themselves are never modified. The ``<desc>`` element records the
update timestamp and data source; it is stripped before comparing, so a rerun
with unchanged inputs leaves the file byte-identical.

Usage:
    render_badges.py version [--out-dir DIR]
    render_badges.py counts --repo OWNER/NAME [--max-age-hours 24]
    render_badges.py all --repo OWNER/NAME [--max-age-hours 24]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "docs" / "icons"
TEMPLATES = ROOT / "docs" / "badges" / "templates"

# Published crates whose download counters make up the crates.io badge.
PUBLISHED_CRATES = (
    "rustmpeg-core",
    "rustmpeg-format",
    "rustmpeg-audio",
    "rustmpeg-image",
    "rustmpeg-video",
    "rustmpeg-vulkan",
    "rustmpeg-cli",
    "rustmpeg",
)

# Source grid of the digits-pack glyphs: 7x9 cells of 3 px, 21x27 px per digit.
PACK_CELL = 3
DIGIT_CELLS_W, DIGIT_CELLS_H = 7, 9

VALUE_FILL = "#ffffff"
OUTLINE_FILL = "#1a0803"

# Outline and white body of each digits-pack white glyph, verbatim from the pack.
DIGITS = {
    "0": (
        "M3 0h15v3h-15zM0 18h3v3h-3zM6 18h9v3h-9zM18 18h3v3h-3zM0 3h6v3h-6zM15 3h6v3h-6zM0 21h6v3h-6zM15 21h6v3h-6zM0 6h3v3h-3zM6 6h9v3h-9zM18 6h3v3h-3zM0 15h3v3h-3zM9 15h6v3h-6zM18 15h3v3h-3zM3 24h15v3h-15zM0 9h3v3h-3zM6 9h6v3h-6zM18 9h3v3h-3zM0 12h3v3h-3zM6 12h3v3h-3zM12 12h3v3h-3zM18 12h3v3h-3z",
        "M3 12h3v3h-3zM9 12h3v3h-3zM15 12h3v3h-3zM3 9h3v3h-3zM12 9h6v3h-6zM6 3h9v3h-9zM6 21h9v3h-9zM3 18h3v3h-3zM15 18h3v3h-3zM3 15h6v3h-6zM15 15h3v3h-3zM3 6h3v3h-3zM15 6h3v3h-3z",
    ),
    "1": (
        "M6 0h9v3h-9zM3 9h6v3h-6zM12 9h3v3h-3zM3 18h6v3h-6zM12 18h6v3h-6zM3 21h3v3h-3zM15 21h3v3h-3zM6 15h3v3h-3zM12 15h3v3h-3zM3 24h15v3h-15zM3 6h3v3h-3zM12 6h3v3h-3zM6 12h3v3h-3zM12 12h3v3h-3zM3 3h6v3h-6zM12 3h3v3h-3z",
        "M9 12h3v3h-3zM6 21h9v3h-9zM9 3h3v3h-3zM9 9h3v3h-3zM9 18h3v3h-3zM6 6h6v3h-6zM9 15h3v3h-3z",
    ),
    "2": (
        "M3 0h15v3h-15zM6 12h6v3h-6zM15 12h6v3h-6zM0 9h15v3h-15zM18 9h3v3h-3zM0 18h6v3h-6zM9 18h12v3h-12zM0 3h6v3h-6zM15 3h6v3h-6zM0 6h3v3h-3zM6 6h9v3h-9zM18 6h3v3h-3zM0 24h21v3h-21zM3 15h6v3h-6zM12 15h6v3h-6zM0 21h3v3h-3zM18 21h3v3h-3z",
        "M12 12h3v3h-3zM3 6h3v3h-3zM15 6h3v3h-3zM6 3h9v3h-9zM3 21h15v3h-15zM6 18h3v3h-3zM15 9h3v3h-3zM9 15h3v3h-3z",
    ),
    "3": (
        "M3 0h15v3h-15zM0 9h15v3h-15zM18 9h3v3h-3zM6 12h3v3h-3zM15 12h6v3h-6zM0 18h3v3h-3zM6 18h9v3h-9zM18 18h3v3h-3zM0 3h6v3h-6zM15 3h6v3h-6zM0 21h6v3h-6zM15 21h6v3h-6zM0 6h3v3h-3zM6 6h9v3h-9zM18 6h3v3h-3zM0 15h15v3h-15zM18 15h3v3h-3zM3 24h15v3h-15z",
        "M9 12h6v3h-6zM3 6h3v3h-3zM15 6h3v3h-3zM15 15h3v3h-3zM6 3h9v3h-9zM6 21h9v3h-9zM3 18h3v3h-3zM15 18h3v3h-3zM15 9h3v3h-3z",
    ),
    "4": (
        "M9 0h9v3h-9zM0 12h3v3h-3zM6 12h6v3h-6zM15 12h6v3h-6zM6 3h6v3h-6zM15 3h3v3h-3zM9 21h3v3h-3zM15 21h3v3h-3zM0 15h3v3h-3zM18 15h3v3h-3zM3 6h6v3h-6zM15 6h3v3h-3zM0 18h12v3h-12zM15 18h6v3h-6zM0 9h6v3h-6zM9 9h3v3h-3zM15 9h3v3h-3zM9 24h9v3h-9z",
        "M3 12h3v3h-3zM12 12h3v3h-3zM3 15h15v3h-15zM6 9h3v3h-3zM12 9h3v3h-3zM12 18h3v3h-3zM9 6h6v3h-6zM12 3h3v3h-3zM12 21h3v3h-3z",
    ),
    "5": (
        "M0 0h21v3h-21zM0 12h15v3h-15zM18 12h3v3h-3zM0 18h3v3h-3zM6 18h9v3h-9zM18 18h3v3h-3zM0 21h6v3h-6zM15 21h6v3h-6zM0 6h3v3h-3zM6 6h15v3h-15zM0 15h9v3h-9zM12 15h3v3h-3zM18 15h3v3h-3zM3 24h15v3h-15zM0 9h3v3h-3zM15 9h6v3h-6zM0 3h3v3h-3zM18 3h3v3h-3z",
        "M3 9h12v3h-12zM3 6h3v3h-3zM15 15h3v3h-3zM3 3h15v3h-15zM6 21h9v3h-9zM15 12h3v3h-3zM3 18h3v3h-3zM15 18h3v3h-3z",
    ),
    "6": (
        "M3 0h15v3h-15zM0 9h3v3h-3zM6 9h12v3h-12zM0 12h3v3h-3zM15 12h6v3h-6zM0 18h3v3h-3zM6 18h9v3h-9zM18 18h3v3h-3zM0 3h6v3h-6zM15 3h3v3h-3zM0 21h6v3h-6zM15 21h6v3h-6zM0 6h3v3h-3zM6 6h12v3h-12zM0 15h3v3h-3zM6 15h9v3h-9zM18 15h3v3h-3zM3 24h15v3h-15z",
        "M3 12h12v3h-12zM3 6h3v3h-3zM3 15h3v3h-3zM15 15h3v3h-3zM6 3h9v3h-9zM6 21h9v3h-9zM3 18h3v3h-3zM15 18h3v3h-3zM3 9h3v3h-3z",
    ),
    "7": (
        "M0 0h21v3h-21zM3 21h3v3h-3zM9 21h3v3h-3zM3 12h6v3h-6zM12 12h6v3h-6zM0 6h15v3h-15zM18 6h3v3h-3zM3 18h3v3h-3zM9 18h3v3h-3zM3 24h9v3h-9zM3 15h3v3h-3zM9 15h6v3h-6zM6 9h6v3h-6zM15 9h6v3h-6zM0 3h3v3h-3zM18 3h3v3h-3z",
        "M3 3h15v3h-15zM9 12h3v3h-3zM12 9h3v3h-3zM6 21h3v3h-3zM6 18h3v3h-3zM6 15h3v3h-3zM15 6h3v3h-3z",
    ),
    "8": (
        "M3 0h15v3h-15zM0 9h3v3h-3zM6 9h9v3h-9zM18 9h3v3h-3zM0 12h6v3h-6zM15 12h6v3h-6zM0 18h3v3h-3zM6 18h9v3h-9zM18 18h3v3h-3zM0 3h6v3h-6zM15 3h6v3h-6zM0 21h6v3h-6zM15 21h6v3h-6zM0 6h3v3h-3zM6 6h9v3h-9zM18 6h3v3h-3zM0 15h3v3h-3zM6 15h9v3h-9zM18 15h3v3h-3zM3 24h15v3h-15z",
        "M6 12h9v3h-9zM3 9h3v3h-3zM15 9h3v3h-3zM3 6h3v3h-3zM15 6h3v3h-3zM3 15h3v3h-3zM15 15h3v3h-3zM6 3h9v3h-9zM6 21h9v3h-9zM3 18h3v3h-3zM15 18h3v3h-3z",
    ),
    "9": (
        "M3 0h15v3h-15zM0 9h3v3h-3zM6 9h9v3h-9zM18 9h3v3h-3zM3 18h12v3h-12zM18 18h3v3h-3zM0 3h6v3h-6zM15 3h6v3h-6zM3 21h3v3h-3zM15 21h6v3h-6zM0 6h3v3h-3zM6 6h9v3h-9zM18 6h3v3h-3zM3 15h12v3h-12zM18 15h3v3h-3zM3 24h15v3h-15zM0 12h6v3h-6zM18 12h3v3h-3z",
        "M6 12h12v3h-12zM3 9h3v3h-3zM15 9h3v3h-3zM3 6h3v3h-3zM15 6h3v3h-3zM15 15h3v3h-3zM6 3h9v3h-9zM6 21h9v3h-9zM15 18h3v3h-3z",
    ),
}

# 5x7 pixel alphabet for the label letters and for letters in the value.
FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "a": ("00000", "00000", "01110", "00001", "01111", "10001", "01111"),
    "b": ("10000", "10000", "10110", "11001", "10001", "10001", "11110"),
    "c": ("00000", "00000", "01110", "10001", "10000", "10001", "01110"),
    "d": ("00001", "00001", "01101", "10011", "10001", "10001", "01111"),
    "e": ("00000", "00000", "01110", "10001", "11111", "10000", "01110"),
    "f": ("00110", "01001", "01000", "11100", "01000", "01000", "01000"),
    "g": ("00000", "01111", "10001", "10001", "01111", "00001", "01110"),
    "h": ("10000", "10000", "10110", "11001", "10001", "10001", "10001"),
    "i": ("00100", "00000", "01100", "00100", "00100", "00100", "01110"),
    "j": ("00010", "00000", "00110", "00010", "00010", "10010", "01100"),
    "k": ("10000", "10000", "10010", "10100", "11000", "10100", "10010"),
    "l": ("01100", "00100", "00100", "00100", "00100", "00100", "01110"),
    "m": ("00000", "00000", "11010", "10101", "10101", "10101", "10101"),
    "n": ("00000", "00000", "10110", "11001", "10001", "10001", "10001"),
    "o": ("00000", "00000", "01110", "10001", "10001", "10001", "01110"),
    "p": ("00000", "00000", "11110", "10001", "11110", "10000", "10000"),
    "q": ("00000", "00000", "01101", "10011", "01111", "00001", "00001"),
    "r": ("00000", "00000", "10110", "11001", "10000", "10000", "10000"),
    "s": ("00000", "00000", "01110", "10000", "01110", "00001", "11110"),
    "t": ("01000", "01000", "11100", "01000", "01000", "01001", "00110"),
    "u": ("00000", "00000", "10001", "10001", "10001", "10011", "01101"),
    "v": ("00000", "00000", "10001", "10001", "10001", "01010", "00100"),
    "w": ("00000", "00000", "10001", "10001", "10101", "10101", "01010"),
    "x": ("00000", "00000", "10001", "01010", "00100", "01010", "10001"),
    "y": ("00000", "00000", "10001", "10001", "01111", "00001", "01110"),
    "z": ("00000", "00000", "11111", "00010", "00100", "01000", "11111"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
}


@dataclass(frozen=True)
class Spec:
    """Geometry of one supplied badge template."""

    template: str
    width: int
    height: int
    grid: int
    cell: int
    panel_left: int
    cap: int
    inner_top: int
    inner_bottom: int


SPECS = {
    "version": Spec("version.svg", 148, 40, 2, 2, 92, 144, 2, 38),
    "crates": Spec("crates.svg", 273, 60, 3, 3, 213, 267, 3, 57),
    "github": Spec("github.svg", 282, 60, 3, 3, 222, 276, 3, 57),
}


def advance(ch: str, cell: int) -> int:
    """Horizontal advance of one value character in pixels."""
    if ch in DIGITS:
        return (DIGIT_CELLS_W + 1) * cell
    if ch == ".":
        return 3 * cell
    if ch == "-":
        return 6 * cell
    return 6 * cell


def value_width(text: str, cell: int) -> int:
    """Pixel width of a value, without the trailing inter-character gap."""
    if not text:
        return 0
    return sum(advance(ch, cell) for ch in text) - cell


def _filled(ch: str) -> set[tuple[int, int]]:
    rows = FONT.get(ch)
    if rows is None:
        return set()
    return {
        (col, row)
        for row, line in enumerate(rows)
        for col, bit in enumerate(line)
        if bit == "1"
    }


def font_markup(ch: str, x: int, y: int, cell: int) -> str:
    """One 5x7 letter or punctuation mark, white with a dark outline."""
    filled = _filled(ch)
    if not filled:
        return ""
    outline = {
        (col + dx, row + dy)
        for col, row in filled
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        if (col + dx, row + dy) not in filled
    }
    parts = [
        f'<rect x="{x + col * cell}" y="{y + row * cell}" width="{cell}" height="{cell}"/>'
        for col, row in sorted(outline)
    ]
    white = [
        f'<rect x="{x + col * cell}" y="{y + row * cell}" width="{cell}" height="{cell}"/>'
        for col, row in sorted(filled)
    ]
    return (
        f'<g fill="{OUTLINE_FILL}">{"".join(parts)}</g>'
        f'<g fill="{VALUE_FILL}">{"".join(white)}</g>'
    )


def value_markup(text: str, x: int, y: int, cell: int) -> str:
    """The generated value: digits-pack glyphs plus outlined pixel letters."""
    scale = cell / PACK_CELL
    parts = []
    cursor = x
    for ch in text:
        if ch in DIGITS:
            outline, white = DIGITS[ch]
            parts.append(
                f'<g transform="translate({cursor} {y}) scale({scale:.7f})">'
                f'<path fill="{OUTLINE_FILL}" d="{outline}"/>'
                f'<path fill="{VALUE_FILL}" d="{white}"/></g>'
            )
        else:
            parts.append(font_markup(ch, cursor, y + 2 * cell, cell))
        cursor += advance(ch, cell)
    return "".join(parts)


def _map_x(value: int, cap: int, delta: int) -> int:
    return value + delta if value >= cap else value


def _shift_path(d: str, cap: int, delta: int) -> str:
    """Translate every x coordinate at or right of the panel cap by ``delta``."""
    out = []
    index = 0
    original_x = 0
    mapped_x = 0
    number = re.compile(r"-?\d+(?:\.\d+)?")

    def read_number() -> float:
        nonlocal index
        while index < len(d) and d[index] in " ,":
            index += 1
        match = number.match(d, index)
        if match is None:
            raise ValueError(f"bad path data: {d!r}")
        index = match.end()
        return float(match.group(0))

    while index < len(d):
        command = d[index]
        index += 1
        if command == "M":
            original_x = read_number()
            y = read_number()
            mapped_x = _map_x(original_x, cap, delta)
            out.append(f"M{int(mapped_x)} {int(y)}")
        elif command == "h":
            original_x += read_number()
            target = _map_x(original_x, cap, delta)
            out.append(f"h{int(target - mapped_x)}")
            mapped_x = target
        elif command == "v":
            out.append(f"v{int(read_number())}")
        elif command == "z":
            out.append("z")
        elif command in " ,\t\n":
            continue
        else:
            raise ValueError(f"unsupported path command {command!r}")
    return "".join(out)


def _rewrite_paths(text: str, cap: int, delta: int) -> str:
    return re.sub(
        r'\sd="([^"]*)"',
        lambda match: f' d="{_shift_path(match.group(1), cap, delta)}"',
        text,
    )


def _rust_period(svg: str) -> int:
    """Horizontal repeat period of the rust texture, from its keyframes."""
    match = re.search(r"@keyframes rust\{to\{transform:translateX\((-?\d+)px\)", svg)
    return int(match.group(1)) if match else 0


def _rust_reach(svg: str) -> int:
    """Rightmost x of one rust tile, relative to its ``<use>`` origin."""
    tile = re.search(r'<g id="st">(.*?)</g>', svg, flags=re.S)
    if tile is None:
        return 0
    return max(int(x) for x in re.findall(r'<rect x="(-?\d+)"', tile.group(1)))


def _extend_tiles(body: str, width: int, period: int, reach: int) -> str:
    """Repeat the rust tile set until it covers the widened value panel."""
    if period <= 0:
        return body

    def extend(match: re.Match[str]) -> str:
        group = match.group(0)
        uses = re.findall(r'<use xlink:href="#(\w+)" x="(-?\d+)" y="(-?\d+)"/>', group)
        if not uses:
            return group
        extra = []
        step = 1
        left = min(int(x) for _, x, _ in uses)
        while reach + left + step * period < width:
            for href, x, y in uses:
                extra.append(
                    f'<use xlink:href="#{href}" x="{int(x) + step * period}" y="{y}"/>'
                )
            step += 1
        return group.replace("</g>", "".join(extra) + "</g>")

    return re.sub(r'<g class="rust">.*?</g>', extend, body, flags=re.S)


def _strip_value_glyphs(body: str, panel_left: int) -> str:
    """Drop the sample digits and their outline from the value panel."""

    def strip(match: re.Match[str]) -> str:
        element = match.group(0)
        if not re.search(r'fill="(?:%s|%s)"' % (OUTLINE_FILL, VALUE_FILL), element):
            return element
        if "fill-opacity" in element:
            return element
        data = re.search(r'\sd="([^"]*)"', element)
        if data is None:
            return element
        kept = []
        for subpath in re.findall(r"M[^M]*", data.group(1)):
            x = int(re.match(r"M(-?\d+)", subpath).group(1))
            if x < panel_left:
                kept.append(subpath)
        if not kept:
            return ""
        return element.replace(data.group(1), "".join(kept))

    return re.sub(r"<path\b[^>]*/>", strip, body)


def load_template(spec: Spec) -> str:
    svg = (TEMPLATES / spec.template).read_text()
    svg = re.sub(r"<metadata>.*?</metadata>", "", svg, flags=re.S)
    svg = svg.replace(' xmlns:c2pa="http://c2pa.org/manifest"', "")
    return svg


def render(spec: Spec, value: str, aria: str, updated: str, source: str) -> str:
    """Return the badge SVG with ``value`` drawn into the supplied design."""
    cell = spec.cell
    inner = spec.cap - spec.panel_left
    need = value_width(value, cell) + 2 * cell
    delta = 0
    if need > inner:
        delta = -(-(need - inner) // spec.grid) * spec.grid
    width = spec.width + delta

    svg = load_template(spec)
    svg = svg.replace(f'viewBox="0 0 {spec.width} {spec.height}"',
                      f'viewBox="0 0 {width} {spec.height}"')
    svg = svg.replace(f'width="{spec.width}" height="{spec.height}"',
                      f'width="{width}" height="{spec.height}"')

    head, _, rest = svg.partition("</defs>")
    head = re.sub(
        r"<clipPath\b.*?</clipPath>",
        lambda match: _rewrite_paths(match.group(0), spec.cap, delta),
        head,
        flags=re.S,
    )
    body = _rewrite_paths("</defs>" + rest, spec.cap, delta)
    body = _extend_tiles(body, width, _rust_period(svg), _rust_reach(svg))
    body = _strip_value_glyphs(body, spec.panel_left)

    content = value_width(value, cell)
    pad = (inner + delta - content) // 2
    x = spec.panel_left + (pad // cell) * cell
    height = DIGIT_CELLS_H * cell
    y = spec.inner_top + ((spec.inner_bottom - spec.inner_top - height) // 2 // cell) * cell

    group = (
        f'  <g id="badge-value" data-cell="{cell}" data-origin="{x} {y}">'
        f"{value_markup(value, x, y, cell)}</g>\n"
        f"  <desc>updated: {updated}; source: {source}</desc>\n"
    )
    svg = head + body
    svg = svg.replace("</svg>", group + "</svg>")

    title = f"{aria}"
    svg = re.sub(r"<title>[^<]*</title>", f"<title>{title}</title>", svg, count=1)
    svg = re.sub(r'aria-label="[^"]*"', f'aria-label="{title}"', svg, count=1)
    return svg


def read_version() -> str:
    with (ROOT / "Cargo.toml").open("rb") as handle:
        manifest = tomllib.load(handle)
    return manifest["workspace"]["package"]["version"]


def read_updated(path: Path) -> datetime | None:
    if not path.exists():
        return None
    match = re.search(r"<desc>updated: ([^;]+);", path.read_text())
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def fetch_json(url: str, headers: dict[str, str]) -> object:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def github_downloads(repo: str, token: str | None) -> int | None:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "rustmpeg-next-badges",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    total = 0
    page = 1
    try:
        while page <= 20:
            data = fetch_json(
                f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}",
                headers,
            )
            if not isinstance(data, list):
                return None
            for release in data:
                for asset in release.get("assets") or []:
                    total += int(asset.get("download_count") or 0)
            if len(data) < 100:
                return total
            page += 1
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return 0
        raise
    return total


def crates_downloads(crates: tuple[str, ...], token: str | None) -> int | None:
    headers = {
        "Accept": "application/json",
        "User-Agent": "rustmpeg-next-badges",
    }
    if token:
        headers["Authorization"] = token
    total = 0
    for crate in crates:
        try:
            data = fetch_json(f"https://crates.io/api/v1/crates/{crate}", headers)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                continue
            raise
        if not isinstance(data, dict):
            return None
        total += int((data.get("crate") or {}).get("downloads") or 0)
        versions = data.get("versions")
        if isinstance(versions, list) and versions:
            total += int((versions[0] or {}).get("downloads") or 0)
    return total


def compact(value: int) -> str:
    if value < 1000:
        return str(value)
    if value < 1_000_000:
        scaled = value / 1000
        suffix = "k"
    else:
        scaled = value / 1_000_000
        suffix = "M"
    text = f"{scaled:.1f}".rstrip("0").rstrip(".")
    return f"{text}{suffix}"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def strip_timestamp(svg: str) -> str:
    return re.sub(r"<desc>.*?</desc>", "<desc/>", svg, flags=re.S)


def write_if_changed(path: Path, svg: str) -> bool:
    if path.exists() and strip_timestamp(path.read_text()) == strip_timestamp(svg):
        return False
    path.write_text(svg)
    return True


def render_version(out_dir: Path) -> bool:
    version = read_version()
    svg = render(
        SPECS["version"],
        version,
        f"version {version}",
        now_iso(),
        "Cargo.toml",
    )
    return write_if_changed(out_dir / "version.svg", svg)


def render_count(
    out_dir: Path,
    name: str,
    value: int,
    source: str,
    max_age_hours: float,
) -> bool:
    path = out_dir / f"{name}.svg"
    updated = read_updated(path)
    if updated is not None:
        age_hours = (datetime.now(timezone.utc) - updated).total_seconds() / 3600
        if age_hours < max_age_hours:
            print(f"{name}: fresh ({age_hours:.1f}h), skipping")
            return False
    text = compact(value)
    svg = render(
        SPECS[name],
        text,
        f"{name} downloads {text}",
        now_iso(),
        source,
    )
    return write_if_changed(path, svg)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("version", "counts", "all"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repo", default="")
    parser.add_argument("--max-age-hours", type=float, default=24.0)
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    changed = False

    if args.mode in ("version", "all"):
        changed |= render_version(out_dir)

    if args.mode in ("counts", "all"):
        if not args.repo:
            parser.error("--repo OWNER/NAME is required for counts")
        token = os.environ.get("GITHUB_TOKEN")
        crates_token = os.environ.get("CRATES_IO_TOKEN")
        for name, source, fetch in (
            (
                "crates",
                "crates.io",
                lambda: crates_downloads(PUBLISHED_CRATES, crates_token),
            ),
            (
                "github",
                "GitHub releases",
                lambda: github_downloads(args.repo, token),
            ),
        ):
            try:
                value = fetch()
            except (urllib.error.URLError, TimeoutError, ValueError) as error:
                print(f"warning: {name} fetch failed ({error}); keeping existing badge")
                continue
            if value is None:
                print(f"warning: {name} returned no data; keeping existing badge")
                continue
            changed |= render_count(
                out_dir,
                name,
                value,
                source,
                args.max_age_hours,
            )

    print("changed" if changed else "unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
