#!/usr/bin/env python3
"""
generate_price_tracks.py

Vertical price tracks renderer.

Input TSV:
  ./price_tiers.tsv

Output PNG:
  ./build_tts/price_tracks.png

Run:
  python generate_price_tracks.py
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont


# ----------------------------
# Fixed paths
# ----------------------------

INPUT_TSV_PATH = Path("price_tiers.tsv")
OUTPUT_PNG_PATH = Path("build_tts") / "price_tracks.png"


# ----------------------------
# Canvas + layout
# ----------------------------

CANVAS_W = 2200
CANVAS_H = 1400

MARGIN_X = 120
MARGIN_Y = 140

TRACK_GAP_X = 420
TRACK_HEIGHT = CANVAS_H - 2 * MARGIN_Y

BIG_DOT_R = 14
SMALL_DOT_R = 6
LINE_W = 5


# ----------------------------
# Model
# ----------------------------

@dataclass(frozen=True)
class Tier:
    price: int
    action_price: int
    intervals_to_next: int


# ----------------------------
# Fonts
# ----------------------------

def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates: List[str] = []

    if sys.platform.startswith("win"):
        if bold:
            candidates += [
                r"C:\Windows\Fonts\arialbd.ttf",
                r"C:\Windows\Fonts\segoeuib.ttf",
            ]
        else:
            candidates += [
                r"C:\Windows\Fonts\arial.ttf",
                r"C:\Windows\Fonts\segoeui.ttf",
            ]

    if bold:
        candidates += ["DejaVuSans-Bold.ttf"]
    else:
        candidates += ["DejaVuSans.ttf"]

    for fp in candidates:
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            continue

    return ImageFont.load_default()


# ----------------------------
# Parsing
# ----------------------------

def _clean(s: Optional[str]) -> str:
    return (s or "").strip()


def _parse_int(cell: str) -> int:
    cell = _clean(cell)
    if cell == "":
        return 0
    return int(cell.replace(",", ""))


def parse_tiers_tsv(path: Path) -> Dict[str, List[Tier]]:
    raw = path.read_text(encoding="utf-8")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    reader = csv.reader(lines, delimiter="\t")

    header = next(reader)
    header_map = {_clean(h): i for i, h in enumerate(header)}

    prefixes = [
        h[:-len("-prices")]
        for h in header
        if h.endswith("-prices") and not h.endswith("-actions-prices")
    ]

    out: Dict[str, List[Tier]] = {p: [] for p in prefixes}

    for row in reader:
        while len(row) < len(header):
            row.append("")
        for p in prefixes:
            out[p].append(
                Tier(
                    price=_parse_int(row[header_map[f"{p}-prices"]]),
                    action_price=_parse_int(row[header_map[f"{p}-actions-prices"]]),
                    intervals_to_next=_parse_int(row[header_map[f"{p}-intervals"]]),
                )
            )

    return out


# ----------------------------
# Drawing
# ----------------------------

def generate_price_tracks() -> None:
    if not INPUT_TSV_PATH.exists():
        raise FileNotFoundError(f"Missing {INPUT_TSV_PATH.resolve()}")

    tiers_by_prefix = parse_tiers_tsv(INPUT_TSV_PATH)

    order = ["pp", "ll", "cc", "tt"]
    names = {
        "pp": "Price Manipulation",
        "ll": "Liquidity Events",
        "cc": "Corporate Espionage",
        "tt": "Technological Leaps",
    }

    img = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    draw = ImageDraw.Draw(img)

    title_font = get_font(56, bold=True)
    label_font = get_font(42, bold=True)
    num_font = get_font(34)

    # Title
    title = "Stock Price Tracks"
    tb = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((CANVAS_W - (tb[2] - tb[0])) // 2, 40),
        title,
        font=title_font,
        fill="black",
    )

    base_y_top = MARGIN_Y
    base_y_bottom = MARGIN_Y + TRACK_HEIGHT

    for i, pref in enumerate(order):
        tiers = tiers_by_prefix[pref]
        n = len(tiers)

        x = MARGIN_X + i * TRACK_GAP_X

        # Track name
        draw.text((x - 80, base_y_top - 70), names[pref], font=label_font, fill="black")

        # Vertical line
        draw.line((x, base_y_top, x, base_y_bottom), fill="black", width=LINE_W)

        step = TRACK_HEIGHT / (n - 1)

        ys = [int(base_y_bottom - j * step) for j in range(n)]

        for j, (y, tier) in enumerate(zip(ys, tiers)):
            # Big dot
            draw.ellipse(
                (x - BIG_DOT_R, y - BIG_DOT_R, x + BIG_DOT_R, y + BIG_DOT_R),
                fill="black",
            )

            # Numbers
            ap = str(tier.action_price)
            pr = str(tier.price)

            ab = draw.textbbox((0, 0), ap, font=num_font)
            draw.text((x - BIG_DOT_R - 14 - (ab[2] - ab[0]), y - 18), ap, font=num_font, fill="black")
            draw.text((x + BIG_DOT_R + 14, y - 18), pr, font=num_font, fill="black")

            # Small dots to next tier
            if j < n - 1:
                n_small = tier.intervals_to_next
                if n_small > 0:
                    y2 = ys[j + 1]
                    seg_top = y - BIG_DOT_R - 10
                    seg_bot = y2 + BIG_DOT_R + 10
                    seg_h = seg_bot - seg_top

                    for k in range(1, n_small + 1):
                        t = k / (n_small + 1)
                        sy = int(seg_top + t * seg_h)
                        draw.ellipse(
                            (x - SMALL_DOT_R, sy - SMALL_DOT_R, x + SMALL_DOT_R, sy + SMALL_DOT_R),
                            fill="black",
                        )

    OUTPUT_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUTPUT_PNG_PATH)
    print(f"Wrote {OUTPUT_PNG_PATH.resolve()}")


if __name__ == "__main__":
    generate_price_tracks()
