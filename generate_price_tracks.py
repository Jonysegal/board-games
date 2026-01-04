#!/usr/bin/env python3
"""
generate_price_tracks.py

Vertical price tracks renderer with:
- Consistent spacing per interval tick
- NO title
- Even horizontal spacing between tracks

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

MARGIN_X = 140
MARGIN_Y = 140

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

    candidates += ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]

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
    if not lines:
        raise ValueError(f"Input TSV is empty: {path}")

    reader = csv.reader(lines, delimiter="\t")
    header = next(reader)

    header_map = {_clean(h): i for i, h in enumerate(header)}

    prefixes = [
        h[:-len("-prices")]
        for h in header
        if _clean(h).endswith("-prices") and not _clean(h).endswith("-actions-prices")
    ]

    out: Dict[str, List[Tier]] = {p: [] for p in prefixes}

    for row in reader:
        while len(row) < len(header):
            row.append("")
        if all(_clean(c) == "" for c in row):
            continue

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
# Geometry helpers
# ----------------------------

def compute_tier_positions_y(
    tiers: List[Tier],
    y_top: int,
    y_bottom: int,
) -> List[int]:
    """
    Compute Y positions for big dots using consistent step spacing.
    Each segment has (intervals + 1) equal steps.
    """
    segment_steps = [max(0, t.intervals_to_next) + 1 for t in tiers[:-1]]
    total_steps = sum(segment_steps)

    if total_steps <= 0:
        step = (y_bottom - y_top) / (len(tiers) - 1)
        return [int(y_top + i * step) for i in range(len(tiers))]

    step_px = (y_bottom - y_top) / total_steps

    ys = [y_top]
    cur = float(y_top)
    for seg in segment_steps:
        cur += seg * step_px
        ys.append(int(round(cur)))

    return ys


# ----------------------------
# Drawing
# ----------------------------

def generate_price_tracks() -> None:
    if not INPUT_TSV_PATH.exists():
        raise FileNotFoundError(f"Missing input TSV: {INPUT_TSV_PATH.resolve()}")

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

    label_font = get_font(42, bold=True)
    num_font = get_font(34)

    y_top = MARGIN_Y
    y_bottom = CANVAS_H - MARGIN_Y

    # Even horizontal spacing
    usable_w = CANVAS_W - 2 * MARGIN_X
    step_x = usable_w / (len(order) - 1)
    xs = [int(MARGIN_X + i * step_x) for i in range(len(order))]

    for x, pref in zip(xs, order):
        tiers = tiers_by_prefix[pref]

        # Track label
        draw.text((x - 90, y_top - 70), names[pref], font=label_font, fill="black")

        # Track line
        draw.line((x, y_top, x, y_bottom), fill="black", width=LINE_W)

        big_ys = compute_tier_positions_y(tiers, y_top, y_bottom)

        # Flip Y positions so higher numeric values appear at the top
        # (reflect positions within the available vertical range).
        big_ys = [int(round(y_top + y_bottom - y)) for y in big_ys]

        for i, (y, tier) in enumerate(zip(big_ys, tiers)):
            # Big dot
            draw.ellipse(
                (x - BIG_DOT_R, y - BIG_DOT_R, x + BIG_DOT_R, y + BIG_DOT_R),
                fill="black",
            )

            # Labels
            ap = str(tier.action_price)
            pr = str(tier.price)

            ab = draw.textbbox((0, 0), ap, font=num_font)
            draw.text(
                (x - BIG_DOT_R - 14 - (ab[2] - ab[0]), y - 18),
                ap,
                font=num_font,
                fill="black",
            )
            draw.text((x + BIG_DOT_R + 14, y - 18), pr, font=num_font, fill="black")

            # Small dots
            if i < len(tiers) - 1:
                n_small = tier.intervals_to_next
                if n_small > 0:
                    y_next = big_ys[i + 1]
                    for k in range(1, n_small + 1):
                        t = k / (n_small + 1)
                        sy = int(round(y + t * (y_next - y)))
                        draw.ellipse(
                            (x - SMALL_DOT_R, sy - SMALL_DOT_R, x + SMALL_DOT_R, sy + SMALL_DOT_R),
                            fill="black",
                        )

    OUTPUT_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUTPUT_PNG_PATH, format="PNG")
    print(f"Wrote {OUTPUT_PNG_PATH.resolve()}")


if __name__ == "__main__":
    generate_price_tracks()
