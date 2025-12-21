#!/usr/bin/env python3
"""
generate_price_tracks.py

Vertical price tracks renderer with CONSISTENT spacing per interval "tick".

Interpretation:
- For each segment between tier i and tier i+1:
    intervals_to_next = N means there are N small dots between the big dots.
  That implies (N + 1) equal steps from big dot to next big dot.
- We make each step the same pixel distance, so segments with more intervals
  take proportionally more vertical space.

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
MARGIN_Y = 160

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
    header = next(reader, None)
    if not header:
        raise ValueError("Missing header row.")

    header_map = {_clean(h): i for i, h in enumerate(header)}

    prefixes = [
        h[:-len("-prices")]
        for h in header
        if _clean(h).endswith("-prices") and not _clean(h).endswith("-actions-prices")
    ]
    if not prefixes:
        raise ValueError("Could not find any '<prefix>-prices' columns (excluding '*-actions-prices').")

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

    for p, tiers in out.items():
        if len(tiers) < 2:
            raise ValueError(f"Prefix '{p}' needs at least 2 tiers.")
        for i, t in enumerate(tiers):
            if t.price <= 0:
                raise ValueError(f"Prefix '{p}' has non-positive price at row {i+1}: {t.price}")

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
    Compute y positions for big dots using consistent step spacing.

    Total "steps" = sum over segments (intervals_to_next + 1).
    (The +1 is the jump from big dot to next big dot, with intervals small dots between.)
    """
    if len(tiers) < 2:
        return [(y_top + y_bottom) // 2]

    segment_steps = [max(0, t.intervals_to_next) + 1 for t in tiers[:-1]]
    total_steps = sum(segment_steps)
    if total_steps <= 0:
        # Fallback: evenly spaced
        n = len(tiers)
        step_px = (y_bottom - y_top) / (n - 1)
        return [int(round(y_top + i * step_px)) for i in range(n)]

    step_px = (y_bottom - y_top) / total_steps

    ys: List[int] = [y_top]
    cur = float(y_top)
    for seg in segment_steps:
        cur += seg * step_px
        ys.append(int(round(cur)))

    # ys length = len(tiers)
    return ys


# ----------------------------
# Drawing
# ----------------------------

def generate_price_tracks() -> None:
    if not INPUT_TSV_PATH.exists():
        cwd = Path.cwd().resolve()
        raise FileNotFoundError(
            f"Missing input TSV: {INPUT_TSV_PATH.resolve()}\n"
            f"Current working directory: {cwd}\n"
            f"Fix: place the file at '{cwd / INPUT_TSV_PATH}'."
        )

    tiers_by_prefix = parse_tiers_tsv(INPUT_TSV_PATH)

    order = ["pp", "ll", "cc", "tt"]
    names = {
        "pp": "Price Manipulation",
        "ll": "Liquidity Events",
        "cc": "Corporate Espionage",
        "tt": "Technological Leaps",
    }

    # Ensure we have all 4 (or fail loudly)
    missing = [p for p in order if p not in tiers_by_prefix]
    if missing:
        raise ValueError(f"Missing required prefixes in TSV: {missing}. Found: {sorted(tiers_by_prefix.keys())}")

    img = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    draw = ImageDraw.Draw(img)

    title_font = get_font(56, bold=True)
    label_font = get_font(42, bold=True)
    num_font = get_font(34, bold=False)

    title = "Stock Price Tracks"
    tb = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((CANVAS_W - (tb[2] - tb[0])) // 2, 40),
        title,
        font=title_font,
        fill="black",
    )

    y_top = MARGIN_Y
    y_bottom = CANVAS_H - MARGIN_Y

    for i, pref in enumerate(order):
        tiers = tiers_by_prefix[pref]
        x = MARGIN_X + i * TRACK_GAP_X

        # Label
        draw.text((x - 90, y_top - 80), names[pref], font=label_font, fill="black")

        # Track line
        draw.line((x, y_top, x, y_bottom), fill="black", width=LINE_W)

        # Big-dot Y positions computed by step counts
        big_ys = compute_tier_positions_y(tiers, y_top=y_top, y_bottom=y_bottom)

        for j, (y, tier) in enumerate(zip(big_ys, tiers)):
            # Big dot
            draw.ellipse(
                (x - BIG_DOT_R, y - BIG_DOT_R, x + BIG_DOT_R, y + BIG_DOT_R),
                fill="black",
            )

            # Labels: left = action_price, right = price
            left_txt = str(tier.action_price)
            right_txt = str(tier.price)

            lb = draw.textbbox((0, 0), left_txt, font=num_font)
            ltw = lb[2] - lb[0]
            draw.text((x - BIG_DOT_R - 14 - ltw, y - 18), left_txt, font=num_font, fill="black")
            draw.text((x + BIG_DOT_R + 14, y - 18), right_txt, font=num_font, fill="black")

            # Small dots between this big dot and next big dot
            if j < len(tiers) - 1:
                n_small = max(0, int(tier.intervals_to_next))
                if n_small > 0:
                    y_next = big_ys[j + 1]

                    # We want (n_small + 1) equal steps from big to next big.
                    # Place small dots at 1..n_small steps along the segment.
                    total_steps = n_small + 1
                    for k in range(1, n_small + 1):
                        t = k / total_steps
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
