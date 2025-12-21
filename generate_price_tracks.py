#!/usr/bin/env python3
"""
generate_price_tracks.py

No-CLI-args version: uses fixed, consistent paths every run.

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
# Fixed paths (edit if needed)
# ----------------------------

INPUT_TSV_PATH = Path("price_tiers.tsv")               # must exist in your current working directory
OUTPUT_PNG_PATH = Path("build_tts") / "price_tracks.png"


# ----------------------------
# Drawing defaults
# ----------------------------

CANVAS_W_PX = 2200
CANVAS_H_PX = 1100
MARGIN_PX = 90
TRACK_GAP_PX = 230

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
# Fonts (portable-ish)
# ----------------------------

def _get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates: List[str] = []

    if sys.platform.startswith("win"):
        if bold:
            candidates += [
                r"C:\Windows\Fonts\arialbd.ttf",
                r"C:\Windows\Fonts\calibrib.ttf",
                r"C:\Windows\Fonts\segoeuib.ttf",
            ]
        else:
            candidates += [
                r"C:\Windows\Fonts\arial.ttf",
                r"C:\Windows\Fonts\calibri.ttf",
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


def _parse_int_cell(cell: str, default: int = 0) -> int:
    cell = _clean(cell)
    if cell == "":
        return default
    cell = cell.replace(",", "")
    return int(cell)


def parse_tiers_tsv(path: Path) -> Dict[str, List[Tier]]:
    raw = path.read_text(encoding="utf-8")
    lines = [ln for ln in raw.splitlines() if ln.strip() != ""]
    if not lines:
        raise ValueError(f"Input TSV is empty (or only whitespace): {path}")

    reader = csv.reader(lines, delimiter="\t")
    header = next(reader, None)
    if not header:
        raise ValueError("Missing header row.")

    # IMPORTANT FIX:
    # Only treat "<prefix>-prices" as the base price column if it is NOT the actions-price column.
    # i.e. include "*-prices" but exclude "*-actions-prices"
    prefixes: List[str] = []
    for h in header:
        hh = _clean(h)
        if hh.endswith("-prices") and not hh.endswith("-actions-prices"):
            prefixes.append(hh[:-len("-prices")])

    if not prefixes:
        raise ValueError(
            "Could not find any '<prefix>-prices' columns in header "
            "(excluding '*-actions-prices')."
        )

    # column indices for each prefix
    col_idx: Dict[str, int] = {}
    header_map = {_clean(h): i for i, h in enumerate(header)}

    for pref in prefixes:
        prices_key = f"{pref}-prices"
        actions_key = f"{pref}-actions-prices"
        intervals_key = f"{pref}-intervals"

        missing = [k for k in [prices_key, actions_key, intervals_key] if k not in header_map]
        if missing:
            raise ValueError(f"Missing required columns for prefix '{pref}': {missing}")

        col_idx[prices_key] = header_map[prices_key]
        col_idx[actions_key] = header_map[actions_key]
        col_idx[intervals_key] = header_map[intervals_key]

    out: Dict[str, List[Tier]] = {pref: [] for pref in prefixes}

    for row_idx, row in enumerate(reader, start=2):
        while len(row) < len(header):
            row.append("")

        if all(_clean(c) == "" for c in row):
            continue

        for pref in prefixes:
            price = _parse_int_cell(row[col_idx[f"{pref}-prices"]], default=0)
            action_price = _parse_int_cell(row[col_idx[f"{pref}-actions-prices"]], default=0)
            intervals = _parse_int_cell(row[col_idx[f"{pref}-intervals"]], default=0)
            out[pref].append(Tier(price=price, action_price=action_price, intervals_to_next=intervals))

    for pref, tiers in out.items():
        if len(tiers) < 2:
            raise ValueError(f"Prefix '{pref}' needs at least 2 tiers to draw a track.")
        for i, t in enumerate(tiers):
            if t.price <= 0:
                raise ValueError(f"Prefix '{pref}' has non-positive price at row {i+1}: {t.price}")

    return out


# ----------------------------
# Drawing
# ----------------------------

def generate_price_tracks() -> None:
    if not INPUT_TSV_PATH.exists():
        cwd = Path.cwd().resolve()
        raise FileNotFoundError(
            f"Missing input TSV: {INPUT_TSV_PATH.resolve()}\n"
            f"Current working directory: {cwd}\n"
            f"Fix: place your tiers file at '{cwd / INPUT_TSV_PATH}', or change INPUT_TSV_PATH in this script."
        )

    tiers_by_prefix = parse_tiers_tsv(INPUT_TSV_PATH)

    preferred = ["pp", "ll", "cc", "tt"]
    prefixes = [p for p in preferred if p in tiers_by_prefix] + [p for p in sorted(tiers_by_prefix) if p not in preferred]
    prefixes = prefixes[:4]
    if len(prefixes) != 4:
        raise ValueError(f"Expected 4 tracks, found {len(prefixes)}: {prefixes}")

    img = Image.new("RGB", (CANVAS_W_PX, CANVAS_H_PX), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_font = _get_font(54, bold=True)
    label_font_bold = _get_font(40, bold=True)
    label_font = _get_font(36, bold=False)

    title = "Stock Price Tracks"
    tb = draw.textbbox((0, 0), title, font=title_font)
    tw, _ = tb[2] - tb[0], tb[3] - tb[1]
    draw.text(((CANVAS_W_PX - tw) // 2, MARGIN_PX // 2), title, font=title_font, fill=(0, 0, 0))

    top_y = MARGIN_PX + 90
    x0 = MARGIN_PX
    x1 = CANVAS_W_PX - MARGIN_PX

    name_map = {
        "pp": "Price Manipulation",
        "ll": "Liquidity Events",
        "cc": "Corporate Espionage",
        "tt": "Technological Leaps",
    }

    for track_i, pref in enumerate(prefixes):
        tiers = tiers_by_prefix[pref]
        y = top_y + track_i * TRACK_GAP_PX

        track_name = name_map.get(pref, pref.upper())
        draw.text((x0, y - 78), track_name, font=label_font_bold, fill=(0, 0, 0))

        draw.line((x0, y, x1, y), fill=(0, 0, 0), width=LINE_W)

        n_tiers = len(tiers)
        inner_pad = 180
        bx0 = x0 + inner_pad
        bx1 = x1 - inner_pad

        step = (bx1 - bx0) / (n_tiers - 1)
        xs = [int(round(bx0 + i * step)) for i in range(n_tiers)]

        for i, (x, tier) in enumerate(zip(xs, tiers)):
            draw.ellipse((x - BIG_DOT_R, y - BIG_DOT_R, x + BIG_DOT_R, y + BIG_DOT_R), fill=(0, 0, 0))

            left_txt = str(tier.action_price)
            right_txt = str(tier.price)

            lb = draw.textbbox((0, 0), left_txt, font=label_font)
            ltw = lb[2] - lb[0]
            draw.text((x - BIG_DOT_R - 12 - ltw, y - 62), left_txt, font=label_font, fill=(0, 0, 0))
            draw.text((x + BIG_DOT_R + 12, y - 62), right_txt, font=label_font, fill=(0, 0, 0))

            if i < n_tiers - 1:
                n_small = max(0, int(tier.intervals_to_next))
                if n_small > 0:
                    x_next = xs[i + 1]
                    seg_left = x + BIG_DOT_R + 10
                    seg_right = x_next - BIG_DOT_R - 10
                    seg_w = max(0, seg_right - seg_left)

                    for k in range(1, n_small + 1):
                        t = k / (n_small + 1)
                        sx = int(round(seg_left + t * seg_w))
                        draw.ellipse((sx - SMALL_DOT_R, y - SMALL_DOT_R, sx + SMALL_DOT_R, y + SMALL_DOT_R), fill=(0, 0, 0))

    OUTPUT_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUTPUT_PNG_PATH, format="PNG")
    print(f"Wrote: {OUTPUT_PNG_PATH.resolve()}")


def main() -> None:
    generate_price_tracks()


if __name__ == "__main__":
    main()
