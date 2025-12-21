#!/usr/bin/env python3
"""
make_price_tracks.py

Generate a single PNG image containing 4 horizontal price tracks (PP, LL, CC, TT)
from a TSV file shaped like:

pp-prices    pp-actions-prices    pp-intervals    ll-prices    ll-actions-prices    ll-intervals    ...

Rules implemented (per your spec):
- For each type, draw a horizontal track with "tiers" represented by BIG dots.
- For each BIG dot:
    - left label  = actions-price value
    - right label = price value
- Between consecutive BIG dots, draw N small dots, where N is the "intervals" value
  from the LEFT tier row (i.e., the interval count associated with that tier).
  If N=0, no small dots.

Output is designed to be imported into your larger pipeline; this file can be called
from your main script.

Usage:
  python make_price_tracks.py input.tsv --out price_tracks.png

Or from code:
  from make_price_tracks import generate_price_tracks
  generate_price_tracks("tiers.tsv", "price_tracks.png")
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont


# ----------------------------
# Model
# ----------------------------

@dataclass(frozen=True)
class Tier:
    price: int
    action_price: int
    intervals_to_next: int  # number of small dots between this tier and next tier


# ----------------------------
# Fonts (portable-ish)
# ----------------------------

def _get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates: List[str] = []

    # Windows common fonts
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

    # Cross-platform fallbacks if present in Pillow bundle / system
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
    # allow commas if you ever use "1,000"
    cell = cell.replace(",", "")
    return int(cell)


def parse_tiers_tsv(path: str | Path) -> Dict[str, List[Tier]]:
    """
    Reads TSV with header columns:
      <prefix>-prices, <prefix>-actions-prices, <prefix>-intervals

    Returns dict { prefix: [Tier, Tier, ...] } in row order.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    lines = [ln for ln in raw.splitlines() if ln.strip() != ""]
    if not lines:
        raise ValueError("Input TSV is empty (or only whitespace).")

    reader = csv.reader(lines, delimiter="\t")
    header = next(reader, None)
    if not header:
        raise ValueError("Missing header row.")

    # Find prefixes by locating "-prices" columns
    # Example header cell: "pp-prices"
    prefixes: List[str] = []
    col_idx: Dict[str, int] = {}
    for i, h in enumerate(header):
        h2 = _clean(h)
        if h2.endswith("-prices"):
            pref = h2[:-len("-prices")]
            prefixes.append(pref)

    if not prefixes:
        raise ValueError("Could not find any '<prefix>-prices' columns in header.")

    # Build lookup for expected columns per prefix
    for pref in prefixes:
        want = {
            f"{pref}-prices": None,
            f"{pref}-actions-prices": None,
            f"{pref}-intervals": None,
        }
        for i, h in enumerate(header):
            hh = _clean(h)
            if hh in want:
                want[hh] = i
        missing = [k for k, v in want.items() if v is None]
        if missing:
            raise ValueError(f"Missing required columns for prefix '{pref}': {missing}")
        col_idx[f"{pref}-prices"] = int(want[f"{pref}-prices"])  # type: ignore[arg-type]
        col_idx[f"{pref}-actions-prices"] = int(want[f"{pref}-actions-prices"])  # type: ignore[arg-type]
        col_idx[f"{pref}-intervals"] = int(want[f"{pref}-intervals"])  # type: ignore[arg-type]

    out: Dict[str, List[Tier]] = {pref: [] for pref in prefixes}

    for row_idx, row in enumerate(reader, start=2):
        # pad row to header length
        while len(row) < len(header):
            row.append("")

        # If the whole row is empty-ish, skip
        if all(_clean(c) == "" for c in row):
            continue

        for pref in prefixes:
            price = _parse_int_cell(row[col_idx[f"{pref}-prices"]], default=0)
            action_price = _parse_int_cell(row[col_idx[f"{pref}-actions-prices"]], default=0)
            intervals = _parse_int_cell(row[col_idx[f"{pref}-intervals"]], default=0)
            out[pref].append(Tier(price=price, action_price=action_price, intervals_to_next=intervals))

    # Validate monotonic tiers (optional, but helps catch TSV mistakes)
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

def generate_price_tracks(
    input_tsv: str | Path,
    out_png: str | Path,
    *,
    # Canvas sizing
    width_px: int = 2200,
    height_px: int = 1100,
    margin_px: int = 90,
    track_gap_px: int = 230,
    # Dot sizing
    big_dot_r: int = 14,
    small_dot_r: int = 6,
    line_w: int = 5,
) -> None:
    tiers_by_prefix = parse_tiers_tsv(input_tsv)

    # Deterministic ordering for your 4 tracks:
    # Prefer pp,ll,cc,tt if present; otherwise alpha.
    preferred = ["pp", "ll", "cc", "tt"]
    prefixes = [p for p in preferred if p in tiers_by_prefix] + [p for p in sorted(tiers_by_prefix) if p not in preferred]

    # Only draw first 4 (per your spec)
    prefixes = prefixes[:4]
    if len(prefixes) != 4:
        raise ValueError(f"Expected 4 tracks, found {len(prefixes)}: {prefixes}")

    img = Image.new("RGB", (width_px, height_px), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    title_font = _get_font(54, bold=True)
    label_font_bold = _get_font(40, bold=True)
    label_font = _get_font(36, bold=False)

    # Title
    title = "Stock Price Tracks"
    tb = draw.textbbox((0, 0), title, font=title_font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text(((width_px - tw) // 2, margin_px // 2), title, font=title_font, fill=(0, 0, 0))

    # Track y positions
    top_y = margin_px + 90
    x0 = margin_px
    x1 = width_px - margin_px

    for track_i, pref in enumerate(prefixes):
        tiers = tiers_by_prefix[pref]
        y = top_y + track_i * track_gap_px

        # Track label on the left
        track_name = {
            "pp": "Price Manipulation",
            "ll": "Liquidity Events",
            "cc": "Corporate Espionage",
            "tt": "Technological Leaps",
        }.get(pref, pref.upper())

        draw.text((x0, y - 78), track_name, font=label_font_bold, fill=(0, 0, 0))

        # Baseline line
        draw.line((x0, y, x1, y), fill=(0, 0, 0), width=line_w)

        # Big dot x positions evenly spaced
        n_tiers = len(tiers)
        # Keep a little internal padding so dots don't sit exactly on margins
        inner_pad = 180
        bx0 = x0 + inner_pad
        bx1 = x1 - inner_pad
        if n_tiers == 1:
            xs = [int((bx0 + bx1) / 2)]
        else:
            step = (bx1 - bx0) / (n_tiers - 1)
            xs = [int(round(bx0 + i * step)) for i in range(n_tiers)]

        # Draw big dots + labels, then small dots between
        for i, (x, tier) in enumerate(zip(xs, tiers)):
            # Big dot
            draw.ellipse((x - big_dot_r, y - big_dot_r, x + big_dot_r, y + big_dot_r), fill=(0, 0, 0))

            # Labels:
            # left label = action_price (slightly above baseline)
            # right label = price (slightly above baseline)
            left_txt = str(tier.action_price)
            right_txt = str(tier.price)

            lb = draw.textbbox((0, 0), left_txt, font=label_font)
            ltw, lth = lb[2] - lb[0], lb[3] - lb[1]
            rb = draw.textbbox((0, 0), right_txt, font=label_font)
            rtw, rth = rb[2] - rb[0], rb[3] - rb[1]

            # Place action_price to the left of big dot
            draw.text((x - big_dot_r - 12 - ltw, y - 62), left_txt, font=label_font, fill=(0, 0, 0))
            # Place price to the right of big dot
            draw.text((x + big_dot_r + 12, y - 62), right_txt, font=label_font, fill=(0, 0, 0))

            # Small dots between this tier and next (using this tier's intervals_to_next)
            if i < n_tiers - 1:
                n_small = max(0, int(tier.intervals_to_next))
                if n_small > 0:
                    x_next = xs[i + 1]
                    seg_left = x + big_dot_r + 10
                    seg_right = x_next - big_dot_r - 10
                    seg_w = max(0, seg_right - seg_left)

                    # Evenly distribute n_small dots across the segment
                    for k in range(1, n_small + 1):
                        # position at k/(n_small+1) to keep away from endpoints
                        t = k / (n_small + 1)
                        sx = int(round(seg_left + t * seg_w))
                        draw.ellipse((sx - small_dot_r, y - small_dot_r, sx + small_dot_r, y + small_dot_r), fill=(0, 0, 0))

    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png, format="PNG")


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_tsv", type=str, help="TSV file with pp/ll/cc/tt tiers")
    ap.add_argument("--out", type=str, default="price_tracks.png", help="Output PNG path")
    ap.add_argument("--w", type=int, default=2200, help="Image width in px")
    ap.add_argument("--h", type=int, default=1100, help="Image height in px")
    args = ap.parse_args()

    generate_price_tracks(args.input_tsv, args.out, width_px=args.w, height_px=args.h)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
