#!/usr/bin/env python3
"""
make_tts_deck.py

Spreadsheet/TSV -> TTS-ready deck assets:
1) Parse TSV-ish input (section rows + card rows with: name, text, image-link, card-count)
2) Expand by card-count
3) Render each card face as a PNG (default 744x1039)
4) Pack faces into one or more sprite sheets (<= 4096x4096 by default)
5) Emit a Tabletop Simulator deck JSON that references those sprite sheets

PUBLIC URL SUPPORT + VERSIONING
- Provide --public-sprites-root to auto-populate FaceURL for each section sheet:
    <public-sprites-root>/<Section_Folder>/deck_01.png
- Provide --public-back-url or let it derive back.png from the sprites root:
    if sprites root ends with /sprites, back defaults to the parent .../back.png
- Provide --url-version to append ?v=<value> (or &v= if query already exists) to ALL FaceURL/BackURL,
  which helps avoid Tabletop Simulator image caching during iteration.

COLORED BACKS FOR MAIN DECKS
- If --public-backs-root is provided, this script will override BackURL per deck for:
    Price Manipulation  -> Red.png
    Liquidity Events    -> Blue.png   (you wrote "Liquidation"—interpreted as Liquidity Events)
    Corporate Espionage -> Green.png
    Technological Leaps -> Yellow.png
  URL scheme:
    <public-backs-root>/<Color>.png
  Example:
    https://raw.githubusercontent.com/Jonysegal/board-games/refs/heads/main/backs/Blue.png

Example (your repo):
  python make_tts_deck.py cards.tsv --out-dir build_tts --per-section --write-back ^
    --public-sprites-root "https://raw.githubusercontent.com/Jonysegal/board-games/main/build_tts/sprites" ^
    --public-backs-root "https://raw.githubusercontent.com/Jonysegal/board-games/refs/heads/main/backs" ^
    --url-version "1"

Dependencies:
  pip install pillow requests cairosvg
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse, urlencode, urlunparse, parse_qs

import requests
from PIL import Image, ImageDraw, ImageFont


# ----------------------------
# Data model
# ----------------------------

@dataclass(frozen=True)
class CardDef:
    section: str
    name: str
    text: str
    image_link: str
    card_count: int


@dataclass(frozen=True)
class PrintableCard:
    section: str
    name: str
    text: str
    image_link: str
    copy_index: int  # 1..N


# ----------------------------
# TSV-ish parsing
# ----------------------------

def _clean(s: Optional[str]) -> str:
    return (s or "").strip()


def parse_card_file_tsvish(path: Path) -> List[CardDef]:
    raw = path.read_text(encoding="utf-8")
    lines = [ln for ln in raw.splitlines() if ln.strip() != ""]
    if not lines:
        raise ValueError("Input file is empty (or only whitespace).")

    reader = csv.reader(lines, delimiter="\t")

    header = next(reader, None)
    if header is None or len(header) < 2:
        raise ValueError("Missing header row or insufficient columns.")

    current_section: Optional[str] = None
    out: List[CardDef] = []

    for row_idx, row in enumerate(reader, start=2):
        while len(row) < 4:
            row.append("")

        name = _clean(row[0])
        text = _clean(row[1])
        image_link = _clean(row[2])
        count_raw = _clean(row[3])

        if not name:
            continue

        # Section row: name only
        if name and not text and not image_link and not count_raw:
            current_section = name
            continue

        if current_section is None:
            raise ValueError(f"Card row before any section header on line {row_idx}: {row}")

        if not text:
            raise ValueError(f"Missing card text on line {row_idx}: {row}")
        if not image_link:
            raise ValueError(f"Missing image-link on line {row_idx}: {row}")
        if not count_raw:
            raise ValueError(f"Missing card-count on line {row_idx}: {row}")

        try:
            card_count = int(count_raw)
        except ValueError:
            raise ValueError(f"Invalid card-count '{count_raw}' on line {row_idx}: {row}")

        out.append(
            CardDef(
                section=current_section,
                name=name,
                text=text,
                image_link=image_link,
                card_count=card_count,
            )
        )

    return out


def expand_counts(cards: List[CardDef]) -> List[PrintableCard]:
    expanded: List[PrintableCard] = []
    for c in cards:
        if c.card_count <= 0:
            continue
        for i in range(1, c.card_count + 1):
            expanded.append(
                PrintableCard(
                    section=c.section,
                    name=c.name,
                    text=c.text,
                    image_link=c.image_link,
                    copy_index=i,
                )
            )
    return expanded


# ----------------------------
# Image download + caching
# ----------------------------

def normalize_wikimedia_url(url: str, width: int = 512) -> str:
    u = url.strip()

    m = re.search(r"(https?://commons\.wikimedia\.org)/wiki/File:(.+)$", u)
    if m:
        base = m.group(1)
        filename = unquote(m.group(2)).replace(" ", "_")
        direct = f"{base}/wiki/Special:FilePath/{filename}"
        return _ensure_width_query(direct, width)

    if "commons.wikimedia.org" in u and "/wiki/Special:FilePath/" in u:
        return _ensure_width_query(u, width)

    return u


def _ensure_width_query(url: str, width: int) -> str:
    parts = list(urlparse(url))
    q = parse_qs(parts[4])
    if "width" not in q:
        q["width"] = [str(width)]
    parts[4] = urlencode(q, doseq=True)
    return urlunparse(parts)


def safe_cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]


def download_bytes(url: str, timeout: int = 25) -> Tuple[bytes, str]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "tts-card-deck-maker/1.0"})
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
    return r.content, ctype


def svg_to_png_bytes(svg_bytes: bytes, png_width: int = 512) -> Optional[bytes]:
    try:
        import cairosvg  # type: ignore
    except Exception:
        return None
    try:
        return cairosvg.svg2png(bytestring=svg_bytes, output_width=png_width)
    except Exception:
        return None


def pil_from_bytes(data: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg.convert("RGB")
    else:
        im = im.convert("RGB")
    return im


def load_icon_image(
    image_link: str,
    cache_dir: Path,
    wikimedia_width: int = 512,
    svg_png_width: int = 512,
) -> Optional[Image.Image]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    normalized = normalize_wikimedia_url(image_link, width=wikimedia_width)
    key = safe_cache_key(normalized)
    meta_path = cache_dir / f"{key}.json"
    bin_path = cache_dir / f"{key}.bin"

    if meta_path.exists() and bin_path.exists():
        try:
            data = bin_path.read_bytes()
            return pil_from_bytes(data)
        except Exception:
            pass

    try:
        data, ctype = download_bytes(normalized)
    except Exception:
        return None

    is_svg = ("svg" in ctype) or data.lstrip().startswith(b"<svg") or (b"<svg" in data[:2000])
    if is_svg:
        png = svg_to_png_bytes(data, png_width=svg_png_width)
        if png is None:
            return None
        data = png

    try:
        meta_path.write_text(json.dumps({"url": normalized, "content_type": ctype}, indent=2), encoding="utf-8")
        bin_path.write_bytes(data)
    except Exception:
        pass

    try:
        return pil_from_bytes(data)
    except Exception:
        return None


# ----------------------------
# Card rendering (PNG per card)
# ----------------------------

def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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


def wrap_text_pil(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    cur: List[str] = []

    def width(s: str) -> int:
        bbox = draw.textbbox((0, 0), s, font=font)
        return bbox[2] - bbox[0]

    for w in words:
        trial = (" ".join(cur + [w])).strip()
        if width(trial) <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def render_card_png(
    card: PrintableCard,
    out_path: Path,
    icon_cache_dir: Path,
    card_size: Tuple[int, int],
    wikimedia_width: int,
    svg_png_width: int,
) -> None:
    W, H = card_size
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    pad = int(0.04 * W)
    name_h = int(0.10 * H)
    art_h = int(0.46 * H)
    type_h = int(0.06 * H)
    gap = int(0.012 * H)

    name_font = get_font(52, bold=True)
    type_font = get_font(36, bold=True)
    rules_font = get_font(32, bold=False)

    name_box = (pad, pad, W - pad, pad + name_h)
    draw.rectangle(name_box, outline=(0, 0, 0), width=3)
    draw.text((pad + 14, pad + 10), card.name, font=name_font, fill=(0, 0, 0))

    art_top = name_box[3] + gap
    art_box = (pad, art_top, W - pad, art_top + art_h)
    draw.rectangle(art_box, outline=(0, 0, 0), width=3)

    icon = load_icon_image(
        card.image_link,
        cache_dir=icon_cache_dir,
        wikimedia_width=wikimedia_width,
        svg_png_width=svg_png_width,
    )

    if icon is not None:
        art_inner_pad = int(0.04 * W)
        target_w = (art_box[2] - art_box[0]) - 2 * art_inner_pad
        target_h = (art_box[3] - art_box[1]) - 2 * art_inner_pad

        iw, ih = icon.size
        if iw > 0 and ih > 0:
            scale = min(target_w / iw, target_h / ih)
            nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
            icon_r = icon.resize((nw, nh), resample=Image.LANCZOS)
            dx = art_box[0] + (art_box[2] - art_box[0] - nw) // 2
            dy = art_box[1] + (art_box[3] - art_box[1] - nh) // 2
            img.paste(icon_r, (dx, dy))
    else:
        ph = "IMAGE"
        bbox = draw.textbbox((0, 0), ph, font=type_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx = art_box[0] + (art_box[2] - art_box[0] - tw) // 2
        cy = art_box[1] + (art_box[3] - art_box[1] - th) // 2
        draw.text((cx, cy), ph, font=type_font, fill=(0, 0, 0))

    type_top = art_box[3] + gap
    type_box = (pad, type_top, W - pad, type_top + type_h)
    draw.rectangle(type_box, outline=(0, 0, 0), width=3)
    draw.text((pad + 14, type_top + 10), card.section, font=type_font, fill=(0, 0, 0))

    draw.line((pad, type_box[3] + 2, W - pad, type_box[3] + 2), fill=(0, 0, 0), width=2)

    rules_top = type_box[3] + gap
    rules_box = (pad, rules_top, W - pad, H - pad)
    draw.rectangle(rules_box, outline=(0, 0, 0), width=3)

    rules_inner_pad = int(0.03 * W)
    rules_x = rules_box[0] + rules_inner_pad
    rules_y = rules_box[1] + rules_inner_pad
    rules_w = (rules_box[2] - rules_box[0]) - 2 * rules_inner_pad
    rules_h = (rules_box[3] - rules_box[1]) - 2 * rules_inner_pad

    lines = wrap_text_pil(draw, card.text, rules_font, rules_w)
    line_h = int(rules_font.size * 1.25) if hasattr(rules_font, "size") else 34
    max_lines = max(1, rules_h // line_h)

    y = rules_y
    for ln in lines[:max_lines]:
        draw.text((rules_x, y), ln, font=rules_font, fill=(0, 0, 0))
        y += line_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")


def make_simple_back_image(path: Path, size: Tuple[int, int]) -> None:
    W, H = size
    img = Image.new("RGB", (W, H), (40, 40, 40))
    draw = ImageDraw.Draw(img)
    font = get_font(64, bold=True)
    text = "STOCKS"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((W - tw) // 2, (H - th) // 2), text, font=font, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


# ----------------------------
# Sprite sheet packing
# ----------------------------

def choose_grid_for_sheet(
    card_size: Tuple[int, int],
    max_sheet_px: int,
    preferred_cols: int,
) -> Tuple[int, int]:
    card_w, card_h = card_size
    max_cols = max(1, max_sheet_px // card_w)
    max_rows = max(1, max_sheet_px // card_h)

    cols = min(preferred_cols, max_cols)
    if cols < 1:
        cols = 1
    return cols, max_rows


def pack_into_sheets(
    card_image_paths: List[Path],
    out_dir: Path,
    card_size: Tuple[int, int],
    max_sheet_px: int = 4096,
    preferred_cols: int = 5,
) -> List[Tuple[Path, int, int, int]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    card_w, card_h = card_size
    cols, max_rows = choose_grid_for_sheet(card_size, max_sheet_px, preferred_cols)
    max_per_sheet = cols * max_rows

    sheets: List[Tuple[Path, int, int, int]] = []
    total = len(card_image_paths)
    sheet_count = math.ceil(total / max_per_sheet) if max_per_sheet > 0 else 1

    idx = 0
    for s in range(sheet_count):
        remaining = total - idx
        take = min(remaining, max_per_sheet)

        rows = math.ceil(take / cols)
        rows = max(1, min(rows, max_rows))

        sheet_w = cols * card_w
        sheet_h = rows * card_h
        sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))

        for j in range(take):
            p = card_image_paths[idx + j]
            im = Image.open(p).convert("RGB")
            x = (j % cols) * card_w
            y = (j // cols) * card_h
            sheet.paste(im, (x, y))

        sheet_path = out_dir / f"deck_{s+1:02d}.png"
        sheet.save(sheet_path, format="PNG")
        sheets.append((sheet_path, cols, rows, take))
        idx += take

    return sheets


# ----------------------------
# Public URL + version helpers
# ----------------------------

def _with_version(url: str, v: str) -> str:
    if not v:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}v={v}"


def _sanitize_folder(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")


def derive_back_url_from_public_sprites_root(public_sprites_root: str) -> str:
    root = public_sprites_root.rstrip("/")
    if root.lower().endswith("/sprites"):
        root = root[: -len("/sprites")]
    return f"{root}/back.png"


def build_face_urls_for_group(
    sheet_specs: List[Tuple[Path, int, int, int]],
    public_sprites_root: str,
    group_folder_name: str,
    url_version: str,
) -> List[str]:
    root = public_sprites_root.rstrip("/")
    urls: List[str] = []
    for (sheet_path, _, _, _) in sheet_specs:
        u = f"{root}/{group_folder_name}/{sheet_path.name}"
        urls.append(_with_version(u, url_version))
    return urls


def colored_back_url_for_group(
    group_name: str,
    public_backs_root: str,
    url_version: str,
    fallback_back_url: str,
) -> str:
    """
    Override back URL for the 4 main decks:
      Price Manipulation  -> Red.png
      Liquidity Events    -> Blue.png
      Corporate Espionage -> Green.png
      Technological Leaps -> Yellow.png
    Otherwise return fallback_back_url.
    """
    if not public_backs_root:
        return fallback_back_url

    g = group_name.strip().lower()

    # Map group name -> color
    # Note: user wrote "Liquidation" but your section is "Liquidity Events"
    mapping = {
        "price manipulation": "Red",
        "price persuasion": "Red",          # tolerate naming drift
        "liquidity events": "Blue",
        "liquidation": "Blue",              # tolerate typo/alt label
        "corporate espionage": "Green",
        "espionage": "Green",
        "technological leaps": "Yellow",
        "technology": "Yellow",
        "technological leap": "Yellow",
    }

    color = mapping.get(g)
    if not color:
        return fallback_back_url

    root = public_backs_root.rstrip("/")
    u = f"{root}/{color}.png"
    return _with_version(u, url_version)


# ----------------------------
# TTS JSON generation
# ----------------------------

def tts_deck_object(
    deck_name: str,
    cards: List[PrintableCard],
    sheet_specs: List[Tuple[Path, int, int, int]],
    face_urls: List[str],
    back_url: str,
    pos: Tuple[float, float, float],
) -> dict:
    if len(sheet_specs) != len(face_urls):
        raise ValueError("sheet_specs and face_urls must be same length.")

    deck_ids: List[int] = []
    contained: List[dict] = []
    custom_deck: Dict[str, dict] = {}

    card_idx = 0
    deck_index = 1

    for (sheet_path, num_w, num_h, count_on_sheet), face_url in zip(sheet_specs, face_urls):
        custom_deck[str(deck_index)] = {
            "FaceURL": face_url,
            "BackURL": back_url,
            "NumWidth": num_w,
            "NumHeight": num_h,
            "BackIsHidden": True,
            "UniqueBack": False,
            "Type": 0,
        }

        for slot in range(count_on_sheet):
            card = cards[card_idx]
            card_id = deck_index * 100 + slot
            deck_ids.append(card_id)

            contained.append(
                {
                    "Name": "Card",
                    "Transform": {
                        "posX": 0,
                        "posY": 0,
                        "posZ": 0,
                        "rotX": 0,
                        "rotY": 180,
                        "rotZ": 0,
                        "scaleX": 1,
                        "scaleY": 1,
                        "scaleZ": 1,
                    },
                    "Nickname": card.name,
                    "Description": card.text,
                    "CardID": card_id,
                    "CustomDeck": {},
                }
            )
            card_idx += 1

        deck_index += 1

    obj = {
        "Name": "Deck",
        "Transform": {
            "posX": pos[0],
            "posY": pos[1],
            "posZ": pos[2],
            "rotX": 0,
            "rotY": 180,
            "rotZ": 0,
            "scaleX": 1,
            "scaleY": 1,
            "scaleZ": 1,
        },
        "Nickname": deck_name,
        "Description": "",
        "DeckIDs": deck_ids,
        "CustomDeck": custom_deck,
        "ContainedObjects": contained,
    }
    return obj


def build_face_urls(sheet_specs: List[Tuple[Path, int, int, int]], face_url_base: str) -> List[str]:
    urls: List[str] = []
    base = face_url_base.rstrip("/")
    for (sheet_path, _, _, _) in sheet_specs:
        urls.append(f"{base}/{sheet_path.name}" if base else f"REPLACE_WITH_PUBLIC_URL/{sheet_path.name}")
    return urls


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_tsv", type=str, help="TSV-ish card file")
    ap.add_argument("--out-dir", type=str, default="build_tts", help="Output directory")

    ap.add_argument("--per-section", action="store_true", help="Create one TTS deck per section (recommended)")
    ap.add_argument("--single-deck", action="store_true", help="Force a single deck containing all cards")

    ap.add_argument("--card-w", type=int, default=744, help="Card face width in px")
    ap.add_argument("--card-h", type=int, default=1039, help="Card face height in px")

    ap.add_argument("--max-sheet-px", type=int, default=4096, help="Max sprite sheet width/height in px")
    ap.add_argument("--preferred-cols", type=int, default=5, help="Preferred number of columns in sprite sheets")

    ap.add_argument("--wikimedia-width", type=int, default=512, help="Width parameter for Commons Special:FilePath")
    ap.add_argument("--svg-png-width", type=int, default=512, help="Raster width when converting SVG->PNG via cairosvg")

    # Legacy/manual hosting
    ap.add_argument("--face-url-base", type=str, default="", help="Base URL where deck_XX.png sheets will be hosted (legacy)")
    ap.add_argument("--back-url", type=str, default="", help="Public URL for back.png (legacy)")
    ap.add_argument("--write-back", action="store_true", help="Write a simple back.png into out-dir (still needs hosting).")

    # Public hosting rooted at sprites directory
    ap.add_argument(
        "--public-sprites-root",
        type=str,
        default="",
        help="Public URL root for sprites directory, e.g. https://raw.githubusercontent.com/<user>/<repo>/main/build_tts/sprites",
    )
    ap.add_argument(
        "--public-back-url",
        type=str,
        default="",
        help="Public URL for default back.png. If omitted and --public-sprites-root is set, derives from it.",
    )
    ap.add_argument(
        "--public-backs-root",
        type=str,
        default="",
        help="Public URL root for colored backs, e.g. https://raw.githubusercontent.com/<user>/<repo>/refs/heads/main/backs",
    )
    ap.add_argument(
        "--url-version",
        type=str,
        default="",
        help="Optional cache-buster appended as ?v=<value> to FaceURL/BackURL (recommended for TTS caching).",
    )

    args = ap.parse_args()

    in_path = Path(args.input_tsv)
    out_dir = Path(args.out_dir)
    faces_dir = out_dir / "faces"
    sprites_dir = out_dir / "sprites"
    cache_dir = out_dir / ".icon_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    card_size = (args.card_w, args.card_h)

    cards_def = parse_card_file_tsvish(in_path)
    expanded_all = expand_counts(cards_def)

    if args.single_deck and args.per_section:
        raise ValueError("Choose only one of --single-deck or --per-section.")
    if not args.single_deck and not args.per_section:
        args.per_section = True

    back_path = out_dir / "back.png"
    if args.write_back:
        make_simple_back_image(back_path, card_size)

    # Default BackURL (public)
    if args.public_back_url.strip():
        default_back_url = args.public_back_url.strip()
    elif args.public_sprites_root.strip():
        default_back_url = derive_back_url_from_public_sprites_root(args.public_sprites_root.strip())
    else:
        default_back_url = args.back_url.strip()

    if not default_back_url:
        default_back_url = "REPLACE_WITH_PUBLIC_URL/back.png"

    default_back_url = _with_version(default_back_url, args.url_version)

    # Partition cards by section if requested
    groups: List[Tuple[str, List[PrintableCard]]] = []
    if args.single_deck:
        groups = [("All Cards", expanded_all)]
    else:
        by_section: Dict[str, List[PrintableCard]] = {}
        for c in expanded_all:
            by_section.setdefault(c.section, []).append(c)
        for sec in sorted(by_section.keys()):
            groups.append((sec, by_section[sec]))

    tts_objects: List[dict] = []
    manifest_rows: List[List[str]] = []

    base_x = 0.0
    base_z = 0.0
    step_x = 3.5
    step_z = 0.0

    for gi, (group_name, group_cards) in enumerate(groups):
        if not group_cards:
            continue

        group_folder = _sanitize_folder(group_name)

        # Render faces
        group_face_dir = faces_dir / group_folder
        group_face_dir.mkdir(parents=True, exist_ok=True)

        face_paths: List[Path] = []
        for idx, card in enumerate(group_cards, start=1):
            safe_section = re.sub(r"[^a-zA-Z0-9_-]+", "_", card.section).strip("_").lower()
            safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", card.name).strip("_").lower()
            fn = f"{idx:04d}_{safe_section}_{safe_name}_{card.copy_index:02d}.png"
            out_path = group_face_dir / fn
            render_card_png(
                card=card,
                out_path=out_path,
                icon_cache_dir=cache_dir,
                card_size=card_size,
                wikimedia_width=args.wikimedia_width,
                svg_png_width=args.svg_png_width,
            )
            face_paths.append(out_path)

        # Pack into sprite sheets
        group_sprite_dir = sprites_dir / group_folder
        sheet_specs = pack_into_sheets(
            card_image_paths=face_paths,
            out_dir=group_sprite_dir,
            card_size=card_size,
            max_sheet_px=args.max_sheet_px,
            preferred_cols=args.preferred_cols,
        )

        # FaceURLs
        if args.public_sprites_root.strip():
            face_urls = build_face_urls_for_group(
                sheet_specs=sheet_specs,
                public_sprites_root=args.public_sprites_root.strip(),
                group_folder_name=group_folder,
                url_version=args.url_version,
            )
        else:
            face_urls = build_face_urls(sheet_specs, args.face_url_base)
            face_urls = [_with_version(u, args.url_version) for u in face_urls]

        # BackURL per deck (colored for 4 mains if requested)
        deck_back_url = colored_back_url_for_group(
            group_name=group_name,
            public_backs_root=args.public_backs_root.strip(),
            url_version=args.url_version,
            fallback_back_url=default_back_url,
        )

        pos = (base_x + gi * step_x, 1.0, base_z + gi * step_z)
        tts_obj = tts_deck_object(
            deck_name=group_name,
            cards=group_cards,
            sheet_specs=sheet_specs,
            face_urls=face_urls,
            back_url=deck_back_url,
            pos=pos,
        )
        tts_objects.append(tts_obj)

        for (sheet_path, num_w, num_h, count_on_sheet), url in zip(sheet_specs, face_urls):
            manifest_rows.append([
                group_name,
                sheet_path.name,
                str(sheet_path),
                str(num_w),
                str(num_h),
                str(count_on_sheet),
                url,
                deck_back_url,
            ])

    out_tts_json = out_dir / "tts_decks.json"
    payload = {"ObjectStates": tts_objects}
    out_tts_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_manifest = out_dir / "sprite_manifest.tsv"
    out_manifest.write_text(
        "deck_name\tsheet_file\tsheet_path\tnum_width\tnum_height\tcards_on_sheet\tface_url\tback_url\n"
        + "\n".join("\t".join(row) for row in manifest_rows),
        encoding="utf-8",
    )

    out_cards_json = out_dir / "cards_expanded.json"
    out_cards_json.write_text(json.dumps([asdict(c) for c in expanded_all], indent=2), encoding="utf-8")

    print(f"Wrote: {out_tts_json}")
    print(f"Wrote: {out_manifest}")
    print(f"Wrote: {out_cards_json}")
    if args.write_back:
        print(f"Wrote: {out_dir / 'back.png'}")
    print(f"Rendered cards: {len(expanded_all)}")
    print(f"Sprite sheets: {len(manifest_rows)}")

    if args.public_sprites_root.strip():
        print(f"Public sprites root: {args.public_sprites_root.strip()}")
    if args.public_backs_root.strip():
        print(f"Public backs root: {args.public_backs_root.strip()}")
    if args.url_version:
        print(f"URL version: v={args.url_version}")

    try:
        import cairosvg  # type: ignore
        print(f"cairosvg detected: {getattr(cairosvg, '__version__', 'unknown')}")
    except Exception as e:
        print("Note: cairosvg is not available in this Python environment. SVG icons may render as placeholders.")
        print(f"Python executable: {sys.executable}")
        print("Install into this interpreter with:")
        print("  python -m pip install cairosvg")
        print(f"Import error: {e}")


if __name__ == "__main__":
    main()
