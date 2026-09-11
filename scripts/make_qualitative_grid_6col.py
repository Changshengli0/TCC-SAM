#!/usr/bin/env python3
"""Build the final 6-column qualitative comparison figure for selected samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "Image & Expression",
    "Ground Truth",
    "OpenWorldSAM",
    "DIT-SAM",
    "SSP-SAM",
    "TCC-SAM (Ours)",
]
GT_COLOR = (230, 139, 0)
GT_EDGE = (164, 93, 0)
PRED_COLOR = (0, 114, 255)
PRED_EDGE = (0, 77, 180)
BORDER = (218, 223, 230)
TEXT = (22, 25, 30)
MUTED = (75, 85, 99)
PILL_BG = (241, 245, 249)
PILL_BORDER = (203, 213, 225)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_font(size: int, bold: bool = False):
    candidates: List[str] = []
    if bold:
        candidates += [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]
    candidates += [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    words = str(text or "").split()
    if not words:
        return [""]
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_size(draw, candidate, font)[0] <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        if text_size(draw, word, font)[0] <= max_width:
            current = word
        else:
            chunk = ""
            for ch in word:
                trial = chunk + ch
                if text_size(draw, trial, font)[0] <= max_width:
                    chunk = trial
                else:
                    if chunk:
                        lines.append(chunk)
                    chunk = ch
            current = chunk
    if current:
        lines.append(current)
    return lines


def load_rgb(path: Path) -> Optional[Image.Image]:
    if not path.exists():
        return None
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def load_mask(path: Optional[Path], size: Tuple[int, int]) -> Optional[Image.Image]:
    if path is None or not path.exists():
        return None
    with Image.open(path) as img:
        mask = img.convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return mask.point(lambda p: 255 if p > 0 else 0, mode="L")


def union_bbox(masks: Sequence[Optional[Image.Image]]) -> Optional[Tuple[int, int, int, int]]:
    union = None
    for mask in masks:
        if mask is None or mask.getbbox() is None:
            continue
        union = mask.copy() if union is None else ImageChops.lighter(union, mask)
    return None if union is None else union.getbbox()


def clamp_box(cx: float, cy: float, w: float, h: float, image_w: int, image_h: int) -> Tuple[int, int, int, int]:
    w = max(1.0, min(float(image_w), w))
    h = max(1.0, min(float(image_h), h))
    x0 = min(max(0.0, cx - w / 2), max(0.0, image_w - w))
    y0 = min(max(0.0, cy - h / 2), max(0.0, image_h - h))
    x1 = x0 + w
    y1 = y0 + h
    ix0, iy0, ix1, iy1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    return ix0, iy0, max(ix0 + 1, min(image_w, ix1)), max(iy0 + 1, min(image_h, iy1))


def match_aspect(w: float, h: float, image_w: int, image_h: int, aspect: float) -> Tuple[float, float]:
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    if w > image_w:
        w = float(image_w)
        h = w / aspect
    if h > image_h:
        h = float(image_h)
        w = h * aspect
    return min(float(image_w), w), min(float(image_h), h)


def compute_crop(size: Tuple[int, int], masks: Sequence[Optional[Image.Image]], args) -> Tuple[int, int, int, int]:
    image_w, image_h = size
    aspect = args.tile_w / args.tile_h
    bbox = union_bbox(masks) if args.crop_mode == "context" else None
    if bbox is None:
        w, h = match_aspect(float(image_w), float(image_h), image_w, image_h, aspect)
        return clamp_box(image_w / 2, image_h / 2, w, h, image_w, image_h)
    x0, y0, x1, y1 = bbox
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    crop_w = max(box_w * (1 + 2 * args.crop_pad), image_w * args.min_crop_ratio)
    crop_h = max(box_h * (1 + 2 * args.crop_pad), image_h * args.min_crop_ratio)
    crop_w, crop_h = match_aspect(crop_w, crop_h, image_w, image_h, aspect)
    return clamp_box((x0 + x1) / 2, (y0 + y1) / 2, crop_w, crop_h, image_w, image_h)


def contour(mask: Image.Image, width: int = 2) -> Image.Image:
    binary = mask.point(lambda p: 255 if p > 0 else 0, mode="L")
    dilated = binary
    eroded = binary
    for _ in range(max(1, width)):
        dilated = dilated.filter(ImageFilter.MaxFilter(3))
        eroded = eroded.filter(ImageFilter.MinFilter(3))
    return ImageChops.subtract(dilated, eroded).filter(ImageFilter.GaussianBlur(0.25))


def render_overlay(image: Image.Image, mask: Optional[Image.Image], fill: Tuple[int, int, int], edge: Tuple[int, int, int]) -> Image.Image:
    if mask is None or mask.getbbox() is None:
        return image.copy()
    base = image.convert("RGBA")
    fill_layer = Image.new("RGBA", base.size, fill + (0,))
    fill_layer.putalpha(mask.point(lambda p: 82 if p > 0 else 0, mode="L"))
    base = Image.alpha_composite(base, fill_layer)
    edge_layer = Image.new("RGBA", base.size, edge + (0,))
    edge_layer.putalpha(contour(mask, 1))
    return Image.alpha_composite(base, edge_layer).convert("RGB")


def crop_tile(image: Image.Image, crop: Tuple[int, int, int, int], tile_size: Tuple[int, int]) -> Image.Image:
    return image.crop(crop).resize(tile_size, Image.Resampling.LANCZOS)


def draw_centered(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], text: str, font, fill=TEXT):
    tw, th = text_size(draw, text, font)
    x0, y0, x1, y1 = box
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2 - 1), text, font=font, fill=fill)


def draw_pill(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], label: str, font):
    x, y = xy
    tw, th = text_size(draw, label, font)
    box = [x, y, x + tw + 14, y + th + 8]
    draw.rounded_rectangle(box, radius=4, fill=PILL_BG, outline=PILL_BORDER, width=1)
    draw.text((x + 7, y + 3), label, font=font, fill=MUTED)
    return box[3]


def collect_rows(args):
    selected = read_csv(resolve(args.selected_csv))
    summary_rows = {r["sample_uid"]: r for r in read_csv(resolve(args.summary_csv))}
    dit_dir = resolve(args.dit_dir)
    ssp_dir = resolve(args.ssp_dir)
    rows = []
    missing: List[Dict[str, str]] = []
    for item in selected:
        uid = item["sample_uid"]
        s = summary_rows.get(uid)
        if s is None:
            missing.append({"sample_uid": uid, "column": "summary", "path": str(resolve(args.summary_csv))})
            continue
        raw_path = resolve(s.get("raw_path") or s.get("image_path") or "")
        image_path = resolve(s.get("image_path") or s.get("raw_path") or "")
        raw = load_rgb(raw_path) or load_rgb(image_path)
        if raw is None:
            missing.append({"sample_uid": uid, "column": "Image & Expression", "path": str(raw_path)})
            continue
        size = raw.size
        paths = {
            "gt_mask": resolve(s.get("gt_mask_path", "")),
            "open_mask": resolve(s.get("open_mask_path", "")),
            "ours_mask": resolve(s.get("ours_mask_path", "")),
            "dit_mask": dit_dir / f"{uid}_mask.png",
            "ssp_mask": ssp_dir / f"{uid}_mask.png",
            "gt_overlay": resolve(s.get("gt_overlay_path", "")),
            "open_overlay": resolve(s.get("open_overlay_path", "")),
            "ours_overlay": resolve(s.get("ours_overlay_path", "")),
            "dit_overlay": dit_dir / f"{uid}_overlay.png",
            "ssp_overlay": ssp_dir / f"{uid}_overlay.png",
        }
        masks = {k: load_mask(v, size) for k, v in paths.items() if k.endswith("_mask")}
        checks = [
            ("Ground Truth", paths["gt_overlay"]),
            ("OpenWorldSAM", paths["open_overlay"]),
            ("DIT-SAM", paths["dit_overlay"]),
            ("SSP-SAM", paths["ssp_overlay"]),
            ("TCC-SAM (Ours)", paths["ours_overlay"]),
        ]
        for col, path in checks:
            if not path.exists():
                missing.append({"sample_uid": uid, "column": col, "path": str(path)})
        rows.append({
            "sample_uid": uid,
            "challenge_label": item.get("challenge_label") or s.get("challenge_type", ""),
            "expression": s.get("expression", ""),
            "raw": raw,
            "masks": masks,
            "paths": paths,
        })
    return rows, missing


def build_figure(rows, missing, args, output_dir: Path):
    tile = (args.tile_w, args.tile_h)
    cols = len(COLUMNS)
    width = args.margin * 2 + cols * args.tile_w + (cols - 1) * args.col_gap
    row_h = args.tile_h + args.caption_h
    height = args.margin * 2 + args.header_h + len(rows) * row_h + max(0, len(rows) - 1) * args.row_gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    header_font = load_font(args.header_font, bold=True)
    label_font = load_font(args.label_font, bold=True)
    expr_font = load_font(args.expr_font, bold=False)
    missing_font = load_font(max(12, args.expr_font - 2), bold=False)

    y = args.margin
    for c, title in enumerate(COLUMNS):
        x = args.margin + c * (args.tile_w + args.col_gap)
        fill = (15, 23, 42) if title != "TCC-SAM (Ours)" else (0, 74, 145)
        draw_centered(draw, (x, y, x + args.tile_w, y + args.header_h), title, header_font, fill=fill)
    y += args.header_h

    for row in rows:
        raw = row["raw"]
        masks = row["masks"]
        crop = compute_crop(raw.size, [masks.get(k) for k in ["gt_mask", "open_mask", "dit_mask", "ssp_mask", "ours_mask"]], args)
        images = [
            raw,
            render_overlay(raw, masks.get("gt_mask"), GT_COLOR, GT_EDGE),
            render_overlay(raw, masks.get("open_mask"), PRED_COLOR, PRED_EDGE),
            render_overlay(raw, masks.get("dit_mask"), PRED_COLOR, PRED_EDGE),
            render_overlay(raw, masks.get("ssp_mask"), PRED_COLOR, PRED_EDGE),
            render_overlay(raw, masks.get("ours_mask"), PRED_COLOR, PRED_EDGE),
        ]
        for c, image in enumerate(images):
            x = args.margin + c * (args.tile_w + args.col_gap)
            tile_img = crop_tile(image, crop, tile)
            canvas.paste(tile_img, (x, y))
            draw.rectangle([x, y, x + args.tile_w - 1, y + args.tile_h - 1], outline=BORDER, width=1)
            if c == 0:
                pill_y = y + args.tile_h + 5
                text_y = draw_pill(draw, (x + 6, pill_y), row["challenge_label"], label_font) + 3
                lines = wrap_text(draw, row["expression"], expr_font, args.tile_w - 12)
                for line in lines[:2]:
                    draw.text((x + 6, text_y), line, font=expr_font, fill=TEXT)
                    text_y += args.expr_font + 4
                if len(lines) > 2:
                    draw.text((x + 6, text_y), "...", font=missing_font, fill=MUTED)
        y += row_h + args.row_gap

    png = output_dir / "qualitative_comparison_6col.png"
    png_600 = output_dir / "qualitative_comparison_6col_600dpi.png"
    pdf = output_dir / "qualitative_comparison_6col.pdf"
    canvas.save(png, dpi=(300, 300))
    canvas.save(png_600, dpi=(600, 600))
    canvas.save(pdf, "PDF", resolution=300)
    report = {
        "selected_count": len(rows) + len([m for m in missing if m["column"] == "summary"]),
        "rendered_rows": len(rows),
        "missing": missing,
        "output_dir": str(output_dir),
        "png": str(png),
        "png_600dpi": str(png_600),
        "pdf": str(pdf),
        "image_size": list(canvas.size),
        "columns": COLUMNS,
    }
    (output_dir / "qualitative_comparison_6col_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--dit-dir", required=True)
    parser.add_argument("--ssp-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tile-w", type=int, default=285)
    parser.add_argument("--tile-h", type=int, default=205)
    parser.add_argument("--caption-h", type=int, default=54)
    parser.add_argument("--header-h", type=int, default=40)
    parser.add_argument("--margin", type=int, default=12)
    parser.add_argument("--col-gap", type=int, default=6)
    parser.add_argument("--row-gap", type=int, default=8)
    parser.add_argument("--header-font", type=int, default=18)
    parser.add_argument("--label-font", type=int, default=15)
    parser.add_argument("--expr-font", type=int, default=16)
    parser.add_argument("--crop-mode", choices=["context", "full"], default="context")
    parser.add_argument("--crop-pad", type=float, default=0.50)
    parser.add_argument("--min-crop-ratio", type=float, default=0.50)
    parser.add_argument("--panel-mode", default="cover")
    args = parser.parse_args()

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, missing = collect_rows(args)
    report = build_figure(rows, missing, args, output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
