#!/usr/bin/env python3
"""Build a compact qualitative comparison grid for six selected samples."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = [
    {
        "sample_uid": "refcocog_val_umd_369509_p000_bfb4441e09a",
        "expression": "big clock face is shown from the side angle.",
    },
    {
        "sample_uid": "refcocog_val_umd_210187_p001_8ad79cd318",
        "expression": "a book on table",
    },
    {
        "sample_uid": "refcocog_val_umd_33581_p001_d93eb03999",
        "expression": "the spoon to the left of the cup",
    },
    {
        "sample_uid": "refcocog_val_umd_223459_p001_a78688869c",
        "expression": "a jet third from the right in a group of identical jets.",
    },
    {
        "sample_uid": "refcocog_val_umd_290185_p001_a69fe29d7",
        "expression": "napkins stacked on a round, brown table.",
    },
    {
        "sample_uid": "refcocog_val_umd_315221_p001_cdc8bff6d7",
        "expression": "the bird in front of the other bird.",
    },
]
COLUMNS = ["Image", "GT", "OpenWorldSAM", "DIT-SAM", "SSP-SAM", "TCC-SAM"]
LETTERS = "abcdefghijklmnopqrstuvwxyz"
GT_COLOR = (235, 102, 28)
GT_EDGE = (180, 61, 12)
PRED_COLOR = (0, 110, 215)
PRED_EDGE = (0, 72, 160)
BORDER = (214, 219, 226)
TEXT = (18, 24, 33)
MUTED = (73, 82, 96)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def norm_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().replace(".", " ").replace(",", " ").split())


def uid_parts(uid: str) -> Tuple[str, str]:
    parts = str(uid or "").split("_")
    image_id = parts[3] if len(parts) > 3 else ""
    prompt = parts[4][1:] if len(parts) > 4 and parts[4].startswith("p") else ""
    return image_id, prompt


def find_col(row: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    lowered = {k.lower(): k for k in row.keys()}
    for name in candidates:
        if name in row:
            return name
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def path_from_row(row: Dict[str, str], candidates: Iterable[str]) -> Optional[Path]:
    col = find_col(row, candidates)
    if col and row.get(col):
        return resolve(row[col])
    return None


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


def text_box(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_centered(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], text: str, font, fill=TEXT) -> None:
    tw, th = text_box(draw, text, font)
    x0, y0, x1, y1 = box
    draw.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2 - 1), text, font=font, fill=fill)


def truncate_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def load_rgb(path: Optional[Path]) -> Optional[Image.Image]:
    if path is None or not path.exists():
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
    return union.getbbox() if union is not None else None


def match_aspect(w: float, h: float, max_w: int, max_h: int, aspect: float) -> Tuple[float, float]:
    if w / h < aspect:
        w = h * aspect
    else:
        h = w / aspect
    if w > max_w:
        w = float(max_w)
        h = w / aspect
    if h > max_h:
        h = float(max_h)
        w = h * aspect
    return min(float(max_w), w), min(float(max_h), h)


def clamp_crop(cx: float, cy: float, w: float, h: float, max_w: int, max_h: int) -> Tuple[int, int, int, int]:
    w = max(1.0, min(w, float(max_w)))
    h = max(1.0, min(h, float(max_h)))
    x0 = min(max(0.0, cx - w / 2), max(0.0, max_w - w))
    y0 = min(max(0.0, cy - h / 2), max(0.0, max_h - h))
    x1 = x0 + w
    y1 = y0 + h
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def compute_crop(size: Tuple[int, int], masks: Sequence[Optional[Image.Image]], args) -> Tuple[int, int, int, int]:
    image_w, image_h = size
    aspect = args.tile_w / args.tile_h
    bbox = union_bbox(masks)
    if bbox is None:
        crop_w, crop_h = match_aspect(float(image_w), float(image_h), image_w, image_h, aspect)
        return clamp_crop(image_w / 2, image_h / 2, crop_w, crop_h, image_w, image_h)
    x0, y0, x1, y1 = bbox
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    crop_w = max(box_w * (1 + 2 * args.crop_pad), image_w * args.min_crop_ratio)
    crop_h = max(box_h * (1 + 2 * args.crop_pad), image_h * args.min_crop_ratio)
    crop_w, crop_h = match_aspect(crop_w, crop_h, image_w, image_h, aspect)
    return clamp_crop((x0 + x1) / 2, (y0 + y1) / 2, crop_w, crop_h, image_w, image_h)


def contour(mask: Image.Image) -> Image.Image:
    binary = mask.point(lambda p: 255 if p > 0 else 0, mode="L")
    dilated = binary.filter(ImageFilter.MaxFilter(3))
    eroded = binary.filter(ImageFilter.MinFilter(3))
    return ImageChops.subtract(dilated, eroded).filter(ImageFilter.GaussianBlur(0.2))


def render_overlay(image: Image.Image, mask: Optional[Image.Image], fill: Tuple[int, int, int], edge: Tuple[int, int, int]) -> Image.Image:
    if mask is None or mask.getbbox() is None:
        return image.copy()
    base = image.convert("RGBA")
    fill_layer = Image.new("RGBA", base.size, fill + (0,))
    fill_layer.putalpha(mask.point(lambda p: 84 if p > 0 else 0, mode="L"))
    base = Image.alpha_composite(base, fill_layer)
    edge_layer = Image.new("RGBA", base.size, edge + (0,))
    edge_layer.putalpha(contour(mask))
    return Image.alpha_composite(base, edge_layer).convert("RGB")


def tile_image(image: Image.Image, crop: Tuple[int, int, int, int], size: Tuple[int, int]) -> Image.Image:
    return image.crop(crop).resize(size, Image.Resampling.LANCZOS)


def resolve_target(target: Dict[str, str], rows: List[Dict[str, str]]) -> Tuple[Optional[Dict[str, str]], str, float]:
    requested_uid = str(target.get("sample_uid", "")).strip()
    by_uid = {str(r.get("sample_uid", "")).strip(): r for r in rows}
    if requested_uid in by_uid:
        return by_uid[requested_uid], "exact_sample_uid", 1.0

    image_id, prompt = uid_parts(requested_uid)
    candidates = []
    for row in rows:
        uid = str(row.get("sample_uid", "")).strip()
        row_image_id, row_prompt = uid_parts(uid)
        uid_score = difflib.SequenceMatcher(None, requested_uid, uid).ratio()
        expr_score = difflib.SequenceMatcher(None, norm_text(target.get("expression", "")), norm_text(row.get("expression", ""))).ratio()
        same_image = row_image_id == image_id and bool(image_id)
        same_prompt = row_prompt == prompt and bool(prompt)
        score = uid_score + 0.30 * expr_score + (0.25 if same_image else 0.0) + (0.10 if same_prompt else 0.0)
        if same_image or uid_score > 0.70:
            candidates.append((score, row))
    if not candidates:
        matches = difflib.get_close_matches(requested_uid, list(by_uid.keys()), n=1, cutoff=0.0)
        if matches:
            return by_uid[matches[0]], "closest_sample_uid", difflib.SequenceMatcher(None, requested_uid, matches[0]).ratio()
        return None, "unresolved", 0.0
    score, row = max(candidates, key=lambda item: item[0])
    method = "fuzzy_image_prompt_expression" if str(row.get("sample_uid", "")) != requested_uid else "exact_sample_uid"
    return row, method, round(score, 6)


def load_targets(path: Optional[str]) -> List[Dict[str, str]]:
    if not path:
        return DEFAULT_TARGETS
    rows = read_csv(resolve(path))
    targets = []
    for row in rows:
        uid_col = find_col(row, ["sample_uid", "uid", "requested_sample_uid"])
        expr_col = find_col(row, ["expression", "expr", "text"])
        if uid_col:
            targets.append({"sample_uid": row[uid_col], "expression": row.get(expr_col or "", "")})
    return targets


def collect_rows(args) -> Tuple[List[Dict[str, object]], List[Dict[str, str]], List[Dict[str, str]]]:
    merged_rows = read_csv(resolve(args.merged_summary))
    main_rows = {r.get("sample_uid", ""): r for r in read_csv(resolve(args.main_summary))}
    targets = load_targets(args.selected_csv)
    dit_dir = resolve(args.dit_dir)
    ssp_dir = resolve(args.ssp_dir)
    render_rows: List[Dict[str, object]] = []
    resolved_rows: List[Dict[str, str]] = []
    missing: List[Dict[str, str]] = []

    for idx, target in enumerate(targets):
        merged, resolved_from, match_score = resolve_target(target, merged_rows)
        if merged is None:
            missing.append({"requested_sample_uid": target.get("sample_uid", ""), "sample_uid": "", "item": "row", "path": ""})
            continue
        uid = str(merged.get("sample_uid", "")).strip()
        main = main_rows.get(uid, {})
        expression = merged.get("expression") or main.get("expression") or target.get("expression", "")
        challenge = merged.get("challenge_label") or main.get("challenge_type", "")

        raw_path = path_from_row(merged, ["raw_path", "image_path"]) or path_from_row(main, ["raw_path", "image_path"])
        raw = load_rgb(raw_path)
        if raw is None:
            image_path = path_from_row(main, ["image_path"])
            raw = load_rgb(image_path)
            raw_path = image_path
        if raw is None:
            missing.append({"requested_sample_uid": target.get("sample_uid", ""), "sample_uid": uid, "item": "Image", "path": str(raw_path or "")})
            continue

        size = raw.size
        paths = {
            "Image": raw_path,
            "GT": path_from_row(merged, ["gt_overlay_path"]) or path_from_row(main, ["gt_overlay_path"]),
            "OpenWorldSAM": path_from_row(merged, ["open_overlay_path"]) or path_from_row(main, ["open_overlay_path"]),
            "DIT-SAM": path_from_row(merged, ["dit_overlay_path"]) or (dit_dir / f"{uid}_overlay.png"),
            "SSP-SAM": path_from_row(merged, ["ssp_overlay_path"]) or (ssp_dir / f"{uid}_overlay.png"),
            "TCC-SAM": path_from_row(merged, ["ours_overlay_path", "tcc_overlay_path"]) or path_from_row(main, ["ours_overlay_path"]),
        }
        mask_paths = {
            "GT": path_from_row(main, ["gt_mask_path"]) or paths["GT"],
            "OpenWorldSAM": path_from_row(main, ["open_mask_path"]) or paths["OpenWorldSAM"],
            "DIT-SAM": dit_dir / f"{uid}_mask.png",
            "SSP-SAM": ssp_dir / f"{uid}_mask.png",
            "TCC-SAM": path_from_row(main, ["ours_mask_path"]) or paths["TCC-SAM"],
        }
        for item, path in paths.items():
            if path is None or not path.exists():
                missing.append({"requested_sample_uid": target.get("sample_uid", ""), "sample_uid": uid, "item": item, "path": str(path or "")})
        masks = {name: load_mask(path, size) for name, path in mask_paths.items()}

        resolved_rows.append({
            "requested_sample_uid": target.get("sample_uid", ""),
            "sample_uid": uid,
            "resolved_from": resolved_from,
            "match_score": str(match_score),
            "challenge_label": challenge,
            "expression": expression,
            "iou_open": merged.get("iou_open", ""),
            "iou_dit": merged.get("iou_dit", ""),
            "iou_ssp": merged.get("iou_ssp", ""),
            "iou_ours": merged.get("iou_ours", ""),
        })
        render_rows.append({
            "letter": LETTERS[idx],
            "sample_uid": uid,
            "expression": expression,
            "challenge_label": challenge,
            "raw": raw,
            "paths": paths,
            "masks": masks,
            "requested_sample_uid": target.get("sample_uid", ""),
            "resolved_from": resolved_from,
        })

    return render_rows, resolved_rows, missing


def draw_caption(draw: ImageDraw.ImageDraw, canvas: Image.Image, args, rows: Sequence[Dict[str, object]], y: int) -> None:
    font = load_font(args.caption_font, bold=False)
    line = "  ".join(f"({row['letter']}) {truncate_text(str(row['expression']), 55)}" for row in rows)
    max_w = canvas.size[0] - 2 * args.margin
    if text_box(draw, line, font)[0] <= max_w:
        draw.text((args.margin, y), line, font=font, fill=MUTED)
        return
    x = args.margin
    line_h = args.caption_font + 5
    for row in rows:
        item = f"({row['letter']}) {truncate_text(str(row['expression']), 55)}"
        iw = text_box(draw, item, font)[0]
        if x > args.margin and x + iw > max_w:
            x = args.margin
            y += line_h
        draw.text((x, y), item, font=font, fill=MUTED)
        x += iw + 18


def build_figure(rows: Sequence[Dict[str, object]], missing: Sequence[Dict[str, str]], args, output_dir: Path) -> Dict[str, object]:
    tile_size = (args.tile_w, args.tile_h)
    width = args.margin * 2 + len(COLUMNS) * args.tile_w + (len(COLUMNS) - 1) * args.col_gap
    caption_h = args.caption_h
    height = args.margin * 2 + args.header_h + len(rows) * args.tile_h + max(0, len(rows) - 1) * args.row_gap + caption_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(args.header_font, bold=True)
    row_font = load_font(args.row_font, bold=True)

    y = args.margin
    for c, title in enumerate(COLUMNS):
        x = args.margin + c * (args.tile_w + args.col_gap)
        fill = (8, 31, 61) if title == "TCC-SAM" else TEXT
        draw_centered(draw, (x, y, x + args.tile_w, y + args.header_h), title, header_font, fill=fill)
    y += args.header_h

    crops: Dict[str, List[int]] = {}
    for row in rows:
        raw = row["raw"]
        assert isinstance(raw, Image.Image)
        masks = row["masks"]
        assert isinstance(masks, dict)
        crop = compute_crop(raw.size, [masks.get(c) for c in COLUMNS[1:]], args)
        crops[str(row["sample_uid"])] = list(crop)
        panels = [
            raw,
            render_overlay(raw, masks.get("GT"), GT_COLOR, GT_EDGE),
            render_overlay(raw, masks.get("OpenWorldSAM"), PRED_COLOR, PRED_EDGE),
            render_overlay(raw, masks.get("DIT-SAM"), PRED_COLOR, PRED_EDGE),
            render_overlay(raw, masks.get("SSP-SAM"), PRED_COLOR, PRED_EDGE),
            render_overlay(raw, masks.get("TCC-SAM"), PRED_COLOR, PRED_EDGE),
        ]
        for c, panel in enumerate(panels):
            x = args.margin + c * (args.tile_w + args.col_gap)
            canvas.paste(tile_image(panel, crop, tile_size), (x, y))
            draw.rectangle([x, y, x + args.tile_w - 1, y + args.tile_h - 1], outline=BORDER, width=1)
            if c == 0:
                label = f"({row['letter']})"
                tw, th = text_box(draw, label, row_font)
                draw.rectangle([x + 4, y + 4, x + tw + 12, y + th + 10], fill=(255, 255, 255), outline=(226, 232, 240))
                draw.text((x + 8, y + 7), label, font=row_font, fill=TEXT)
        y += args.tile_h + args.row_gap

    draw_caption(draw, canvas, args, rows, height - args.margin - caption_h + 8)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / "qualitative_simple_grid_6methods.png"
    png_600 = output_dir / "qualitative_simple_grid_6methods_600dpi.png"
    pdf = output_dir / "qualitative_simple_grid_6methods.pdf"
    canvas.save(png, dpi=(300, 300))
    canvas.save(png_600, dpi=(600, 600))
    canvas.save(pdf, "PDF", resolution=300)
    return {
        "output_dir": str(output_dir),
        "png": str(png),
        "png_600dpi": str(png_600),
        "pdf": str(pdf),
        "image_size": list(canvas.size),
        "columns": COLUMNS,
        "rendered_rows": len(rows),
        "missing": list(missing),
        "crops": crops,
    }


def write_expressions(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in rows:
        lines.append(f"({row['letter']}) {row['sample_uid']} | {row['expression']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-summary", required=True)
    parser.add_argument("--main-summary", required=True)
    parser.add_argument("--dit-dir", required=True)
    parser.add_argument("--ssp-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selected-csv", default=None)
    parser.add_argument("--resolved-csv", default="outputs/paper_figs/qualitative_comparison/review_sheets_4methods/resolved_selected6_simple.csv")
    parser.add_argument("--expressions-txt", default="outputs/paper_figs/qualitative_comparison/review_sheets_4methods/final_simple_grid_selected6_expressions.txt")
    parser.add_argument("--tile-w", type=int, default=210)
    parser.add_argument("--tile-h", type=int, default=155)
    parser.add_argument("--header-h", type=int, default=34)
    parser.add_argument("--margin", type=int, default=8)
    parser.add_argument("--col-gap", type=int, default=5)
    parser.add_argument("--row-gap", type=int, default=5)
    parser.add_argument("--header-font", type=int, default=18)
    parser.add_argument("--row-font", type=int, default=12)
    parser.add_argument("--caption-font", type=int, default=11)
    parser.add_argument("--caption-h", type=int, default=48)
    parser.add_argument("--crop-pad", type=float, default=0.55)
    parser.add_argument("--min-crop-ratio", type=float, default=0.48)
    args = parser.parse_args()

    output_dir = resolve(args.output_dir)
    rows, resolved_rows, missing = collect_rows(args)
    report = build_figure(rows, missing, args, output_dir)

    resolved_csv = resolve(args.resolved_csv)
    write_csv(
        resolved_csv,
        resolved_rows,
        [
            "requested_sample_uid",
            "sample_uid",
            "resolved_from",
            "match_score",
            "challenge_label",
            "expression",
            "iou_open",
            "iou_dit",
            "iou_ssp",
            "iou_ours",
        ],
    )
    expressions_txt = resolve(args.expressions_txt)
    write_expressions(expressions_txt, rows)
    output_expr = output_dir / "final_simple_grid_selected6_expressions.txt"
    if output_expr != expressions_txt:
        write_expressions(output_expr, rows)

    report.update({
        "requested_count": len(load_targets(args.selected_csv)),
        "resolved_count": len(resolved_rows),
        "resolved_csv": str(resolved_csv),
        "expressions_txt": str(expressions_txt),
        "output_expressions_txt": str(output_expr),
        "resolved": resolved_rows,
    })
    report_path = output_dir / "qualitative_simple_grid_6methods_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
