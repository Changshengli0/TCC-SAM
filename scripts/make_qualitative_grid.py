#!/usr/bin/env python3
"""Build the final 4-column qualitative comparison grid."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps


COLUMN_TITLES = ["Image & Expression", "Ground Truth", "OpenWorldSAM", "OPC-SAM (Ours)"]
OUTPUT_BASENAME = "qualitative_comparison_grid"
GRID_ROWS = 6
GRID_COLS = 4
PANEL_BORDER = "#E5E7EB"
PANEL_PAD = 0
PILL_BG = "#EEF2F7"
PILL_BORDER = "#CBD5E1"
PILL_TEXT = "#334155"
BODY_TEXT = "#111827"
HEADER_TEXT = "#111827"
OURS_BLUE = "#1F4E79"
GT_STYLE = {
    "fill": "#E69F00",
    "alpha": 0.28,
    "contour": "#A16207",
}
PRED_STYLE = {
    "fill": "#2F6FDB",
    "alpha": 0.28,
    "contour": "#1D4ED8",
}


def read_csv_rows(path: Any) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend([
            "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ])
    candidates.extend([
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ])
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return int(box[2] - box[0]), int(box[3] - box[1])


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = str(text or "").split()
    if not words:
        return [""]
    lines: List[str] = []
    cur: List[str] = []
    for word in words:
        trial = " ".join(cur + [word])
        if text_size(draw, trial, font)[0] <= max_width or not cur:
            cur.append(word)
        else:
            lines.append(" ".join(cur))
            cur = [word]
    if cur:
        lines.append(" ".join(cur))
    return lines


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"expected 6-digit hex color, got {value!r}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def load_rgb_image(path: Any) -> Image.Image:
    if not path:
        raise ValueError("missing image path")
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def load_binary_mask(path: Any, image_size: Tuple[int, int]) -> Optional[Image.Image]:
    if not path:
        return None
    mask_path = Path(str(path))
    if not mask_path.exists():
        return None
    with Image.open(mask_path) as mask:
        mask = mask.convert("L")
    if mask.size != image_size:
        mask = mask.resize(image_size, resample=Image.Resampling.NEAREST)
    return mask.point(lambda p: 255 if p > 0 else 0, mode="L")


def union_bbox(masks: Sequence[Optional[Image.Image]]) -> Optional[Tuple[int, int, int, int]]:
    union: Optional[Image.Image] = None
    for mask in masks:
        if mask is None or mask.getbbox() is None:
            continue
        union = mask.copy() if union is None else ImageChops.lighter(union, mask)
    return None if union is None else union.getbbox()


def clamp_center_box(
    cx: float,
    cy: float,
    crop_w: float,
    crop_h: float,
    image_w: int,
    image_h: int,
) -> Tuple[int, int, int, int]:
    crop_w = min(float(image_w), max(1.0, crop_w))
    crop_h = min(float(image_h), max(1.0, crop_h))
    x0 = cx - crop_w / 2.0
    y0 = cy - crop_h / 2.0
    x0 = min(max(0.0, x0), max(0.0, image_w - crop_w))
    y0 = min(max(0.0, y0), max(0.0, image_h - crop_h))
    x1 = x0 + crop_w
    y1 = y0 + crop_h
    ix0 = int(round(x0))
    iy0 = int(round(y0))
    ix1 = int(round(x1))
    iy1 = int(round(y1))
    ix1 = min(image_w, max(ix0 + 1, ix1))
    iy1 = min(image_h, max(iy0 + 1, iy1))
    return ix0, iy0, ix1, iy1


def aspect_limited_size(
    crop_w: float,
    crop_h: float,
    image_w: int,
    image_h: int,
    target_aspect: float,
) -> Tuple[float, float]:
    crop_w = max(1.0, float(crop_w))
    crop_h = max(1.0, float(crop_h))
    if target_aspect > 0:
        if crop_w / crop_h < target_aspect:
            crop_w = crop_h * target_aspect
        else:
            crop_h = crop_w / target_aspect
    if crop_w > image_w:
        crop_w = float(image_w)
        crop_h = crop_w / target_aspect if target_aspect > 0 else crop_h
    if crop_h > image_h:
        crop_h = float(image_h)
        crop_w = crop_h * target_aspect if target_aspect > 0 else crop_w
    crop_w = min(float(image_w), max(1.0, crop_w))
    crop_h = min(float(image_h), max(1.0, crop_h))
    return crop_w, crop_h


def full_cover_crop_box(image_size: Tuple[int, int], target_aspect: float) -> Tuple[int, int, int, int]:
    image_w, image_h = image_size
    if target_aspect <= 0:
        return 0, 0, image_w, image_h
    crop_w, crop_h = aspect_limited_size(image_w, image_h, image_w, image_h, target_aspect)
    return clamp_center_box(image_w / 2.0, image_h / 2.0, crop_w, crop_h, image_w, image_h)


def compute_crop_box(
    image_size: Tuple[int, int],
    masks: Sequence[Optional[Image.Image]],
    crop_mode: str,
    crop_pad: float,
    min_crop_ratio: float,
    target_aspect: float,
) -> Tuple[int, int, int, int]:
    image_w, image_h = image_size
    full = full_cover_crop_box(image_size, target_aspect)
    if crop_mode == "full":
        return full
    bbox = union_bbox(masks)
    if bbox is None:
        return full
    x0, y0, x1, y1 = bbox
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    pad = max(0.0, float(crop_pad))
    min_ratio = min(1.0, max(0.0, float(min_crop_ratio)))
    crop_w = max(box_w * (1.0 + 2.0 * pad), image_w * min_ratio)
    crop_h = max(box_h * (1.0 + 2.0 * pad), image_h * min_ratio)
    crop_w, crop_h = aspect_limited_size(crop_w, crop_h, image_w, image_h, target_aspect)
    return clamp_center_box(cx, cy, crop_w, crop_h, image_w, image_h)


def contour_mask(mask: Image.Image, contour_width: float) -> Image.Image:
    binary = mask.point(lambda p: 255 if p > 0 else 0, mode="L")
    # Close only the visual contour to avoid noisy one-pixel gaps in rendered outlines.
    closed = binary.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    radius = max(1, int(round(float(contour_width) / 2.0)))
    dilated = closed
    eroded = closed
    for _ in range(radius):
        dilated = dilated.filter(ImageFilter.MaxFilter(3))
        eroded = eroded.filter(ImageFilter.MinFilter(3))
    edge = ImageChops.subtract(dilated, eroded)
    return edge.filter(ImageFilter.GaussianBlur(0.35))


def render_overlay(
    image: Image.Image,
    mask: Optional[Image.Image],
    style: Dict[str, Any],
    contour_width: float,
) -> Image.Image:
    if mask is None or mask.getbbox() is None:
        return image.convert("RGB")
    if mask.size != image.size:
        mask = mask.resize(image.size, resample=Image.Resampling.NEAREST)
    mask = mask.point(lambda p: 255 if p > 0 else 0, mode="L")
    base = image.convert("RGBA")

    fill_layer = Image.new("RGBA", base.size, hex_to_rgb(str(style["fill"])) + (0,))
    fill_alpha = mask.point(lambda p: int(round(255 * float(style["alpha"]))) if p > 0 else 0, mode="L")
    fill_layer.putalpha(fill_alpha)
    base = Image.alpha_composite(base, fill_layer)

    outline = contour_mask(mask, contour_width)
    contour_layer = Image.new("RGBA", base.size, hex_to_rgb(str(style["contour"])) + (0,))
    contour_layer.putalpha(outline)
    base = Image.alpha_composite(base, contour_layer)
    return base.convert("RGB")


def cover_resize(image: Image.Image, box_size: Tuple[int, int], resample: int) -> Image.Image:
    return ImageOps.fit(image, box_size, method=resample, centering=(0.5, 0.5))


def make_image_panel(
    image: Image.Image,
    box_size: Tuple[int, int],
    mask: Optional[Image.Image] = None,
    style: Optional[Dict[str, Any]] = None,
    panel_pad: int = PANEL_PAD,
    panel_mode: str = "cover",
) -> Image.Image:
    panel = Image.new("RGB", box_size, "white")
    inner_w = max(1, box_size[0] - 2 * panel_pad)
    inner_h = max(1, box_size[1] - 2 * panel_pad)
    inner_size = (inner_w, inner_h)
    if panel_mode == "contain":
        fitted = ImageOps.contain(image.convert("RGB"), inner_size, method=Image.Resampling.LANCZOS)
    else:
        fitted = cover_resize(image.convert("RGB"), inner_size, Image.Resampling.LANCZOS)
    if mask is not None and style is not None:
        if panel_mode == "contain":
            fitted_mask = mask.resize(fitted.size, resample=Image.Resampling.NEAREST)
        else:
            fitted_mask = cover_resize(mask.convert("L"), fitted.size, Image.Resampling.NEAREST)
        fitted = render_overlay(fitted, fitted_mask, style, contour_width=1.2)
    x = (box_size[0] - fitted.width) // 2
    y = (box_size[1] - fitted.height) // 2
    panel.paste(fitted, (x, y))
    return panel


def draw_image_cell(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    panel: Image.Image,
    box: Tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    canvas.paste(panel, (x0, y0))
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=PANEL_BORDER, width=1)


def index_summary(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {str(row.get("sample_uid", "")): row for row in rows}


def selected_rows(summary_rows: Sequence[Dict[str, str]], selected: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    index = index_summary(summary_rows)
    out: List[Dict[str, str]] = []
    for item in selected:
        uid = str(item.get("sample_uid", "")).strip()
        if uid not in index:
            raise KeyError(f"selected sample_uid not found in summary: {uid}")
        row = dict(index[uid])
        label = str(item.get("challenge_label", "")).strip()
        if label:
            row["selected_challenge_label"] = label
        out.append(row)
    return out


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = HEADER_TEXT,
) -> Tuple[int, int, int, int]:
    w, h = text_size(draw, text, font)
    x0, y0, x1, y1 = box
    tx = x0 + (x1 - x0 - w) // 2
    ty = y0 + (y1 - y0 - h) // 2
    draw.text((tx, ty), text, font=font, fill=fill)
    return tx, ty, tx + w, ty + h


def draw_header(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    title: str,
    font: ImageFont.ImageFont,
    is_ours: bool = False,
) -> None:
    fill = OURS_BLUE if is_ours else HEADER_TEXT
    draw_centered_text(draw, box, title, font, fill=fill)


def ellipsize_line(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    ellipsis = "..."
    if text_size(draw, text.rstrip() + ellipsis, font)[0] <= max_width:
        return text.rstrip() + ellipsis
    base = text
    while base and text_size(draw, base.rstrip() + ellipsis, font)[0] > max_width:
        base = base[:-1]
    return (base.rstrip() or text[:1]) + ellipsis


def draw_row_caption(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    label: str,
    expression: str,
    label_font: ImageFont.ImageFont,
    expr_font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = box
    pad_x = 0
    label = str(label or "").strip()
    expression = str(expression or "").strip()
    label_w, label_h = text_size(draw, label, label_font)
    pill_h = label_h + 7
    pill_w = label_w + 16
    pill_x0 = x0 + pad_x
    pill_y0 = y0 + max(0, (y1 - y0 - pill_h) // 2)
    pill_box = (pill_x0, pill_y0, pill_x0 + pill_w, pill_y0 + pill_h)
    try:
        draw.rounded_rectangle(pill_box, radius=5, fill=PILL_BG, outline=PILL_BORDER, width=1)
    except Exception:
        draw.rectangle(pill_box, fill=PILL_BG, outline=PILL_BORDER, width=1)
    draw.text((pill_x0 + 8, pill_y0 + 3), label, font=label_font, fill=PILL_TEXT)

    expr_x = pill_x0 + pill_w + 12
    max_text_w = max(1, x1 - expr_x)
    line_h = text_size(draw, "Ag", expr_font)[1] + 5
    max_lines = max(1, min(2, (y1 - y0) // max(1, line_h)))
    lines = wrap_text(draw, expression, expr_font, max_text_w)
    shown = lines[:max_lines]
    if len(lines) > max_lines and shown:
        shown[-1] = ellipsize_line(draw, shown[-1], expr_font, max_text_w)
    text_h = len(shown) * line_h - 5
    expr_y = y0 + max(0, (y1 - y0 - text_h) // 2)
    for line in shown:
        draw.text((expr_x, expr_y), line, font=expr_font, fill=BODY_TEXT)
        expr_y += line_h


def first_existing_path(*values: Any) -> Any:
    fallback = ""
    for value in values:
        if not value:
            continue
        if not fallback:
            fallback = value
        if Path(str(value)).exists():
            return value
    return fallback


def prepare_row_panels(
    row: Dict[str, str],
    args: argparse.Namespace,
    panel_size: Tuple[int, int],
) -> Tuple[List[Image.Image], Tuple[int, int, int, int]]:
    raw_path = first_existing_path(row.get("raw_path"), row.get("image_path"))
    image = load_rgb_image(raw_path)
    image_size = image.size
    gt_mask = load_binary_mask(row.get("gt_mask_path"), image_size)
    open_mask = load_binary_mask(row.get("open_mask_path"), image_size)
    ours_mask = load_binary_mask(row.get("ours_mask_path"), image_size)
    crop_box = compute_crop_box(
        image_size,
        [gt_mask, open_mask, ours_mask],
        str(args.crop_mode),
        float(args.crop_pad),
        float(args.min_crop_ratio),
        target_aspect=panel_size[0] / max(1, panel_size[1]),
    )

    image_crop = image.crop(crop_box)
    gt_crop = gt_mask.crop(crop_box) if gt_mask is not None else None
    open_crop = open_mask.crop(crop_box) if open_mask is not None else None
    ours_crop = ours_mask.crop(crop_box) if ours_mask is not None else None
    panels = [
        make_image_panel(image_crop, panel_size, panel_mode=str(args.panel_mode)),
        make_image_panel(image_crop, panel_size, gt_crop, GT_STYLE, panel_mode=str(args.panel_mode)),
        make_image_panel(image_crop, panel_size, open_crop, PRED_STYLE, panel_mode=str(args.panel_mode)),
        make_image_panel(image_crop, panel_size, ours_crop, PRED_STYLE, panel_mode=str(args.panel_mode)),
    ]
    return panels, crop_box


def build_grid(args: argparse.Namespace) -> Image.Image:
    summary = read_csv_rows(args.summary_csv)
    selected = read_csv_rows(args.selected_csv)
    rows = selected_rows(summary, selected)
    if len(rows) != GRID_ROWS:
        raise ValueError(f"selected CSV must contain exactly 6 rows; got {len(rows)}")

    tile_w = max(160, int(args.tile_w))
    tile_h = max(120, int(args.tile_h))
    caption_h = max(34, int(args.caption_h))
    header_h = max(40, int(args.header_h))
    margin = max(0, int(args.margin))
    col_gap = max(0, int(args.col_gap))
    row_gap = max(0, int(args.row_gap))
    header_gap = max(4, row_gap // 2)
    row_h = caption_h + tile_h
    total_w = margin * 2 + GRID_COLS * tile_w + (GRID_COLS - 1) * col_gap
    total_h = margin * 2 + header_h + header_gap + row_h * len(rows) + row_gap * (len(rows) - 1)

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(int(args.header_font), bold=True)
    label_font = load_font(int(args.label_font), bold=True)
    body_font = load_font(int(args.expr_font), bold=False)

    for col, title in enumerate(COLUMN_TITLES):
        x0 = margin + col * (tile_w + col_gap)
        y0 = margin
        draw_header(draw, (x0, y0, x0 + tile_w, y0 + header_h), title, header_font, is_ours=(col == 3))

    for row_idx, row in enumerate(rows):
        y0 = margin + header_h + header_gap + row_idx * (row_h + row_gap)
        label = row.get("selected_challenge_label") or row.get("challenge_type", "").split("|")[0]
        expression = row.get("expression", "")
        caption_box = (margin, y0, total_w - margin, y0 + caption_h)
        draw_row_caption(draw, caption_box, str(label), str(expression), label_font, body_font)

        tile_y0 = y0 + caption_h
        panels, _ = prepare_row_panels(row, args, (tile_w, tile_h))
        for col, panel in enumerate(panels):
            x0 = margin + col * (tile_w + col_gap)
            image_box = (
                x0,
                tile_y0,
                x0 + tile_w,
                tile_y0 + tile_h,
            )
            draw_image_cell(canvas, draw, panel, image_box)

    return canvas


def save_outputs(canvas: Image.Image, args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(args.dpi)
    png_600 = output_dir / f"{OUTPUT_BASENAME}_600dpi.png"
    png = output_dir / f"{OUTPUT_BASENAME}.png"
    pdf = output_dir / f"{OUTPUT_BASENAME}.pdf"

    canvas.save(png_600, dpi=(dpi, dpi))
    normal_dpi = int(args.png_dpi)
    if normal_dpi > 0 and normal_dpi < dpi:
        scale = normal_dpi / float(dpi)
        normal = canvas.resize(
            (max(1, int(canvas.width * scale)), max(1, int(canvas.height * scale))),
            resample=Image.Resampling.LANCZOS,
        )
        normal.save(png, dpi=(normal_dpi, normal_dpi))
    else:
        canvas.save(png, dpi=(dpi, dpi))
    canvas.save(pdf, "PDF", resolution=float(dpi))
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    print(f"wrote {png_600}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--selected-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--png-dpi", type=int, default=300)
    parser.add_argument("--tile-w", type=int, default=390)
    parser.add_argument("--tile-h", type=int, default=250)
    parser.add_argument("--caption-h", type=int, default=52)
    parser.add_argument("--header-h", type=int, default=42)
    parser.add_argument("--margin", type=int, default=14)
    parser.add_argument("--col-gap", type=int, default=8)
    parser.add_argument("--row-gap", type=int, default=10)
    parser.add_argument("--header-font", type=int, default=20)
    parser.add_argument("--label-font", type=int, default=18)
    parser.add_argument("--expr-font", type=int, default=19)
    parser.add_argument("--crop-mode", choices=("full", "context"), default="context")
    parser.add_argument("--crop-pad", type=float, default=0.50)
    parser.add_argument("--min-crop-ratio", type=float, default=0.50)
    parser.add_argument("--panel-mode", choices=("cover", "contain"), default="cover")
    parser.add_argument("--cell-w", type=int, default=860, help=argparse.SUPPRESS)
    parser.add_argument("--img-h", type=int, default=430, help=argparse.SUPPRESS)
    parser.add_argument("--expr-h", type=int, default=92, help=argparse.SUPPRESS)
    parser.add_argument("--gap", type=int, default=7, help=argparse.SUPPRESS)
    parser.add_argument("--width-inches", type=float, default=7.2, help=argparse.SUPPRESS)
    parser.add_argument("--image-ratio", type=float, default=0.72, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    canvas = build_grid(args)
    save_outputs(canvas, args)


if __name__ == "__main__":
    main()
