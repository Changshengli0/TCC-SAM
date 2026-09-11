#!/usr/bin/env python3
import argparse
import csv
import json
import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_font(size, bold=False):
    candidates = []
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
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_width(draw, text, font):
    if not text:
        return 0
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def wrap_text(draw, text, font, max_width):
    words = str(text).split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            if text_width(draw, word, font) <= max_width:
                current = word
            else:
                chunks = textwrap.wrap(word, width=max(8, int(max_width / max(font.size * 0.55, 1))))
                lines.extend(chunks[:-1])
                current = chunks[-1] if chunks else ""
    if current:
        lines.append(current)
    return lines or [""]


def fit_image(image, box_size):
    target_w, target_h = box_size
    canvas = Image.new("RGB", box_size, "white")
    img = image.convert("RGB")
    img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    x = (target_w - img.width) // 2
    y = (target_h - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def read_rows(selected_csv, summary_csv, image_dir, image_suffix):
    selected = list(csv.DictReader(open(selected_csv, newline="")))
    summary = {
        row["sample_uid"]: row
        for row in csv.DictReader(open(summary_csv, newline=""))
    }
    rows = []
    missing = []
    for selected_row in selected:
        uid = selected_row["sample_uid"]
        summary_row = summary.get(uid, {})
        image_path = image_dir / f"{uid}{image_suffix}"
        if not image_path.exists():
            missing.append(uid)
            continue
        rows.append({
            "sample_uid": uid,
            "challenge_label": selected_row.get("challenge_label", ""),
            "expression": summary_row.get("expression", ""),
            "image_path": image_path,
        })
    return rows, missing


def draw_grid(rows, title, output_base, dpi=300, save_pdf=True):
    cols = 3 if len(rows) > 4 else 2
    rows_n = max(1, math.ceil(len(rows) / cols))
    panel_w = 640
    image_h = 430
    caption_h = 126
    gap = 24
    margin_x = 42
    margin_top = 76 if title else 36
    margin_bottom = 36
    panel_h = image_h + caption_h
    width = margin_x * 2 + cols * panel_w + (cols - 1) * gap
    height = margin_top + rows_n * panel_h + (rows_n - 1) * gap + margin_bottom

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(32, bold=True)
    label_font = load_font(21, bold=True)
    expr_font = load_font(20, bold=False)
    small_font = load_font(17, bold=False)

    if title:
        tw = text_width(draw, title, title_font)
        draw.text(((width - tw) // 2, 24), title, fill=(20, 20, 20), font=title_font)

    border = (210, 210, 210)
    text_color = (24, 24, 24)
    muted = (82, 82, 82)
    for idx, row in enumerate(rows):
        r = idx // cols
        c = idx % cols
        x = margin_x + c * (panel_w + gap)
        y = margin_top + r * (panel_h + gap)
        draw.rectangle([x, y, x + panel_w - 1, y + panel_h - 1], outline=border, width=1)

        img = fit_image(Image.open(row["image_path"]), (panel_w - 2, image_h - 2))
        canvas.paste(img, (x + 1, y + 1))
        draw.line([x, y + image_h, x + panel_w - 1, y + image_h], fill=border, width=1)

        tx = x + 18
        ty = y + image_h + 14
        max_text_w = panel_w - 36
        label = row["challenge_label"]
        draw.text((tx, ty), label, fill=text_color, font=label_font)
        ty += 30
        expr_lines = wrap_text(draw, row["expression"], expr_font, max_text_w)
        max_expr_lines = 3
        for line in expr_lines[:max_expr_lines]:
            draw.text((tx, ty), line, fill=muted, font=expr_font)
            ty += 24
        if len(expr_lines) > max_expr_lines:
            draw.text((tx, ty), "...", fill=muted, font=small_font)

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    canvas.save(png_path, dpi=(dpi, dpi))
    if save_pdf:
        canvas.save(pdf_path, "PDF", resolution=dpi)
    return png_path, pdf_path if save_pdf else None, canvas.size


def main():
    parser = argparse.ArgumentParser(description="Build a single-method selected preview grid.")
    parser.add_argument("--selected-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--image-suffix", default="_overlay.png")
    parser.add_argument("--title", default="")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    selected_csv = resolve(args.selected_csv)
    summary_csv = resolve(args.summary_csv)
    image_dir = resolve(args.image_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, missing = read_rows(selected_csv, summary_csv, image_dir, args.image_suffix)
    base = output_dir / "sspsam_selected6_preview"
    png_path, pdf_path, size = draw_grid(rows, args.title, base, dpi=300)
    png_600_path, _, size_600 = draw_grid(rows, args.title, output_dir / "sspsam_selected6_preview_600dpi", dpi=600, save_pdf=False)

    report = {
        "input_count": len(rows) + len(missing),
        "rendered_count": len(rows),
        "missing_count": len(missing),
        "missing_sample_uids": missing,
        "png_path": str(png_path),
        "png_600dpi_path": str(png_600_path),
        "pdf_path": str(pdf_path),
        "size": list(size),
        "size_600dpi": list(size_600),
    }
    (output_dir / "sspsam_selected6_preview_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
