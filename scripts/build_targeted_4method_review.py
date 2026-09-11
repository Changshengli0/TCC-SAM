#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATS = [
    ("Multi-instance", "multi_instance_top10_review.png"),
    ("Attribute", "attribute_top10_review.png"),
    ("Spatial", "spatial_top10_review.png"),
    ("Occlusion / Boundary", "occlusion_top10_review.png"),
    ("Small object", "small_object_top10_review.png"),
    ("Long expression", "long_expression_top10_review.png"),
]
COLS = ["Image", "Ground Truth", "OpenWorldSAM", "DIT-SAM", "SSP-SAM", "TCC-SAM (Ours)"]


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_font(size, bold=False):
    cands = []
    if bold:
        cands += ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
    cands += ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]
    for c in cands:
        if os.path.exists(c):
            return ImageFont.truetype(c, size=size)
    return ImageFont.load_default()


def mask_arr(path, size=None):
    path = resolve(path)
    if not path.exists():
        return None
    with Image.open(path) as img:
        m = img.convert("L")
    if size and m.size != size:
        m = m.resize(size, Image.Resampling.NEAREST)
    return np.array(m) > 0


def iou(gt_path, pred_path):
    gt_p = resolve(gt_path)
    pred_p = resolve(pred_path)
    if not gt_p.exists() or not pred_p.exists():
        return None
    with Image.open(gt_p) as g:
        size = g.size
    gt = mask_arr(gt_p, size)
    pred = mask_arr(pred_p, size)
    if gt is None or pred is None:
        return None
    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return float(inter / union) if union else 0.0


def fval(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except Exception:
        return default


def challenge_membership(challenge_type):
    out = set()
    text = challenge_type or ""
    for cat, _ in CATS:
        if cat in text:
            out.add(cat)
    return out


def passes_for_cat(row, cat):
    ours = row["iou_ours"]
    dopen = row["delta_vs_open"]
    dbest = row["delta_vs_best_other"]
    wc = row["word_count"]
    if dopen < 0.12:
        return False
    if cat in ["Multi-instance", "Attribute", "Spatial"]:
        return ours >= 0.72 and dbest >= 0.08
    if cat == "Occlusion / Boundary":
        return ours >= 0.68 and dbest >= 0.06
    if cat == "Small object":
        return ours >= 0.60 and dbest >= 0.07
    if cat == "Long expression":
        return wc >= 10 and ours >= 0.68 and dbest >= 0.06
    return ours >= 0.70 and dbest >= 0.06


def score_row(row, cat):
    wc = row["word_count"]
    challenge_bonus = 1.0 if cat in challenge_membership(row["challenge_label"]) else 0.0
    text_complexity_bonus = min(1.0, max(0.0, (wc - 5) / 10.0))
    return (
        0.45 * row["delta_vs_best_other"]
        + 0.20 * row["delta_vs_open"]
        + 0.20 * row["iou_ours"]
        + 0.10 * challenge_bonus
        + 0.05 * text_complexity_bonus
    )


def collect(args):
    pool = read_csv(resolve(args.pool_csv))
    summary = {r["sample_uid"]: r for r in read_csv(resolve(args.summary_csv))}
    dit_dir = resolve(args.dit_dir)
    ssp_dir = resolve(args.ssp_dir)
    rows = []
    missing = []
    for item in pool:
        uid = item["sample_uid"]
        s = summary.get(uid)
        if not s:
            missing.append({"sample_uid": uid, "missing": "summary"})
            continue
        gt = s["gt_mask_path"]
        open_mask = s["open_mask_path"]
        ours_mask = s["ours_mask_path"]
        dit_mask = dit_dir / f"{uid}_mask.png"
        ssp_mask = ssp_dir / f"{uid}_mask.png"
        iou_open = fval(s, "iou_open")
        iou_ours = fval(s, "iou_ours")
        iou_dit = iou(gt, dit_mask)
        iou_ssp = iou(gt, ssp_mask)
        if iou_dit is None:
            missing.append({"sample_uid": uid, "missing": "DIT-SAM mask", "path": str(dit_mask)})
            iou_dit = -1.0
        if iou_ssp is None:
            missing.append({"sample_uid": uid, "missing": "SSP-SAM mask", "path": str(ssp_mask)})
            iou_ssp = -1.0
        best_other = max(iou_open, iou_dit, iou_ssp)
        row = {
            "sample_uid": uid,
            "challenge_label": item.get("challenge_label") or s.get("challenge_type", ""),
            "expression": s.get("expression", ""),
            "iou_open": iou_open,
            "iou_dit": iou_dit,
            "iou_ssp": iou_ssp,
            "iou_ours": iou_ours,
            "best_other_iou": best_other,
            "delta_vs_open": iou_ours - iou_open,
            "delta_vs_best_other": iou_ours - best_other,
            "gt_area_ratio": fval(s, "gt_area_ratio"),
            "word_count": int(float(s.get("word_count") or 0)),
            "raw_path": s.get("raw_path") or s.get("image_path", ""),
            "gt_overlay_path": s.get("gt_overlay_path", ""),
            "open_overlay_path": s.get("open_overlay_path", ""),
            "dit_overlay_path": str(dit_dir / f"{uid}_overlay.png"),
            "ssp_overlay_path": str(ssp_dir / f"{uid}_overlay.png"),
            "ours_overlay_path": s.get("ours_overlay_path", ""),
            "gt_mask_path": gt,
            "open_mask_path": open_mask,
            "dit_mask_path": str(dit_mask),
            "ssp_mask_path": str(ssp_mask),
            "ours_mask_path": ours_mask,
        }
        for cat, _ in CATS:
            if cat in challenge_membership(row["challenge_label"]):
                row[f"score_{cat}"] = score_row(row, cat)
        rows.append(row)
    return rows, missing


def write_merged(rows, path):
    fields = [
        "sample_uid", "challenge_label", "expression", "iou_open", "iou_dit", "iou_ssp", "iou_ours",
        "best_other_iou", "delta_vs_open", "delta_vs_best_other", "gt_area_ratio", "word_count",
        "raw_path", "gt_overlay_path", "open_overlay_path", "dit_overlay_path", "ssp_overlay_path", "ours_overlay_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fields}
            for k in ["iou_open", "iou_dit", "iou_ssp", "iou_ours", "best_other_iou", "delta_vs_open", "delta_vs_best_other", "gt_area_ratio"]:
                out[k] = f"{float(out[k]):.6f}"
            w.writerow(out)


def text_width(draw, text, font):
    b = draw.textbbox((0, 0), text, font=font)
    return b[2] - b[0]


def wrap(draw, text, font, width, max_lines=2):
    words = str(text).split()
    lines = []
    cur = ""
    for word in words:
        trial = word if not cur else cur + " " + word
        if text_width(draw, trial, font) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        return lines[:max_lines-1] + [lines[max_lines-1] + " ..."]
    return lines


def fit_img(path, size):
    p = resolve(path)
    if not p.exists():
        img = Image.new("RGB", size, "#F8FAFC")
        d = ImageDraw.Draw(img)
        d.text((12, size[1]//2-8), "missing", fill=(120, 120, 120), font=load_font(14))
        return img
    with Image.open(p) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    return ImageOps.fit(im, size, Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def draw_sheet(cat, rows, path):
    top = rows[:10]
    tile_w, tile_h = 205, 145
    meta_h, header_h = 76, 34
    margin, gap, row_gap = 14, 5, 10
    width = margin*2 + len(COLS)*tile_w + (len(COLS)-1)*gap
    height = margin*2 + header_h + len(top)*(tile_h+meta_h) + max(0, len(top)-1)*row_gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    hf = load_font(16, True); sf = load_font(11); mf = load_font(12); bf = load_font(12, True)
    y = margin
    for i,c in enumerate(COLS):
        x = margin + i*(tile_w+gap)
        color = (0,74,145) if c == "TCC-SAM (Ours)" else (20,24,30)
        tw = text_width(draw, c, hf)
        draw.text((x+(tile_w-tw)/2, y+6), c, font=hf, fill=color)
    y += header_h
    for r in top:
        paths = [r["raw_path"], r["gt_overlay_path"], r["open_overlay_path"], r["dit_overlay_path"], r["ssp_overlay_path"], r["ours_overlay_path"]]
        for i,pth in enumerate(paths):
            x = margin + i*(tile_w+gap)
            canvas.paste(fit_img(pth, (tile_w, tile_h)), (x,y))
            draw.rectangle([x,y,x+tile_w-1,y+tile_h-1], outline=(218,223,230), width=1)
        meta_y = y + tile_h + 5
        draw.text((margin, meta_y), r["sample_uid"], font=bf, fill=(15,23,42))
        metrics = f"open {r['iou_open']:.3f} | dit {r['iou_dit']:.3f} | ssp {r['iou_ssp']:.3f} | ours {r['iou_ours']:.3f} | d_best {r['delta_vs_best_other']:.3f}"
        draw.text((margin + 390, meta_y), metrics, font=mf, fill=(40,50,65))
        expr_y = meta_y + 18
        for line in wrap(draw, r["expression"], sf, width - 2*margin, 2):
            draw.text((margin, expr_y), line, font=sf, fill=(45,55,72))
            expr_y += 15
        y += tile_h + meta_h + row_gap
    canvas.save(path, dpi=(220,220))
    return canvas.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-csv", required=True)
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--dit-dir", required=True)
    ap.add_argument("--ssp-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = resolve(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows, missing = collect(args)
    merged = out / "merged_4methods_summary.csv"
    write_merged(rows, merged)
    by_cat = {}
    counts = {}
    sheet_paths = {}
    recommendations = {}
    for cat, filename in CATS:
        cand = [r for r in rows if cat in challenge_membership(r["challenge_label"]) and passes_for_cat(r, cat)]
        cand.sort(key=lambda r: score_row(r, cat), reverse=True)
        by_cat[cat] = cand
        counts[cat] = len(cand)
        path = out / filename
        draw_sheet(cat, cand, path)
        sheet_paths[cat] = str(path)
        recommendations[cat] = [
            {
                "sample_uid": r["sample_uid"],
                "score": round(score_row(r, cat), 6),
                "expression": r["expression"],
                "iou_open": round(r["iou_open"], 4),
                "iou_dit": round(r["iou_dit"], 4),
                "iou_ssp": round(r["iou_ssp"], 4),
                "iou_ours": round(r["iou_ours"], 4),
                "delta_vs_best_other": round(r["delta_vs_best_other"], 4),
            }
            for r in cand[:3]
        ]
    report = {
        "pool_size": len(rows),
        "merged_summary": str(merged),
        "hard_filter_counts": counts,
        "review_sheets": sheet_paths,
        "recommendations_top3": recommendations,
        "missing": missing,
    }
    (out / "targeted_4method_review_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
