#!/usr/bin/env python3
"""Build Candidate Calibration Case Study figures from rerank case exports."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps


COLUMN_TITLES = [
    "Image & Expression",
    "Ground Truth",
    "Base-selected Candidate",
    "Calibrated Candidate",
    "OPC-SAM Final",
]
OUTPUT_BASENAME = "candidate_calibration_cases"
PANEL_BORDER = "#E5E7EB"
PILL_BG = "#EEF2F7"
PILL_BORDER = "#CBD5E1"
PILL_TEXT = "#334155"
BODY_TEXT = "#111827"
HEADER_TEXT = "#111827"
OURS_BLUE = "#1F4E79"
BASE_RED = "#9F1239"
GT_STYLE = {"fill": "#E69F00", "alpha": 0.28, "contour": "#A16207"}
PRED_STYLE = {"fill": "#2F6FDB", "alpha": 0.28, "contour": "#1D4ED8"}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: List[str] = []
    if bold:
        candidates.extend([
            "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
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
    box = draw.textbbox((0, 0), str(text), font=font)
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


def ellipsize_line(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    ellipsis = "..."
    if text_size(draw, text.rstrip() + ellipsis, font)[0] <= max_width:
        return text.rstrip() + ellipsis
    base = text
    while base and text_size(draw, base.rstrip() + ellipsis, font)[0] > max_width:
        base = base[:-1]
    return (base.rstrip() or text[:1]) + ellipsis


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.strip().lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def read_csv_rows(path: Any) -> List[Dict[str, str]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def load_rgb_image(path: Any) -> Image.Image:
    if not path:
        raise ValueError("missing image path")
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def load_mask_path(path: Any, image_size: Tuple[int, int]) -> Optional[Image.Image]:
    if not path:
        return None
    p = Path(str(path))
    if not p.exists():
        return None
    with Image.open(p) as mask:
        mask = mask.convert("L")
    if mask.size != image_size:
        mask = mask.resize(image_size, resample=Image.Resampling.NEAREST)
    return mask.point(lambda px: 255 if px > 0 else 0, mode="L")


def tensor_to_mask(value: Any, image_size: Tuple[int, int], logits: bool = False) -> Optional[Image.Image]:
    if value is None or not torch.is_tensor(value):
        return None
    tensor = value.detach().float().cpu()
    while tensor.ndim > 2 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        return None
    if logits:
        tensor = torch.sigmoid(tensor)
    arr = ((tensor >= 0.5).to(torch.uint8).numpy() * 255).astype(np.uint8)
    mask = Image.fromarray(arr, mode="L")
    if mask.size != image_size:
        mask = mask.resize(image_size, resample=Image.Resampling.NEAREST)
    return mask


def mask_iou(a: Optional[Image.Image], b: Optional[Image.Image]) -> Optional[float]:
    if a is None or b is None:
        return None
    if a.size != b.size:
        b = b.resize(a.size, resample=Image.Resampling.NEAREST)
    aa = np.asarray(a) > 0
    bb = np.asarray(b) > 0
    union = np.logical_or(aa, bb).sum()
    if union <= 0:
        return 0.0
    inter = np.logical_and(aa, bb).sum()
    return float(inter) / float(union)


def mask_area_ratio(mask: Optional[Image.Image]) -> float:
    if mask is None:
        return 0.0
    arr = np.asarray(mask) > 0
    return float(arr.mean()) if arr.size else 0.0


def scalar_at(value: Any, idx: int) -> Optional[float]:
    if value is None or idx < 0:
        return None
    if torch.is_tensor(value):
        flat = value.detach().float().cpu().view(-1)
        if idx >= flat.numel():
            return None
        return float(flat[idx].item())
    try:
        return float(value[idx])
    except Exception:
        return None


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
    x0 = min(max(0.0, cx - crop_w / 2.0), max(0.0, image_w - crop_w))
    y0 = min(max(0.0, cy - crop_h / 2.0), max(0.0, image_h - crop_h))
    x1 = x0 + crop_w
    y1 = y0 + crop_h
    ix0 = int(round(x0))
    iy0 = int(round(y0))
    ix1 = min(image_w, max(ix0 + 1, int(round(x1))))
    iy1 = min(image_h, max(iy0 + 1, int(round(y1))))
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
    return min(float(image_w), crop_w), min(float(image_h), crop_h)


def full_cover_crop_box(image_size: Tuple[int, int], target_aspect: float) -> Tuple[int, int, int, int]:
    image_w, image_h = image_size
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
    binary = mask.point(lambda px: 255 if px > 0 else 0, mode="L")
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
    contour_width: float = 1.2,
) -> Image.Image:
    if mask is None or mask.getbbox() is None:
        return image.convert("RGB")
    if mask.size != image.size:
        mask = mask.resize(image.size, resample=Image.Resampling.NEAREST)
    mask = mask.point(lambda px: 255 if px > 0 else 0, mode="L")
    base = image.convert("RGBA")
    fill_layer = Image.new("RGBA", base.size, hex_to_rgb(str(style["fill"])) + (0,))
    fill_alpha = mask.point(lambda px: int(round(255 * float(style["alpha"]))) if px > 0 else 0, mode="L")
    fill_layer.putalpha(fill_alpha)
    base = Image.alpha_composite(base, fill_layer)
    contour_layer = Image.new("RGBA", base.size, hex_to_rgb(str(style["contour"])) + (0,))
    contour_layer.putalpha(contour_mask(mask, contour_width))
    base = Image.alpha_composite(base, contour_layer)
    return base.convert("RGB")


def make_panel(image: Image.Image, mask: Optional[Image.Image], style: Optional[Dict[str, Any]], tile_size: Tuple[int, int]) -> Image.Image:
    fitted = ImageOps.fit(image.convert("RGB"), tile_size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    if mask is None or style is None:
        return fitted
    fitted_mask = ImageOps.fit(mask.convert("L"), tile_size, method=Image.Resampling.NEAREST, centering=(0.5, 0.5))
    return render_overlay(fitted, fitted_mask, style)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = HEADER_TEXT,
) -> None:
    w, h = text_size(draw, text, font)
    x0, y0, x1, y1 = box
    draw.text((x0 + (x1 - x0 - w) // 2, y0 + (y1 - y0 - h) // 2), text, font=font, fill=fill)


def draw_row_caption(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    label: str,
    expression: str,
    label_font: ImageFont.ImageFont,
    expr_font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = box
    label = str(label or "Calibration case").strip()
    expression = str(expression or "").strip()
    label_w, label_h = text_size(draw, label, label_font)
    pill_h = label_h + 7
    pill_w = label_w + 16
    pill_y0 = y0 + max(0, (y1 - y0 - pill_h) // 2)
    pill_box = (x0, pill_y0, x0 + pill_w, pill_y0 + pill_h)
    draw.rounded_rectangle(pill_box, radius=5, fill=PILL_BG, outline=PILL_BORDER, width=1)
    draw.text((x0 + 8, pill_y0 + 3), label, font=label_font, fill=PILL_TEXT)
    expr_x = x0 + pill_w + 12
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


def summary_indices(rows: Sequence[Dict[str, str]]) -> Tuple[Dict[str, Dict[str, str]], Dict[Tuple[str, str], Dict[str, str]]]:
    by_uid: Dict[str, Dict[str, str]] = {}
    by_pair: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        uid = str(row.get("sample_uid", "")).strip()
        if uid:
            by_uid[uid] = row
        image_id = str(row.get("image_id", "")).strip()
        prompt_idx = str(row.get("prompt_idx", row.get("prompt_index", ""))).strip()
        if image_id and prompt_idx:
            by_pair[(image_id, prompt_idx)] = row
    return by_uid, by_pair


def match_summary(rec: Dict[str, Any], by_uid: Dict[str, Dict[str, str]], by_pair: Dict[Tuple[str, str], Dict[str, str]]) -> Optional[Dict[str, str]]:
    uid = str(rec.get("sample_uid", "")).strip()
    if uid and uid in by_uid:
        return by_uid[uid]
    image_id = str(rec.get("image_id", "")).strip()
    prompt_idx = str(rec.get("prompt_index", "")).strip()
    return by_pair.get((image_id, prompt_idx))


def list_pt_files(path: Any) -> List[Path]:
    p = Path(path)
    if p.is_file() and p.suffix == ".pt":
        return [p]
    return sorted(p.glob("*.pt"))


def load_case(path: Path, summary_row: Optional[Dict[str, str]], fallback_final_to_candidates: bool) -> Optional[Dict[str, Any]]:
    rec = torch.load(path, map_location="cpu")
    cand = rec.get("candidate_masks")
    if not torch.is_tensor(cand) or cand.ndim != 3:
        return None
    k_base = int(rec.get("top1_iou_pred_index", -1))
    k_final = int(rec.get("top1_final_index", -1))
    if k_base < 0 or k_final < 0 or k_base >= cand.shape[0] or k_final >= cand.shape[0] or k_base == k_final:
        return None

    file_name = None
    if summary_row:
        file_name = summary_row.get("raw_path") or summary_row.get("image_path")
    file_name = file_name or rec.get("file_name")
    image = load_rgb_image(file_name)
    image_size = image.size
    gt_mask = None
    if summary_row:
        gt_mask = load_mask_path(summary_row.get("gt_mask_path"), image_size)
    if gt_mask is None:
        gt_mask = tensor_to_mask(rec.get("gt_mask"), image_size, logits=False)

    base_candidate = tensor_to_mask(cand[k_base], image_size, logits=True)
    final_candidate = tensor_to_mask(cand[k_final], image_size, logits=True)
    open_final = load_mask_path(summary_row.get("open_mask_path"), image_size) if summary_row else None
    ours_final = load_mask_path(summary_row.get("ours_mask_path"), image_size) if summary_row else None
    final_source = "summary_masks" if open_final is not None and ours_final is not None else "missing"
    if final_source == "missing" and fallback_final_to_candidates:
        open_final = base_candidate
        ours_final = final_candidate
        final_source = "candidate_fallback"
    if open_final is None or ours_final is None:
        return None

    base_candidate_gt_iou = scalar_at(rec.get("gt_iou"), k_base)
    final_candidate_gt_iou = scalar_at(rec.get("gt_iou"), k_final)
    if base_candidate_gt_iou is None:
        base_candidate_gt_iou = mask_iou(base_candidate, gt_mask)
    if final_candidate_gt_iou is None:
        final_candidate_gt_iou = mask_iou(final_candidate, gt_mask)
    if base_candidate_gt_iou is None or final_candidate_gt_iou is None:
        return None

    iou_open = mask_iou(open_final, gt_mask)
    iou_ours = mask_iou(ours_final, gt_mask)
    if iou_open is None:
        iou_open = base_candidate_gt_iou
    if iou_ours is None:
        iou_ours = final_candidate_gt_iou

    sample_uid = str(rec.get("sample_uid") or "")
    if summary_row and summary_row.get("sample_uid"):
        sample_uid = summary_row["sample_uid"]
    image_id = rec.get("image_id", summary_row.get("image_id") if summary_row else "")
    prompt_idx = rec.get("prompt_index", summary_row.get("prompt_idx") if summary_row else "")
    case_uid = f"{image_id}:{prompt_idx}:{path.stem}"
    expression = str((summary_row or {}).get("expression") or rec.get("expression") or "")

    base_score = scalar_at(rec.get("iou_pred"), k_base)
    final_candidate_base_score = scalar_at(rec.get("iou_pred"), k_final)
    base_candidate_calib_score = scalar_at(rec.get("final_score"), k_base)
    final_score = scalar_at(rec.get("final_score"), k_final)
    evidence_base = scalar_at(rec.get("evidence_score"), k_base)
    evidence_final = scalar_at(rec.get("evidence_score"), k_final)
    candidate_pair_iou = mask_iou(base_candidate, final_candidate)

    return {
        "case_uid": case_uid,
        "source_pt": str(path),
        "sample_uid": sample_uid,
        "image_id": image_id,
        "prompt_idx": prompt_idx,
        "file_name": str(file_name),
        "expression": expression,
        "k_base": k_base,
        "k_final": k_final,
        "base_score": base_score,
        "final_candidate_base_score": final_candidate_base_score,
        "base_candidate_calib_score": base_candidate_calib_score,
        "final_score": final_score,
        "calib_score": final_score,
        "evidence_base": evidence_base,
        "evidence_final": evidence_final,
        "base_candidate_evidence_score": evidence_base,
        "evidence_score": evidence_final,
        "base_candidate_gt_iou": float(base_candidate_gt_iou),
        "final_candidate_gt_iou": float(final_candidate_gt_iou),
        "candidate_iou_gain": float(final_candidate_gt_iou) - float(base_candidate_gt_iou),
        "candidate_pair_iou": candidate_pair_iou,
        "iou_open": float(iou_open),
        "iou_ours": float(iou_ours),
        "delta_iou": float(iou_ours) - float(iou_open),
        "open_mask_area_ratio": mask_area_ratio(open_final),
        "final_source": final_source,
        "summary_matched": bool(summary_row),
        "_image": image,
        "_gt_mask": gt_mask,
        "_base_candidate": base_candidate,
        "_final_candidate": final_candidate,
        "_open_final": open_final,
        "_ours_final": ours_final,
    }

def collect_cases(args: argparse.Namespace) -> List[Dict[str, Any]]:
    summary = read_csv_rows(args.summary_csv)
    by_uid, by_pair = summary_indices(summary)
    cases: List[Dict[str, Any]] = []
    for path in list_pt_files(args.rerank_dir):
        rec = torch.load(path, map_location="cpu")
        summary_row = match_summary(rec, by_uid, by_pair)
        case = load_case(path, summary_row, fallback_final_to_candidates=bool(args.fallback_final_to_candidates))
        if case is not None:
            cases.append(case)
    for case in cases:
        annotate_case(case, args)
    cases.sort(key=lambda r: float(r.get("candidate_iou_gain", 0.0)), reverse=True)
    return cases


def is_extreme_failure(case: Dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        float(case.get("iou_open", 0.0)) < float(args.extreme_failure_iou)
        or float(case.get("open_mask_area_ratio", 0.0)) < float(args.extreme_failure_area)
    )


def score_improves(case: Dict[str, Any]) -> bool:
    evidence_base = case.get("evidence_base")
    evidence_final = case.get("evidence_final")
    calib_base = case.get("base_candidate_calib_score")
    final_score = case.get("final_score")
    if evidence_base is None or evidence_final is None or calib_base is None or final_score is None:
        return False
    return float(evidence_final) > float(evidence_base) and float(final_score) > float(calib_base)


def annotate_case(case: Dict[str, Any], args: argparse.Namespace) -> None:
    base_iou = float(case.get("base_candidate_gt_iou", 0.0))
    final_iou = float(case.get("final_candidate_gt_iou", 0.0))
    gain = float(case.get("candidate_iou_gain", final_iou - base_iou))
    pair_iou = case.get("candidate_pair_iou")
    pair_iou_value = 0.0 if pair_iou is None else float(pair_iou)
    case["passes_candidate_switch"] = bool(int(case.get("k_base", -1)) != int(case.get("k_final", -1)))
    case["passes_iou_gain"] = bool(gain >= float(args.min_candidate_iou_gain))
    case["passes_candidate_quality"] = bool(
        final_iou >= float(args.min_final_candidate_iou)
        and float(args.min_base_candidate_iou) <= base_iou <= float(args.max_base_candidate_iou)
    )
    case["passes_score_order"] = bool(score_improves(case))
    case["passes_final_source"] = bool(case.get("final_source") == "summary_masks")
    case["passes_extreme_filter"] = not is_extreme_failure(case, args)
    case["passes_visual_difference"] = bool(pair_iou_value <= float(args.max_candidate_pair_iou))
    case["passes_strict_filter"] = bool(
        case["passes_candidate_switch"]
        and case["passes_iou_gain"]
        and case["passes_candidate_quality"]
        and case["passes_score_order"]
        and case["passes_final_source"]
        and case["passes_extreme_filter"]
        and case["passes_visual_difference"]
    )


def auto_select_cases(cases: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    strict = [case for case in cases if case.get("passes_strict_filter")]
    return strict[: args.rows]

def select_cases(cases: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected_rows = read_csv_rows(args.selected_csv)
    if not selected_rows:
        return auto_select_cases(cases, args)
    by_case = {str(c["case_uid"]): c for c in cases}
    by_sample = {str(c["sample_uid"]): c for c in cases if c.get("sample_uid")}
    by_source = {Path(str(c["source_pt"])).name: c for c in cases}
    out: List[Dict[str, Any]] = []
    for row in selected_rows:
        key = str(row.get("case_uid") or row.get("sample_uid") or row.get("source_pt") or "").strip()
        if not key:
            continue
        case = by_case.get(key) or by_sample.get(key) or by_source.get(Path(key).name)
        if case is None:
            raise KeyError(f"selected case not found: {key}")
        out.append(case)
    return out[: args.rows]


def strip_internal(case: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in case.items() if not k.startswith("_")}


def fmt_score(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def score_lines(case: Dict[str, Any], calibrated: bool) -> Tuple[str, str]:
    if calibrated:
        return (
            f"k={case['k_final']} | s_calib={fmt_score(case.get('final_score'))}",
            f"e={fmt_score(case.get('evidence_final'))} | IoU={fmt_score(case.get('final_candidate_gt_iou'))}",
        )
    return (
        f"k={case['k_base']} | s_base={fmt_score(case.get('base_score'))}",
        f"e={fmt_score(case.get('evidence_base'))} | IoU={fmt_score(case.get('base_candidate_gt_iou'))}",
    )


def draw_score_strip(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    lines: Tuple[str, str],
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    x0, y0, x1, y1 = box
    line_h = text_size(draw, "Ag", font)[1] + 3
    total_h = line_h * len(lines) - 3
    y = y0 + max(0, (y1 - y0 - total_h) // 2)
    for line in lines:
        w, _ = text_size(draw, line, font)
        draw.text((x0 + (x1 - x0 - w) // 2, y), line, font=font, fill=fill)
        y += line_h

def draw_figure(cases: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Image.Image:
    cols = len(COLUMN_TITLES)
    tile_w = max(120, int(args.tile_w))
    tile_h = max(90, int(args.tile_h))
    caption_h = max(34, int(args.caption_h))
    score_h = max(30, int(args.score_h))
    header_h = max(34, int(args.header_h))
    margin = max(0, int(args.margin))
    col_gap = max(0, int(args.col_gap))
    row_gap = max(0, int(args.row_gap))
    header_gap = max(4, row_gap // 2)
    row_h = caption_h + tile_h + score_h
    total_w = margin * 2 + cols * tile_w + (cols - 1) * col_gap
    total_h = margin * 2 + header_h + header_gap + len(cases) * row_h + max(0, len(cases) - 1) * row_gap
    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    header_font = load_font(args.header_font, bold=True)
    label_font = load_font(args.label_font, bold=True)
    expr_font = load_font(args.expr_font, bold=False)
    score_font = load_font(args.score_font, bold=False)

    for col, title in enumerate(COLUMN_TITLES):
        x0 = margin + col * (tile_w + col_gap)
        fill = BASE_RED if col == 2 else (OURS_BLUE if col in (3, 4) else HEADER_TEXT)
        draw_centered_text(draw, (x0, margin, x0 + tile_w, margin + header_h), title, header_font, fill=fill)

    for row_idx, case in enumerate(cases):
        y0 = margin + header_h + header_gap + row_idx * (row_h + row_gap)
        draw_row_caption(
            draw,
            (margin, y0, total_w - margin, y0 + caption_h),
            f"Calibration case {row_idx + 1}",
            str(case.get("expression", "")),
            label_font,
            expr_font,
        )
        image = case["_image"]
        crop_box = compute_crop_box(
            image.size,
            [
                case.get("_gt_mask"),
                case.get("_base_candidate"),
                case.get("_final_candidate"),
                case.get("_open_final"),
                case.get("_ours_final"),
            ],
            args.crop_mode,
            args.crop_pad,
            args.min_crop_ratio,
            tile_w / max(1, tile_h),
        )
        image_crop = image.crop(crop_box)
        masks = [
            None,
            case["_gt_mask"].crop(crop_box) if case.get("_gt_mask") is not None else None,
            case["_base_candidate"].crop(crop_box) if case.get("_base_candidate") is not None else None,
            case["_final_candidate"].crop(crop_box) if case.get("_final_candidate") is not None else None,
            case["_ours_final"].crop(crop_box) if case.get("_ours_final") is not None else None,
        ]
        styles = [None, GT_STYLE, PRED_STYLE, PRED_STYLE, PRED_STYLE]
        tile_y0 = y0 + caption_h
        for col in range(cols):
            x0 = margin + col * (tile_w + col_gap)
            panel = make_panel(image_crop, masks[col], styles[col], (tile_w, tile_h))
            canvas.paste(panel, (x0, tile_y0))
            draw.rectangle((x0, tile_y0, x0 + tile_w - 1, tile_y0 + tile_h - 1), outline=PANEL_BORDER, width=1)
            strip_y0 = tile_y0 + tile_h
            if col == 2:
                draw_score_strip(draw, (x0, strip_y0, x0 + tile_w, strip_y0 + score_h), score_lines(case, calibrated=False), score_font, BASE_RED)
            elif col == 3:
                draw_score_strip(draw, (x0, strip_y0, x0 + tile_w, strip_y0 + score_h), score_lines(case, calibrated=True), score_font, OURS_BLUE)

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
    parser.add_argument("--rerank-dir", default="outputs/rerank_vis/rerank_case_exports")
    parser.add_argument("--summary-csv", default="")
    parser.add_argument("--selected-csv", default="")
    parser.add_argument("--output-dir", default="outputs/paper_figs/candidate_calibration_cases")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--tile-w", type=int, default=340)
    parser.add_argument("--tile-h", type=int, default=220)
    parser.add_argument("--caption-h", type=int, default=50)
    parser.add_argument("--score-h", type=int, default=42)
    parser.add_argument("--header-h", type=int, default=38)
    parser.add_argument("--margin", type=int, default=14)
    parser.add_argument("--col-gap", type=int, default=7)
    parser.add_argument("--row-gap", type=int, default=10)
    parser.add_argument("--header-font", type=int, default=16)
    parser.add_argument("--label-font", type=int, default=15)
    parser.add_argument("--expr-font", type=int, default=16)
    parser.add_argument("--score-font", type=int, default=14)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--png-dpi", type=int, default=300)
    parser.add_argument("--crop-mode", choices=("full", "context"), default="context")
    parser.add_argument("--crop-pad", type=float, default=0.50)
    parser.add_argument("--min-crop-ratio", type=float, default=0.50)
    parser.add_argument("--min-candidate-iou-gain", type=float, default=0.25)
    parser.add_argument("--min-base-candidate-iou", type=float, default=0.20)
    parser.add_argument("--max-base-candidate-iou", type=float, default=0.70)
    parser.add_argument("--min-final-candidate-iou", type=float, default=0.70)
    parser.add_argument("--max-candidate-pair-iou", type=float, default=0.90)
    parser.add_argument("--extreme-failure-iou", type=float, default=0.02)
    parser.add_argument("--extreme-failure-area", type=float, default=0.0005)
    parser.add_argument("--fallback-final-to-candidates", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    cases = collect_cases(args)
    if not cases:
        raise SystemExit(
            "No usable rerank cases found. Check --rerank-dir, or enable "
            "MODEL.OpenWorldSAM2.RERANK_CASE_EXPORT_ON during eval-only."
        )
    summary_fields = [
        "case_uid",
        "source_pt",
        "sample_uid",
        "image_id",
        "prompt_idx",
        "file_name",
        "expression",
        "k_base",
        "k_final",
        "base_score",
        "base_candidate_calib_score",
        "final_candidate_base_score",
        "final_score",
        "calib_score",
        "evidence_base",
        "evidence_final",
        "base_candidate_evidence_score",
        "evidence_score",
        "base_candidate_gt_iou",
        "final_candidate_gt_iou",
        "candidate_iou_gain",
        "candidate_pair_iou",
        "iou_open",
        "iou_ours",
        "delta_iou",
        "open_mask_area_ratio",
        "final_source",
        "summary_matched",
        "passes_candidate_switch",
        "passes_iou_gain",
        "passes_candidate_quality",
        "passes_score_order",
        "passes_final_source",
        "passes_extreme_filter",
        "passes_visual_difference",
        "passes_strict_filter",
    ]
    write_csv(output_dir / "candidate_cases_summary.csv", [strip_internal(c) for c in cases], summary_fields)
    selected = select_cases(cases, args)
    if not selected:
        raise SystemExit("No candidate cases selected after filtering.")
    write_csv(output_dir / "candidate_cases_selected.csv", [strip_internal(c) for c in selected], summary_fields)
    canvas = draw_figure(selected, args)
    save_outputs(canvas, args)


if __name__ == "__main__":
    main()
