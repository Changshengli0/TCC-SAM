#!/usr/bin/env python3
"""Export and rank qualitative RefCOCO comparison samples for paper figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


GT_FILL = "#F59E0B"
GT_CONTOUR = "#B45309"
PRED_FILL = "#1E88E5"
PRED_CONTOUR = "#0B3D91"
OVERLAY_ALPHA = 0.43
CONTOUR_WIDTH = 2

SUMMARY_FIELDS = [
    "dataset",
    "sample_uid",
    "image_id",
    "prompt_idx",
    "image_path",
    "expression",
    "gt_area_ratio",
    "iou_open",
    "iou_ours",
    "delta_iou",
    "challenge_type",
    "raw_path",
    "gt_overlay_path",
    "open_overlay_path",
    "ours_overlay_path",
    "open_mask_path",
    "ours_mask_path",
]

EXTRA_SUMMARY_FIELDS = [
    "gt_mask_path",
    "word_count",
    "avoid_reason",
]

CHALLENGE_SPECS = [
    ("Multi-instance", "Multi-instance"),
    ("Attribute", "Attribute"),
    ("Spatial relation", "Spatial"),
    ("Occlusion / Boundary", "Occlusion / Boundary"),
    ("Small object", "Small object"),
    ("Long expression", "Long expression"),
]

CHALLENGE_FILENAMES = {
    "Multi-instance": "Multi-instance",
    "Attribute": "Attribute",
    "Spatial relation": "Spatial",
    "Occlusion / Boundary": "Occlusion-boundary",
    "Small object": "Small-object",
    "Long expression": "Long-expression",
}

MULTI_INSTANCE_TERMS = [
    "person", "man", "woman", "boy", "girl", "people",
    "dog", "cat", "car", "bus", "horse", "cow", "sheep",
    "food", "sandwich", "chair", "table", "plate", "bottle", "cup",
]

ATTRIBUTE_TERMS = [
    "red", "blue", "white", "black", "green", "yellow", "orange", "brown",
    "gray", "grey", "small", "large", "big", "little", "tiny", "striped",
    "wearing", "shirt", "hat", "jacket", "coat", "dress", "shorts",
    "pants", "jeans", "glasses",
]

SPATIAL_TERMS = [
    "left", "right", "front", "back", "behind", "next to", "near", "beside",
    "on", "under", "above", "between", "in front of", "on top of", "middle",
    "center",
]


def _clean_remainder(opts: Sequence[str] | None) -> List[str]:
    opts = list(opts or [])
    if opts and opts[0] == "--":
        opts = opts[1:]
    return opts


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return str(value)


def _append_jsonl(path: Path, rec: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    color = color.strip().lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _safe_component(value: Any, fallback: str = "none") -> str:
    text = str(value if value is not None else fallback)
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", text).strip("_")
    return text or fallback


def _expression_hash(expression: Any) -> str:
    data = str(expression or "").strip().lower().encode("utf-8", errors="ignore")
    return hashlib.sha1(data).hexdigest()[:10]


def make_sample_uid(dataset: str, image_id: Any, prompt_idx: int, expression: Any) -> str:
    dataset_part = _safe_component(dataset, "dataset")
    image_part = _safe_component(image_id, "image")
    return f"{dataset_part}_{image_part}_p{int(prompt_idx):03d}_{_expression_hash(expression)}"


def _tensor_or_array_to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if torch.is_tensor(value):
        arr = value.detach().float().cpu().numpy()
    else:
        try:
            arr = np.asarray(value)
        except Exception:
            return None
    if arr.size == 0:
        return None
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    while arr.ndim > 2:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr[0]
    if arr.ndim != 2:
        return None
    return arr


def _resize_bool_mask(mask: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if tuple(mask.shape) == tuple(out_hw):
        return mask
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.resize((int(out_hw[1]), int(out_hw[0])), resample=Image.Resampling.NEAREST)
    return np.asarray(img, dtype=np.uint8) > 127


def pred_to_binary_mask(value: Any, out_hw: Tuple[int, int], threshold: float = 0.5) -> Optional[np.ndarray]:
    arr = _tensor_or_array_to_numpy(value)
    if arr is None:
        return None
    if float(np.nanmin(arr)) < 0.0 or float(np.nanmax(arr)) > 1.0:
        arr = 1.0 / (1.0 + np.exp(-np.clip(arr, -60.0, 60.0)))
    mask = arr >= float(threshold)
    return _resize_bool_mask(mask, out_hw)


def gt_to_binary_mask(value: Any, out_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    arr = _tensor_or_array_to_numpy(value)
    if arr is None:
        return None
    return _resize_bool_mask(arr >= 0.5, out_hw)


def load_binary_mask(path: Any, out_hw: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    if path is None or str(path) == "":
        return None
    try:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("L"), dtype=np.uint8) > 127
    except Exception:
        return None
    if out_hw is not None:
        arr = _resize_bool_mask(arr, out_hw)
    return arr


def mask_iou(mask_a: Optional[np.ndarray], mask_b: Optional[np.ndarray]) -> Optional[float]:
    if mask_a is None or mask_b is None:
        return None
    a = np.asarray(mask_a).astype(bool)
    b = np.asarray(mask_b).astype(bool)
    if a.shape != b.shape:
        b = _resize_bool_mask(b, (int(a.shape[0]), int(a.shape[1])))
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    union = np.logical_or(a, b).sum(dtype=np.float64)
    return float(inter / max(float(union), 1e-6))


def _shift_bool(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    y0_src = max(0, -dy)
    y1_src = min(h, h - dy)
    x0_src = max(0, -dx)
    x1_src = min(w, w - dx)
    y0_dst = max(0, dy)
    y1_dst = min(h, h + dy)
    x0_dst = max(0, dx)
    x1_dst = min(w, w + dx)
    if y0_src < y1_src and x0_src < x1_src:
        out[y0_dst:y1_dst, x0_dst:x1_dst] = mask[y0_src:y1_src, x0_src:x1_src]
    return out


def mask_contour(mask: np.ndarray, width: int = 2) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            eroded &= _shift_bool(mask, dy, dx)
    edge = mask & ~eroded
    radius = max(int(width), 1)
    dilated = edge.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                dilated |= _shift_bool(edge, dy, dx)
    return dilated


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    fill: str,
    contour: str,
    alpha: float = OVERLAY_ALPHA,
    contour_width: int = CONTOUR_WIDTH,
) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    mask_bool = np.asarray(mask).astype(bool)
    fill_rgb = np.asarray(_hex_to_rgb(fill), dtype=np.float32).reshape(1, 3)
    contour_rgb = np.asarray(_hex_to_rgb(contour), dtype=np.float32).reshape(1, 3)
    if mask_bool.any():
        base[mask_bool] = base[mask_bool] * (1.0 - float(alpha)) + fill_rgb * float(alpha)
        edge = mask_contour(mask_bool, width=contour_width)
        base[edge] = contour_rgb
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def prompt_expression(sample: Dict[str, Any], prompt_idx: int) -> Optional[str]:
    prompt = sample.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, (list, tuple)) and 0 <= prompt_idx < len(prompt):
        return str(prompt[prompt_idx])
    groundings = sample.get("groundings")
    if isinstance(groundings, dict):
        texts = groundings.get("texts")
        if isinstance(texts, (list, tuple)) and 0 <= prompt_idx < len(texts):
            cur = texts[prompt_idx]
            if isinstance(cur, str):
                return cur
            if isinstance(cur, (list, tuple)) and cur:
                return str(cur[0])
    grounding_info = sample.get("grounding_info")
    if isinstance(grounding_info, (list, tuple)) and 0 <= prompt_idx < len(grounding_info):
        ann = grounding_info[prompt_idx]
        if isinstance(ann, dict):
            sentences = ann.get("sentences")
            if isinstance(sentences, (list, tuple)) and sentences:
                sent = sentences[0]
                if isinstance(sent, dict):
                    return str(sent.get("raw", sent.get("sent", "")))
                return str(sent)
    return None


def _num_prompts_from_sample_output(sample: Dict[str, Any], output: Dict[str, Any]) -> int:
    pred = output.get("grounding_mask") if isinstance(output, dict) else None
    pred_n = int(pred.shape[0]) if torch.is_tensor(pred) and pred.ndim >= 3 else 0
    groundings = sample.get("groundings") if isinstance(sample, dict) else None
    gt = groundings.get("masks") if isinstance(groundings, dict) else None
    gt_n = int(gt.shape[0]) if torch.is_tensor(gt) and gt.ndim >= 3 else (1 if torch.is_tensor(gt) and gt.ndim == 2 else 0)
    prompt = sample.get("prompt") if isinstance(sample, dict) else None
    prompt_n = len(prompt) if isinstance(prompt, (list, tuple)) else (1 if isinstance(prompt, str) else 0)
    candidates = [x for x in (pred_n, gt_n, prompt_n) if x > 0]
    return min(candidates) if candidates else 0


def export_prompt_assets(
    *,
    args: argparse.Namespace,
    sample: Dict[str, Any],
    output: Dict[str, Any],
    prompt_idx: int,
    output_root: Path,
) -> Dict[str, Any]:
    file_name = sample.get("file_name")
    if not file_name:
        raise ValueError("sample has no file_name")
    image_path = str(file_name)
    with Image.open(image_path) as img:
        image = img.convert("RGB")
    out_hw = (int(image.height), int(image.width))

    pred_all = output.get("grounding_mask")
    if not torch.is_tensor(pred_all) or prompt_idx >= int(pred_all.shape[0]):
        raise ValueError("output grounding_mask is missing or too short")
    pred_mask = pred_to_binary_mask(pred_all[prompt_idx], out_hw, threshold=float(args.pred_threshold))
    if pred_mask is None:
        raise ValueError("failed to convert prediction mask")

    groundings = sample.get("groundings")
    gt_all = groundings.get("masks") if isinstance(groundings, dict) else None
    gt_value = gt_all if torch.is_tensor(gt_all) and gt_all.ndim == 2 else gt_all[prompt_idx]
    gt_mask = gt_to_binary_mask(gt_value, out_hw)
    if gt_mask is None:
        raise ValueError("failed to convert GT mask")

    expression = prompt_expression(sample, prompt_idx) or ""
    image_id = sample.get("image_id", sample.get("image_index", "none"))
    sample_uid = make_sample_uid(args.dataset, image_id, prompt_idx, expression)
    prefix = _safe_component(sample_uid)

    raw_dir = output_root / "raw"
    overlay_dir = output_root / "overlays"
    mask_dir = output_root / "masks"
    for directory in (raw_dir, overlay_dir, mask_dir):
        directory.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{prefix}_image.png"
    gt_mask_path = mask_dir / f"{prefix}_gt.png"
    pred_mask_path = mask_dir / f"{prefix}_pred.png"
    gt_overlay_path = overlay_dir / f"{prefix}_gt_overlay.png"
    pred_overlay_path = overlay_dir / f"{prefix}_pred_overlay.png"

    image.save(raw_path)
    Image.fromarray((gt_mask.astype(np.uint8) * 255), mode="L").save(gt_mask_path)
    Image.fromarray((pred_mask.astype(np.uint8) * 255), mode="L").save(pred_mask_path)
    overlay_mask(image, gt_mask, GT_FILL, GT_CONTOUR).save(gt_overlay_path)
    overlay_mask(image, pred_mask, PRED_FILL, PRED_CONTOUR).save(pred_overlay_path)

    iou = mask_iou(pred_mask, gt_mask)
    gt_area_ratio = float(gt_mask.sum(dtype=np.float64) / max(float(gt_mask.size), 1.0))
    pred_area_ratio = float(pred_mask.sum(dtype=np.float64) / max(float(pred_mask.size), 1.0))

    return {
        "dataset": args.dataset,
        "model_name": args.name,
        "sample_uid": sample_uid,
        "legacy_sample_uid": f"{image_id}:{int(prompt_idx)}",
        "image_id": image_id,
        "prompt_idx": int(prompt_idx),
        "prompt_index": int(prompt_idx),
        "image_path": image_path,
        "file_name": image_path,
        "expression": expression,
        "height": int(image.height),
        "width": int(image.width),
        "gt_area_ratio": gt_area_ratio,
        "pred_area_ratio": pred_area_ratio,
        "iou": None if iou is None else float(iou),
        "raw_path": str(raw_path),
        "gt_mask_path": str(gt_mask_path),
        "pred_mask_path": str(pred_mask_path),
        "gt_overlay_path": str(gt_overlay_path),
        "pred_overlay_path": str(pred_overlay_path),
    }


def _log_export_error(errors_path: Path, sample: Dict[str, Any], prompt_idx: Optional[int], exc: BaseException) -> None:
    rec = {
        "image_id": sample.get("image_id") if isinstance(sample, dict) else None,
        "file_name": sample.get("file_name") if isinstance(sample, dict) else None,
        "prompt_idx": prompt_idx,
        "expression": prompt_expression(sample, prompt_idx) if isinstance(sample, dict) and prompt_idx is not None else None,
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }
    _append_jsonl(errors_path, rec)


def run_export(args: argparse.Namespace) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import MetadataCatalog
    from train_net import Trainer, setup

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / "predictions.jsonl"
    errors_path = output_root / "errors.jsonl"
    for path in (predictions_path, errors_path):
        if path.exists() and not args.append:
            path.unlink()
    for sub in ("raw", "overlays", "masks"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    opts = [
        "DATASETS.TEST",
        f"('{args.dataset}',)",
        "MODEL.WEIGHTS",
        args.weights,
    ] + _clean_remainder(args.opts)
    setup_args = SimpleNamespace(
        config_file=args.config_file,
        opts=opts,
        run_idx=args.run_idx,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_only=True,
        resume=False,
        num_gpus=1,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
    )
    cfg = setup(setup_args)
    model = Trainer.build_model(cfg)
    model.metadata = MetadataCatalog.get(args.dataset)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    model.eval()
    loader = Trainer.build_test_loader(cfg, args.dataset)

    exported = 0
    seen_images = 0
    limit = int(args.max_samples) if args.max_samples is not None and int(args.max_samples) > 0 else None
    with torch.no_grad():
        for batch in loader:
            if limit is not None and exported >= limit:
                break
            try:
                outputs = model(batch)
            except Exception as exc:
                for sample in batch:
                    _log_export_error(errors_path, sample, None, exc)
                continue
            for sample, output in zip(batch, outputs):
                seen_images += 1
                num_prompts = _num_prompts_from_sample_output(sample, output)
                for prompt_idx in range(num_prompts):
                    if limit is not None and exported >= limit:
                        break
                    try:
                        rec = export_prompt_assets(
                            args=args,
                            sample=sample,
                            output=output,
                            prompt_idx=prompt_idx,
                            output_root=output_root,
                        )
                    except Exception as exc:
                        _log_export_error(errors_path, sample, prompt_idx, exc)
                        continue
                    _append_jsonl(predictions_path, rec)
                    exported += 1
            if limit is not None and exported >= limit:
                break

    print(
        f"exported {exported} image-expression samples from {seen_images} images to {output_root}",
        flush=True,
    )
    print(f"predictions: {predictions_path}", flush=True)
    print(f"errors: {errors_path}", flush=True)


def read_jsonl(path: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def index_by_sample_uid(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for rec in rows:
        key = rec.get("sample_uid")
        if key is None:
            key = make_sample_uid(
                str(rec.get("dataset", "dataset")),
                rec.get("image_id", "image"),
                int(rec.get("prompt_idx", rec.get("prompt_index", 0)) or 0),
                rec.get("expression", ""),
            )
        out[str(key)] = rec
    return out


def _word_count(expression: Any) -> int:
    return len(str(expression or "").strip().split())


def _contains_term(text: str, terms: Sequence[str]) -> bool:
    text_l = f" {str(text or '').lower()} "
    for term in terms:
        term_l = term.lower()
        if " " in term_l:
            if term_l in text_l:
                return True
        elif re.search(rf"\b{re.escape(term_l)}\b", text_l):
            return True
    return False


def classify_challenges(row: Dict[str, Any]) -> str:
    expression = str(row.get("expression", ""))
    dataset = str(row.get("dataset", "")).lower()
    iou_open = float(row.get("iou_open", 0.0) or 0.0)
    iou_ours = float(row.get("iou_ours", 0.0) or 0.0)
    delta = float(row.get("delta_iou", 0.0) or 0.0)
    gt_area_ratio = float(row.get("gt_area_ratio", 0.0) or 0.0)
    words = _word_count(expression)
    tags: List[str] = []
    if _contains_term(expression, MULTI_INSTANCE_TERMS) and delta >= 0.25 and iou_ours >= 0.65:
        tags.append("Multi-instance")
    if _contains_term(expression, ATTRIBUTE_TERMS) and delta >= 0.20 and iou_ours >= 0.65:
        tags.append("Attribute")
    if _contains_term(expression, SPATIAL_TERMS) and delta >= 0.25 and iou_ours >= 0.65:
        tags.append("Spatial")
    if 0.30 <= iou_open <= 0.70 and iou_ours >= 0.65 and delta >= 0.15:
        tags.append("Occlusion / Boundary")
    if gt_area_ratio < 0.05 and delta >= 0.15 and iou_ours >= 0.50:
        tags.append("Small object")
    elif gt_area_ratio <= 0.10 and delta >= 0.20 and iou_ours >= 0.55:
        tags.append("Small object")
    if (("refcocog" in dataset and words >= 8) or words >= 10) and delta >= 0.18 and iou_ours >= 0.65:
        tags.append("Long expression")
    return "|".join(tags) if tags else "Uncategorized"


def avoid_reason(row: Dict[str, Any]) -> str:
    reasons: List[str] = []
    expression = str(row.get("expression", ""))
    iou_open = float(row.get("iou_open", 0.0) or 0.0)
    iou_ours = float(row.get("iou_ours", 0.0) or 0.0)
    gt_area_ratio = float(row.get("gt_area_ratio", 0.0) or 0.0)
    if iou_open <= 0.03:
        reasons.append("openworldsam_extreme_failure")
    if iou_ours < 0.50:
        reasons.append("ours_low_iou")
    if gt_area_ratio < 0.005:
        reasons.append("too_small_after_scaling")
    if _word_count(expression) <= 2:
        reasons.append("very_short_expression")
    return "|".join(reasons)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def build_summary_row(open_rec: Dict[str, Any], ours_rec: Dict[str, Any]) -> Dict[str, Any]:
    gt_mask_path = _first_present(ours_rec.get("gt_mask_path"), open_rec.get("gt_mask_path"))
    open_mask_path = open_rec.get("pred_mask_path")
    ours_mask_path = ours_rec.get("pred_mask_path")
    gt_mask = load_binary_mask(gt_mask_path)
    open_mask = load_binary_mask(open_mask_path, None if gt_mask is None else tuple(gt_mask.shape))
    ours_mask = load_binary_mask(ours_mask_path, None if gt_mask is None else tuple(gt_mask.shape))
    iou_open = mask_iou(open_mask, gt_mask)
    iou_ours = mask_iou(ours_mask, gt_mask)
    if iou_open is None:
        iou_open = float(open_rec.get("iou", 0.0) or 0.0)
    if iou_ours is None:
        iou_ours = float(ours_rec.get("iou", 0.0) or 0.0)
    delta = float(iou_ours) - float(iou_open)
    if gt_mask is not None:
        gt_area_ratio = float(gt_mask.sum(dtype=np.float64) / max(float(gt_mask.size), 1.0))
    else:
        gt_area_ratio = float(_first_present(ours_rec.get("gt_area_ratio"), open_rec.get("gt_area_ratio"), 0.0) or 0.0)
    expression = _first_present(ours_rec.get("expression"), open_rec.get("expression"), "")
    row = {
        "dataset": _first_present(ours_rec.get("dataset"), open_rec.get("dataset"), ""),
        "sample_uid": _first_present(ours_rec.get("sample_uid"), open_rec.get("sample_uid"), ""),
        "image_id": _first_present(ours_rec.get("image_id"), open_rec.get("image_id"), ""),
        "prompt_idx": int(_first_present(ours_rec.get("prompt_idx"), open_rec.get("prompt_idx"), 0) or 0),
        "image_path": _first_present(ours_rec.get("image_path"), open_rec.get("image_path"), ""),
        "expression": expression,
        "gt_area_ratio": gt_area_ratio,
        "iou_open": float(iou_open),
        "iou_ours": float(iou_ours),
        "delta_iou": float(delta),
        "raw_path": _first_present(ours_rec.get("raw_path"), open_rec.get("raw_path"), ""),
        "gt_overlay_path": _first_present(ours_rec.get("gt_overlay_path"), open_rec.get("gt_overlay_path"), ""),
        "open_overlay_path": open_rec.get("pred_overlay_path", ""),
        "ours_overlay_path": ours_rec.get("pred_overlay_path", ""),
        "open_mask_path": open_mask_path or "",
        "ours_mask_path": ours_mask_path or "",
        "gt_mask_path": gt_mask_path or "",
        "word_count": _word_count(expression),
    }
    row["challenge_type"] = classify_challenges(row)
    row["avoid_reason"] = avoid_reason(row)
    return row


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_compare(args: argparse.Namespace) -> None:
    open_rows = index_by_sample_uid(read_jsonl(args.open_jsonl))
    ours_rows = index_by_sample_uid(read_jsonl(args.ours_jsonl))
    keys = sorted(set(open_rows) & set(ours_rows))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [build_summary_row(open_rows[key], ours_rows[key]) for key in keys]
    rows.sort(key=lambda row: (float(row.get("delta_iou", 0.0)), float(row.get("iou_ours", 0.0))), reverse=True)
    fields = SUMMARY_FIELDS + EXTRA_SUMMARY_FIELDS
    write_csv(output / "summary.csv", rows, fields)
    with (output / "summary.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
    print(f"wrote {len(rows)} merged samples to {output}", flush=True)


def read_csv_rows(path: Any) -> List[Dict[str, Any]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_has_challenge(row: Dict[str, Any], label: str) -> bool:
    target = dict(CHALLENGE_SPECS).get(label, label)
    tags = str(row.get("challenge_type", "")).split("|")
    return target in tags


def row_sort_key(row: Dict[str, Any]) -> Tuple[int, float, float, float]:
    avoid = 1 if str(row.get("avoid_reason", "")) else 0
    return (
        avoid,
        -float(row.get("delta_iou", 0.0) or 0.0),
        -float(row.get("iou_ours", 0.0) or 0.0),
        float(row.get("iou_open", 0.0) or 0.0),
    )


def run_topk(args: argparse.Namespace) -> None:
    rows = read_csv_rows(args.summary_csv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fields = SUMMARY_FIELDS + EXTRA_SUMMARY_FIELDS
    for label, _tag in CHALLENGE_SPECS:
        cur = [row for row in rows if row_has_challenge(row, label)]
        cur.sort(key=row_sort_key)
        name = f"{CHALLENGE_FILENAMES[label]}_top{int(args.topn)}.csv"
        write_csv(output / name, cur[: int(args.topn)], fields)
    print(f"wrote top-{int(args.topn)} candidate CSVs to {output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_export = subparsers.add_parser("export", help="Run a limited eval loop and export masks/overlays.")
    p_export.add_argument("--config-file", required=True)
    p_export.add_argument("--dataset", required=True)
    p_export.add_argument("--weights", required=True)
    p_export.add_argument("--name", required=True, choices=["openworldsam", "opcsam"])
    p_export.add_argument("--output", required=True)
    p_export.add_argument("--max-samples", type=int, default=None)
    p_export.add_argument("--pred-threshold", type=float, default=0.5)
    p_export.add_argument("--append", action="store_true")
    p_export.add_argument("--run-idx", type=int, default=0)
    p_export.add_argument("--batch-size", type=int, default=8)
    p_export.add_argument("--lr", type=float, default=1e-4)
    p_export.add_argument("opts", nargs=argparse.REMAINDER)
    p_export.set_defaults(func=run_export)

    p_compare = subparsers.add_parser("compare", help="Merge OpenWorldSAM and OPC-SAM exports.")
    p_compare.add_argument("--open-jsonl", required=True)
    p_compare.add_argument("--ours-jsonl", required=True)
    p_compare.add_argument("--output", required=True)
    p_compare.set_defaults(func=run_compare)

    p_topk = subparsers.add_parser("topk", help="Write per-challenge top-N candidate CSVs.")
    p_topk.add_argument("--summary-csv", required=True)
    p_topk.add_argument("--output", required=True)
    p_topk.add_argument("--topn", type=int, default=30)
    p_topk.set_defaults(func=run_topk)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
