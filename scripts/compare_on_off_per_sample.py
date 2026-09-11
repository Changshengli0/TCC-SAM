#!/usr/bin/env python3
"""Run ON/OFF eval exports and build per-sample IoU comparison tables."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIG = "configs/refcoco/Open-World-SAM2-CrossAttention-refcoco.yaml"
DEFAULT_ON_WEIGHTS = "output/run_158/model_best_80p5723_backup.pth"
DEFAULT_OFF_WEIGHTS = "output/run_157/model_best.pth"
DEFAULT_EVF_CONFIG = "/home/zxing/code/SAM_RIS_1/checkpoints/evf-sam2-multitask"
DEFAULT_TOKENIZER_CONFIG = "/home/zxing/code/SAM_RIS_1/checkpoints/evf-sam2-multitask"
DEFAULT_VISION_PRETRAINED = "/home/zxing/code/SAM_RIS_1/sam2_hiera_large.pt"


def run_eval(args, weights, output_jsonl):
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if output_jsonl.exists():
        output_jsonl.unlink()
    cmd = [
        args.python,
        "train_net.py",
        "--config-file",
        args.config_file,
        "--eval-only",
        "DATASETS.TEST",
        f"('{args.dataset}',)",
        "MODEL.WEIGHTS",
        weights,
        "MODEL.OpenWorldSAM2.EVF_CONFIG",
        DEFAULT_EVF_CONFIG,
        "MODEL.OpenWorldSAM2.TOKENIZER_CONFIG",
        DEFAULT_TOKENIZER_CONFIG,
        "MODEL.OpenWorldSAM2.VISION_PRETRAINED",
        DEFAULT_VISION_PRETRAINED,
        "MODEL.OpenWorldSAM2.FUSION.EVIDENCE_RERANK",
        "True",
        "MODEL.OpenWorldSAM2.FUSION.EVIDENCE_ALPHA",
        str(args.evidence_alpha),
        "MODEL.OpenWorldSAM2.FUSION.WEAK_MAP_FALLBACK_THR",
        str(args.weak_map_fallback_thr),
        "MODEL.OpenWorldSAM2.EVAL_EXPORT_PER_SAMPLE_ON",
        "True",
        "MODEL.OpenWorldSAM2.EVAL_EXPORT_PER_SAMPLE_PATH",
        str(output_jsonl),
    ]
    cmd.extend(args.extra_opts)
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    subprocess.run(cmd, check=True, env=env)


def read_jsonl(path):
    rows = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = rec.get("sample_uid")
            if key is None:
                key = f"{rec.get('image_id')}:{rec.get('prompt_index')}"
            rows[str(key)] = rec
    return rows


def build_rows(on_rows, off_rows):
    rows = []
    for key in sorted(set(on_rows) & set(off_rows)):
        on = on_rows[key]
        off = off_rows[key]
        iou_on = float(on.get("iou", on.get("top1_iou", 0.0)))
        iou_off = float(off.get("iou", off.get("top1_iou", 0.0)))
        rows.append(
            {
                "sample_uid": key,
                "image_id": on.get("image_id", off.get("image_id")),
                "prompt_index": on.get("prompt_index", off.get("prompt_index")),
                "expression": on.get("expression", off.get("expression")),
                "file_name": on.get("file_name", off.get("file_name")),
                "IoU_ON": iou_on,
                "IoU_OFF": iou_off,
                "delta": iou_on - iou_off,
            }
        )
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_uid", "image_id", "prompt_index", "expression", "file_name", "IoU_ON", "IoU_OFF", "delta"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", default=DEFAULT_CONFIG)
    parser.add_argument("--on-weights", default=DEFAULT_ON_WEIGHTS)
    parser.add_argument("--off-weights", default=DEFAULT_OFF_WEIGHTS)
    parser.add_argument("--dataset", default="refcoco_val_unc")
    parser.add_argument("--output-dir", default="outputs/on_off_analysis")
    parser.add_argument("--evidence-alpha", type=float, default=4.5)
    parser.add_argument("--weak-map-fallback-thr", type=float, default=1e-3)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--on-jsonl", default=None)
    parser.add_argument("--off-jsonl", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("extra_opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    on_jsonl = Path(args.on_jsonl) if args.on_jsonl else output_dir / "on_per_sample.jsonl"
    off_jsonl = Path(args.off_jsonl) if args.off_jsonl else output_dir / "off_per_sample.jsonl"
    if not args.skip_eval:
        run_eval(args, args.on_weights, on_jsonl)
        run_eval(args, args.off_weights, off_jsonl)

    rows = build_rows(read_jsonl(on_jsonl), read_jsonl(off_jsonl))
    rows_sorted = sorted(rows, key=lambda r: r["delta"], reverse=True)
    write_csv(output_dir / "per_sample_on_off.csv", rows_sorted)
    write_csv(output_dir / "top20_gain.csv", rows_sorted[:20])
    write_csv(output_dir / "top20_drop.csv", list(reversed(rows_sorted[-20:])))
    print(f"wrote {len(rows_sorted)} joined samples to {output_dir}")


if __name__ == "__main__":
    main()
