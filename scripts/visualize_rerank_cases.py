#!/usr/bin/env python3
"""Export and visualize samples where evidence reranking changes top-1 mask choice."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_CONFIG = "configs/refcoco/Open-World-SAM2-CrossAttention-refcoco.yaml"
DEFAULT_WEIGHTS = "output/run_158/model_best_80p5723_backup.pth"
DEFAULT_EVF_CONFIG = "/home/zxing/code/SAM_RIS_1/checkpoints/evf-sam2-multitask"
DEFAULT_TOKENIZER_CONFIG = "/home/zxing/code/SAM_RIS_1/checkpoints/evf-sam2-multitask"
DEFAULT_VISION_PRETRAINED = "/home/zxing/code/SAM_RIS_1/sam2_hiera_large.pt"


def run_eval(args):
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python,
        "train_net.py",
        "--config-file",
        args.config_file,
        "--eval-only",
        "DATASETS.TEST",
        f"('{args.dataset}',)",
        "MODEL.WEIGHTS",
        args.weights,
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
        "MODEL.OpenWorldSAM2.RERANK_CASE_EXPORT_ON",
        "True",
        "MODEL.OpenWorldSAM2.RERANK_CASE_EXPORT_MAX_SAMPLES",
        str(args.max_cases),
        "MODEL.OpenWorldSAM2.RERANK_CASE_EXPORT_DIR",
        str(export_dir),
    ]
    cmd.extend(args.extra_opts)
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    subprocess.run(cmd, check=True, env=env)


def tensor_to_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu()
        while value.ndim > 2 and value.shape[0] == 1:
            value = value[0]
        return value.float().numpy()
    return np.asarray(value)


def load_image(path):
    if path:
        p = Path(path)
        if p.exists():
            return mpimg.imread(p)
    return np.ones((512, 512, 3), dtype=np.float32)


def overlay_mask(ax, image, logits, title):
    ax.imshow(image)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    if logits is None:
        return
    arr = tensor_to_numpy(logits)
    if arr is None:
        return
    prob = 1.0 / (1.0 + np.exp(-arr))
    h, w = image.shape[:2]
    ax.imshow(prob >= 0.5, cmap="Reds", alpha=0.42, extent=(0, w, h, 0))


def score_text(rec, idx):
    iou_pred = rec.get("iou_pred")
    evidence = rec.get("evidence_score")
    final = rec.get("final_score")
    gt_iou = rec.get("gt_iou")
    parts = []
    if torch.is_tensor(iou_pred):
        parts.append(f"iou_pred={float(iou_pred[idx]):.4f}")
    if torch.is_tensor(evidence):
        parts.append(f"evidence={float(evidence[idx]):.4f}")
    if torch.is_tensor(final):
        parts.append(f"final={float(final[idx]):.4f}")
    if torch.is_tensor(gt_iou):
        parts.append(f"GT IoU={float(gt_iou[idx]):.4f}")
    return "\n".join(parts)


def draw_candidate(ax, image, masks, rec, idx, title):
    overlay_mask(ax, image, masks[idx], title)
    ax.text(
        0.02,
        0.98,
        score_text(rec, idx),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
    )


def visualize_record(path, output_dir):
    rec = torch.load(path, map_location="cpu")
    image = load_image(rec.get("file_name"))
    masks = rec["candidate_masks"]
    top_iou = [int(x) for x in rec.get("top3_iou_pred_indices", [])]
    top_final = [int(x) for x in rec.get("top3_final_indices", [])]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    title = (
        f"image={rec.get('image_id')} prompt={rec.get('prompt_index')} "
        f"iou_top1={rec.get('top1_iou_pred_index')} final_top1={rec.get('top1_final_index')} "
        f"{rec.get('expression') or ''}"
    )
    fig.suptitle(title, fontsize=10)
    axes = axes.reshape(-1)
    axes[0].imshow(image)
    axes[0].set_title("image", fontsize=8)
    axes[0].axis("off")
    gt_panel = tensor_to_numpy(rec.get("gt_mask"))
    axes[1].imshow(image)
    axes[1].set_title("GT mask", fontsize=8)
    axes[1].axis("off")
    if gt_panel is not None:
        h, w = image.shape[:2]
        axes[1].imshow(gt_panel, cmap="Greens", alpha=0.42, extent=(0, w, h, 0))
    for offset, idx in enumerate(top_iou[:3], start=2):
        draw_candidate(axes[offset], image, masks, rec, idx, f"iou_pred rank {offset - 1} idx={idx}")
    for offset, idx in enumerate(top_final[:3], start=5):
        draw_candidate(axes[offset], image, masks, rec, idx, f"final rank {offset - 4} idx={idx}")
    out_name = Path(path).with_suffix(".png").name
    fig.savefig(output_dir / out_name, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", default=DEFAULT_CONFIG)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--dataset", default="refcoco_val_unc")
    parser.add_argument("--output-dir", default="outputs/rerank_vis")
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--max-cases", type=int, default=20)
    parser.add_argument("--evidence-alpha", type=float, default=4.5)
    parser.add_argument("--weak-map-fallback-thr", type=float, default=1e-3)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("extra_opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.export_dir = args.export_dir or str(output_dir / "rerank_case_exports")
    if not args.skip_eval:
        run_eval(args)

    files = sorted(Path(args.export_dir).glob("*.pt"))[: args.max_cases]
    if not files:
        raise SystemExit(f"No rerank case .pt files found in {args.export_dir}")
    for path in files:
        visualize_record(path, output_dir)
    print(f"saved {len(files)} figures to {output_dir}")


if __name__ == "__main__":
    main()
