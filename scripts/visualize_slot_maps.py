#!/usr/bin/env python3
"""Export and visualize slot evidence maps for RefCOCO-style eval samples."""

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
        "MODEL.OpenWorldSAM2.FUSION.EVIDENCE_ALPHA",
        str(args.evidence_alpha),
        "MODEL.OpenWorldSAM2.FUSION.WEAK_MAP_FALLBACK_THR",
        str(args.weak_map_fallback_thr),
        "MODEL.OpenWorldSAM2.FUSION.EXPORT_SLOT_MECH_DIAG_ON",
        "True",
        "MODEL.OpenWorldSAM2.FUSION.EXPORT_SLOT_MECH_DIAG_MAX_SAMPLES",
        str(args.max_samples),
        "MODEL.OpenWorldSAM2.FUSION.EXPORT_SLOT_MECH_DIAG_DIR",
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


def overlay(ax, image, mask, title, cmap="magma", alpha=0.55, threshold=False):
    ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if mask is None:
        return
    arr = tensor_to_numpy(mask)
    if arr is None:
        return
    if threshold:
        arr = (1.0 / (1.0 + np.exp(-arr))) >= 0.5
    h, w = image.shape[:2]
    ax.imshow(arr, cmap=cmap, alpha=alpha, extent=(0, w, h, 0))


def first_slot(slot_maps, index):
    if slot_maps is None:
        return None
    slots = tensor_to_numpy(slot_maps)
    if slots is None:
        return None
    if slots.ndim == 4 and slots.shape[1] == 1:
        slots = slots[:, 0]
    if slots.ndim == 3 and index < slots.shape[0]:
        return slots[index]
    return None


def visualize_record(path, output_dir):
    rec = torch.load(path, map_location="cpu")
    image = load_image(rec.get("file_name"))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    title = f"image={rec.get('image_id')} prompt={rec.get('prompt_index')} {rec.get('expression') or ''}"
    fig.suptitle(title, fontsize=10)
    axes = axes.reshape(-1)
    axes[0].imshow(image)
    axes[0].set_title("image", fontsize=9)
    axes[0].axis("off")
    overlay(axes[1], image, rec.get("gt_mask"), "GT mask", cmap="Greens", alpha=0.45)
    overlay(axes[2], image, rec.get("pred_mask"), "pred final mask", cmap="Reds", alpha=0.45, threshold=True)
    overlay(axes[3], image, first_slot(rec.get("slot_maps"), 0), "slot 0 evidence", cmap="magma", alpha=0.55)
    overlay(axes[4], image, first_slot(rec.get("slot_maps"), 1), "slot 1 evidence", cmap="magma", alpha=0.55)
    union = rec.get("slot_union_prob")
    if union is None and rec.get("slot_union_logits") is not None:
        union = torch.sigmoid(rec["slot_union_logits"])
    overlay(axes[5], image, union, "slot union evidence", cmap="viridis", alpha=0.55)
    out_name = Path(path).with_suffix(".png").name
    fig.savefig(output_dir / out_name, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", default=DEFAULT_CONFIG)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--dataset", default="refcoco_val_unc")
    parser.add_argument("--output-dir", default="outputs/slot_vis")
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--evidence-alpha", type=float, default=4.5)
    parser.add_argument("--weak-map-fallback-thr", type=float, default=1e-3)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("extra_opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.export_dir = args.export_dir or str(output_dir / "slot_mech_exports")
    if not args.skip_eval:
        run_eval(args)

    files = sorted(Path(args.export_dir).glob("*.pt"))[: args.max_samples]
    if not files:
        raise SystemExit(f"No slot export .pt files found in {args.export_dir}")
    for path in files:
        visualize_record(path, output_dir)
    print(f"saved {len(files)} figures to {output_dir}")


if __name__ == "__main__":
    main()
