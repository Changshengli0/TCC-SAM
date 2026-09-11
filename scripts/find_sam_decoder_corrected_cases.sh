#!/usr/bin/env bash
set -e

cd /home/zxing/code/SAM_RIS_1/OpenWorldSAM-main

SRC_ROOT="outputs/vis_gt_81p0133_refcoco_val_all"
DST_ROOT="outputs/vis_gt_81p0133_refcoco_val_sam_decoder_corrected"

rm -rf "${DST_ROOT}"
mkdir -p "${DST_ROOT}/corrected_by_score_head"
mkdir -p "${DST_ROOT}/sam_final_diff_but_not_improved"
mkdir -p "${DST_ROOT}/sam_calib_disagree"

python - <<'PY'
import glob, json, os, shutil, csv
import numpy as np
from PIL import Image, ImageDraw

src_root = "outputs/vis_gt_81p0133_refcoco_val_all"
dst_root = "outputs/vis_gt_81p0133_refcoco_val_sam_decoder_corrected"

corrected_dir = os.path.join(dst_root, "corrected_by_score_head")
not_improved_dir = os.path.join(dst_root, "sam_final_diff_but_not_improved")
disagree_dir = os.path.join(dst_root, "sam_calib_disagree")

for d in [corrected_dir, not_improved_dir, disagree_dir]:
    os.makedirs(d, exist_ok=True)

def argmax_or_none(x):
    if not x:
        return None
    return int(np.argmax(x))

def load_mask(path):
    if not os.path.exists(path):
        return None
    arr = np.array(Image.open(path).convert("L"))
    return arr > 127

def mask_iou(a, b):
    if a is None or b is None:
        return None
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)

paths = sorted(glob.glob(os.path.join(src_root, "sample_*", "meta.json")))
rows = []

for meta_path in paths:
    sample_dir = os.path.dirname(meta_path)
    sample_name = os.path.basename(sample_dir)

    with open(meta_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    iou_pred = m.get("iou_pred_list") or []
    base_scores = m.get("base_score_list") or []
    calib_scores = m.get("calib_score_list") or []
    final_idx = m.get("final_idx")

    sam_idx = argmax_or_none(iou_pred)
    base_idx = m.get("base_idx")
    if base_idx is None:
        base_idx = argmax_or_none(base_scores)

    calib_idx = m.get("calib_idx")
    if calib_idx is None:
        calib_idx = argmax_or_none(calib_scores)

    if sam_idx is None or calib_idx is None or final_idx is None:
        continue

    gt = load_mask(os.path.join(sample_dir, "gt_mask.png"))
    sam_mask = load_mask(os.path.join(sample_dir, f"cand_{sam_idx}_mask.png"))
    final_mask = load_mask(os.path.join(sample_dir, "final_mask.png"))
    calib_mask = load_mask(os.path.join(sample_dir, f"cand_{calib_idx}_mask.png"))

    iou_sam_gt = mask_iou(sam_mask, gt)
    iou_final_gt = mask_iou(final_mask, gt)
    iou_calib_gt = mask_iou(calib_mask, gt)

    if iou_sam_gt is None or iou_final_gt is None:
        continue

    gain_vs_sam = iou_final_gt - iou_sam_gt

    # 严格定义：SAM decoder 原本选的候选 != 最终候选，
    # 且 Candidate Score Head 推荐了最终候选，
    # 且最终 IoU 比 SAM 原始选择更高。
    is_score_head_corrected_sam = (
        sam_idx != final_idx
        and final_idx == calib_idx
        and gain_vs_sam > 1e-6
    )

    # 也记录 SAM 与 Final 不一致但没有提升的反例，方便你确认
    is_sam_final_diff_not_improved = (
        sam_idx != final_idx
        and final_idx == calib_idx
        and gain_vs_sam <= 1e-6
    )

    # 记录 score head 与 SAM 发生分歧的所有情况
    is_sam_calib_disagree = (sam_idx != calib_idx)

    group = None
    target_parent = None

    if is_score_head_corrected_sam:
        group = "corrected_by_score_head"
        target_parent = corrected_dir
    elif is_sam_final_diff_not_improved:
        group = "sam_final_diff_but_not_improved"
        target_parent = not_improved_dir
    elif is_sam_calib_disagree:
        group = "sam_calib_disagree"
        target_parent = disagree_dir
    else:
        continue

    dst_dir = os.path.join(target_parent, sample_name)
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(sample_dir, dst_dir)

    with open(os.path.join(dst_dir, "sam_decoder_correction.txt"), "w", encoding="utf-8") as f:
        f.write(f"sample: {sample_name}\n")
        f.write(f"group: {group}\n")
        f.write(f"expression: {m.get('expression')}\n")
        f.write(f"sam_idx_argmax_iou_pred: {sam_idx}\n")
        f.write(f"base_idx: {base_idx}\n")
        f.write(f"calib_idx: {calib_idx}\n")
        f.write(f"final_idx: {final_idx}\n")
        f.write(f"iou_sam_with_gt: {iou_sam_gt}\n")
        f.write(f"iou_calib_with_gt: {iou_calib_gt}\n")
        f.write(f"iou_final_with_gt: {iou_final_gt}\n")
        f.write(f"gain_final_vs_sam: {gain_vs_sam}\n")
        f.write(f"iou_pred_list: {iou_pred}\n")
        f.write(f"base_score_list: {base_scores}\n")
        f.write(f"calib_score_list: {calib_scores}\n")
        f.write(f"evidence_score_list: {m.get('evidence_score_list')}\n")

    rows.append({
        "sample": sample_name,
        "group": group,
        "src_dir": sample_dir,
        "dst_dir": dst_dir,
        "expression": m.get("expression"),
        "sam_idx": sam_idx,
        "base_idx": base_idx,
        "calib_idx": calib_idx,
        "final_idx": final_idx,
        "iou_sam_with_gt": iou_sam_gt,
        "iou_calib_with_gt": iou_calib_gt,
        "iou_final_with_gt": iou_final_gt,
        "gain_final_vs_sam": gain_vs_sam,
        "iou_pred_list": json.dumps(iou_pred),
        "base_score_list": json.dumps(base_scores),
        "calib_score_list": json.dumps(calib_scores),
        "evidence_score_list": json.dumps(m.get("evidence_score_list")),
    })

summary_path = os.path.join(dst_root, "sam_decoder_correction_summary.csv")
rows_sorted = sorted(rows, key=lambda r: r["gain_final_vs_sam"], reverse=True)

with open(summary_path, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "sample", "group", "src_dir", "dst_dir", "expression",
        "sam_idx", "base_idx", "calib_idx", "final_idx",
        "iou_sam_with_gt", "iou_calib_with_gt", "iou_final_with_gt",
        "gain_final_vs_sam",
        "iou_pred_list", "base_score_list", "calib_score_list", "evidence_score_list",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_sorted)

from collections import Counter
cnt = Counter(r["group"] for r in rows)

print("total meta:", len(paths))
print("sam_calib_disagree total:", cnt.get("sam_calib_disagree", 0) + cnt.get("corrected_by_score_head", 0) + cnt.get("sam_final_diff_but_not_improved", 0))
print("corrected_by_score_head:", cnt.get("corrected_by_score_head", 0))
print("sam_final_diff_but_not_improved:", cnt.get("sam_final_diff_but_not_improved", 0))
print("sam_calib_disagree_only:", cnt.get("sam_calib_disagree", 0))
print("summary:", summary_path)

print("\nTop corrected cases:")
for r in rows_sorted[:20]:
    if r["group"] != "corrected_by_score_head":
        continue
    print(
        f"gain={r['gain_final_vs_sam']:.4f}",
        f"expr={r['expression']}",
        f"sam={r['sam_idx']}",
        f"calib={r['calib_idx']}",
        f"final={r['final_idx']}",
        f"dir={r['dst_dir']}",
    )

# 生成 corrected_by_score_head 的 top contact sheet
def make_sheet(root, out_name, max_samples=80):
    sample_dirs = []
    for meta_path in glob.glob(os.path.join(root, "sample_*", "meta.json")):
        d = os.path.dirname(meta_path)
        info_path = os.path.join(d, "sam_decoder_correction.txt")
        gain = 0.0
        if os.path.exists(info_path):
            txt = open(info_path, "r", encoding="utf-8").read()
            for line in txt.splitlines():
                if line.startswith("gain_final_vs_sam:"):
                    gain = float(line.split(":", 1)[1].strip())
                    break
        sample_dirs.append((gain, d))
    sample_dirs = [d for _, d in sorted(sample_dirs, reverse=True)[:max_samples]]

    if not sample_dirs:
        print("no samples for sheet:", root)
        return

    thumb_w = 1250
    thumb_h = 260
    header_h = 48
    rows_n = len(sample_dirs)
    canvas = Image.new("RGB", (thumb_w, rows_n * (thumb_h + header_h)), "white")
    draw = ImageDraw.Draw(canvas)

    for i, d in enumerate(sample_dirs):
        meta = json.load(open(os.path.join(d, "meta.json"), "r", encoding="utf-8"))
        info_path = os.path.join(d, "sam_decoder_correction.txt")
        info = {}
        if os.path.exists(info_path):
            for line in open(info_path, "r", encoding="utf-8"):
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    info[k.strip()] = v.strip()

        label = (
            f"{os.path.basename(d)} | "
            f"sam:{info.get('sam_idx_argmax_iou_pred')} "
            f"base:{info.get('base_idx')} "
            f"calib:{info.get('calib_idx')} "
            f"final:{info.get('final_idx')} | "
            f"IoU sam:{info.get('iou_sam_with_gt')} "
            f"final:{info.get('iou_final_with_gt')} "
            f"gain:{info.get('gain_final_vs_sam')} | "
            f"{meta.get('expression')}"
        )

        y = i * (thumb_h + header_h)
        draw.text((5, y + 8), label[:200], fill=(0, 0, 0))

        panel = os.path.join(d, "comparison_panel.png")
        im = Image.open(panel).convert("RGB")
        im.thumbnail((thumb_w - 10, thumb_h))
        canvas.paste(im, (5, y + header_h))

    canvas.save(out_name, quality=95)
    print("saved sheet:", out_name)

make_sheet(corrected_dir, os.path.join(dst_root, "contact_sheet_sam_decoder_corrected.jpg"))
PY

echo ""
echo "================ Final outputs ================"
echo "/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/outputs/vis_gt_81p0133_refcoco_val_sam_decoder_corrected/corrected_by_score_head"
echo "/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/outputs/vis_gt_81p0133_refcoco_val_sam_decoder_corrected/contact_sheet_sam_decoder_corrected.jpg"
echo "/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/outputs/vis_gt_81p0133_refcoco_val_sam_decoder_corrected/sam_decoder_correction_summary.csv"
