#!/usr/bin/env bash
set -e

cd /home/zxing/code/SAM_RIS_1/OpenWorldSAM-main

echo "================ Step 0: py_compile check ================"
python -m py_compile train_net.py model/open_world_sam2.py model/config.py

echo "================ Step 1: clean old outputs ================"
rm -rf outputs/vis_gt_81p0133_refcoco_val_all
rm -rf outputs/vis_gt_81p0133_refcoco_val_gate_cases
rm -rf output/eval_vis_gt_81p0133_refcoco_val_all

export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "================ Step 2: export all samples with GT ================"

CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --eval-only \
  --num-gpus 1 \
  --config-file configs/refcoco/Open-World-SAM2-CrossAttention-refcoco.yaml \
  DATASETS.TEST "('refcoco_val_unc',)" \
  MODEL.WEIGHTS /home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/output/cand_score_head_gapw_from807787_lr2p5e5_4k/run_0/model_best_80p7958_backup.pth \
  MODEL.OpenWorldSAM2.EVF_CONFIG /home/zxing/code/SAM_RIS_1/checkpoints/evf-sam2-multitask \
  MODEL.OpenWorldSAM2.TOKENIZER_CONFIG /home/zxing/code/SAM_RIS_1/checkpoints/evf-sam2-multitask \
  MODEL.OpenWorldSAM2.VISION_PRETRAINED /home/zxing/code/SAM_RIS_1/sam2_hiera_large.pt \
  MODEL.OpenWorldSAM2.STA_ON True \
  MODEL.OpenWorldSAM2.STA_TRAIN_ON False \
  MODEL.OpenWorldSAM2.STA_VARIANT opm_v1 \
  MODEL.OpenWorldSAM2.STA_GATE_INIT -8.0 \
  MODEL.OpenWorldSAM2.STA_INJECT_BLOCKS "[20]" \
  MODEL.OpenWorldSAM2.STA_NUM_TEXT_VIEWS 3 \
  MODEL.OpenWorldSAM2.STA_USE_COMPETITIVE_ROUTING True \
  MODEL.OpenWorldSAM2.STA_USE_FILM False \
  MODEL.OpenWorldSAM2.STA_USE_COMPLEXITY_GATE True \
  MODEL.OpenWorldSAM2.STA_MAX_RESIDUAL_RATIO 0.05 \
  MODEL.OpenWorldSAM2.STA_MAX_MOD_SCALE 1.0 \
  MODEL.OpenWorldSAM2.STA_MIN_BLOCK_GATE 0.0 \
  MODEL.OpenWorldSAM2.FUSION.ON True \
  MODEL.OpenWorldSAM2.FUSION.USE_REL False \
  MODEL.OpenWorldSAM2.FUSION.SLOT_ON True \
  MODEL.OpenWorldSAM2.FUSION.SLOT_NUM 2 \
  MODEL.OpenWorldSAM2.FUSION.SLOT_GATE_TYPE residual \
  MODEL.OpenWorldSAM2.FUSION.EVIDENCE_USE_SLOTS True \
  MODEL.OpenWorldSAM2.FUSION.EVIDENCE_USE_CAT True \
  MODEL.OpenWorldSAM2.FUSION.EVIDENCE_USE_ATTR True \
  MODEL.OpenWorldSAM2.FUSION.EVIDENCE_USE_REL False \
  MODEL.OpenWorldSAM2.FUSION.EVIDENCE_RERANK True \
  MODEL.OpenWorldSAM2.FUSION.EVIDENCE_ALPHA 5.3 \
  MODEL.OpenWorldSAM2.FUSION.RESIDUAL_BETA_OVERRIDE 0.02 \
  MODEL.OpenWorldSAM2.FUSION.WEAK_MAP_FALLBACK_THR 1e-3 \
  MODEL.OpenWorldSAM2.FUSION.ADAPTIVE_ALPHA_ON True \
  MODEL.OpenWorldSAM2.FUSION.ADAPTIVE_ALPHA_SHORT_SCALE 0.5 \
  MODEL.OpenWorldSAM2.FUSION.ADAPTIVE_ALPHA_MID_SCALE 1.0 \
  MODEL.OpenWorldSAM2.FUSION.ADAPTIVE_ALPHA_LONG_SCALE 1.2 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_ON True \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ON False \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ONLY False \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_CONSERVATIVE_ON True \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_DELTA_SCALE 0.24 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TOPGAP_THR 0.15 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_EVIDENCE_DROP_THR 0.06 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_MIN_DELTA_GAP 0.00 \
  MODEL.OpenWorldSAM2.VIS_EXPORT_ON True \
  MODEL.OpenWorldSAM2.VIS_EXPORT_DIR outputs/vis_gt_81p0133_refcoco_val_all \
  MODEL.OpenWorldSAM2.VIS_EXPORT_MAX_SAMPLES 5000 \
  MODEL.OpenWorldSAM2.VIS_EXPORT_EVERY 1 \
  OUTPUT_DIR ./output/eval_vis_gt_81p0133_refcoco_val_all

echo "================ Step 3: filter GT gate cases ================"

python - <<'PY'
import glob, json, os, shutil, csv
import numpy as np

src_root = "outputs/vis_gt_81p0133_refcoco_val_all"
dst_root = "outputs/vis_gt_81p0133_refcoco_val_gate_cases"

dirs = {
    "accepted_improved": os.path.join(dst_root, "accepted_improved"),
    "accepted_not_improved": os.path.join(dst_root, "accepted_not_improved"),
    "rejected_keep_base": os.path.join(dst_root, "rejected_keep_base"),
    "base_calib_same": os.path.join(dst_root, "base_calib_same"),
    "other": os.path.join(dst_root, "other"),
}
for d in dirs.values():
    os.makedirs(d, exist_ok=True)

paths = sorted(glob.glob(os.path.join(src_root, "sample_*", "meta.json")))

def argmax_or_none(x):
    if not x:
        return None
    return int(np.argmax(x))

rows = []

for meta_path in paths:
    with open(meta_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    sample_dir = os.path.dirname(meta_path)
    sample_name = os.path.basename(sample_dir)

    base = m.get("base_score_list") or []
    calib = m.get("calib_score_list") or []
    final_idx = m.get("final_idx")
    gt_available = bool(m.get("gt_available"))

    base_idx = m.get("base_idx")
    calib_idx = m.get("calib_idx")

    if base_idx is None:
        base_idx = argmax_or_none(base)
    if calib_idx is None:
        calib_idx = argmax_or_none(calib)

    if base_idx is None or calib_idx is None or final_idx is None:
        continue

    iou_base = m.get("iou_base_with_gt")
    iou_calib = m.get("iou_calib_with_gt")
    iou_final = m.get("iou_final_with_gt")
    gate_improved = m.get("gate_improved")

    # 分类逻辑
    if base_idx == calib_idx:
        group = "base_calib_same"
    elif final_idx == calib_idx:
        if gt_available and gate_improved is True:
            group = "accepted_improved"
        else:
            group = "accepted_not_improved"
    elif final_idx == base_idx:
        group = "rejected_keep_base"
    else:
        group = "other"

    # 只复制 gate 相关和 improved 样本，base_calib_same 不全复制，避免目录过大
    copy_this = group != "base_calib_same"
    if copy_this:
        dst_dir = os.path.join(dirs[group], sample_name)
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(sample_dir, dst_dir)

        with open(os.path.join(dst_dir, "gate_gt_decision.txt"), "w", encoding="utf-8") as f:
            f.write(f"sample: {sample_name}\n")
            f.write(f"expression: {m.get('expression')}\n")
            f.write(f"group: {group}\n")
            f.write(f"gt_available: {gt_available}\n")
            f.write(f"gt_source: {m.get('gt_source')}\n")
            f.write(f"base_idx: {base_idx}\n")
            f.write(f"calib_idx: {calib_idx}\n")
            f.write(f"final_idx: {final_idx}\n")
            f.write(f"iou_base_with_gt: {iou_base}\n")
            f.write(f"iou_calib_with_gt: {iou_calib}\n")
            f.write(f"iou_final_with_gt: {iou_final}\n")
            f.write(f"gate_improved: {gate_improved}\n")
            f.write(f"base_score_list: {m.get('base_score_list')}\n")
            f.write(f"calib_score_list: {m.get('calib_score_list')}\n")
            f.write(f"iou_pred_list: {m.get('iou_pred_list')}\n")
            f.write(f"evidence_score_list: {m.get('evidence_score_list')}\n")

    rows.append({
        "sample": sample_name,
        "src_dir": sample_dir,
        "group": group,
        "expression": m.get("expression"),
        "gt_available": gt_available,
        "gt_source": m.get("gt_source"),
        "base_idx": base_idx,
        "calib_idx": calib_idx,
        "final_idx": final_idx,
        "iou_base_with_gt": iou_base,
        "iou_calib_with_gt": iou_calib,
        "iou_final_with_gt": iou_final,
        "gate_improved": gate_improved,
        "base_score_list": json.dumps(m.get("base_score_list")),
        "calib_score_list": json.dumps(m.get("calib_score_list")),
        "iou_pred_list": json.dumps(m.get("iou_pred_list")),
        "evidence_score_list": json.dumps(m.get("evidence_score_list")),
    })

summary_path = os.path.join(dst_root, "gate_gt_cases_summary.csv")
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
        "sample", "src_dir", "group", "expression", "gt_available", "gt_source",
        "base_idx", "calib_idx", "final_idx",
        "iou_base_with_gt", "iou_calib_with_gt", "iou_final_with_gt",
        "gate_improved",
        "base_score_list", "calib_score_list", "iou_pred_list", "evidence_score_list",
    ])
    writer.writeheader()
    writer.writerows(rows)

from collections import Counter
cnt = Counter(r["group"] for r in rows)
print("total meta:", len(paths))
print("total rows:", len(rows))
for k in ["accepted_improved", "accepted_not_improved", "rejected_keep_base", "base_calib_same", "other"]:
    print(k + ":", cnt.get(k, 0))
print("saved to:", dst_root)
print("summary:", summary_path)
PY

echo "================ Step 4: make contact sheets ================"

python - <<'PY'
from PIL import Image, ImageDraw
import glob, os, json
import numpy as np

def make_sheet(root, out_name, max_samples=80):
    sample_dirs = sorted(glob.glob(os.path.join(root, "sample_*")))[:max_samples]
    if not sample_dirs:
        print("no samples in", root)
        return

    # 优先使用 comparison_panel，因为里面已经有 Image/Evidence/GT/M_base/M_calib/Final
    panels = []
    for d in sample_dirs:
        p = os.path.join(d, "comparison_panel.png")
        if os.path.exists(p):
            panels.append((d, p))

    if not panels:
        print("no comparison panels in", root)
        return

    thumb_w = 1100
    thumb_h = 230
    header_h = 28
    rows = len(panels)
    canvas = Image.new("RGB", (thumb_w, rows * (thumb_h + header_h)), "white")
    draw = ImageDraw.Draw(canvas)

    for i, (d, p) in enumerate(panels):
        meta_path = os.path.join(d, "meta.json")
        label = os.path.basename(d)
        if os.path.exists(meta_path):
            m = json.load(open(meta_path, "r", encoding="utf-8"))
            label = (
                f"{os.path.basename(d)} | "
                f"base:{m.get('base_idx')} calib:{m.get('calib_idx')} final:{m.get('final_idx')} | "
                f"IoU base:{m.get('iou_base_with_gt')} calib:{m.get('iou_calib_with_gt')} final:{m.get('iou_final_with_gt')} | "
                f"improved:{m.get('gate_improved')} | "
                f"{m.get('expression')}"
            )
        y = i * (thumb_h + header_h)
        draw.text((5, y + 6), label[:180], fill=(0, 0, 0))

        im = Image.open(p).convert("RGB")
        im.thumbnail((thumb_w - 10, thumb_h))
        canvas.paste(im, (5, y + header_h))

    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    canvas.save(out_name, quality=95)
    print("saved:", out_name)

base = "outputs/vis_gt_81p0133_refcoco_val_gate_cases"
make_sheet(os.path.join(base, "accepted_improved"), os.path.join(base, "contact_sheet_accepted_improved.jpg"))
make_sheet(os.path.join(base, "accepted_not_improved"), os.path.join(base, "contact_sheet_accepted_not_improved.jpg"))
make_sheet(os.path.join(base, "rejected_keep_base"), os.path.join(base, "contact_sheet_rejected_keep_base.jpg"))
PY

echo "================ Step 5: final check ================"

echo ""
echo "All GT-exported meta count:"
find outputs/vis_gt_81p0133_refcoco_val_all -name meta.json | wc -l

echo ""
echo "GT available count:"
python - <<'PY'
import glob, json
paths = glob.glob("outputs/vis_gt_81p0133_refcoco_val_all/*/meta.json")
ok = 0
for p in paths:
    ok += int(bool(json.load(open(p, "r", encoding="utf-8")).get("gt_available")))
print(ok, "/", len(paths))
PY

echo ""
echo "Gate GT summary:"
cat outputs/vis_gt_81p0133_refcoco_val_gate_cases/gate_gt_cases_summary.csv | head -20

echo ""
echo "Key outputs:"
echo "/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/outputs/vis_gt_81p0133_refcoco_val_all"
echo "/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/outputs/vis_gt_81p0133_refcoco_val_gate_cases/accepted_improved"
echo "/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/outputs/vis_gt_81p0133_refcoco_val_gate_cases/accepted_not_improved"
echo "/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/outputs/vis_gt_81p0133_refcoco_val_gate_cases/rejected_keep_base"
echo "/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/outputs/vis_gt_81p0133_refcoco_val_gate_cases/gate_gt_cases_summary.csv"
echo "/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/outputs/vis_gt_81p0133_refcoco_val_gate_cases/contact_sheet_accepted_improved.jpg"
echo "/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/outputs/vis_gt_81p0133_refcoco_val_gate_cases/contact_sheet_accepted_not_improved.jpg"
echo "/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/outputs/vis_gt_81p0133_refcoco_val_gate_cases/contact_sheet_rejected_keep_base.jpg"

echo ""
echo "DONE."
