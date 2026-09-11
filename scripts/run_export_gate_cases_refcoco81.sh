#!/usr/bin/env bash
set -e

cd /home/zxing/code/SAM_RIS_1/OpenWorldSAM-main

echo "================ Step 0: py_compile check ================"
python -m py_compile train_net.py model/open_world_sam2.py model/config.py

echo "================ Step 1: clean old outputs ================"
rm -rf outputs/vis_81p0133_refcoco_val_all
rm -rf outputs/vis_81p0133_refcoco_val_gate_cases
rm -rf output/eval_vis_81p0133_refcoco_val_all

export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "================ Step 2: export all visualization samples ================"

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
  MODEL.OpenWorldSAM2.VIS_EXPORT_DIR outputs/vis_81p0133_refcoco_val_all \
  MODEL.OpenWorldSAM2.VIS_EXPORT_MAX_SAMPLES 5000 \
  MODEL.OpenWorldSAM2.VIS_EXPORT_EVERY 1 \
  OUTPUT_DIR ./output/eval_vis_81p0133_refcoco_val_all

echo "================ Step 3: filter Conservative Gate cases ================"

python - <<'PY'
import glob, json, os, shutil, csv
import numpy as np

src_root = "outputs/vis_81p0133_refcoco_val_all"
dst_root = "outputs/vis_81p0133_refcoco_val_gate_cases"

accepted_dir = os.path.join(dst_root, "accepted_switch")
rejected_dir = os.path.join(dst_root, "rejected_keep_base")
other_dir = os.path.join(dst_root, "other_gate_cases")

for d in [accepted_dir, rejected_dir, other_dir]:
    os.makedirs(d, exist_ok=True)

paths = sorted(glob.glob(os.path.join(src_root, "sample_*", "meta.json")))

rows = []
accepted = []
rejected = []
other = []

def argmax_or_none(x):
    if not x:
        return None
    return int(np.argmax(x))

for meta_path in paths:
    with open(meta_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    sample_dir = os.path.dirname(meta_path)
    sample_name = os.path.basename(sample_dir)

    iou = m.get("iou_pred_list") or []
    evidence = m.get("evidence_score_list") or []
    base = m.get("base_score_list") or []
    calib = m.get("calib_score_list") or []
    final_idx = m.get("final_idx")

    sam_idx = argmax_or_none(iou)
    base_idx = argmax_or_none(base)
    calib_idx = argmax_or_none(calib)

    if base_idx is None or calib_idx is None or final_idx is None:
        continue

    # Gate 有作用的核心样本：base scoring 和 calibrated scoring 的候选不一致
    if base_idx == calib_idx:
        continue

    if final_idx == calib_idx:
        gate_action = "accepted_switch"
        target_parent = accepted_dir
        accepted.append(sample_name)
    elif final_idx == base_idx:
        gate_action = "rejected_keep_base"
        target_parent = rejected_dir
        rejected.append(sample_name)
    else:
        gate_action = "other"
        target_parent = other_dir
        other.append(sample_name)

    dst_dir = os.path.join(target_parent, sample_name)
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(sample_dir, dst_dir)

    with open(os.path.join(dst_dir, "gate_decision.txt"), "w", encoding="utf-8") as f:
        f.write(f"sample: {sample_name}\n")
        f.write(f"expression: {m.get('expression')}\n")
        f.write(f"gate_action: {gate_action}\n")
        f.write(f"sam_idx: {sam_idx}\n")
        f.write(f"base_idx: {base_idx}\n")
        f.write(f"calib_idx: {calib_idx}\n")
        f.write(f"final_idx: {final_idx}\n")
        f.write(f"iou_pred_list: {iou}\n")
        f.write(f"evidence_score_list: {evidence}\n")
        f.write(f"base_score_list: {base}\n")
        f.write(f"calib_score_list: {calib}\n")

    rows.append({
        "sample": sample_name,
        "dir": dst_dir,
        "expression": m.get("expression"),
        "gate_action": gate_action,
        "sam_idx": sam_idx,
        "base_idx": base_idx,
        "calib_idx": calib_idx,
        "final_idx": final_idx,
        "iou_pred_list": json.dumps(iou),
        "evidence_score_list": json.dumps(evidence),
        "base_score_list": json.dumps(base),
        "calib_score_list": json.dumps(calib),
    })

summary_path = os.path.join(dst_root, "gate_cases_summary.csv")
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "sample", "dir", "expression", "gate_action",
        "sam_idx", "base_idx", "calib_idx", "final_idx",
        "iou_pred_list", "evidence_score_list", "base_score_list", "calib_score_list"
    ])
    writer.writeheader()
    writer.writerows(rows)

print("total meta:", len(paths))
print("gate cases:", len(rows))
print("accepted_switch:", len(accepted))
print("rejected_keep_base:", len(rejected))
print("other:", len(other))
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

    items = [
        ("image.png", "Image"),
        ("evidence_overlay.png", "Evidence"),
        ("cand_0_overlay.png", "M0"),
        ("cand_1_overlay.png", "M1"),
        ("cand_2_overlay.png", "M2"),
        ("final_overlay.png", "Final"),
    ]

    cell_w, cell_h = 190, 165
    header_h = 40
    cols = len(items)
    rows = len(sample_dirs)

    canvas = Image.new("RGB", (cols * cell_w, header_h + rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)

    for j, (_, title) in enumerate(items):
        draw.text((j * cell_w + 10, 12), title, fill=(0, 0, 0))

    for i, d in enumerate(sample_dirs):
        meta_path = os.path.join(d, "meta.json")
        expr = ""
        base_idx = calib_idx = final_idx = None

        if os.path.exists(meta_path):
            m = json.load(open(meta_path, "r", encoding="utf-8"))
            expr = m.get("expression", "")
            base = m.get("base_score_list") or []
            calib = m.get("calib_score_list") or []
            base_idx = int(np.argmax(base)) if base else None
            calib_idx = int(np.argmax(calib)) if calib else None
            final_idx = m.get("final_idx")

        y0 = header_h + i * cell_h
        label = f"{os.path.basename(d)}  base:{base_idx} calib:{calib_idx} final:{final_idx}"
        draw.text((5, y0 + 3), label[:95], fill=(0, 0, 0))

        for j, (fname, _) in enumerate(items):
            p = os.path.join(d, fname)
            if not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGB")
            im.thumbnail((cell_w - 10, cell_h - 35))
            x = j * cell_w + (cell_w - im.width) // 2
            y = y0 + 25
            canvas.paste(im, (x, y))

    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    canvas.save(out_name, quality=95)
    print("saved:", out_name)

base = "outputs/vis_81p0133_refcoco_val_gate_cases"
make_sheet(os.path.join(base, "accepted_switch"), os.path.join(base, "contact_sheet_accepted_switch.jpg"))
make_sheet(os.path.join(base, "rejected_keep_base"), os.path.join(base, "contact_sheet_rejected_keep_base.jpg"))
make_sheet(os.path.join(base, "other_gate_cases"), os.path.join(base, "contact_sheet_other_gate_cases.jpg"))
PY

echo "================ Step 5: final check ================"

echo ""
echo "Exported all samples:"
find outputs/vis_81p0133_refcoco_val_all -name meta.json | wc -l

echo ""
echo "Gate case summary:"
cat outputs/vis_81p0133_refcoco_val_gate_cases/gate_cases_summary.csv | head -20

echo ""
echo "Accepted switch count:"
find outputs/vis_81p0133_refcoco_val_gate_cases/accepted_switch -maxdepth 1 -type d | wc -l

echo ""
echo "Rejected keep-base count:"
find outputs/vis_81p0133_refcoco_val_gate_cases/rejected_keep_base -maxdepth 1 -type d | wc -l

echo ""
echo "Key outputs:"
echo "outputs/vis_81p0133_refcoco_val_all"
echo "outputs/vis_81p0133_refcoco_val_gate_cases"
echo "outputs/vis_81p0133_refcoco_val_gate_cases/gate_cases_summary.csv"
echo "outputs/vis_81p0133_refcoco_val_gate_cases/contact_sheet_accepted_switch.jpg"
echo "outputs/vis_81p0133_refcoco_val_gate_cases/contact_sheet_rejected_keep_base.jpg"

echo ""
echo "DONE."
