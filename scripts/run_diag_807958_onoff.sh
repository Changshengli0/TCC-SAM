#!/usr/bin/env bash
set -e

cd /home/zxing/code/SAM_RIS_1/OpenWorldSAM-main

export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT=/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/output/cand_score_head_gapw_from807787_lr2p5e5_4k/run_0/model_best_80p7958_backup.pth

mkdir -p outputs

COMMON=(
  --num-gpus 1
  --config-file configs/refcoco/Open-World-SAM2-CrossAttention-refcoco.yaml
  MODEL.WEIGHTS "$CKPT"
  MODEL.OpenWorldSAM2.EVF_CONFIG /home/zxing/code/SAM_RIS_1/checkpoints/evf-sam2-multitask
  MODEL.OpenWorldSAM2.TOKENIZER_CONFIG /home/zxing/code/SAM_RIS_1/checkpoints/evf-sam2-multitask
  MODEL.OpenWorldSAM2.VISION_PRETRAINED /home/zxing/code/SAM_RIS_1/sam2_hiera_large.pt

  MODEL.OpenWorldSAM2.STA_ON True
  MODEL.OpenWorldSAM2.STA_TRAIN_ON False
  MODEL.OpenWorldSAM2.STA_VARIANT opm_v1
  MODEL.OpenWorldSAM2.STA_GATE_INIT -8.0
  MODEL.OpenWorldSAM2.STA_INJECT_BLOCKS "[20]"
  MODEL.OpenWorldSAM2.STA_NUM_TEXT_VIEWS 3
  MODEL.OpenWorldSAM2.STA_USE_COMPETITIVE_ROUTING True
  MODEL.OpenWorldSAM2.STA_USE_FILM False
  MODEL.OpenWorldSAM2.STA_USE_COMPLEXITY_GATE True
  MODEL.OpenWorldSAM2.STA_MAX_RESIDUAL_RATIO 0.05
  MODEL.OpenWorldSAM2.STA_MAX_MOD_SCALE 1.0
  MODEL.OpenWorldSAM2.STA_MIN_BLOCK_GATE 0.0

  MODEL.OpenWorldSAM2.FUSION.ON True
  MODEL.OpenWorldSAM2.FUSION.USE_REL False
  MODEL.OpenWorldSAM2.FUSION.SLOT_ON True
  MODEL.OpenWorldSAM2.FUSION.SLOT_NUM 2
  MODEL.OpenWorldSAM2.FUSION.SLOT_GATE_TYPE residual
  MODEL.OpenWorldSAM2.FUSION.EVIDENCE_USE_SLOTS True
  MODEL.OpenWorldSAM2.FUSION.EVIDENCE_USE_CAT True
  MODEL.OpenWorldSAM2.FUSION.EVIDENCE_USE_ATTR True
  MODEL.OpenWorldSAM2.FUSION.EVIDENCE_USE_REL False
  MODEL.OpenWorldSAM2.FUSION.EVIDENCE_RERANK True
  MODEL.OpenWorldSAM2.FUSION.EVIDENCE_ALPHA 5.3
  MODEL.OpenWorldSAM2.FUSION.RESIDUAL_BETA_OVERRIDE 0.02
  MODEL.OpenWorldSAM2.FUSION.WEAK_MAP_FALLBACK_THR 1e-3
  MODEL.OpenWorldSAM2.FUSION.ADAPTIVE_ALPHA_ON True
  MODEL.OpenWorldSAM2.FUSION.ADAPTIVE_ALPHA_SHORT_SCALE 0.5
  MODEL.OpenWorldSAM2.FUSION.ADAPTIVE_ALPHA_MID_SCALE 1.0
  MODEL.OpenWorldSAM2.FUSION.ADAPTIVE_ALPHA_LONG_SCALE 1.2

  MODEL.OpenWorldSAM2.CAND_RANK_LOSS_ON False
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ON False
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ONLY False
)

echo "==================== [1] score head OFF export ===================="
rm -f outputs/per_sample_807958_scorehead_off.jsonl
rm -rf output/eval_807958_scorehead_off_export

CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --eval-only \
  "${COMMON[@]}" \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_ON False \
  MODEL.OpenWorldSAM2.EVAL_EXPORT_PER_SAMPLE_ON True \
  MODEL.OpenWorldSAM2.EVAL_EXPORT_PER_SAMPLE_PATH outputs/per_sample_807958_scorehead_off.jsonl \
  OUTPUT_DIR ./output/eval_807958_scorehead_off_export

echo "==================== [2] score head ON export ===================="
rm -f outputs/per_sample_807958_scorehead_on.jsonl
rm -rf output/eval_807958_scorehead_on_export

CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --eval-only \
  "${COMMON[@]}" \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_ON True \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_CONSERVATIVE_ON True \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_DELTA_SCALE 0.10 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TOPGAP_THR 0.05 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_EVIDENCE_DROP_THR 0.03 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_MIN_DELTA_GAP 0.00 \
  MODEL.OpenWorldSAM2.EVAL_EXPORT_PER_SAMPLE_ON True \
  MODEL.OpenWorldSAM2.EVAL_EXPORT_PER_SAMPLE_PATH outputs/per_sample_807958_scorehead_on.jsonl \
  OUTPUT_DIR ./output/eval_807958_scorehead_on_export

echo "==================== [3] analyze ON/OFF + strict oracle ===================="

python - <<'PY'
import json, os, math, statistics
import numpy as np

off_path = "outputs/per_sample_807958_scorehead_off.jsonl"
on_path  = "outputs/per_sample_807958_scorehead_on.jsonl"
out_sum  = "outputs/diag_807958_scorehead_onoff_summary.txt"
out_imp  = "outputs/diag_807958_top_improved.txt"
out_wor  = "outputs/diag_807958_top_worsened.txt"

def load(path):
    data = {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            key = d.get("sample_uid")
            if key is None:
                key = f'{d.get("image_id")}:{d.get("prompt_index")}'
            data[key] = d
    return data

def get_iou(d):
    for k in ["iou", "top1_iou", "current_iou"]:
        if k in d and d[k] is not None:
            return float(d[k])
    return None

def reconstruct_ciou(records):
    inter_sum = 0.0
    union_sum = 0.0
    ious = []
    for d in records:
        iou = get_iou(d)
        if iou is None:
            continue
        pred = float(d.get("pred_area", 0.0))
        gt = float(d.get("gt_area", 0.0))
        if pred <= 0 and gt <= 0:
            continue
        # iou = inter / (pred + gt - inter)
        inter = iou * (pred + gt) / (1.0 + iou + 1e-12)
        union = pred + gt - inter
        inter_sum += inter
        union_sum += union
        ious.append(iou)
    ciou = inter_sum / union_sum * 100 if union_sum > 0 else float("nan")
    miou = statistics.mean(ious) * 100 if ious else float("nan")
    return ciou, miou, len(ious)

def strict_stats(records):
    cur = []
    ora = []
    gaps = []
    valid = 0
    for d in records:
        if not d.get("strict_oracle_valid", True):
            continue

        c = d.get("current_iou_from_candidate", None)
        o = d.get("oracle_best_of_k_strict", None)

        if c is None:
            c = d.get("current_iou", None)
        if o is None:
            o = d.get("oracle_best_iou", d.get("oracle_best_of_k", None))

        if c is None and "candidate_iou_list" in d and "current_idx" in d:
            try:
                c = d["candidate_iou_list"][int(d["current_idx"])]
            except Exception:
                c = None

        if o is None and "candidate_iou_list" in d:
            try:
                o = max(float(x) for x in d["candidate_iou_list"])
            except Exception:
                o = None

        if c is None or o is None:
            continue

        c = float(c)
        o = float(o)
        cur.append(c)
        ora.append(o)
        gaps.append(o - c)
        valid += 1

    if not cur:
        return None

    gaps_np = np.array(gaps)
    return {
        "valid": valid,
        "strict_current": float(np.mean(cur)) * 100,
        "strict_oracle": float(np.mean(ora)) * 100,
        "strict_headroom": float(np.mean(gaps)) * 100,
        "gap_gt_001": int((gaps_np > 0.01).sum()),
        "gap_gt_003": int((gaps_np > 0.03).sum()),
        "gap_gt_005": int((gaps_np > 0.05).sum()),
        "gap_gt_010": int((gaps_np > 0.10).sum()),
    }

off = load(off_path)
on = load(on_path)
keys = sorted(set(off) & set(on))

lines = []
def add(s=""):
    lines.append(str(s))

add("==================== Basic export counts ====================")
add(f"off records = {len(off)}")
add(f"on  records = {len(on)}")
add(f"matched     = {len(keys)}")

add("")
add("==================== Reconstructed metrics ====================")
for name, data in [("OFF", off), ("ON", on)]:
    ciou, miou, n = reconstruct_ciou(list(data.values()))
    add(f"{name}: reconstructed cIoU={ciou:.4f}, mean IoU={miou:.4f}, n={n}")

add("")
add("==================== Strict oracle headroom ====================")
for name, data in [("OFF", off), ("ON", on)]:
    st = strict_stats(list(data.values()))
    if st is None:
        add(f"{name}: strict fields not found")
    else:
        add(f"{name}: valid={st['valid']}")
        add(f"  strict current avg IoU = {st['strict_current']:.4f}")
        add(f"  strict oracle  avg IoU = {st['strict_oracle']:.4f}")
        add(f"  strict headroom        = {st['strict_headroom']:.4f}")
        add(f"  oracle better >0.01    = {st['gap_gt_001']}")
        add(f"  oracle better >0.03    = {st['gap_gt_003']}")
        add(f"  oracle better >0.05    = {st['gap_gt_005']}")
        add(f"  oracle better >0.10    = {st['gap_gt_010']}")

add("")
add("==================== ON vs OFF per-sample comparison ====================")
rows = []
changed_idx = 0
idx_available = 0

for k in keys:
    a = off[k]
    b = on[k]
    iou_a = get_iou(a)
    iou_b = get_iou(b)
    if iou_a is None or iou_b is None:
        continue
    diff = iou_b - iou_a
    gt = float(a.get("gt_area", 0.0))
    pred_a = float(a.get("pred_area", 0.0))
    pred_b = float(b.get("pred_area", 0.0))
    expr = a.get("expression", "")
    image_id = a.get("image_id", "")
    prompt_index = a.get("prompt_index", "")

    cur_a = a.get("current_idx", None)
    cur_b = b.get("current_idx", None)
    if cur_a is not None and cur_b is not None:
        idx_available += 1
        if int(cur_a) != int(cur_b):
            changed_idx += 1

    rows.append({
        "key": k,
        "off": iou_a,
        "on": iou_b,
        "diff": diff,
        "gt": gt,
        "pred_off": pred_a,
        "pred_on": pred_b,
        "expr": expr,
        "image_id": image_id,
        "prompt_index": prompt_index,
        "idx_off": cur_a,
        "idx_on": cur_b,
    })

diffs = np.array([r["diff"] for r in rows], dtype=float)
gts = np.array([r["gt"] for r in rows], dtype=float)

add(f"valid comparison samples = {len(rows)}")
add(f"mean diff ON-OFF        = {diffs.mean()*100:.4f}")
add(f"improved >0.01          = {int((diffs > 0.01).sum())}")
add(f"worsened  <-0.01        = {int((diffs < -0.01).sum())}")
add(f"improved >0.03          = {int((diffs > 0.03).sum())}")
add(f"worsened  <-0.03        = {int((diffs < -0.03).sum())}")
add(f"improved >0.05          = {int((diffs > 0.05).sum())}")
add(f"worsened  <-0.05        = {int((diffs < -0.05).sum())}")

if idx_available:
    add(f"current_idx changed     = {changed_idx}/{idx_available} ({changed_idx/idx_available*100:.2f}%)")
else:
    add("current_idx changed     = unavailable")

add("")
add("==================== Diff by GT area bins ====================")
if len(gts) > 0:
    q1, q2 = np.quantile(gts, [0.33, 0.66])
    bins = [
        ("small bottom33", gts <= q1),
        ("middle 33-66", (gts > q1) & (gts <= q2)),
        ("large top33", gts > q2),
    ]
    for name, mask in bins:
        d = diffs[mask]
        add(f"{name}: n={len(d)}, mean_diff={d.mean()*100:.4f}, improved>0.01={int((d>0.01).sum())}, worsened<-0.01={int((d<-0.01).sum())}")

def format_row(r):
    return (
        f"{r['key']} diff={r['diff']:+.4f} off={r['off']:.4f} on={r['on']:.4f} "
        f"idx={r['idx_off']}->{r['idx_on']} gt={r['gt']:.0f} "
        f"img={r['image_id']} prompt={r['prompt_index']} expr={r['expr']}"
    )

top_imp = sorted(rows, key=lambda x: -x["diff"])[:80]
top_wor = sorted(rows, key=lambda x: x["diff"])[:80]

with open(out_imp, "w") as f:
    for r in top_imp:
        f.write(format_row(r) + "\n")

with open(out_wor, "w") as f:
    for r in top_wor:
        f.write(format_row(r) + "\n")

add("")
add("==================== Top examples saved ====================")
add(f"top improved saved to: {out_imp}")
add(f"top worsened  saved to: {out_wor}")

os.makedirs("outputs", exist_ok=True)
with open(out_sum, "w") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print("")
print("saved summary:", out_sum)
print("saved improved:", out_imp)
print("saved worsened:", out_wor)
PY

echo "==================== [4] eval log summary ===================="
echo "----- OFF eval -----"
grep -n "copypaste: [0-9]" output/eval_807958_scorehead_off_export/run_0/log.txt | tail -1
echo "----- ON eval -----"
grep -n "copypaste: [0-9]" output/eval_807958_scorehead_on_export/run_0/log.txt | tail -1

echo "DONE."
