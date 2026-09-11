#!/usr/bin/env bash
set -e

cd /home/zxing/code/SAM_RIS_1/OpenWorldSAM-main

export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT=/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/output/cand_score_head_gapw_from807787_lr2p5e5_4k/run_0/model_best_80p7958_backup.pth

run_eval () {
  NAME=$1
  TOPGAP=$2
  EVDROP=$3

  OUT=./output/eval_807958_gate_${NAME}
  EXPORT=outputs/per_sample_807958_gate_${NAME}.jsonl

  echo ""
  echo "================ ${NAME}: TOPGAP=${TOPGAP}, EVDROP=${EVDROP} ================"

  rm -rf "${OUT}"
  rm -f "${EXPORT}"

  CUDA_VISIBLE_DEVICES=0 python train_net.py \
    --eval-only \
    --num-gpus 1 \
    --config-file configs/refcoco/Open-World-SAM2-CrossAttention-refcoco.yaml \
    MODEL.WEIGHTS ${CKPT} \
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
    MODEL.OpenWorldSAM2.CAND_RANK_LOSS_ON False \
    MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_ON True \
    MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ON False \
    MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ONLY False \
    MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_CONSERVATIVE_ON True \
    MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_DELTA_SCALE 0.10 \
    MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TOPGAP_THR ${TOPGAP} \
    MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_EVIDENCE_DROP_THR ${EVDROP} \
    MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_MIN_DELTA_GAP 0.00 \
    MODEL.OpenWorldSAM2.EVAL_EXPORT_PER_SAMPLE_ON True \
    MODEL.OpenWorldSAM2.EVAL_EXPORT_PER_SAMPLE_PATH ${EXPORT} \
    OUTPUT_DIR ${OUT}
}

run_eval A_current 0.05 0.03
run_eval B_looser  0.08 0.03
run_eval C_safe    0.08 0.01

echo ""
echo "================ Gate sweep eval summary ================"
for NAME in A_current B_looser C_safe; do
  LOG=output/eval_807958_gate_${NAME}/run_0/log.txt
  echo "----- ${NAME} -----"
  grep -n "copypaste: [0-9]" "$LOG" | tail -1
done

echo ""
echo "================ Gate sweep changed-ratio summary ================"
python - <<'PY'
import json, os, statistics
import numpy as np

off_path = "outputs/per_sample_807958_scorehead_off.jsonl"
names = ["A_current", "B_looser", "C_safe"]

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

if not os.path.exists(off_path):
    print("OFF export not found, skip changed-ratio summary:", off_path)
    raise SystemExit

off = load(off_path)

for name in names:
    path = f"outputs/per_sample_807958_gate_{name}.jsonl"
    if not os.path.exists(path):
        print(name, "missing export")
        continue

    on = load(path)
    keys = sorted(set(off) & set(on))
    diffs = []
    changed_idx = 0
    idx_avail = 0
    improved = worsened = 0

    for k in keys:
        a = off[k]
        b = on[k]
        ia = get_iou(a)
        ib = get_iou(b)
        if ia is None or ib is None:
            continue
        diff = ib - ia
        diffs.append(diff)
        if diff > 0.01:
            improved += 1
        if diff < -0.01:
            worsened += 1

        ca = a.get("current_idx", None)
        cb = b.get("current_idx", None)
        if ca is not None and cb is not None:
            idx_avail += 1
            if int(ca) != int(cb):
                changed_idx += 1

    print(f"----- {name} -----")
    print(f"matched={len(diffs)}")
    print(f"mean_diff={np.mean(diffs)*100:.4f}")
    print(f"improved>0.01={improved}")
    print(f"worsened<-0.01={worsened}")
    if idx_avail:
        print(f"current_idx_changed={changed_idx}/{idx_avail} ({changed_idx/idx_avail*100:.2f}%)")
PY

echo ""
echo "DONE."
