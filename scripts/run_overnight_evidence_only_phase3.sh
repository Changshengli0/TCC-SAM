#!/usr/bin/env bash
set -euo pipefail

cd /home/zxing/code/SAM_RIS_1/OpenWorldSAM-main

export PYTHONPATH=/home/zxing/code/SAM_RIS_1:/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m py_compile model/open_world_sam2.py model/config.py train_net.py && echo "[OK] compile"

ROOT_TS=$(date +%Y%m%d_%H%M%S)
SUMMARY="outputs/overnight_evidence_only_phase3_summary_${ROOT_TS}.txt"
echo "SUMMARY=$SUMMARY"
echo "ROOT_TS=$ROOT_TS" | tee -a "$SUMMARY"

BASE_ARGS=(
  --num-gpus 1
  --eval-only
  --config-file configs/refcoco/Open-World-SAM2-CrossAttention-refcoco.yaml
  MODEL.WEIGHTS output/cand_score_head_gapw_from807787_lr2p5e5_4k/run_0/model_best_80p7958_backup.pth
  DATASETS.TEST "('refcoco_val_unc',)"
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_ON True
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_CONSERVATIVE_ON True
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_DELTA_SCALE 0.24
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TOPGAP_THR 0.15
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_EVIDENCE_DROP_THR 0.06
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_MIN_DELTA_GAP 0.0
  MODEL.OpenWorldSAM2.VIS_EXPORT_ON False
)

run_eval () {
  TAG="$1"
  shift

  TS=$(date +%Y%m%d_%H%M%S)
  OUT_DIR="output/${TAG}_${TS}/run_0"
  LOG_FILE="outputs/${TAG}_${TS}.log"

  echo "" | tee -a "$SUMMARY"
  echo "==================================================" | tee -a "$SUMMARY"
  echo "[RUN] $TAG" | tee -a "$SUMMARY"
  echo "OUT_DIR=$OUT_DIR" | tee -a "$SUMMARY"
  echo "LOG_FILE=$LOG_FILE" | tee -a "$SUMMARY"
  echo "==================================================" | tee -a "$SUMMARY"

  CUDA_VISIBLE_DEVICES=0 python train_net.py \
    "${BASE_ARGS[@]}" \
    "$@" \
    OUTPUT_DIR "$OUT_DIR" \
    2>&1 | tee "$LOG_FILE"

  echo "" | tee -a "$SUMMARY"
  echo "[DONE] $TAG" | tee -a "$SUMMARY"
  grep -n "FUSION-EVIDENCE-ONLY\|copypaste\|cIoU\|mIoU" "$LOG_FILE" | tail -50 | tee -a "$SUMMARY" || true
}

# 1) 原 81 主线复现：不显式打开 FUSION.ON，不开 expansion
run_eval "eval_replay_81_noexpand" \
  MODEL.OpenWorldSAM2.FUSION.EVIDENCE_RERANK True \
  MODEL.OpenWorldSAM2.FUSION.EVIDENCE_ALPHA 5.3 \
  MODEL.OpenWorldSAM2.FUSION.WEAK_MAP_FALLBACK_THR 0.001 \
  MODEL.OpenWorldSAM2.FUSION.ADAPTIVE_ALPHA_ON True \
  MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_ON False \
  MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_GENERATE_ON False

# 2) Evidence-only no expansion + rerank ON：检查 bypass 是否恢复主线
run_eval "eval_evidence_only_noexpand_rerankon" \
  MODEL.OpenWorldSAM2.FUSION.ON True \
  MODEL.OpenWorldSAM2.FUSION.EVIDENCE_ONLY_ON True \
  MODEL.OpenWorldSAM2.FUSION.SLOT_ON True \
  MODEL.OpenWorldSAM2.FUSION.EVIDENCE_USE_SLOTS True \
  MODEL.OpenWorldSAM2.FUSION.EVIDENCE_RERANK True \
  MODEL.OpenWorldSAM2.FUSION.EVIDENCE_ALPHA 5.3 \
  MODEL.OpenWorldSAM2.FUSION.WEAK_MAP_FALLBACK_THR 0.001 \
  MODEL.OpenWorldSAM2.FUSION.ADAPTIVE_ALPHA_ON True \
  MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_ON False \
  MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_GENERATE_ON False

# 3) Evidence-only + expansion + bonus sweep，mode=calib
for B in 0.00 0.005 0.01 0.02 0.03 0.05; do
  TAG_B=$(echo "$B" | sed 's/\./p/g')
  TS_IN=$(date +%Y%m%d_%H%M%S)
  DEBUG_JSONL="outputs/evidence_only_expand_bonus_calib_${TAG_B}_${TS_IN}_debug.jsonl"
  ORACLE_JSONL="outputs/evidence_only_expand_bonus_calib_${TAG_B}_${TS_IN}_oracle.jsonl"

  run_eval "eval_evidence_only_expand_bonus_calib_${TAG_B}" \
    MODEL.OpenWorldSAM2.FUSION.ON True \
    MODEL.OpenWorldSAM2.FUSION.EVIDENCE_ONLY_ON True \
    MODEL.OpenWorldSAM2.FUSION.SLOT_ON True \
    MODEL.OpenWorldSAM2.FUSION.EVIDENCE_USE_SLOTS True \
    MODEL.OpenWorldSAM2.FUSION.EVIDENCE_RERANK True \
    MODEL.OpenWorldSAM2.FUSION.EVIDENCE_ALPHA 5.3 \
    MODEL.OpenWorldSAM2.FUSION.WEAK_MAP_FALLBACK_THR 0.001 \
    MODEL.OpenWorldSAM2.FUSION.ADAPTIVE_ALPHA_ON True \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_ON True \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_GENERATE_ON True \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_TOPK 3 \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_MAX_EXTRA 3 \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_POINT_BATCH_SIZE 4 \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_EMPTY_CACHE_ON_OOM True \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_MIN_SCORE 0.15 \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_NMS_RADIUS 12 \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_DEDUP_IOU_THR 0.90 \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_SCORE_BONUS_ON True \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_SCORE_BONUS "$B" \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_SCORE_BONUS_MODE calib \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_EXPORT_ON True \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_EXPORT_PATH "$DEBUG_JSONL" \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_ORACLE_EXPORT_ON True \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_ORACLE_EXPORT_PATH "$ORACLE_JSONL"
done

# 4) 额外只测两个 mode，对比 base/both 是否比 calib 好
for MODE in base both; do
  B=0.03
  TAG_B=$(echo "$B" | sed 's/\./p/g')
  TS_IN=$(date +%Y%m%d_%H%M%S)
  DEBUG_JSONL="outputs/evidence_only_expand_bonus_${MODE}_${TAG_B}_${TS_IN}_debug.jsonl"
  ORACLE_JSONL="outputs/evidence_only_expand_bonus_${MODE}_${TAG_B}_${TS_IN}_oracle.jsonl"

  run_eval "eval_evidence_only_expand_bonus_${MODE}_${TAG_B}" \
    MODEL.OpenWorldSAM2.FUSION.ON True \
    MODEL.OpenWorldSAM2.FUSION.EVIDENCE_ONLY_ON True \
    MODEL.OpenWorldSAM2.FUSION.SLOT_ON True \
    MODEL.OpenWorldSAM2.FUSION.EVIDENCE_USE_SLOTS True \
    MODEL.OpenWorldSAM2.FUSION.EVIDENCE_RERANK True \
    MODEL.OpenWorldSAM2.FUSION.EVIDENCE_ALPHA 5.3 \
    MODEL.OpenWorldSAM2.FUSION.WEAK_MAP_FALLBACK_THR 0.001 \
    MODEL.OpenWorldSAM2.FUSION.ADAPTIVE_ALPHA_ON True \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_ON True \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_GENERATE_ON True \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_TOPK 3 \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_MAX_EXTRA 3 \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_POINT_BATCH_SIZE 4 \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_EMPTY_CACHE_ON_OOM True \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_MIN_SCORE 0.15 \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_NMS_RADIUS 12 \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_DEDUP_IOU_THR 0.90 \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_SCORE_BONUS_ON True \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_SCORE_BONUS "$B" \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_SCORE_BONUS_MODE "$MODE" \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_EXPORT_ON True \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_EXPORT_PATH "$DEBUG_JSONL" \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_ORACLE_EXPORT_ON True \
    MODEL.OpenWorldSAM2.INSTANCE_CAND_EXPAND_ORACLE_EXPORT_PATH "$ORACLE_JSONL"
done

echo "" | tee -a "$SUMMARY"
echo "==================================================" | tee -a "$SUMMARY"
echo "[ALL DONE] ROOT_TS=$ROOT_TS" | tee -a "$SUMMARY"
echo "SUMMARY=$SUMMARY" | tee -a "$SUMMARY"
echo "==================================================" | tee -a "$SUMMARY"

# 自动汇总 oracle 和 log 指标
python - <<'PY' | tee -a "$SUMMARY"
import json, glob, re, os

log_paths = sorted(glob.glob("outputs/eval_*evidence_only*.log") + glob.glob("outputs/eval_replay_81_noexpand*.log"))
oracle_paths = sorted(glob.glob("outputs/evidence_only_expand_*_oracle.jsonl"))

def parse_metric(log):
    ciou = miou = None
    with open(log, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "copypaste:" in line and "precision@0.5" in line:
            if i + 1 < len(lines):
                vals = lines[i + 1].strip().split("copypaste:")[-1].strip().split(",")
                if len(vals) >= 7:
                    ciou = float(vals[5])
                    miou = float(vals[6])
    return ciou, miou

print("\n===== METRIC SUMMARY =====")
for log in log_paths:
    ciou, miou = parse_metric(log)
    if ciou is not None:
        print({"log": log, "cIoU": ciou, "mIoU": miou})

print("\n===== ORACLE SUMMARY =====")
for p in oracle_paths:
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    extra = [r.get("extra_candidate_count", 0) or 0 for r in rows]
    sel = [r for r in rows if r.get("extra_selected_by_final")]
    gains = [r.get("oracle_gain") for r in rows if r.get("oracle_gain") is not None]
    print({
        "oracle": p,
        "rows": len(rows),
        "extra>0": f"{sum(x > 0 for x in extra)}/{len(rows)}",
        "extra_selected": f"{len(sel)}/{len(rows)}",
        "oracle_gain_mean": round(sum(gains)/len(gains), 6) if gains else None,
        "oracle_gain>0.05": f"{sum(x > 0.05 for x in gains)}/{len(gains)}" if gains else None,
        "oracle_gain>0.10": f"{sum(x > 0.10 for x in gains)}/{len(gains)}" if gains else None,
        "oracle_gain_max": round(max(gains), 6) if gains else None,
    })
PY

