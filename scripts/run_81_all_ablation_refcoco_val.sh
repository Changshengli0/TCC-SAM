#!/usr/bin/env bash
set -euo pipefail

cd /home/zxing/code/SAM_RIS_1/OpenWorldSAM-main

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

CFG="configs/refcoco/Open-World-SAM2-CrossAttention-refcoco.yaml"
DATASET="refcoco_val_unc"

WEIGHT="/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/output/cand_score_head_gapw_from807787_lr2p5e5_4k/run_0/model_best_80p7958_backup.pth"
EVF="/home/zxing/code/SAM_RIS_1/checkpoints/evf-sam2-multitask"
SAM2="/home/zxing/code/SAM_RIS_1/sam2_hiera_large.pt"

RUN_ROOT="output/abl_81_refcoco_val_all_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"

COMMON=(
  --eval-only
  --num-gpus 1
  --config-file "$CFG"
  DATASETS.TEST "('$DATASET',)"
  MODEL.WEIGHTS "$WEIGHT"
  MODEL.OpenWorldSAM2.EVF_CONFIG "$EVF"
  MODEL.OpenWorldSAM2.TOKENIZER_CONFIG "$EVF"
  MODEL.OpenWorldSAM2.VISION_PRETRAINED "$SAM2"
  MODEL.OpenWorldSAM2.CAND_RANK_LOSS_ON False
  MODEL.OpenWorldSAM2.VIS_EXPORT_ON False
)

OPM_OFF=(
  MODEL.OpenWorldSAM2.STA_ON False
  MODEL.OpenWorldSAM2.STA_TRAIN_ON False
)

OPM_ON=(
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
)

EVIDENCE_OFF=(
  MODEL.OpenWorldSAM2.FUSION.ON False
  MODEL.OpenWorldSAM2.FUSION.EVIDENCE_RERANK False
)

EVIDENCE_ON=(
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
)

CSH_OFF=(
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_ON False
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ON False
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ONLY False
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_CONSERVATIVE_ON False
)

CSH_ON_NO_GATE=(
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_ON True
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ON False
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ONLY False
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_CONSERVATIVE_ON False
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_DELTA_SCALE 0.24
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TOPGAP_THR 0.15
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_EVIDENCE_DROP_THR 0.06
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_MIN_DELTA_GAP 0.00
)

CSH_ON_GATE=(
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_ON True
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ON False
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ONLY False
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_CONSERVATIVE_ON True
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_DELTA_SCALE 0.24
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TOPGAP_THR 0.15
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_EVIDENCE_DROP_THR 0.06
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_MIN_DELTA_GAP 0.00
)

run_eval () {
  NAME="$1"
  shift
  OUT="$RUN_ROOT/$NAME"

  echo ""
  echo "============================================================"
  echo "Running $NAME"
  echo "Output: $OUT"
  echo "============================================================"
  echo ""

  python train_net.py \
    "${COMMON[@]}" \
    "$@" \
    OUTPUT_DIR "$OUT"

  echo ""
  echo "Finished $NAME"
  echo ""
}

# A. Base：关闭 OPM、Evidence、Candidate Score Head、Conservative Gate
run_eval "A_base" \
  "${OPM_OFF[@]}" \
  "${EVIDENCE_OFF[@]}" \
  "${CSH_OFF[@]}"

# B. + OPM：只开启 OPM，关闭 Evidence、Candidate Score Head、Gate
run_eval "B_opm" \
  "${OPM_ON[@]}" \
  "${EVIDENCE_OFF[@]}" \
  "${CSH_OFF[@]}"

# C. + OPM + Evidence：开启 OPM 和 Evidence，关闭 Candidate Score Head、Gate
run_eval "C_opm_evidence" \
  "${OPM_ON[@]}" \
  "${EVIDENCE_ON[@]}" \
  "${CSH_OFF[@]}"

# D. + OPM + Evidence + Candidate Score Head，关闭 Conservative Gate
run_eval "D_opm_evidence_csh_no_gate" \
  "${OPM_ON[@]}" \
  "${EVIDENCE_ON[@]}" \
  "${CSH_ON_NO_GATE[@]}"

# E. Full：完整模型，复现 81.0133
run_eval "E_full" \
  "${OPM_ON[@]}" \
  "${EVIDENCE_ON[@]}" \
  "${CSH_ON_GATE[@]}"

# F. Full w/o evidence safety：去掉 evidence safety，EVIDENCE_DROP_THR 设成极大
run_eval "F_full_no_evidence_safety" \
  "${OPM_ON[@]}" \
  "${EVIDENCE_ON[@]}" \
  "${CSH_ON_GATE[@]}" \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_EVIDENCE_DROP_THR 999.0

# G. Full w/o gap safety：去掉 base uncertainty 限制，TOPGAP_THR 设成极大
run_eval "G_full_no_gap_safety" \
  "${OPM_ON[@]}" \
  "${EVIDENCE_ON[@]}" \
  "${CSH_ON_GATE[@]}" \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TOPGAP_THR 999.0

echo ""
echo "============================================================"
echo "All ablations finished."
echo "RUN_ROOT=$RUN_ROOT"
echo "============================================================"
echo ""

SUMMARY="$RUN_ROOT/summary_copypaste.txt"
: > "$SUMMARY"

for d in "$RUN_ROOT"/*; do
  [ -d "$d" ] || continue

  if [ -f "$d/run_0/log.txt" ]; then
    LOG="$d/run_0/log.txt"
  elif [ -f "$d/log.txt" ]; then
    LOG="$d/log.txt"
  else
    echo "===== $(basename "$d") =====" | tee -a "$SUMMARY"
    echo "NO LOG FOUND" | tee -a "$SUMMARY"
    continue
  fi

  echo "===== $(basename "$d") =====" | tee -a "$SUMMARY"
  grep -n "copypaste: [0-9]" "$LOG" | tail -1 | tee -a "$SUMMARY" || echo "NO COPYPASTE FOUND" | tee -a "$SUMMARY"
done

echo ""
echo "Saved summary to:"
echo "$SUMMARY"
echo ""
cat "$SUMMARY"
