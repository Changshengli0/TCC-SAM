#!/usr/bin/env bash
set -euo pipefail

cd /home/zxing/code/SAM_RIS_1/OpenWorldSAM-main

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

CFG="configs/refcoco/Open-World-SAM2-CrossAttention-refcoco.yaml"
EVF="/home/zxing/code/SAM_RIS_1/checkpoints/evf-sam2-multitask"
SAM2="/home/zxing/code/SAM_RIS_1/sam2_hiera_large.pt"

PREV_ROOT="output/official_init_ablation_refcoco_10k_20260521_132429"
BASE_FT="$PREV_ROOT/01_base_ft_no_opm_no_evidence_no_csh/run_0/model_best.pth"

if [ ! -f "$BASE_FT" ]; then
  echo "ERROR: Base-FT checkpoint not found:"
  echo "$BASE_FT"
  exit 1
fi

RUN_ROOT="output/stage_ablation_from_baseft_refcoco_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"

echo "BASE_FT=$BASE_FT"
echo "RUN_ROOT=$RUN_ROOT"

COMMON_TRAIN=(
  --num-gpus 1
  -b 1
  --lr 2.5e-6
  --config-file "$CFG"
  DATASETS.TRAIN "('refcoco_train_unc',)"
  DATASETS.TEST "('refcoco_val_unc',)"
  SOLVER.MAX_ITER 4000
  TEST.EVAL_PERIOD 1000
  MODEL.OpenWorldSAM2.EVF_CONFIG "$EVF"
  MODEL.OpenWorldSAM2.TOKENIZER_CONFIG "$EVF"
  MODEL.OpenWorldSAM2.VISION_PRETRAINED "$SAM2"
  MODEL.OpenWorldSAM2.CAND_RANK_LOSS_ON False
  MODEL.OpenWorldSAM2.VIS_EXPORT_ON False
)

OPM_ON=(
  MODEL.OpenWorldSAM2.STA_ON True
  MODEL.OpenWorldSAM2.STA_TRAIN_ON True
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

EVIDENCE_LOW=(
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
  MODEL.OpenWorldSAM2.FUSION.EVIDENCE_ALPHA 0.5
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

echo ""
echo "============================================================"
echo "04: OPM only from Base-FT"
echo "============================================================"

python train_net.py \
  "${COMMON_TRAIN[@]}" \
  MODEL.WEIGHTS "$BASE_FT" \
  "${OPM_ON[@]}" \
  "${EVIDENCE_OFF[@]}" \
  "${CSH_OFF[@]}" \
  OUTPUT_DIR "$RUN_ROOT/04_opm_only_from_baseft"

OPM_CKPT="$RUN_ROOT/04_opm_only_from_baseft/run_0/model_best.pth"

if [ ! -f "$OPM_CKPT" ]; then
  echo "ERROR: OPM checkpoint not found:"
  echo "$OPM_CKPT"
  exit 1
fi

echo ""
echo "============================================================"
echo "05: OPM + Evidence low-alpha from OPM checkpoint"
echo "============================================================"

python train_net.py \
  "${COMMON_TRAIN[@]}" \
  MODEL.WEIGHTS "$OPM_CKPT" \
  "${OPM_ON[@]}" \
  "${EVIDENCE_LOW[@]}" \
  "${CSH_OFF[@]}" \
  OUTPUT_DIR "$RUN_ROOT/05_opm_evidence_alpha0p5_from_opm"

echo ""
echo "============================================================"
echo "Collecting results"
echo "============================================================"

SUMMARY="$RUN_ROOT/summary_copypaste.txt"
: > "$SUMMARY"

for d in "$RUN_ROOT"/*; do
  [ -d "$d" ] || continue
  LOG="$d/run_0/log.txt"
  echo "===== $(basename "$d") =====" | tee -a "$SUMMARY"
  grep -n "copypaste: [0-9]" "$LOG" | tail -1 | tee -a "$SUMMARY" || echo "NO COPYPASTE FOUND" | tee -a "$SUMMARY"
  echo "OOM skip count:" | tee -a "$SUMMARY"
  grep -c "OOM skip" "$LOG" | tee -a "$SUMMARY" || true
  echo "" | tee -a "$SUMMARY"
done

cat "$SUMMARY"
