#!/usr/bin/env bash
set -e

cd /home/zxing/code/SAM_RIS_1/OpenWorldSAM-main

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT=/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/output/cand_rank_softce_w005_500eval_B/run_0/model_best_80p7402_alpha53_backup.pth

COMMON_ARGS="\
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
  MODEL.OpenWorldSAM2.STA_INJECT_BLOCKS [20] \
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
  MODEL.OpenWorldSAM2.CAND_RANK_LOSS_ON False"

echo "==================== [0] py_compile ===================="
python -m py_compile train_net.py model/open_world_sam2.py model/config.py

echo ""
echo "==================== [1] eval-only: untrained conservative equivalence ===================="
rm -rf output/eval_cand_score_cons_untrained_equiv

CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --eval-only \
  ${COMMON_ARGS} \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_ON True \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ON False \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ONLY False \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_CONSERVATIVE_ON True \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_DELTA_SCALE 0.10 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TOPGAP_THR 0.05 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_EVIDENCE_DROP_THR 0.03 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_MIN_DELTA_GAP 0.00 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_DEBUG True \
  OUTPUT_DIR ./output/eval_cand_score_cons_untrained_equiv

echo ""
echo "==================== [2] smoke120: conservative + area-aware train-only ===================="
rm -rf output/cand_score_head_cons_area_smoke120

CUDA_VISIBLE_DEVICES=0 python train_net.py \
  ${COMMON_ARGS} \
  SOLVER.MAX_ITER 120 \
  TEST.EVAL_PERIOD 0 \
  SOLVER.IMS_PER_BATCH 1 \
  SOLVER.BASE_LR 1e-4 \
  SOLVER.LORA_LR 1e-4 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_ON True \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ON True \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ONLY True \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_CONSERVATIVE_ON True \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_AREA_AWARE_LOSS True \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_AREA_REF 50000.0 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_AREA_WEIGHT_POWER 0.5 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_AREA_MIN_WEIGHT 1.0 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_MAX_AREA_WEIGHT 3.0 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_WEIGHT 0.01 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_DELTA_SCALE 0.10 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_SOFT_TAU 0.05 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_SCORE_TAU 0.10 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_HIDDEN_DIM 256 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TOPGAP_THR 0.05 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_EVIDENCE_DROP_THR 0.03 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_MIN_DELTA_GAP 0.00 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_DEBUG True \
  OUTPUT_DIR ./output/cand_score_head_cons_area_smoke120

echo ""
echo "==================== [3] summary ===================="

echo ""
echo "----- eval conservative untrained equivalence -----"
LOG=output/eval_cand_score_cons_untrained_equiv/run_0/log.txt
grep -n "copypaste: precision@0.5" "$LOG" | tail -3 || true
grep -n "copypaste: [0-9]" "$LOG" | tail -3 || true
echo ""
echo "[CAND-SCORE / CONS debug]"
grep -n "\[CAND-SCORE-CONS\]\|\[CAND-SCORE\]" "$LOG" | head -80 || true
echo ""
echo "[errors]"
grep -niE "nan|traceback|cuda error|runtimeerror|ValueError" "$LOG" | tail -80 || true

echo ""
echo "----- smoke120 conservative + area-aware -----"
LOG=output/cand_score_head_cons_area_smoke120/run_0/log.txt
echo ""
echo "[CAND-SCORE debug]"
grep -n "\[CAND-SCORE\]" "$LOG" | head -120 || true
echo ""
echo "[area-aware / loss]"
grep -n "cand_score_area_weight_mean\|cand_score_large_obj_loss\|cand_score_small_obj_loss\|loss_cand_score_head\|cand_score_head_loss" "$LOG" | tail -120 || true
echo ""
echo "[optimizer / trainable]"
grep -n "optimizer includes cand_score_head\|train_only=True\|allow empty LoRA\|cand_score_head" "$LOG" | head -160 || true
echo ""
echo "[errors]"
grep -niE "nan|traceback|cuda error|runtimeerror|ValueError" "$LOG" | tail -80 || true

echo ""
echo "==================== DONE ===================="
