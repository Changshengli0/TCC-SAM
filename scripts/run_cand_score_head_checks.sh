#!/usr/bin/env bash
set -e

cd /home/zxing/code/SAM_RIS_1/OpenWorldSAM-main

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

echo "===== [0] py_compile ====="
python -m py_compile model/open_world_sam2.py model/config.py

echo ""
echo "===== [1] Eval baseline: CAND_SCORE_HEAD_ON=False ====="
rm -rf output/eval_cand_score_off_baseline807402

CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --eval-only \
  ${COMMON_ARGS} \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_ON False \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ON False \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ONLY False \
  OUTPUT_DIR ./output/eval_cand_score_off_baseline807402

echo ""
echo "===== [2] Eval zero-head equivalence: CAND_SCORE_HEAD_ON=True but untrained ====="
rm -rf output/eval_cand_score_on_untrained_equiv

CUDA_VISIBLE_DEVICES=0 python train_net.py \
  --eval-only \
  ${COMMON_ARGS} \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_ON True \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ON False \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ONLY False \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_DELTA_SCALE 0.10 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_DEBUG True \
  OUTPUT_DIR ./output/eval_cand_score_on_untrained_equiv

echo ""
echo "===== Eval equivalence summary ====="
for D in eval_cand_score_off_baseline807402 eval_cand_score_on_untrained_equiv; do
  LOG=output/${D}/run_0/log.txt
  echo "----- ${D} -----"
  grep -n "copypaste: [0-9]" "$LOG" | tail -1 || true
  grep -n "\[CAND-SCORE\]" "$LOG" | head -20 || true
  grep -niE "nan|traceback|cuda error|runtimeerror|(^|[^a-zA-Z])inf([^a-zA-Z]|$)" "$LOG" | tail -20 || true
done

echo ""
echo "===== [3] Train-only smoke120: only cand_score_head trainable ====="
rm -rf output/cand_score_head_trainonly_smoke120

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
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_WEIGHT 0.01 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_DELTA_SCALE 0.10 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_SOFT_TAU 0.05 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_SCORE_TAU 0.10 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_HIDDEN_DIM 256 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_DEBUG True \
  OUTPUT_DIR ./output/cand_score_head_trainonly_smoke120

SMOKE_LOG=output/cand_score_head_trainonly_smoke120/run_0/log.txt

echo ""
echo "===== Smoke120 check ====="
echo "----- CAND-SCORE debug -----"
grep -n "\[CAND-SCORE\]" "$SMOKE_LOG" | head -120 || true

echo ""
echo "----- trainable / optimizer check -----"
grep -n "CAND-SCORE\|cand_score_head\|Trainable parameters\|Parameter Name\|optimizer\|OPTIM" "$SMOKE_LOG" | head -200 || true

echo ""
echo "----- loss_cand_score_head -----"
grep -n "loss_cand_score_head\|cand_score_head_loss" "$SMOKE_LOG" | tail -80 || true

echo ""
echo "----- real errors only -----"
grep -niE "nan|traceback|cuda error|runtimeerror|(^|[^a-zA-Z])inf([^a-zA-Z]|$)" "$SMOKE_LOG" | tail -80 || true

echo ""
echo "===== [4] Train-only 500 iter + eval ====="
rm -rf output/cand_score_head_trainonly_w001_lr1e4_500eval

CUDA_VISIBLE_DEVICES=0 python train_net.py \
  ${COMMON_ARGS} \
  SOLVER.MAX_ITER 500 \
  TEST.EVAL_PERIOD 500 \
  SOLVER.CHECKPOINT_PERIOD 500 \
  SOLVER.IMS_PER_BATCH 1 \
  SOLVER.BASE_LR 1e-4 \
  SOLVER.LORA_LR 1e-4 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_ON True \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ON True \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ONLY True \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_WEIGHT 0.01 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_DELTA_SCALE 0.10 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_SOFT_TAU 0.05 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_SCORE_TAU 0.10 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_HIDDEN_DIM 256 \
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_DEBUG True \
  OUTPUT_DIR ./output/cand_score_head_trainonly_w001_lr1e4_500eval

LOG=output/cand_score_head_trainonly_w001_lr1e4_500eval/run_0/log.txt

echo ""
echo "===== 500eval summary ====="
grep -n "copypaste: precision@0.5" "$LOG" | tail -5 || true
grep -n "copypaste: [0-9]" "$LOG" | tail -5 || true
grep -n "BestCheck" "$LOG" | tail -10 || true

echo ""
echo "===== cand score loss ====="
grep -n "loss_cand_score_head\|cand_score_head_loss" "$LOG" | tail -80 || true

echo ""
echo "===== CAND-SCORE debug ====="
grep -n "\[CAND-SCORE\]" "$LOG" | head -80 || true

echo ""
echo "===== real errors only ====="
grep -niE "nan|traceback|cuda error|runtimeerror|(^|[^a-zA-Z])inf([^a-zA-Z]|$)" "$LOG" | tail -80 || true

echo ""
echo "===== DONE ====="
