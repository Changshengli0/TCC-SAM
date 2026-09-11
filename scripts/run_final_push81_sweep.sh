#!/usr/bin/env bash
set -e

cd /home/zxing/code/SAM_RIS_1/OpenWorldSAM-main

export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT=/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/output/cand_score_head_gapw_from807787_lr2p5e5_4k/run_0/model_best_80p7958_backup.pth

run_eval () {
  NAME=$1
  DELTA=$2
  TOPGAP=$3
  EVDROP=$4

  OUT=./output/eval_807958_final81_${NAME}

  echo ""
  echo "================ ${NAME}: DELTA=${DELTA}, TOPGAP=${TOPGAP}, EVDROP=${EVDROP} ================"

  rm -rf "${OUT}"

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
    MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_DELTA_SCALE ${DELTA} \
    MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TOPGAP_THR ${TOPGAP} \
    MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_EVIDENCE_DROP_THR ${EVDROP} \
    MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_MIN_DELTA_GAP 0.00 \
    OUTPUT_DIR ${OUT}
}

# 围绕当前最优 delta=0.20 做细扫
run_eval d018_t012_e006 0.18 0.12 0.06
run_eval d020_t012_e006 0.20 0.12 0.06
run_eval d022_t012_e006 0.22 0.12 0.06
run_eval d024_t012_e006 0.24 0.12 0.06
run_eval d026_t012_e006 0.26 0.12 0.06

# 轻微放宽 topgap，看看是否能多救一点
run_eval d020_t015_e006 0.20 0.15 0.06
run_eval d022_t015_e006 0.22 0.15 0.06
run_eval d024_t015_e006 0.24 0.15 0.06

echo ""
echo "================ Final push-81 sweep summary ================"
for NAME in d018_t012_e006 d020_t012_e006 d022_t012_e006 d024_t012_e006 d026_t012_e006 d020_t015_e006 d022_t015_e006 d024_t015_e006; do
  LOG=output/eval_807958_final81_${NAME}/run_0/log.txt
  echo "----- ${NAME} -----"
  grep -n "copypaste: [0-9]" "$LOG" | tail -1
done

echo "DONE."
