#!/usr/bin/env bash
set -euo pipefail

cd /home/zxing/code/SAM_RIS_1/OpenWorldSAM-main

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

CFG="configs/refcoco/Open-World-SAM2-CrossAttention-refcoco.yaml"
OFFICIAL_WEIGHT="/home/zxing/Downloads/model_final_refcocog.pth"
EVF="/home/zxing/code/SAM_RIS_1/checkpoints/evf-sam2-multitask"
SAM2="/home/zxing/code/SAM_RIS_1/sam2_hiera_large.pt"

TRAIN_SET="refcoco_train_unc"
TEST_SET="refcoco_val_unc"

RUN_ROOT="output/official_init_ablation_refcoco_10k_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"

echo "============================================================"
echo "RUN_ROOT=$RUN_ROOT"
echo "OFFICIAL_WEIGHT=$OFFICIAL_WEIGHT"
echo "============================================================"

# 1. Audit 官方权重是否含有我们后期模块 key
python - <<PY | tee "$RUN_ROOT/audit_official_weight.txt"
import torch
from pathlib import Path

pth = Path("$OFFICIAL_WEIGHT")
patterns = [
    "sta", "opm",
    "cand_score", "candidate_score",
    "conservative",
    "evidence_head", "residual_head",
    "rerank",
]

print("checkpoint:", pth)
ckpt = torch.load(str(pth), map_location="cpu")
sd = ckpt.get("model", ckpt)
keys = list(sd.keys())

print("num_keys:", len(keys))
bad = [k for k in keys if any(p in k.lower() for p in patterns)]
print("num_suspicious_keys:", len(bad))

groups = {
    "sta/opm": [k for k in keys if ("sta" in k.lower() or "opm" in k.lower())],
    "candidate_score": [k for k in keys if ("cand_score" in k.lower() or "candidate_score" in k.lower())],
    "conservative": [k for k in keys if "conservative" in k.lower()],
    "evidence_or_residual": [k for k in keys if ("evidence_head" in k.lower() or "residual_head" in k.lower() or "rerank" in k.lower())],
}

for name, arr in groups.items():
    print("\\n[%s] count=%d" % (name, len(arr)))
    for k in arr[:80]:
        print("  ", k)
PY

COMMON_EVAL=(
  --eval-only
  --num-gpus 1
  --config-file "$CFG"
  DATASETS.TEST "('$TEST_SET',)"
  MODEL.WEIGHTS "$OFFICIAL_WEIGHT"
  MODEL.OpenWorldSAM2.EVF_CONFIG "$EVF"
  MODEL.OpenWorldSAM2.TOKENIZER_CONFIG "$EVF"
  MODEL.OpenWorldSAM2.VISION_PRETRAINED "$SAM2"
  MODEL.OpenWorldSAM2.CAND_RANK_LOSS_ON False
  MODEL.OpenWorldSAM2.VIS_EXPORT_ON False
)

COMMON_TRAIN=(
  --num-gpus 1
  -b 1
  --lr 5e-6
  --config-file "$CFG"
  DATASETS.TRAIN "('$TRAIN_SET',)"
  DATASETS.TEST "('$TEST_SET',)"
  SOLVER.MAX_ITER 10000
  TEST.EVAL_PERIOD 2000
  MODEL.WEIGHTS "$OFFICIAL_WEIGHT"
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

CSH_ON_GATE=(
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_ON True
  MODEL.OpenWorldSAM2.CAND_SCORE_HEAD_TRAIN_ON True
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
  echo "Eval $NAME"
  echo "Output: $OUT"
  echo "============================================================"

  python train_net.py \
    "${COMMON_EVAL[@]}" \
    "$@" \
    OUTPUT_DIR "$OUT"
}

run_train () {
  NAME="$1"
  shift
  OUT="$RUN_ROOT/$NAME"

  echo ""
  echo "============================================================"
  echo "Train $NAME"
  echo "Output: $OUT"
  echo "============================================================"

  python train_net.py \
    "${COMMON_TRAIN[@]}" \
    "$@" \
    OUTPUT_DIR "$OUT"
}

# 2. 官方 OpenWorldSAM baseline：不训练，直接 eval
run_eval "00_official_openworldsam_eval" \
  "${OPM_OFF[@]}" \
  "${EVIDENCE_OFF[@]}" \
  "${CSH_OFF[@]}"

# 3. Base-FT：从官方权重继续训练，但关闭我们的新模块
run_train "01_base_ft_no_opm_no_evidence_no_csh" \
  "${OPM_OFF[@]}" \
  "${EVIDENCE_OFF[@]}" \
  "${CSH_OFF[@]}"

# 4. + OPM + Evidence：验证 evidence map + mask-evidence pooling
run_train "02_opm_evidence" \
  "${OPM_ON[@]}" \
  "${EVIDENCE_ON[@]}" \
  "${CSH_OFF[@]}"

# 5. Full：OPM + Evidence + Candidate Score Head + Conservative Gate
run_train "03_full_opm_evidence_csh_gate" \
  "${OPM_ON[@]}" \
  "${EVIDENCE_ON[@]}" \
  "${CSH_ON_GATE[@]}"

echo ""
echo "============================================================"
echo "All runs finished. Collecting results..."
echo "============================================================"

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

  echo "BestCheck lines:" | tee -a "$SUMMARY"
  grep -n "BestCheck\|best" "$LOG" | tail -5 | tee -a "$SUMMARY" || true
  echo "" | tee -a "$SUMMARY"
done

echo ""
echo "Saved summary to:"
echo "$SUMMARY"
echo ""
cat "$SUMMARY"
