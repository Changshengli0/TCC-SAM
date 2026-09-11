#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main"
cd "${REPO_DIR}"

CONFIG_FILE="/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/configs/refcoco/Open-World-SAM2-CrossAttention-refcoco.yaml"
CHECKPOINT="/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/output/ft_refcocog_betaOnly_lr1e-4_3k_local/run_0/model_best.pth"
OUTPUT_ROOT="${REPO_DIR}/output/eval_evidence_component_sweep_refcocog"
DATASET_TEST="('refcocog_val_umd',)"
EVF_LOCAL="/home/zxing/code/SAM_RIS_1/checkpoints/evf-sam2-multitask"
VISION_LOCAL="/home/zxing/code/SAM_RIS_1/sam2_hiera_large.pt"

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/zxing/code/SAM_RIS_1:/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_ROOT}"

extract_metrics() {
  local log_file="$1"
  python - "$log_file" <<'PY'
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
lines = text.splitlines()
idxs = [i for i, ln in enumerate(lines) if "copypaste:" in ln]

header_line = None
value_line = None
for i in range(len(idxs) - 1):
    h = lines[idxs[i]]
    v = lines[idxs[i + 1]]
    h_cols = [x.strip() for x in h.split("copypaste:", 1)[1].split(",")] if "copypaste:" in h else []
    if "cIoU" in h_cols and "mIoU" in h_cols and "copypaste:" in v:
        header_line = h
        value_line = v

if header_line is None or value_line is None:
    print("NA,NA,NA,NA,NA")
    sys.exit(0)

headers = [x.strip() for x in header_line.split("copypaste:", 1)[1].split(",")]
vals = [x.strip() for x in value_line.split("copypaste:", 1)[1].split(",")]
metric_map = {k: v for k, v in zip(headers, vals)}
print(",".join([
    metric_map.get("cIoU", "NA"),
    metric_map.get("mIoU", "NA"),
    metric_map.get("precision@0.5", "NA"),
    metric_map.get("precision@0.7", "NA"),
    metric_map.get("precision@0.9", "NA"),
]))
PY
}

run_mode() {
  local mode="$1"
  local use_cat="$2"
  local use_attr="$3"
  local log_file="${OUTPUT_ROOT}/${mode}.log"
  local run_dir="${OUTPUT_ROOT}/${mode}"
  mkdir -p "${run_dir}"

  python "${REPO_DIR}/train_net.py" \
    --eval-only \
    --config-file "${CONFIG_FILE}" \
    OUTPUT_DIR "${run_dir}" \
    MODEL.WEIGHTS "${CHECKPOINT}" \
    DATASETS.TEST "${DATASET_TEST}" \
    MODEL.OpenWorldSAM2.FUSION.ON True \
    MODEL.OpenWorldSAM2.FUSION.USE_REL False \
    MODEL.OpenWorldSAM2.FUSION.EVIDENCE_RERANK True \
    MODEL.OpenWorldSAM2.FUSION.EVIDENCE_ALPHA 5.0 \
    MODEL.OpenWorldSAM2.FUSION.WEAK_MAP_FALLBACK_THR 1e-3 \
    MODEL.OpenWorldSAM2.FUSION.EVIDENCE_USE_CAT "${use_cat}" \
    MODEL.OpenWorldSAM2.FUSION.EVIDENCE_USE_ATTR "${use_attr}" \
    MODEL.OpenWorldSAM2.EVF_CONFIG "${EVF_LOCAL}" \
    MODEL.OpenWorldSAM2.TOKENIZER_CONFIG "${EVF_LOCAL}" \
    MODEL.OpenWorldSAM2.VISION_PRETRAINED "${VISION_LOCAL}" \
    2>&1 | tee "${log_file}"

  IFS=',' read -r ciou miou p05 p07 p09 <<< "$(extract_metrics "${log_file}")"
  printf "%-10s %-10s %-10s %-10s %-10s %-10s\n" "${mode}" "${ciou}" "${miou}" "${p05}" "${p07}" "${p09}"
}

printf "%-10s %-10s %-10s %-10s %-10s %-10s\n" "mode" "cIoU" "mIoU" "p@0.5" "p@0.7" "p@0.9"
printf "%-10s %-10s %-10s %-10s %-10s %-10s\n" "----------" "----------" "----------" "----------" "----------" "----------"
run_mode "cat_only" "True" "False"
run_mode "attr_only" "False" "True"
run_mode "cat_attr" "True" "True"
