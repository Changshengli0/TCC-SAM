#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

CONFIG_FILE="${1:-${REPO_DIR}/configs/refcoco/Open-World-SAM2-CrossAttention-refcoco.yaml}"
OUTPUT_ROOT="${2:-${REPO_DIR}/output/eval_alpha_sweep_refcocog}"
CHECKPOINT="/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/output/ft_refcocog_betaOnly_lr1e-4_3k_local/run_0/model_best.pth"
DATASET_TEST="('refcocog_val_umd',)"
EVF_LOCAL="/home/zxing/code/SAM_RIS_1/checkpoints/evf-sam2-multitask"
VISION_LOCAL="/home/zxing/code/SAM_RIS_1/sam2_hiera_large.pt"
ALPHAS=(4.0 4.5 5.0 5.5 6.0)

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/zxing/code/SAM_RIS_1:/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_ROOT}"

extract_metrics() {
  local log_file="$1"
  python - "$log_file" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
text = log_path.read_text(encoding="utf-8", errors="ignore")
lines = text.splitlines()
idxs = [i for i, ln in enumerate(lines) if "copypaste:" in ln]
if len(idxs) < 2:
    print("NA,NA,NA,NA,NA")
    sys.exit(0)

header_line = None
value_line = None
for i in range(len(idxs) - 1):
    h = lines[idxs[i]]
    v = lines[idxs[i + 1]]
    if "copypaste:" in h and "copypaste:" in v:
        h_cols = [x.strip() for x in h.split("copypaste:", 1)[1].split(",")]
        if "cIoU" in h_cols and "mIoU" in h_cols:
            header_line = h
            value_line = v

if header_line is None or value_line is None:
    print("NA,NA,NA,NA,NA")
    sys.exit(0)

headers = [x.strip() for x in header_line.split("copypaste:", 1)[1].split(",")]
vals = [x.strip() for x in value_line.split("copypaste:", 1)[1].split(",")]
metric_map = {}
for k, v in zip(headers, vals):
    metric_map[k] = v

out = [
    metric_map.get("cIoU", "NA"),
    metric_map.get("mIoU", "NA"),
    metric_map.get("precision@0.5", "NA"),
    metric_map.get("precision@0.7", "NA"),
    metric_map.get("precision@0.9", "NA"),
]
print(",".join(out))
PY
}

printf "%-6s %-10s %-10s %-10s %-10s %-10s\n" "alpha" "cIoU" "mIoU" "p@0.5" "p@0.7" "p@0.9"
printf "%-6s %-10s %-10s %-10s %-10s %-10s\n" "-----" "----------" "----------" "----------" "----------" "----------"

for alpha in "${ALPHAS[@]}"; do
  run_dir="${OUTPUT_ROOT}/alpha_${alpha}"
  log_file="${OUTPUT_ROOT}/alpha_${alpha}.log"
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
    MODEL.OpenWorldSAM2.FUSION.EVIDENCE_ALPHA "${alpha}" \
    MODEL.OpenWorldSAM2.EVF_CONFIG "${EVF_LOCAL}" \
    MODEL.OpenWorldSAM2.TOKENIZER_CONFIG "${EVF_LOCAL}" \
    MODEL.OpenWorldSAM2.VISION_PRETRAINED "${VISION_LOCAL}" \
    2>&1 | tee "${log_file}"

  IFS=',' read -r ciou miou p05 p07 p09 <<< "$(extract_metrics "${log_file}")"
  printf "%-6s %-10s %-10s %-10s %-10s %-10s\n" "${alpha}" "${ciou}" "${miou}" "${p05}" "${p07}" "${p09}"
done
