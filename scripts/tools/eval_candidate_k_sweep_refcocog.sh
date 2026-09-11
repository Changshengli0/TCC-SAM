#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main"
cd "${REPO_DIR}"

CONFIG_FILE="/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/configs/refcoco/Open-World-SAM2-CrossAttention-refcoco.yaml"
CHECKPOINT="/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main/output/ft_refcocog_betaOnly_lr1e-4_3k_local/run_0/model_best.pth"
OUTPUT_ROOT="${REPO_DIR}/output/eval_candidate_k_sweep_refcocog"
DATASET_TEST="('refcocog_val_umd',)"
EVF_LOCAL="/home/zxing/code/SAM_RIS_1/checkpoints/evf-sam2-multitask"
VISION_LOCAL="/home/zxing/code/SAM_RIS_1/sam2_hiera_large.pt"
KS=("3" "5" "7")

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/zxing/code/SAM_RIS_1:/home/zxing/code/SAM_RIS_1/OpenWorldSAM-main:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_ROOT}"
SUMMARY_CSV="${OUTPUT_ROOT}/k_sweep_summary.csv"
echo "requested_k,requested_prompt_views,effective_candidate_k,cIoU,mIoU,p@0.5,p@0.7,p@0.9,oracle_best_of_k,top1_oracle_gap,pairwise_mask_iou_mean,candidate_entropy" > "${SUMMARY_CSV}"

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

extract_analysis_metrics() {
  local summary_json="$1"
  local export_jsonl="$2"
  python - "$summary_json" "$export_jsonl" <<'PY'
import json
import sys
from pathlib import Path

summary = Path(sys.argv[1])
export_jsonl = Path(sys.argv[2])
def fmt(v):
    return "NA" if v is None else str(v)

prompt_recs = []
eff_k = None
if export_jsonl.exists():
    with export_jsonl.open("r", encoding="utf-8") as fin:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("record_type") == "prompt":
                prompt_recs.append(rec)
                if eff_k is None and "K" in rec:
                    eff_k = rec.get("K")

def mean_key(rows, key):
    vals = []
    for r in rows:
        v = r.get(key)
        if isinstance(v, bool):
            vals.append(float(v))
        elif isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return None
    return sum(vals) / len(vals)

oracle = mean_key(prompt_recs, "oracle_best_of_k")
gap = mean_key(prompt_recs, "top1_oracle_gap")
pair_iou = mean_key(prompt_recs, "pairwise_mask_iou_mean")
ent = mean_key(prompt_recs, "candidate_entropy")

# fallback to analyze summary if prompt-level export is missing
if (oracle is None or gap is None or pair_iou is None or ent is None) and summary.exists():
    obj = json.loads(summary.read_text(encoding="utf-8"))
    overall = obj.get("overall", {})
    oracle = overall.get("oracle_best_of_k", oracle)
    gap = overall.get("top1_oracle_gap", gap)
    pair_iou = overall.get("pairwise_mask_iou_mean", pair_iou)
    ent = overall.get("candidate_entropy", ent)

req_views = mean_key(prompt_recs, "requested_prompt_views")
print(",".join([
    "NA" if req_views is None else str(int(round(req_views))),
    "NA" if eff_k is None else str(eff_k),
    fmt(oracle),
    fmt(gap),
    fmt(pair_iou),
    fmt(ent),
]))
PY
}

printf "%-11s %-9s %-12s %-10s %-10s %-10s %-10s %-10s %-14s %-14s %-18s %-16s\n" \
  "request_k" "views" "effective_k" "cIoU" "mIoU" "p@0.5" "p@0.7" "p@0.9" "oracle_best" "top1_gap" "pairwise_iou" "cand_entropy"
printf "%-11s %-9s %-12s %-10s %-10s %-10s %-10s %-10s %-14s %-14s %-18s %-16s\n" \
  "-----------" "---------" "------------" "----------" "----------" "----------" "----------" "----------" "--------------" "--------------" "------------------" "----------------"

for k in "${KS[@]}"; do
  run_dir="${OUTPUT_ROOT}/k_${k}"
  log_file="${run_dir}/eval.log"
  export_jsonl="${run_dir}/candidate_extra_analysis.jsonl"
  summary_dir="${run_dir}/candidate_extra_summary"
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
    MODEL.OpenWorldSAM2.FUSION.EVAL_CANDIDATE_K "${k}" \
    MODEL.OpenWorldSAM2.FUSION.MULTI_PROMPT_UNION_ON True \
    MODEL.OpenWorldSAM2.FUSION.MULTI_PROMPT_UNION_DEDUP_IOU_THR 0.95 \
    MODEL.OpenWorldSAM2.FUSION.EXTERNAL_PROPOSAL_ON False \
    MODEL.OpenWorldSAM2.FUSION.DECODER_NATIVE_EXPAND_ON False \
    MODEL.OpenWorldSAM2.FUSION.REAL_CANDIDATE_EXPAND_ON False \
    MODEL.OpenWorldSAM2.FUSION.EXTRA_ANALYSIS_ON True \
    MODEL.OpenWorldSAM2.FUSION.EXTRA_ANALYSIS_EXPORT_ON True \
    MODEL.OpenWorldSAM2.FUSION.EXTRA_ANALYSIS_EXPORT_PATH "${export_jsonl}" \
    MODEL.OpenWorldSAM2.FUSION.SAME_IMAGE_HARD_SUBSET_ON True \
    MODEL.OpenWorldSAM2.LOSS.CAND_ALIGN_ON False \
    MODEL.OpenWorldSAM2.LOSS.CAND_DIVERSITY_ON False \
    MODEL.OpenWorldSAM2.LOSS.SAME_IMAGE_HARDNEG_ON False \
    MODEL.OpenWorldSAM2.FUSION.LEARNABLE_FALLBACK_GATE_ON False \
    MODEL.OpenWorldSAM2.FUSION.LEARNABLE_FALLBACK_GATE_INFER_ON False \
    MODEL.OpenWorldSAM2.EVF_CONFIG "${EVF_LOCAL}" \
    MODEL.OpenWorldSAM2.TOKENIZER_CONFIG "${EVF_LOCAL}" \
    MODEL.OpenWorldSAM2.VISION_PRETRAINED "${VISION_LOCAL}" \
    2>&1 | tee "${log_file}"

  python tools/analyze_candidate_extra.py --input "${export_jsonl}" --outdir "${summary_dir}" >/dev/null

  IFS=',' read -r ciou miou p05 p07 p09 <<< "$(extract_metrics "${log_file}")"
  IFS=',' read -r req_views effk oracle gap pair_iou ent <<< "$(extract_analysis_metrics "${summary_dir}/summary.json" "${export_jsonl}")"

  echo "[K-SWEEP] requested_k=${k} requested_prompt_views=${req_views} effective_k=${effk}" | tee -a "${log_file}"
  printf "%-11s %-9s %-12s %-10s %-10s %-10s %-10s %-10s %-14s %-14s %-18s %-16s\n" \
    "${k}" "${req_views}" "${effk}" "${ciou}" "${miou}" "${p05}" "${p07}" "${p09}" "${oracle}" "${gap}" "${pair_iou}" "${ent}"

  echo "${k},${req_views},${effk},${ciou},${miou},${p05},${p07},${p09},${oracle},${gap},${pair_iou},${ent}" >> "${SUMMARY_CSV}"
done

python - "${SUMMARY_CSV}" <<'PY'
import csv
import math
import sys
from pathlib import Path

p = Path(sys.argv[1])
rows = []
with p.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

def to_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan

valid = [r for r in rows if not math.isnan(to_float(r.get("oracle_best_of_k", ""))) and not math.isnan(to_float(r.get("cIoU", "")))]
if len(valid) < 2:
    print("\n[K-SWEEP Conclusion] insufficient valid rows for trend judgment")
    raise SystemExit(0)

valid.sort(key=lambda r: to_float(r.get("effective_candidate_k", "0")))
first = valid[0]
last = valid[-1]
oracle_delta = to_float(last["oracle_best_of_k"]) - to_float(first["oracle_best_of_k"])
ciou_delta = to_float(last["cIoU"]) - to_float(first["cIoU"])
effks = [to_float(r.get("effective_candidate_k", "")) for r in rows if not math.isnan(to_float(r.get("effective_candidate_k", "")))]

oracle_up = oracle_delta > 1e-6
ciou_up = ciou_delta > 1e-6

print("\n[K-SWEEP Conclusion]")
if effks and len(set(effks)) == 1:
    print("- current sweep is clipping-only / no real candidate expansion")
print(f"- oracle_best_of_k trend: {'up' if oracle_up else 'not up'} (delta={oracle_delta:.6f})")
print(f"- cIoU trend: {'up' if ciou_up else 'not up'} (delta={ciou_delta:.6f})")
if oracle_up and (not ciou_up):
    print("- interpretation: oracle rises but cIoU does not -> bottleneck is more likely selection/alignment")
elif not oracle_up:
    print("- interpretation: oracle does not rise -> bottleneck is more likely candidate generation")
else:
    print("- interpretation: oracle and cIoU both rise -> candidate generation helps and selection is not the only bottleneck")
PY

echo "\nSaved summary CSV: ${SUMMARY_CSV}"
