#!/usr/bin/env bash
# =============================================================================
# scripts/run_ablation.sh
# Run the complete ablation study for SAM2 Lung Nodule Segmentation.
#
# Trains and evaluates four model variants in sequence:
#   1. base             — full model (TC + unfrozen encoder)
#   2. ablation_no_tc   — no temporal consistency loss
#   3. ablation_tc_dice — TC with Dice consistency metric instead of L2
#   4. ablation_frozen  — frozen SAM2 encoder
#
# Usage:
#   bash scripts/run_ablation.sh [OPTIONS]
#
# Options:
#   --data-dir   <dir>   Processed data root   (default: data)
#   --runs-dir   <dir>   Run output root        (default: runs/ablation)
#   --results-dir <dir>  Results root           (default: results)
#   --gpu        <id>    CUDA device ID         (default: 0)
#   --epochs     <int>   Training epochs        (default: 50)
#   --skip-train         Skip training, evaluate existing checkpoints
#   --dry-run            Print commands and exit
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_DIR="data"
RUNS_DIR="runs/ablation"
RESULTS_DIR="results"
GPU="0"
EPOCHS=50
SKIP_TRAIN=false
DRY_RUN=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-dir)    DATA_DIR="$2";    shift 2 ;;
    --runs-dir)    RUNS_DIR="$2";    shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --gpu)         GPU="$2";         shift 2 ;;
    --epochs)      EPOCHS="$2";      shift 2 ;;
    --skip-train)  SKIP_TRAIN=true;  shift ;;
    --dry-run)     DRY_RUN=true;     shift ;;
    -h|--help)
      sed -n '4,20p' "$0"; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
mkdir -p "${RESULTS_DIR}" "${RUNS_DIR}"

# ── Ablation variant definitions ──────────────────────────────────────────────
# Format: "variant_name|config_file"
VARIANTS=(
  "base|configs/base.yaml"
  "no_tc|configs/ablation_no_tc.yaml"
  "tc_dice|configs/ablation_tc_dice.yaml"
  "frozen|configs/ablation_frozen.yaml"
)

# ── CSV header ────────────────────────────────────────────────────────────────
SUMMARY_CSV="${RESULTS_DIR}/ablation_results_${TIMESTAMP}.csv"
echo "variant,dice,iou,precision,recall,ece,brier,uncertainty_auc" > "${SUMMARY_CSV}"

echo "=================================================="
echo " SAM2 Lung Nodule Segmentation — Ablation Study"
echo "  Variants  : ${#VARIANTS[@]}"
echo "  GPU       : ${GPU}"
echo "  Epochs    : ${EPOCHS}"
echo "  Runs dir  : ${RUNS_DIR}"
echo "  Results   : ${RESULTS_DIR}"
echo "=================================================="

for entry in "${VARIANTS[@]}"; do
  VARIANT="${entry%%|*}"
  CONFIG="${entry##*|}"
  RUN_DIR="${RUNS_DIR}/${VARIANT}"
  CKPT="${RUN_DIR}/checkpoints/best_model.pt"
  EVAL_OUT="${RESULTS_DIR}/eval_${VARIANT}"

  echo ""
  echo "── Variant: ${VARIANT} ──────────────────────────"
  echo "   Config   : ${CONFIG}"
  echo "   Run dir  : ${RUN_DIR}"

  # ── Training ────────────────────────────────────────────────────────────────
  if ${SKIP_TRAIN}; then
    echo "   [skip-train] Skipping training for ${VARIANT}"
  elif [[ -f "${CKPT}" ]]; then
    echo "   [skip-train] Checkpoint already exists: ${CKPT}"
  else
    TRAIN_CMD=(
      bash scripts/run_training.sh
        --config   "${CONFIG}"
        --run-dir  "${RUN_DIR}"
        --gpu      "${GPU}"
    )
    echo "   Training: ${TRAIN_CMD[*]}"
    if ! ${DRY_RUN}; then
      "${TRAIN_CMD[@]}" || {
        echo "   ERROR: training failed for ${VARIANT}" >&2; continue
      }
    fi
  fi

  # ── Evaluation ──────────────────────────────────────────────────────────────
  if [[ ! -f "${CKPT}" ]] && ! ${DRY_RUN}; then
    echo "   ERROR: checkpoint not found after training: ${CKPT}" >&2
    continue
  fi

  EVAL_CMD=(
    bash scripts/run_evaluation.sh
      --checkpoint "${CKPT}"
      --config     "${CONFIG}"
      --output-dir "${EVAL_OUT}"
      --gpu        "${GPU}"
  )
  echo "   Evaluating: ${EVAL_CMD[*]}"
  if ! ${DRY_RUN}; then
    "${EVAL_CMD[@]}" || {
      echo "   ERROR: evaluation failed for ${VARIANT}" >&2; continue
    }
  fi

  # ── Collect metrics into summary CSV ────────────────────────────────────────
  METRICS_JSON="${EVAL_OUT}/metrics.json"
  if [[ -f "${METRICS_JSON}" ]] && ! ${DRY_RUN}; then
    python - << PYEOF
import json, csv, sys
with open("${METRICS_JSON}") as f:
    m = json.load(f)
row = {
    "variant":         "${VARIANT}",
    "dice":            m.get("segmentation",{}).get("dice",          "N/A"),
    "iou":             m.get("segmentation",{}).get("iou",           "N/A"),
    "precision":       m.get("segmentation",{}).get("precision",     "N/A"),
    "recall":          m.get("segmentation",{}).get("recall",        "N/A"),
    "ece":             m.get("calibration",{}).get("ece",            "N/A"),
    "brier":           m.get("calibration",{}).get("brier",          "N/A"),
    "uncertainty_auc": m.get("calibration",{}).get("uncertainty_auc","N/A"),
}
with open("${SUMMARY_CSV}", "a", newline="") as fout:
    w = csv.DictWriter(fout, fieldnames=list(row.keys()))
    w.writerow(row)
print(f"   Appended metrics for ${VARIANT}: dice={row['dice']}")
PYEOF
  else
    echo "   [dry-run] Would append metrics to ${SUMMARY_CSV}"
  fi

done

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo " Ablation study complete"
echo "  Summary CSV : ${SUMMARY_CSV}"
if [[ -f "${SUMMARY_CSV}" ]]; then
  echo ""
  column -t -s',' "${SUMMARY_CSV}" 2>/dev/null || cat "${SUMMARY_CSV}"
fi
echo "=================================================="
