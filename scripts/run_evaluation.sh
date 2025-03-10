#!/usr/bin/env bash
# =============================================================================
# scripts/run_evaluation.sh
# Evaluate a trained SAM2 Lung Nodule Segmentation checkpoint.
#
# Usage:
#   bash scripts/run_evaluation.sh [OPTIONS]
#
# Options:
#   --checkpoint  <path>   Model checkpoint (.pt)          [REQUIRED]
#   --config      <yaml>   Config file            (default: configs/base.yaml)
#   --split       <name>   Dataset split to eval  (default: test)
#   --output-dir  <dir>    Results output dir     (default: results/eval_<ts>)
#   --n-mc        <int>    MC Dropout samples     (default: 25)
#   --threshold   <float>  Decision threshold     (default: 0.5)
#   --rad-csv     <csv>    Radiologist labels CSV (skip if absent)
#   --gpu         <id>     CUDA device ID         (default: 0)
#   --dry-run              Print command and exit
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CHECKPOINT=""
CONFIG="configs/base.yaml"
SPLIT="test"
OUTPUT_DIR=""
N_MC=25
THRESHOLD=0.5
RAD_CSV=""
GPU="0"
DRY_RUN=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --checkpoint)  CHECKPOINT="$2";  shift 2 ;;
    --config)      CONFIG="$2";      shift 2 ;;
    --split)       SPLIT="$2";       shift 2 ;;
    --output-dir)  OUTPUT_DIR="$2";  shift 2 ;;
    --n-mc)        N_MC="$2";        shift 2 ;;
    --threshold)   THRESHOLD="$2";   shift 2 ;;
    --rad-csv)     RAD_CSV="$2";     shift 2 ;;
    --gpu)         GPU="$2";         shift 2 ;;
    --dry-run)     DRY_RUN=true;     shift ;;
    -h|--help)
      sed -n '4,18p' "$0"; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── Validate required args ────────────────────────────────────────────────────
if [[ -z "${CHECKPOINT}" ]]; then
  echo "ERROR: --checkpoint is required." >&2
  echo "  Example: bash scripts/run_evaluation.sh --checkpoint runs/sam2_lung_seg_v1/checkpoints/best_model.pt"
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

# ── Set output dir default ────────────────────────────────────────────────────
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_DIR="${OUTPUT_DIR:-results/eval_${SPLIT}_${TIMESTAMP}}"
mkdir -p "${OUTPUT_DIR}"

# ── Environment ───────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ── Build command ─────────────────────────────────────────────────────────────
CMD=(
  python -m evaluation.evaluate
    --checkpoint   "${CHECKPOINT}"
    --config       "${CONFIG}"
    --split        "${SPLIT}"
    --output-dir   "${OUTPUT_DIR}"
    --n-mc         "${N_MC}"
    --threshold    "${THRESHOLD}"
)

if [[ -n "${RAD_CSV}" ]]; then
  CMD+=(--radiologist-csv "${RAD_CSV}")
fi

# ── Print info ────────────────────────────────────────────────────────────────
echo "================================================"
echo " SAM2 Lung Nodule Segmentation — Evaluation"
echo "================================================"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Config     : ${CONFIG}"
echo "  Split      : ${SPLIT}"
echo "  MC samples : ${N_MC}"
echo "  Threshold  : ${THRESHOLD}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Rad CSV    : ${RAD_CSV:-(none)}"
echo "  GPU        : ${GPU}"
echo "------------------------------------------------"
echo "  Command    : ${CMD[*]}"
echo "================================================"

if ${DRY_RUN}; then
  echo "[dry-run] Exiting without running."
  exit 0
fi

# ── Run ───────────────────────────────────────────────────────────────────────
"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/eval.log"

EXIT_CODE="${PIPESTATUS[0]}"

echo ""
if [[ "${EXIT_CODE}" -eq 0 ]]; then
  echo "================================================"
  echo " Evaluation complete!"
  echo "  JSON metrics  : ${OUTPUT_DIR}/metrics.json"
  echo "  Per-case CSV  : ${OUTPUT_DIR}/per_case_metrics.csv"
  echo "  Calib plot    : ${OUTPUT_DIR}/calibration_diagram.png"
  echo "  Log           : ${OUTPUT_DIR}/eval.log"
  echo "================================================"
else
  echo "ERROR: Evaluation exited with code ${EXIT_CODE}." >&2
  exit "${EXIT_CODE}"
fi
