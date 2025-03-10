#!/usr/bin/env bash
# =============================================================================
# scripts/run_training.sh
# Launch SAM2 Lung Nodule Segmentation training.
#
# Usage:
#   bash scripts/run_training.sh [OPTIONS]
#
# Options:
#   --config    <yaml>   Config file path (default: configs/base.yaml)
#   --run-dir   <dir>    Output directory  (default: runs/sam2_lung_seg_v1)
#   --gpu       <id>     CUDA device ID    (default: 0)
#   --resume    <ckpt>   Resume from checkpoint
#   --no-amp             Disable automatic mixed precision
#   --wandb              Enable Weights & Biases logging
#   --dry-run            Print command and exit without running
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CONFIG="configs/base.yaml"
RUN_DIR="runs/sam2_lung_seg_v1"
GPU="0"
RESUME=""
AMP="--amp"
WANDB=""
DRY_RUN=false

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)   CONFIG="$2";    shift 2 ;;
    --run-dir)  RUN_DIR="$2";   shift 2 ;;
    --gpu)      GPU="$2";       shift 2 ;;
    --resume)   RESUME="--resume $2"; shift 2 ;;
    --no-amp)   AMP="";         shift ;;
    --wandb)    WANDB="--wandb"; shift ;;
    --dry-run)  DRY_RUN=true;   shift ;;
    -h|--help)
      sed -n '4,17p' "$0"; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ── Environment setup ─────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ── Timestamp for run identification ─────────────────────────────────────────
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${RUN_DIR}/logs/train_${TIMESTAMP}.log"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/checkpoints"

# ── Build the command ─────────────────────────────────────────────────────────
CMD=(
  python -m training.train
    --config        "${CONFIG}"
    --run-dir       "${RUN_DIR}"
    ${AMP}
    ${WANDB}
    ${RESUME}
)

# ── Print info ────────────────────────────────────────────────────────────────
echo "================================================"
echo " SAM2 Lung Nodule Segmentation — Training"
echo "================================================"
echo "  Config    : ${CONFIG}"
echo "  Run dir   : ${RUN_DIR}"
echo "  GPU       : ${GPU}"
echo "  AMP       : ${AMP:-(disabled)}"
echo "  W&B       : ${WANDB:-(disabled)}"
echo "  Resume    : ${RESUME:-(from scratch)}"
echo "  Log file  : ${LOG_FILE}"
echo "------------------------------------------------"
echo "  Command   : ${CMD[*]}"
echo "================================================"

if ${DRY_RUN}; then
  echo "[dry-run] Exiting without running."
  exit 0
fi

# ── Check CUDA availability ───────────────────────────────────────────────────
python - << 'EOF'
import torch
avail = torch.cuda.is_available()
device = torch.cuda.get_device_name(0) if avail else "CPU"
print(f"  PyTorch {torch.__version__}  |  CUDA {'available' if avail else 'NOT available'}  |  Device: {device}")
EOF

echo ""
echo "Starting training … (logging to ${LOG_FILE})"
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"

EXIT_CODE="${PIPESTATUS[0]}"

echo ""
if [[ "${EXIT_CODE}" -eq 0 ]]; then
  echo "================================================"
  echo " Training complete!"
  echo "  Best checkpoint : ${RUN_DIR}/checkpoints/best_model.pt"
  echo "  Logs            : ${LOG_FILE}"
  echo "================================================"
else
  echo "ERROR: Training exited with code ${EXIT_CODE}. Check ${LOG_FILE} for details." >&2
  exit "${EXIT_CODE}"
fi
