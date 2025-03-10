#!/usr/bin/env bash
# =============================================================================
# scripts/download_luna16.sh
# Download and prepare the LUNA16 dataset for training.
#
# Usage:
#   bash scripts/download_luna16.sh [--data-dir <path>] [--subsets 0,1,2]
#
# Requirements:
#   - wget or curl
#   - ~130 GB free disk space (full 10-subset dataset)
#   - LUNA16 access token (register at https://luna16.grand-challenge.org/)
#
# The script downloads the following from the LUNA16 challenge server:
#   subset0.zip … subset9.zip   CT scans (MHD + RAW)
#   annotations.csv             Nodule coordinates & diameter
#   candidates_V2.csv           False-positive reduction candidates
#
# After extraction the directory layout will be:
#   <data-dir>/luna16/
#     annotations.csv
#     candidates_V2.csv
#     subsets/
#       subset0/  … subset9/
#         *.mhd   (header files, CT volume metadata)
#         *.raw   (raw voxel data)
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-data}"
SUBSETS="${SUBSETS:-0,1,2,3,4,5,6,7,8,9}"   # comma-separated subset IDs
BASE_URL="https://zenodo.org/record/3723295/files"
ANNO_URL="https://luna16.grand-challenge.org/serve/public_html/pretraining/annotations.csv"
CAND_URL="https://luna16.grand-challenge.org/serve/public_html/pretraining/candidates_V2.csv"

LUNA_DIR="${DATA_DIR}/luna16"
SUBSET_DIR="${LUNA_DIR}/subsets"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-dir)  DATA_DIR="$2";   shift 2 ;;
    --subsets)   SUBSETS="$2";    shift 2 ;;
    -h|--help)
      sed -n '4,20p' "$0"; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

LUNA_DIR="${DATA_DIR}/luna16"
SUBSET_DIR="${LUNA_DIR}/subsets"

# ── Helper: prefer wget, fall back to curl ───────────────────────────────────
download() {
  local url="$1" dest="$2"
  if command -v wget &>/dev/null; then
    wget --continue --progress=bar:force -O "$dest" "$url"
  elif command -v curl &>/dev/null; then
    curl -L --progress-bar -C - -o "$dest" "$url"
  else
    echo "ERROR: neither wget nor curl found." >&2; exit 1
  fi
}

# ── Disk space check (~130G for all 10 subsets) ───────────────────────────────
NEEDED_GB=15   # approximate per-subset
AVAIL_GB=$(df -BG "${DATA_DIR}" 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4); print $4}' || echo 999)
echo "Available disk space in ${DATA_DIR}: ${AVAIL_GB} GB"

# ── Create directories ────────────────────────────────────────────────────────
mkdir -p "${SUBSET_DIR}"

# ── Download annotation CSVs ──────────────────────────────────────────────────
echo ""
echo "==> Downloading annotations …"
for url in "${ANNO_URL}" "${CAND_URL}"; do
  fname="$(basename "$url")"
  dest="${LUNA_DIR}/${fname}"
  if [[ -f "${dest}" ]]; then
    echo "  [skip] ${fname} already exists"
  else
    download "${url}" "${dest}"
    echo "  [done] ${fname}"
  fi
done

# ── Download and extract subsets ──────────────────────────────────────────────
IFS=',' read -ra SUBSET_LIST <<< "${SUBSETS}"

for sid in "${SUBSET_LIST[@]}"; do
  sid="${sid// /}"    # strip whitespace
  echo ""
  echo "==> Subset ${sid} …"

  zip_url="${BASE_URL}/subset${sid}.zip"
  zip_dest="${LUNA_DIR}/subset${sid}.zip"
  out_dir="${SUBSET_DIR}/subset${sid}"

  # -- Download
  if [[ -f "${zip_dest}" ]]; then
    echo "  [skip] subset${sid}.zip already downloaded"
  else
    echo "  Downloading subset${sid}.zip …"
    download "${zip_url}" "${zip_dest}"
  fi

  # -- Check extracted files
  if [[ -d "${out_dir}" ]] && [[ -n "$(ls -A "${out_dir}" 2>/dev/null)" ]]; then
    echo "  [skip] subset${sid} already extracted"
  else
    echo "  Extracting subset${sid}.zip …"
    mkdir -p "${out_dir}"
    unzip -q "${zip_dest}" -d "${out_dir}"
    echo "  [done] extracted to ${out_dir}"
  fi

  # -- Verify at least one MHD file exists
  mhd_count=$(find "${out_dir}" -name "*.mhd" 2>/dev/null | wc -l)
  if [[ "${mhd_count}" -gt 0 ]]; then
    echo "  Verified: ${mhd_count} .mhd files in subset${sid}"
  else
    echo "  WARNING: no .mhd files found in ${out_dir} — extraction may have failed" >&2
  fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " LUNA16 download complete"
echo "  Data directory : ${LUNA_DIR}"
echo "  Subsets        : ${SUBSETS}"
echo ""
echo " Next steps:"
echo "   python -m data.luna16_preprocessing \\"
echo "     --luna-dir ${LUNA_DIR} \\"
echo "     --out-dir ${DATA_DIR}/processed \\"
echo "     --patch-size 96"
echo "=========================================="
