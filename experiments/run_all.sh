#!/bin/bash
# Run the FORCE stability experiments in order. Logs go to logs/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG="${SCRIPT_DIR}/config.yaml"
LOGDIR="${SCRIPT_DIR}/logs"
FORCE_FLAG=""

if [[ "${1:-}" == "--force" ]]; then
    FORCE_FLAG="--force"
fi

mkdir -p "${LOGDIR}"
cd "${SCRIPT_DIR}"

echo "=============================================="
echo " FORCE stability experiments"
echo " Config: ${CONFIG}"
echo " Started: $(date)"
echo "=============================================="

run() {
    echo ""
    echo ">>> $1"
    $PYTHON "scripts/$2" --config "$CONFIG" $FORCE_FLAG 2>&1 | tee "${LOGDIR}/${2%.py}.log"
}

run "Exp 2 — library resampling"      exp2_library_resampling.py
run "Exp 1 — neighborhood concentration" exp1_neighborhood.py
run "Exp 6 — split-half library"      exp6_split_half.py
run "Exp 3 — input-noise perturbation" exp3_input_noise.py
run "Exp 4 — K sweep"                 exp4_k_sweep.py
run "Exp 5 — beta sweep"              exp5_beta_sweep.py
run "Exp 7 — DTI reference"           exp7_dti_reference.py

echo ""
echo "=============================================="
echo " All experiments complete: $(date)"
echo "=============================================="
