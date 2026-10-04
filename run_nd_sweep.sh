#!/usr/bin/env bash
# Sweep breadth N and depth D for one model config.
#
# Usage:
#   bash run_nd_sweep.sh configs/<model>.conf
#
# The config sets MODEL_NAME, N_VALUES, D_VALUES, DATASETS, NO_CRITIQUE and
# NO_CORRECTION (see configs/). Each (N, D) pair runs run_paired_ablation.sh.
# Commands run through "uv run" by default; override with RUNNER="" to use the
# active Python environment instead.
set -euo pipefail

CONFIG="${1:?Usage: bash run_nd_sweep.sh configs/<model>.conf}"
# shellcheck disable=SC1090
source "${CONFIG}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
RUNNER="${RUNNER-uv run --}"

ABLATION_FLAGS=()
[[ "${NO_CRITIQUE:-0}"   == "1" ]] && ABLATION_FLAGS+=(--no_critique)
[[ "${NO_CORRECTION:-0}" == "1" ]] && ABLATION_FLAGS+=(--no_correction)

for N in "${N_VALUES[@]}"; do
    for D in "${D_VALUES[@]}"; do
        echo "=== N=${N} D=${D} ==="
        ${RUNNER} bash "${ROOT_DIR}/run_paired_ablation.sh" \
            --model_name "${MODEL_NAME}" \
            --cimr_n_rollouts "${N}" \
            --cimr_depth "${D}" \
            "${ABLATION_FLAGS[@]}" \
            "${DATASETS[@]}"
    done
done
