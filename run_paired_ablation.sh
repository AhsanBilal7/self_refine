#!/usr/bin/env bash
# Run breadth-depth refinement for one model over one or more benchmarks.
#
# Usage:
#   bash run_paired_ablation.sh [--model_name M] [--cimr_n_rollouts N] [--cimr_depth D]
#                               [--no_critique] [--no_correction] [--no_prm] [dataset ...]
#
# Datasets default to: aime24 aime25 amc23 math500 olympiadbench (gsm8k is also accepted).
# Results go to experiments_results/results_paired/<model>_mnt<T>_t<temp>_p<top_p>/.
# Set PYTHON (e.g. PYTHON="uv run python") or GPUS (e.g. GPUS=0) in the environment.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# ── Defaults (the paper's configuration) ─────────────────────────────────────
MODEL_NAME="Qwen/Qwen2.5-Math-7B-Instruct"
VERIFIER_MODEL="Qwen/Qwen2.5-Math-PRM-7B"   # diagnostic PRM only
BATCH_SIZE=32

CIMR_N_ROLLOUTS=8
CIMR_DEPTH=4
CIMR_REASONING_MAX_NEW_TOKENS=2048
CIMR_REASONING_TEMPERATURE=0.7
CIMR_REASONING_TOP_P=0.95
CIMR_CRITIQUE_MAX_TOKENS=384
CIMR_CRITIQUE_MAX_TRACE_CHARS=4000
CIMR_CORRECTION_MAX_TOKENS=1024
CIMR_CORRECTION_MAX_TRACE_CHARS=4000
CIMR_CORRECTION_MAX_CRITIQUE_CHARS=1500
CIMR_CONTINUATION_MAX_PRIOR_CHARS=4000

NO_CRITIQUE=0
NO_CORRECTION=0
NO_PRM=0

# Problems per benchmark (empty = full split).
MAX_SAMPLES_AIME="30"
MAX_SAMPLES_AMC="40"
MAX_SAMPLES_MATH="500"
MAX_SAMPLES_OLYMPIAD=""
MAX_SAMPLES_GSM=""

PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-}"
BASE_OUT="${ROOT_DIR}/experiments_results/results_paired"

# ── Arguments ────────────────────────────────────────────────────────────────
DATASET_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name|--model)     MODEL_NAME="$2";            shift 2 ;;
        --model_name=*|--model=*) MODEL_NAME="${1#*=}";       shift   ;;
        --cimr_n_rollouts)        CIMR_N_ROLLOUTS="$2";       shift 2 ;;
        --cimr_n_rollouts=*)      CIMR_N_ROLLOUTS="${1#*=}";  shift   ;;
        --cimr_depth)             CIMR_DEPTH="$2";            shift 2 ;;
        --cimr_depth=*)           CIMR_DEPTH="${1#*=}";       shift   ;;
        --no_critique)            NO_CRITIQUE=1;              shift   ;;
        --no_correction)          NO_CORRECTION=1;            shift   ;;
        --no_prm)                 NO_PRM=1;                   shift   ;;
        --help|-h)                sed -n '2,10p' "$0"; exit 0 ;;
        *)                        DATASET_ARGS+=("$1");       shift   ;;
    esac
done

MODEL_SHORT=$(basename "${MODEL_NAME}")
OUT="${BASE_OUT}/${MODEL_SHORT}_mnt${CIMR_REASONING_MAX_NEW_TOKENS}_t${CIMR_REASONING_TEMPERATURE}_p${CIMR_REASONING_TOP_P}"
mkdir -p "${OUT}/logs"

SUFFIX=""
[[ "${NO_CRITIQUE}"   == "1" ]] && SUFFIX="${SUFFIX}_no_critique"
[[ "${NO_CORRECTION}" == "1" ]] && SUFFIX="${SUFFIX}_no_correction"

COMMON_ARGS=(
    --model                              "${MODEL_NAME}"
    --verifier_model                     "${VERIFIER_MODEL}"
    --batch_size                         "${BATCH_SIZE}"
    --output_dir                         "${OUT}"
    --cimr_n_rollouts                    "${CIMR_N_ROLLOUTS}"
    --cimr_depth                         "${CIMR_DEPTH}"
    --cimr_reasoning_max_new_tokens      "${CIMR_REASONING_MAX_NEW_TOKENS}"
    --cimr_reasoning_temperature         "${CIMR_REASONING_TEMPERATURE}"
    --cimr_reasoning_top_p               "${CIMR_REASONING_TOP_P}"
    --cimr_critique_max_tokens           "${CIMR_CRITIQUE_MAX_TOKENS}"
    --cimr_critique_max_trace_chars      "${CIMR_CRITIQUE_MAX_TRACE_CHARS}"
    --cimr_correction_max_tokens         "${CIMR_CORRECTION_MAX_TOKENS}"
    --cimr_correction_max_trace_chars    "${CIMR_CORRECTION_MAX_TRACE_CHARS}"
    --cimr_correction_max_critique_chars "${CIMR_CORRECTION_MAX_CRITIQUE_CHARS}"
    --cimr_continuation_max_prior_chars  "${CIMR_CONTINUATION_MAX_PRIOR_CHARS}"
)
[[ -n "${GPUS}" ]]               && COMMON_ARGS+=(--gpus "${GPUS}")
[[ "${NO_CRITIQUE}"   == "1" ]] && COMMON_ARGS+=(--no_critique)
[[ "${NO_CORRECTION}" == "1" ]] && COMMON_ARGS+=(--no_correction)
[[ "${NO_PRM}"        == "1" ]] && COMMON_ARGS+=(--no_prm)

run_dataset() {
    local DATASET="$1" MAX_SAMPLES="$2"
    local TAG="${DATASET}_N${CIMR_N_ROLLOUTS}_d${CIMR_DEPTH}${SUFFIX}"
    local MAX_FLAG=()
    [[ -n "${MAX_SAMPLES}" ]] && MAX_FLAG=(--max_samples "${MAX_SAMPLES}")

    echo "  dataset=${DATASET}  N=${CIMR_N_ROLLOUTS}  D=${CIMR_DEPTH}  ablation=${SUFFIX:-none}"
    ${PYTHON} run_paired_ablation.py \
        "${COMMON_ARGS[@]}" \
        --dataset "${DATASET}" \
        "${MAX_FLAG[@]}" \
        2>&1 | tee "${OUT}/logs/paired_${TAG}.log"
}

DATASETS=("${DATASET_ARGS[@]}")
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=(aime24 aime25 amc23 math500 olympiadbench)

echo "model=${MODEL_NAME}  N=${CIMR_N_ROLLOUTS}  D=${CIMR_DEPTH}  datasets=${DATASETS[*]}"
for DATASET in "${DATASETS[@]}"; do
    case "${DATASET}" in
        aime24|aime25) run_dataset "${DATASET}" "${MAX_SAMPLES_AIME}"     ;;
        amc23)         run_dataset "${DATASET}" "${MAX_SAMPLES_AMC}"      ;;
        math500)       run_dataset "${DATASET}" "${MAX_SAMPLES_MATH}"     ;;
        olympiadbench) run_dataset "${DATASET}" "${MAX_SAMPLES_OLYMPIAD}" ;;
        gsm8k)         run_dataset "${DATASET}" "${MAX_SAMPLES_GSM}"      ;;
        *) echo "Unknown dataset: '${DATASET}'"; exit 1 ;;
    esac
done
