#!/usr/bin/env bash
set -euo pipefail

# Script to generate dataset with consistent naming conventions.
# Usage: ./generate_dataset.sh --dataset <gsm8k|bigmath> --num-budget-copies N --budget-values "V1 V2 ..." --budget-window W [options]

usage() {
    echo "Usage: $0 --dataset <gsm8k|bigmath> --num-budget-copies N --budget-values \"V1 V2 ...\" --budget-window W [options]"
    echo ""
    echo "Required arguments:"
    echo "  --dataset              Dataset type: 'gsm8k' or 'bigmath'"
    echo "  --num-budget-copies    Number of budget copies"
    echo "  --budget-values        Space-separated budget values (quoted)"
    echo "  --budget-window        Window around budget where no penalty is applied"
    echo ""
    echo "Optional arguments:"
    echo "  --format-only-answer   Enable format-only answer instruction"
    echo "  --decreasing-budgets   Sort examples by budget (descending) for curriculum training"
    echo "  --bigmath-extra-args   Extra arguments for bigmath_token_budget.py"
    echo "  --data-base-dir        Base directory for data (default: /workspace/data)"
    echo "  --repo-dir             Repository directory (default: /workspace/rl_cot_monitorability)"
    exit 1
}

# Default values
DATASET=""
NUM_BUDGET_COPIES=""
BUDGET_VALUES=""
BUDGET_WINDOW=""
FORMAT_ONLY_ANSWER=false
DECREASING_BUDGETS=false
BIGMATH_EXTRA_ARGS=""
DATA_BASE_DIR="/workspace/data"
REPO_DIR="/workspace/rl_cot_monitorability"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --num-budget-copies)
            NUM_BUDGET_COPIES="$2"
            shift 2
            ;;
        --budget-values)
            BUDGET_VALUES="$2"
            shift 2
            ;;
        --budget-window)
            BUDGET_WINDOW="$2"
            shift 2
            ;;
        --format-only-answer)
            FORMAT_ONLY_ANSWER=true
            shift
            ;;
        --decreasing-budgets)
            DECREASING_BUDGETS=true
            shift
            ;;
        --bigmath-extra-args)
            BIGMATH_EXTRA_ARGS="$2"
            shift 2
            ;;
        --data-base-dir)
            DATA_BASE_DIR="$2"
            shift 2
            ;;
        --repo-dir)
            REPO_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$DATASET" ] || [ -z "$NUM_BUDGET_COPIES" ] || [ -z "$BUDGET_VALUES" ] || [ -z "$BUDGET_WINDOW" ]; then
    echo "Error: Missing required arguments"
    usage
fi

if [ "$DATASET" != "gsm8k" ] && [ "$DATASET" != "bigmath" ]; then
    echo "Error: Unknown dataset '${DATASET}'. Valid options are 'gsm8k' or 'bigmath'."
    exit 1
fi

# Build path components
BUDGET_VALUES_FORMATTED=$(echo ${BUDGET_VALUES} | tr ' ' '_')
FORMAT_STRING=""
DECREASING_STRING=""
if [ "${FORMAT_ONLY_ANSWER}" = true ]; then FORMAT_STRING="_format"; fi
if [ "${DECREASING_BUDGETS}" = true ]; then DECREASING_STRING="_dec"; fi

# Construct data directory path
mkdir -p "${DATA_BASE_DIR}"
DATA_DIR="${DATA_BASE_DIR}/${DATASET}_${NUM_BUDGET_COPIES}_${BUDGET_VALUES_FORMATTED}_w${BUDGET_WINDOW}${FORMAT_STRING}${DECREASING_STRING}"

# Check if dataset already exists
if [ -f "${DATA_DIR}/train.parquet" ] && [ -f "${DATA_DIR}/test.parquet" ]; then
    echo "Dataset already exists at ${DATA_DIR}, skipping generation..."
    echo "${DATA_DIR}"
    exit 0
fi

echo "Generating ${DATASET} dataset..."

# Dataset-specific script
if [ "${DATASET}" = "gsm8k" ]; then
    SCRIPT="${REPO_DIR}/scripts/gsm8k_token_budget.py"
    EXTRA_ARGS=""
elif [ "${DATASET}" = "bigmath" ]; then
    SCRIPT="${REPO_DIR}/scripts/bigmath_token_budget.py"
    EXTRA_ARGS="${BIGMATH_EXTRA_ARGS}"
fi

# Build command
CMD="python ${SCRIPT} --local-save-dir ${DATA_DIR} --num-budget-copies ${NUM_BUDGET_COPIES} --budget-values ${BUDGET_VALUES} --budget-window ${BUDGET_WINDOW} ${EXTRA_ARGS}"
if [ "${FORMAT_ONLY_ANSWER}" = true ]; then CMD="${CMD} --format-only-answer"; fi
if [ "${DECREASING_BUDGETS}" = true ]; then CMD="${CMD} --decreasing-budgets"; fi

eval ${CMD}

echo "✓ ${DATASET} dataset generated at ${DATA_DIR}"
echo "${DATA_DIR}"

