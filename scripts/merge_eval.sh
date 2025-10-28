#!/usr/bin/env bash
set -euo pipefail
set -x

# Parse command-line arguments
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <actor_model> <lora_alpha> <project_name> <experiment_name> <data_dir> <output_dir>"
    echo "Example: $0 deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 64 verl_gsm8k_length 25_10_20_r1qwen15b_grpo_budget200_lr5e-4_alpha64_exp2 /workspace/data/gsm8k_1 /workspace/25_10_20_r1qwen15b_grpo_budget200_lr5e-4_alpha64_exp2_step24"
    exit 1
fi

ACTOR="$1"
LORA_ALPHA="$2"
PROJECT="$3"
EXP="$4"
DATA_DIR="$5"
OUTPUT_DIR="$6"

# Find the highest-numbered checkpoint
CHECKPOINT_BASE="checkpoints/${PROJECT}/${EXP}"
echo "Looking for checkpoints in ${CHECKPOINT_BASE}..."
FINAL_STEP=$(ls -d ${CHECKPOINT_BASE}/global_step_* 2>/dev/null | \
    sed 's/.*global_step_//' | \
    sort -n | \
    tail -1)

if [ -z "${FINAL_STEP}" ]; then
    echo "Error: No checkpoint directories found in ${CHECKPOINT_BASE}"
    exit 1
fi

echo "Found highest checkpoint: global_step_${FINAL_STEP}"

# Paths
CHECKPOINT_DIR="${CHECKPOINT_BASE}/global_step_${FINAL_STEP}/actor"
ADAPTER_CONFIG="${OUTPUT_DIR}/lora_adapter/adapter_config.json"
EVAL_OUTPUT="${OUTPUT_DIR}/${EXP}_step${FINAL_STEP}_eval.jsonl"

# Step 1: Merge model
echo "Merging model from ${CHECKPOINT_DIR} to ${OUTPUT_DIR}..."
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${CHECKPOINT_DIR}" \
    --target_dir "${OUTPUT_DIR}"

# Step 2: Fix adapter_config.json
echo "Fixing adapter_config.json at ${ADAPTER_CONFIG}..."
python3 - << PY
import json

config_path = "${ADAPTER_CONFIG}"

with open(config_path, 'r') as f:
    config = json.load(f)

# Update fields that verl doesn't set correctly
config['base_model_name_or_path'] = "${ACTOR}"
config['lora_alpha'] = ${LORA_ALPHA}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)

print(f"Updated adapter_config.json:")
print(f"  base_model_name_or_path: {config['base_model_name_or_path']}")
print(f"  lora_alpha: {config['lora_alpha']}")
PY

# Step 3: Run evaluation
echo "Running evaluation..."
python3 /workspace/rl_cot_monitorability/scripts/eval_gsm8k_length.py \
    --model "${ACTOR}" \
    --lora-path "${OUTPUT_DIR}/lora_adapter" \
    --data "${DATA_DIR}/test.parquet" \
    --out "${EVAL_OUTPUT}" \
    --temperature 1.0 \
    --max-new-tokens 1024 \
    --max-model-len 2048 \
    --n-samples 1

echo "Post-processing complete!"
echo "Merged model saved to: ${OUTPUT_DIR}"
echo "Evaluation results saved to: ${EVAL_OUTPUT}"

