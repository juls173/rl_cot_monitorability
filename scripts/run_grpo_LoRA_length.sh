#!/usr/bin/env bash
set -euo pipefail
set -x

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export VLLM_USE_V1=0

# Dataset configuration
DATASET="bigmath"  # Options: "gsm8k" or "bigmath"
NUM_BUDGET_COPIES=1
BUDGET_VALUES="control"
BUDGET_WINDOW=0
FORMAT_ONLY_ANSWER=false
DECREASING_BUDGETS=false
BUDGET_RANGE=false  # If true, interpret budget values as range (2 values) or interpolated range (4 values with curriculum)
CURRICULUM_WARMUP=""  # Fraction of dataset using initial budget (curriculum mode only), e.g., "0.2"
CURRICULUM_COOLDOWN=""  # Fraction of dataset using final budget (curriculum mode only), e.g., "0.2"
# BIGMATH_EXTRA_ARGS="--numerical-only --solve-rate-min 0.1 --solve-rate-max 0.9 --max-samples 65000 --train-fraction 0.98"  # Extra arguments for bigmath_token_budget.py (e.g., "--numerical-only --max-samples 10000")
BIGMATH_EXTRA_ARGS="--numerical-only --solve-rate-min 0.2 --max-samples 65000 --train-fraction 0.98"  # Extra arguments for bigmath_token_budget.py (e.g., "--numerical-only --max-samples 10000")
DATA_BASE_DIR="/workspace/data"
REPO_DIR="/workspace/rl_cot_monitorability"

# Training configuration
PENALTY_WARMUP=0
LR=5e-5
LORA_RANK=32
LORA_ALPHA=64
TEMPERATURE=1
EPOCHS=1

# Build path components (must match generate_dataset.sh logic)
BUDGET_VALUES_FORMATTED=$(echo ${BUDGET_VALUES} | tr ' ' '_')
if [ "${FORMAT_ONLY_ANSWER}" = true ]; then
    FORMAT_STRING="_format"
else
    FORMAT_STRING=""
fi
if [ "${DECREASING_BUDGETS}" = true ]; then
    DECREASING_STRING="_dec"
else
    DECREASING_STRING=""
fi
if [ "${BUDGET_RANGE}" = true ]; then
    RANGE_STRING="_range"
else
    RANGE_STRING=""
fi
if [ -n "${CURRICULUM_WARMUP}" ] || [ -n "${CURRICULUM_COOLDOWN}" ]; then
    WARMUP_VAL="${CURRICULUM_WARMUP:-0}"
    COOLDOWN_VAL="${CURRICULUM_COOLDOWN:-0}"
    CURRICULUM_STRING="_cur${WARMUP_VAL}_${COOLDOWN_VAL}"
else
    CURRICULUM_STRING=""
fi
DATA_DIR="${DATA_BASE_DIR}/${DATASET}_${NUM_BUDGET_COPIES}_${BUDGET_VALUES_FORMATTED}_w${BUDGET_WINDOW}${FORMAT_STRING}${DECREASING_STRING}${RANGE_STRING}${CURRICULUM_STRING}"

# Generate dataset if it doesn't exist
if [ ! -f "${DATA_DIR}/train.parquet" ] || [ ! -f "${DATA_DIR}/test.parquet" ]; then
    echo "Dataset not found at ${DATA_DIR}, generating..."
    GENERATE_CMD="${REPO_DIR}/scripts/generate_dataset.sh --dataset ${DATASET} --num-budget-copies ${NUM_BUDGET_COPIES} --budget-values \"${BUDGET_VALUES}\" --budget-window ${BUDGET_WINDOW} --data-base-dir ${DATA_BASE_DIR} --repo-dir ${REPO_DIR}"
    if [ "${FORMAT_ONLY_ANSWER}" = true ]; then GENERATE_CMD="${GENERATE_CMD} --format-only-answer"; fi
    if [ "${DECREASING_BUDGETS}" = true ]; then GENERATE_CMD="${GENERATE_CMD} --decreasing-budgets"; fi
    if [ "${BUDGET_RANGE}" = true ]; then GENERATE_CMD="${GENERATE_CMD} --budget-range"; fi
    if [ -n "${CURRICULUM_WARMUP}" ]; then GENERATE_CMD="${GENERATE_CMD} --curriculum-warmup ${CURRICULUM_WARMUP}"; fi
    if [ -n "${CURRICULUM_COOLDOWN}" ]; then GENERATE_CMD="${GENERATE_CMD} --curriculum-cooldown ${CURRICULUM_COOLDOWN}"; fi
    if [ -n "${BIGMATH_EXTRA_ARGS}" ]; then GENERATE_CMD="${GENERATE_CMD} --bigmath-extra-args \"${BIGMATH_EXTRA_ARGS}\""; fi
    eval ${GENERATE_CMD}
fi
REWARD_FN_PATH=${REPO_DIR}/scripts/reward_length.py
REWARD_FN_NAME=compute_score
ACTOR=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# ACTOR=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
PROJECT=verl_${DATASET}_length
EXP="25_12_08_r1qwen15b_grpo_${NUM_BUDGET_COPIES}_${BUDGET_VALUES_FORMATTED}_w${BUDGET_WINDOW}${FORMAT_STRING}${DECREASING_STRING}${RANGE_STRING}${CURRICULUM_STRING}_warmup${PENALTY_WARMUP}_lr${LR}_alpha${LORA_ALPHA}_temp${TEMPERATURE}"


# # 1x H100 80GB for 1.5B model
# TRAIN_BATCH_SIZE=256
# PPO_MINI_BATCH_SIZE=64
# PPO_MICRO_BATCH_SIZE=32
# LOG_PROB_MICRO_BATCH_SIZE=32
# N_GPUS=1

# # 1x H200 140GB for 7B model
# TRAIN_BATCH_SIZE=288
# PPO_MINI_BATCH_SIZE=96
# PPO_MICRO_BATCH_SIZE=48
# LOG_PROB_MICRO_BATCH_SIZE=48
# N_GPUS=1

# 2x H100 80GB for 7B model
# TRAIN_BATCH_SIZE=256
# PPO_MINI_BATCH_SIZE=64
# PPO_MICRO_BATCH_SIZE=32
# LOG_PROB_MICRO_BATCH_SIZE=32
# N_GPUS=2

# 1x H200 140GB for 1.5B model
TRAIN_BATCH_SIZE=256
PPO_MINI_BATCH_SIZE=128
PPO_MICRO_BATCH_SIZE=64
LOG_PROB_MICRO_BATCH_SIZE=64
N_GPUS=1

# # 2x H200 140GB for 1.5B model
# TRAIN_BATCH_SIZE=256
# PPO_MINI_BATCH_SIZE=128
# PPO_MICRO_BATCH_SIZE=64
# LOG_PROB_MICRO_BATCH_SIZE=64
# N_GPUS=2

ROLLOUT_N=5

# Export configuration for length-aware reward function
# export LENGTH_PENALTY=0.01
export LENGTH_PENALTY=0.002
export TOKENIZER_MODEL_NAME=${ACTOR}
if [ "${FORMAT_ONLY_ANSWER}" = true ]; then
    export FORMAT_ONLY_ANSWER_PENALTY=1.0
else
    export FORMAT_ONLY_ANSWER_PENALTY=0.0
fi
export PENALTY_WARMUP_PERIOD=${PENALTY_WARMUP}

# Disable shuffle if using decreasing budgets (curriculum training)
if [ "${DECREASING_BUDGETS}" = true ]; then
    DATA_SHUFFLE=False
else
    DATA_SHUFFLE=True
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files=${DATA_DIR}/train.parquet \
  data.val_files=${DATA_DIR}/test.parquet \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.max_prompt_length=256 \
  data.max_response_length=768 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=${DATA_SHUFFLE} \
  actor_rollout_ref.model.path=${ACTOR} \
  actor_rollout_ref.model.lora_rank=${LORA_RANK} \
  actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
  actor_rollout_ref.model.target_modules="all-linear" \
  actor_rollout_ref.actor.optim.lr=${LR} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.checkpoint.save_contents="['optimizer', 'extra']" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1\
  actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE} \
  actor_rollout_ref.rollout.load_format="safetensors" \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.disable_log_stats=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE} \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.actor.entropy_checkpointing=True \
  custom_reward_function.path=${REWARD_FN_PATH} \
  custom_reward_function.name=${REWARD_FN_NAME} \
  trainer.critic_warmup=0 \
  trainer.logger=[console,wandb] \
  trainer.project_name=${PROJECT} \
  trainer.experiment_name=${EXP} \
  trainer.n_gpus_per_node=${N_GPUS} \
  trainer.nnodes=1 \
  trainer.save_freq=2 \
  trainer.test_freq=2 \
  trainer.total_epochs=${EPOCHS} \
  trainer.log_val_generations=15 \
  trainer.rollout_data_dir=/workspace/training_logs/${EXP}/training \
  trainer.validation_data_dir=/workspace/training_logs/${EXP}/validation \
#   actor_rollout_ref.rollout.max_num_seqs=4096 \
#   actor_rollout_ref.rollout.max_model_len=1024 \
#   actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
