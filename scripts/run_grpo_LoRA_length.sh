#!/usr/bin/env bash
set -euo pipefail
set -x

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export VLLM_USE_V1=0

NUM_BUDGET_COPIES=1
BUDGET_VALUES="50"
BUDGET_VALUES_FORMATTED=$(echo ${BUDGET_VALUES} | tr ' ' '_')
BUDGET_WINDOW=0
PENALTY_WARMUP=10000
FORMAT_ONLY_ANSWER=false
if [ "${FORMAT_ONLY_ANSWER}" = true ]; then
    FORMAT_STRING="_format"
else
    FORMAT_STRING=""
fi
LR=5e-4
LORA_RANK=32
LORA_ALPHA=64
EPOCHS=1
# DATA_DIR=/workspace/data/gsm8k_${NUM_BUDGET_COPIES}_${BUDGET_VALUES_FORMATTED}_w${BUDGET_WINDOW}${FORMAT_STRING}
DATA_DIR=/workspace/data/bigmath_${NUM_BUDGET_COPIES}_${BUDGET_VALUES_FORMATTED}_w${BUDGET_WINDOW}${FORMAT_STRING}
REWARD_FN_PATH=/workspace/rl_cot_monitorability/scripts/reward_length.py
REWARD_FN_NAME=compute_score
# ACTOR=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
ACTOR=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# PROJECT=verl_gsm8k_length
PROJECT=verl_bigmath_length
EXP="25_11_24_r1qwen7b_grpo_budget_${NUM_BUDGET_COPIES}_${BUDGET_VALUES_FORMATTED}_w${BUDGET_WINDOW}_warmup${PENALTY_WARMUP}_lr${LR}_alpha${LORA_ALPHA}${FORMAT_STRING}"


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
TRAIN_BATCH_SIZE=256
PPO_MINI_BATCH_SIZE=64
PPO_MICRO_BATCH_SIZE=32
LOG_PROB_MICRO_BATCH_SIZE=32
N_GPUS=2

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
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.checkpoint.save_contents="['optimizer', 'extra']" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1\
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE} \
  actor_rollout_ref.rollout.load_format="safetensors" \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE} \
  actor_rollout_ref.ref.strategy=fsdp2 \
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
  trainer.rollout_data_dir=/workspace/training_logs/${EXP}/generation \
  trainer.validation_data_dir=/workspace/training_logs/${EXP}/validation \

#  actor_rollout_ref.rollout.max_num_seqs=2560 \
#  actor_rollout_ref.rollout.max_model_len=1536 \
#  actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
