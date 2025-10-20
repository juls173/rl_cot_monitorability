#!/usr/bin/env bash
set -euo pipefail
set -x


python3 - << 'PY'
import os, wandb
wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)
print("SDK:", wandb.__version__)
print("Project:", os.getenv("WANDB_PROJECT"))
# Let entity default to your logged-in username
run = wandb.init(project=os.getenv("WANDB_PROJECT"))
wandb.log({"_ping": 1})
print("Run URL:", run.url)
run.finish()
PY

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=0

NUM_BUDGET_COPIES=2
LR=5e-4
LORA_RANK=32
LORA_ALPHA=64
EPOCHS=1
DATA_DIR=/workspace/data/gsm8k_${NUM_BUDGET_COPIES}
REWARD_FN_PATH=/workspace/rl_cot_monitorability/scripts/gsm8k_reward_length.py
REWARD_FN_NAME=compute_score
ACTOR=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
PROJECT=verl_gsm8k_length
EXP="25_10_17_r1qwen15b_grpo_${NUM_BUDGET_COPIES}_copies_lr${LR}_alpha${LORA_ALPHA}_exp2"

# 1x H100 80GB
# TRAIN_BATCH_SIZE=256
# PPO_MINI_BATCH_SIZE=64
# MICRO_BATCH_SIZE=8
# N_GPUS=1

# 1x H200 140GB
TRAIN_BATCH_SIZE=512
PPO_MINI_BATCH_SIZE=256
MICRO_BATCH_SIZE=32
N_GPUS=1

# Export configuration for length-aware reward function
export LENGTH_REWARD=1
export LENGTH_EXPONENT=2
export TOKENIZER_MODEL_NAME=${ACTOR}

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files=${DATA_DIR}/train.parquet \
  data.val_files=${DATA_DIR}/test.parquet \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.max_prompt_length=768 \
  data.max_response_length=768 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=${ACTOR} \
  actor_rollout_ref.model.lora_rank=${LORA_RANK} \
  actor_rollout_ref.model.lora_alpha=${LORA_ALPHA} \
  actor_rollout_ref.model.target_modules="all-linear" \
  actor_rollout_ref.actor.optim.lr=${LR} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1\
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
  actor_rollout_ref.rollout.load_format="safetensors" \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.max_num_seqs=5120 \
  actor_rollout_ref.rollout.max_model_len=1536 \
  actor_rollout_ref.rollout.max_num_batched_tokens=131072 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
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
  trainer.total_epochs=${EPOCHS}

