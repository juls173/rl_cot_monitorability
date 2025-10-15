#!/usr/bin/env bash
set -euo pipefail
set -x

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=0

DATA_DIR=/workspace/data/gsm8k
REWARD_FN_PATH=/workspace/rl_cot_monitorability/scripts/gsm8k_reward.py
REWARD_FN_NAME=compute_score
ACTOR=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
PROJECT=verl_gsm8k_v2
EXP=r1qwen15b_grpo_temp09_topp95_entropy001

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files=${DATA_DIR}/train.parquet \
  data.val_files=${DATA_DIR}/test.parquet \
  data.train_batch_size=64 \
  data.max_prompt_length=180 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=${ACTOR} \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules="all-linear" \
  actor_rollout_ref.actor.optim.lr=8e-5 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.01 \
  actor_rollout_ref.model.enable_gradient_checkpointing=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.temperature=0.9 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.93 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.load_format="safetensors" \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.max_num_seqs=100 \
  actor_rollout_ref.rollout.max_model_len=1204 \
  actor_rollout_ref.rollout.max_num_batched_tokens=65000 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  custom_reward_function.path=${REWARD_FN_PATH} \
  custom_reward_function.name=${REWARD_FN_NAME} \
  trainer.critic_warmup=0 \
  trainer.logger=[console,wandb] \
  trainer.project_name=${PROJECT} \
  trainer.experiment_name=${EXP} \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=42 \
  trainer.test_freq=14 \
  trainer.total_epochs=1 \
