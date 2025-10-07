#!/usr/bin/env bash
set -euo pipefail
set -x

DATA_DIR=/workspace/data/gsm8k
ACTOR=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
CRITIC=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
PROJECT=verl_gsm8k_hotroll
EXP=r1qwen7b_temp13_topp95

python3 -m verl.trainer.main_ppo \
  data.train_files=${DATA_DIR}/train.parquet \
  data.val_files=${DATA_DIR}/test.parquet \
  data.train_batch_size=256 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path=${ACTOR} \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.1 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.val_kwargs={do_sample:false} \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  critic.model.path=${CRITIC} \
  critic.optim.lr=1e-5 \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.ppo_mini_batch_size=64 \
  critic.model.enable_gradient_checkpointing=True \
  critic.model.fsdp_config.param_offload=False \
  algorithm.kl_ctrl.type=fixed \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger=[console,wandb] \
  trainer.project_name=${PROJECT} \
  trainer.experiment_name=${EXP} \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.total_epochs=8
