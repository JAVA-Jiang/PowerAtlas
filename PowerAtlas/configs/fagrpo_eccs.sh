#!/bin/bash
# ============================================================
# ECCS 纯文本训练脚本（Qwen3-4B-Instruct + 本地知识库 RAG + GRPO）
# 与 train.sh 的关键区别：
#   1) 删掉了 data.user_prompt_round_1 / after_image_search / after_text_search 三行
#      —— user_prompt_round_1 缺省即为 null，dataset.py 走"纯文本、不加占位图"分支。
#   2) search.websearch=False（仅本地 kg_server 检索，不联网）。
#   3) 模型换成 Qwen3-4B-Instruct（纯文本，无视觉）。
# 先决条件：kg_server(8101) + mem_server(8102) 已起（bash scripts/serve.sh --kg_dir ...），
#           且 eccs_train.parquet / eccs_test.parquet 已在服务器上用 build_eccs_parquet.py 生成。
# ============================================================
while [[ $# -gt 0 ]]; do
  case $1 in
    --train_data)  TRAIN_DATA="$2"; shift 2 ;;
    --val_data)    VAL_DATA="$2"; shift 2 ;;
    --model_path)  MODEL_PATH="$2"; shift 2 ;;
    --gpus)        GPUS="$2"; shift 2 ;;
    --wandb_project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb_exp)   WANDB_EXP="$2"; shift 2 ;;
    --cuda_devices)  CUDA_DEVICES="$2"; shift 2 ;;
    --ray_port)      RAY_PORT="$2"; shift 2 ;;
    --mem_port)      MEM_PORT="$2"; shift 2 ;;
    --memory_dir)    MEMORY_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$TRAIN_DATA" || -z "$VAL_DATA" || -z "$MODEL_PATH" || -z "$GPUS" || -z "$WANDB_PROJECT" || -z "$WANDB_EXP" ]]; then
  echo "Usage: bash $0 --train_data PATH --val_data PATH --model_path Qwen3-4B --gpus N --wandb_project NAME --wandb_exp NAME"
  exit 1
fi

CUDA_DEVICES=${CUDA_DEVICES:-"0"}
RAY_PORT=${RAY_PORT:-6379}
MEM_PORT=${MEM_PORT:-8102}
MEMORY_DIR=${MEMORY_DIR:-./expr/MEMORY}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-./checkpoints}
KG_PORT=8101

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export HYDRA_FULL_ERROR=1
export MMGRAPH_MEMORY_DIR=$MEMORY_DIR
# [讲解] 单机单卡不预起 Ray 集群、也不设 RAY_ADDRESS，让 ray.init() 自己起本地集群，
# [讲解]   避免 "Failed to connect to Ray cluster / No node info" 的地址不匹配问题。

check_servers() {
  local ok=1
  for port in $KG_PORT $MEM_PORT; do
    if curl -s --max-time 5 http://localhost:$port > /dev/null 2>&1; then
      echo "[OK] Port $port is running"
    else
      echo "[ERROR] Port $port is NOT running!"; ok=0
    fi
  done
  [[ $ok -eq 0 ]] && { echo "[ABORT] Start API servers first (scripts/serve.sh)"; exit 1; }
}
check_servers

python3 -m mmgraph_r1.trainer.main \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=8 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.image_key=images \
    data.shuffle=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_multi_turn_response_mask=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_multiturn_mmgraph \
    actor_rollout_ref.rollout.limit_images=0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.max_gen_round=8 \
    actor_rollout_ref.rollout.response_length_total=4096 \
    actor_rollout_ref.rollout.search.topk=5 \
    actor_rollout_ref.rollout.search.websearch=False \
    actor_rollout_ref.rollout.search.mem_port=$MEM_PORT \
    actor_rollout_ref.rollout.search.image_search_limit=1 \
    actor_rollout_ref.rollout.search.text_search_limit=2 \
    actor_rollout_ref.rollout.search.parallel_tool_call=True \
    actor_rollout_ref.rollout.search.parallel_tool_call_threads=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$WANDB_EXP \
    trainer.n_gpus_per_node=$GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=75 \
    trainer.test_freq=125 \
    trainer.total_epochs=1 \
    +trainer.search_penalty=0 \
    +trainer.format_penalty=0.1 \
    +trainer.reward_mode="ECCS" \
    +trainer.val_before_train=True \
    +algorithm.filter_groups.enable=False \
    trainer.default_local_dir=${CHECKPOINT_DIR}/${WANDB_PROJECT}/${WANDB_EXP} \
    trainer.resume_mode=${RESUME_MODE:-disable} \
    trainer.resume_from_path=${RESUME_FROM_PATH:-False}
