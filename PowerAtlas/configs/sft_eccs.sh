#!/bin/bash
# =====================================================================================
# ECCS v2 —— SFT 预热(GRPO 之前的第一步)
# 目的:用最优 dispatch 标签,教 Qwen3-1.7B 学会输出【合法的 TASKS/POWER 联合计划】,
#       破解 120 维连续 dispatch 的 GRPO 冷启动(否则组内 reward 无方差、学不动)。
# 损失:公式(4) 逐 token 交叉熵(NLL),只在 response 上算(prompt 被 mask)。
# 产出:HF 格式 checkpoint -> 作为 GRPO 的 model.path 起点。
#
# 在服务器(A800 + Qwen 权重)上跑。从 MMGraph_Agent_code/ 目录运行,确保 verl 可 import、
# 且 sft_train.parquet / sft_val.parquet 在可访问路径。
#   bash scripts/sft_eccs.sh
# =====================================================================================
set -e

# ---------- 按你的环境填 ----------
MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/Qwen3-4B-Instruct}   # 基座权重(2×A800 全参)
TRAIN=${TRAIN:-sft_train.parquet}                        # 由 build_eccs_sft.py 产出
VAL=${VAL:-sft_val.parquet}
OUT=${OUT:-checkpoints/eccs-sft/qwen4b}                   # SFT checkpoint 落地目录
NGPU=${NGPU:-2}                                           # 2×A800:4B 全参 FSDP 分片,每卡~32GB
# ----------------------------------

torchrun --standalone --nproc_per_node="$NGPU" -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$TRAIN" \
  data.val_files="$VAL" \
  data.prompt_key=question \
  data.response_key=response \
  data.max_length=4096 \
  data.truncation=error \
  data.train_batch_size=16 \
  data.micro_batch_size_per_gpu=4 \
  data.balance_dp_token=False \
  model.partial_pretrain="$MODEL_PATH" \
  model.lora_rank=0 \
  model.trust_remote_code=True \
  model.strategy=fsdp \
  model.enable_gradient_checkpointing=True \
  optim.lr=1e-5 \
  optim.warmup_steps_ratio=0.1 \
  ulysses_sequence_parallel_size=1 \
  use_remove_padding=False \
  trainer.total_epochs=8 \
  trainer.n_gpus_per_node="$NGPU" \
  trainer.nnodes=1 \
  trainer.project_name=eccs-sft \
  trainer.experiment_name=qwen4b \
  trainer.logger="[console]" \
  trainer.save_freq=-1 \
  trainer.test_freq=9 \
  trainer.default_local_dir="$OUT"

echo ""
echo "SFT 完成。checkpoint 在: $OUT/global_step_<最后一步>/"
echo "下一步 GRPO:把 train_eccs.sh 的 --model_path 指向这个 checkpoint 即可。"
