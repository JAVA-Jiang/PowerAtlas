#!/usr/bin/env bash
# Frontier baselines through any OpenAI-compatible endpoint. The key is read from the environment.
set -e

export API_KEY=${API_KEY:?set API_KEY first}
BASE_URL=${BASE_URL:-https://api.deepseek.com}
MODEL=${MODEL:-deepseek-chat}
PARQUET=${PARQUET:-./parquet/test.parquet}
PRED=${PRED:-./pred_${MODEL}}

python baselines/query_openai_compatible.py \
    --model "$MODEL" --base_url "$BASE_URL" --parquet "$PARQUET" --out_dir "$PRED"

DATA=${DATA:-./dataset} PRED="$PRED" LABEL="$MODEL" bash scripts/evaluate.sh
