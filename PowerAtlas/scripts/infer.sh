#!/usr/bin/env bash
# Greedy single-pass inference with a local checkpoint; no solver is called.
#   CKPT=checkpoints/... PARQUET=parquet/test.parquet bash scripts/infer.sh
set -e

CKPT=${CKPT:?set CKPT to a HF checkpoint directory}
PARQUET=${PARQUET:-./parquet/test.parquet}
PRED=${PRED:-./pred_run}

python poweratlas/infer_local.py --model "$CKPT" --parquet "$PARQUET" --out_dir "$PRED"
