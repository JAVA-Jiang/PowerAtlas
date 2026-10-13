#!/usr/bin/env bash
# Score a directory of stored answers with the grid-aware evaluator.
#   PRED=pred_run DATA=./dataset bash scripts/evaluate.sh
set -e

export RTS_ROOT=${RTS_ROOT:-./RTS-GMLC-master}
DATA=${DATA:-./dataset}
PRED=${PRED:-./pred_run}
LABEL=${LABEL:-run}

python ecbench/evaluator.py --data_dir "$DATA" --pred_dir "$PRED" --label "$LABEL"
