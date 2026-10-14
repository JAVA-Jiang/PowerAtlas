#!/usr/bin/env bash
# Stage I  : supervised initialization on oracle decisions
# Stage II : FA-GRPO with the deterministic verifier in the loop
set -e

case "$1" in
  sft)    bash configs/sft_eccs.sh    ;;
  fagrpo) bash configs/fagrpo_eccs.sh ;;
  *)      echo "usage: bash scripts/train.sh {sft|fagrpo}"; exit 1 ;;
esac
