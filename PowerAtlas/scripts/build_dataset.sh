#!/usr/bin/env bash
# Build ECBench from RTS-GMLC. Needs gurobipy with a licence; every instance is solved to
# optimality and the run is resumable, so it can be interrupted and restarted.
set -e

export RTS_ROOT=${RTS_ROOT:-./RTS-GMLC-master}
export ECBENCH_OUT=${ECBENCH_OUT:-./dataset}

python ecbench/generate_instances.py

# training views of the same instances (see each script's header for its own flags)
python ecbench/build_prompts.py
python ecbench/build_sft_targets.py
python ecbench/build_retrieval_corpus.py
python ecbench/build_knowledge.py
