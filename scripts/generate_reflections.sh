#!/usr/bin/env bash
set -euo pipefail

# Example:
#   DATASET=musique MUSIQUE_PATH=data/MuSiQue/musique_ans_v1.0_train.jsonl \
#   OUTPUT=data/musique_reflections.jsonl LIMIT=100 \
#   bash scripts/generate_reflections.sh
#
# DATASET=narrativeqa uses NARRATIVEQA_DIR (default data/NarrativeQA).
# DATASET=hotpotqa needs neither -- it loads straight from Hugging Face.
# MODE=prompt (default) just exports prompts, no LLM calls; MODE=openai actually generates.

DATASET=${DATASET:-hotpotqa}
OUTPUT=${OUTPUT:-data/${DATASET}_reflections.jsonl}
MODE=${MODE:-openai}
LIMIT=${LIMIT:-100}

EXTRA_ARGS=()
if [ "${DATASET}" = "narrativeqa" ]; then
    EXTRA_ARGS+=(--narrativeqa_dir "${NARRATIVEQA_DIR:-data/NarrativeQA}")
elif [ "${DATASET}" = "musique" ]; then
    EXTRA_ARGS+=(--musique_path "${MUSIQUE_PATH:?Set MUSIQUE_PATH to a musique JSON/JSONL file}")
fi

python generate_reflections.py \
    --dataset "${DATASET}" \
    --mode "${MODE}" \
    --output "${OUTPUT}" \
    --limit "${LIMIT}" \
    "${EXTRA_ARGS[@]}"
