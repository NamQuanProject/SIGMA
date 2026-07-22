#!/usr/bin/env bash
set -euo pipefail

# Example:
#   DATASET=musique MUSIQUE_PATH=data/MuSiQue/musique_ans_v1.0_train.jsonl \
#   OUTPUT=data/musique_memo_reflections.jsonl LIMIT=50 \
#   bash scripts/generate_memo_reflections.sh

DATASET=${DATASET:-hotpotqa}
OUTPUT=${OUTPUT:-data/${DATASET}_memo_reflections.jsonl}
LIMIT=${LIMIT:-50}

EXTRA_ARGS=()
if [ "${DATASET}" = "narrativeqa" ]; then
    EXTRA_ARGS+=(--narrativeqa_dir "${NARRATIVEQA_DIR:-data/NarrativeQA}")
elif [ "${DATASET}" = "musique" ]; then
    EXTRA_ARGS+=(--musique_path "${MUSIQUE_PATH:?Set MUSIQUE_PATH to a musique JSON/JSONL file}")
fi

python generate_memo_reflections.py \
    --dataset "${DATASET}" \
    --output "${OUTPUT}" \
    --limit "${LIMIT}" \
    "${EXTRA_ARGS[@]}"
