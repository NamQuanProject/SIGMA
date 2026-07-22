#!/usr/bin/env bash
set -euo pipefail

# Example:
#   DATASET=musique MUSIQUE_DIR=data/MuSiQue \
#   OUTPUT=data/musique_reflections.jsonl LIMIT=100 \
#   bash scripts/generate_reflections.sh
#
# DATASET=narrativeqa uses NARRATIVEQA_DIR (default data/NarrativeQA).
# DATASET=musique uses MUSIQUE_DIR (default data/MuSiQue).
# Both must already contain the chunked corpus/questions JSONL produced by
# process_narrativeqa.py / process_musique.py -- run those first.
# DATASET=hotpotqa needs neither -- it loads straight from Hugging Face.
# MODE=prompt (default) just exports stage-1 prompts, no LLM calls; MODE=openai runs
# the full MEMO-aligned pipeline for real.

DATASET=${DATASET:-hotpotqa}
OUTPUT=${OUTPUT:-data/${DATASET}_reflections.jsonl}
MODE=${MODE:-openai}
LIMIT=${LIMIT:-100}

EXTRA_ARGS=()
if [ "${DATASET}" = "narrativeqa" ]; then
    EXTRA_ARGS+=(--narrativeqa_dir "${NARRATIVEQA_DIR:-data/NarrativeQA}")
elif [ "${DATASET}" = "musique" ]; then
    EXTRA_ARGS+=(--musique_dir "${MUSIQUE_DIR:-data/MuSiQue}")
fi

python generate_reflections.py \
    --dataset "${DATASET}" \
    --mode "${MODE}" \
    --output "${OUTPUT}" \
    --limit "${LIMIT}" \
    "${EXTRA_ARGS[@]}"
