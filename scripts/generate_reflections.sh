#!/usr/bin/env bash
set -euo pipefail

# Example:
#   DATASET=musique OUTPUT=data/musique_reflections.jsonl LIMIT=100 \
#   bash scripts/generate_reflections.sh

DATASET=${DATASET:-hotpotqa}
OUTPUT=${OUTPUT:-data/${DATASET}_reflections.jsonl}
MODE=${MODE:-openai}
LIMIT=${LIMIT:-100}

python generate_reflections.py \
    --dataset "${DATASET}" \
    --mode "${MODE}" \
    --output "${OUTPUT}" \
    --limit "${LIMIT}"
