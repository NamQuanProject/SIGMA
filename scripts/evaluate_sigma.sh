#!/usr/bin/env bash
set -euo pipefail

# Example:
#   DATASET=musique CORPUS_PATH=data/MuSiQue/dev/musique_corpus_chunks.jsonl \
#   QNS_PATH=data/MuSiQue/dev/musique_questions_chunks.jsonl \
#   bash scripts/evaluate_sigma.sh

MEMORY_ENTRY=${MEMORY_ENTRY:-runs/memory_entry.pt}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B}
DATASET=${DATASET:-musique}
LIMIT=${LIMIT:-200}

python -m sigma.evaluate_sigma \
    --memory_entry_path "${MEMORY_ENTRY}" \
    --model_name_or_path "${MODEL}" \
    --dataset "${DATASET}" \
    --corpus_path "${CORPUS_PATH:?Set CORPUS_PATH to the chunked corpus JSONL}" \
    --qns_path "${QNS_PATH:?Set QNS_PATH to the chunked questions JSONL}" \
    --limit "${LIMIT}"
