#!/usr/bin/env bash
set -euo pipefail

# Example:
#   MUSIQUE_PATH=data/MuSiQue/musique_ans_v1.0_train.jsonl \
#   OUTPUT_DIR=data/MuSiQue bash scripts/process_musique.sh
#
# MUSIQUE_PATH must point at a raw MuSiQue JSON/JSONL file (see
# data/MuSiQue/README.md for the download link).

MUSIQUE_PATH=${MUSIQUE_PATH:?Set MUSIQUE_PATH to a raw MuSiQue JSON/JSONL file}
OUTPUT_DIR=${OUTPUT_DIR:-data/MuSiQue}
CHUNK_SIZE=${CHUNK_SIZE:-6400}
OVERLAP=${OVERLAP:-640}

python -m sigma.data_process.process_musique \
    --musique_path "${MUSIQUE_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --chunk_size "${CHUNK_SIZE}" \
    --overlap "${OVERLAP}"
