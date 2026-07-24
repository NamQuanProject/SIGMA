#!/usr/bin/env bash
set -euo pipefail

# Example:
#   NARRATIVEQA_DIR=data/NarrativeQA SPLIT=train bash scripts/process_narrativeqa.sh
#
# Requires a git clone of https://github.com/google-deepmind/narrativeqa at
# NARRATIVEQA_DIR (documents.csv, qaps.csv, third_party/wikipedia/summaries.csv) -- see
# data/NarrativeQA/README.md.

NARRATIVEQA_DIR=${NARRATIVEQA_DIR:-data/NarrativeQA}
SPLIT=${SPLIT:-train}
CHUNK_SIZE=${CHUNK_SIZE:-6400}
OVERLAP=${OVERLAP:-640}

python -m sigma.data_process.process_narrativeqa \
    --narrativeqa_dir "${NARRATIVEQA_DIR}" \
    --split "${SPLIT}" \
    --chunk_size "${CHUNK_SIZE}" \
    --overlap "${OVERLAP}"
