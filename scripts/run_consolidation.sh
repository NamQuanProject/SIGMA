#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAP_DIR=${BOOTSTRAP_DIR:-runs/bootstrap}
REFLECTIONS=${REFLECTIONS:-data/musique_reflections.jsonl}
OUTPUT_PATH=${OUTPUT_PATH:-runs/memory_entry.pt}
METHOD=${METHOD:-pca}

sigma-consolidate \
    --bootstrap_dir "${BOOTSTRAP_DIR}" \
    --reflections_path "${REFLECTIONS}" \
    --output_path "${OUTPUT_PATH}" \
    --consolidation_method "${METHOD}"
