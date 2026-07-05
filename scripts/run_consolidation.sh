#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAP_DIR=${BOOTSTRAP_DIR:-runs/bootstrap}
REFLECTIONS=${REFLECTIONS:-data/hotpotqa_reflections.jsonl}
OUTPUT_PATH=${OUTPUT_PATH:-runs/memory_entry.pt}
METHOD=${METHOD:-pca}

python run_consolidation.py \
    --bootstrap_dir "${BOOTSTRAP_DIR}" \
    --reflections_path "${REFLECTIONS}" \
    --output_path "${OUTPUT_PATH}" \
    --consolidation_method "${METHOD}"
