#!/usr/bin/env bash
set -euo pipefail

REFLECTIONS=${REFLECTIONS:-data/hotpotqa_reflections.jsonl}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B}
OUTPUT_DIR=${OUTPUT_DIR:-runs/bootstrap}
NUM_ADAPTERS=${NUM_ADAPTERS:-8}
LORA_RANK=${LORA_RANK:-8}

sigma-bootstrap \
    --reflections_path "${REFLECTIONS}" \
    --model_name_or_path "${MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_adapters "${NUM_ADAPTERS}" \
    --lora_rank "${LORA_RANK}"
