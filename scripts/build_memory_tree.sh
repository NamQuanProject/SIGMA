#!/usr/bin/env bash
set -euo pipefail

# Example (bridge/comparison pseudo-task demo -- see README):
#   TASKS="bridge=runs/bridge/memory_entry.pt comparison=runs/comparison/memory_entry.pt" \
#   OUTPUT_PATH=runs/memory_tree.pt \
#   bash scripts/build_memory_tree.sh

TASKS=${TASKS:?Set TASKS to a space-separated list of NAME=PATH pairs}
OUTPUT_PATH=${OUTPUT_PATH:-runs/memory_tree.pt}

TASK_ARGS=()
for task in ${TASKS}; do
    TASK_ARGS+=(--task "${task}")
done

python build_memory_tree.py \
    "${TASK_ARGS[@]}" \
    --output_path "${OUTPUT_PATH}"
