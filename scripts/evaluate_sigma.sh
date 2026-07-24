#!/usr/bin/env bash
set -euo pipefail

MEMORY_ENTRY=${MEMORY_ENTRY:-runs/memory_entry.pt}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B}
LIMIT=${LIMIT:-200}

sigma-evaluate \
    --memory_entry_path "${MEMORY_ENTRY}" \
    --model_name_or_path "${MODEL}" \
    --limit "${LIMIT}"
