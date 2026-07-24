#!/usr/bin/env bash
set -euo pipefail

# Example:
#   DATASET=musique CORPUS_PATH=data/MuSiQue/musique_corpus_chunks.jsonl \
#   QNS_PATH=data/MuSiQue/musique_questions_chunks.jsonl \
#   OUTPUT=data/musique_reflections.jsonl LIMIT=100 \
#   bash scripts/generate_reflections.sh
#
# DATASET is narrativeqa or musique -- both need CORPUS_PATH + QNS_PATH, two explicit
# file paths matching MEMO's own --corpus_path/--qns_path convention exactly (not a
# directory with an implied filename). Both must already contain the chunked
# corpus/questions JSONL produced by python -m sigma.data_process.process_narrativeqa
# or python -m sigma.data_process.process_musique -- run those first.
# MODE=prompt (default) just exports stage-1 prompts, no LLM calls; MODE=openai runs the
# full MEMO-aligned pipeline against the OpenAI API; MODE=hf runs it against a local,
# open-source instruction-tuned model instead (MODEL default: Qwen/Qwen2.5-7B-Instruct).
#
#   MODE=hf MODEL=Qwen/Qwen2.5-7B-Instruct DTYPE=bf16 bash scripts/generate_reflections.sh

DATASET=${DATASET:-musique}
OUTPUT=${OUTPUT:-data/${DATASET}_reflections.jsonl}
MODE=${MODE:-openai}
LIMIT=${LIMIT:-100}
DTYPE=${DTYPE:-auto}

EXTRA_ARGS=(
    --corpus_path "${CORPUS_PATH:?Set CORPUS_PATH to the chunked corpus JSONL}"
    --qns_path "${QNS_PATH:?Set QNS_PATH to the chunked questions JSONL}"
)

if [ -n "${MODEL:-}" ]; then
    EXTRA_ARGS+=(--model "${MODEL}")
fi

if [ "${MODE}" = "hf" ]; then
    EXTRA_ARGS+=(--dtype "${DTYPE}")
fi

python -m sigma.reflections \
    --dataset "${DATASET}" \
    --mode "${MODE}" \
    --output "${OUTPUT}" \
    --limit "${LIMIT}" \
    "${EXTRA_ARGS[@]}"
