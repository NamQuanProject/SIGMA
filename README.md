# SIGMA_DEV

## HotpotQA reflection generation

Generate prompt records:

```bash
python generate_hotpotqa_reflections.py --output data/hotpotqa_prompts.jsonl --limit 100
```

Generate reflections with OpenAI:

```bash
python generate_hotpotqa_reflections.py --mode openai --output data/hotpotqa_reflections.jsonl --limit 100
```

Put `OPENAI_API_KEY=...` in the repo root `.env` file, or in `.env/.env` if you prefer a `.env` folder layout.

The script loads HotpotQA from Hugging Face using the `hotpot_qa` dataset name and
tries the common `distractor` and `fullwiki` configs if no config is provided.

## SIGMA bootstrap-and-consolidate (single task: HotpotQA)

Implements section 4.2.1 of `ideas/sigma proposal v1.pdf`: bootstrap many shared-frozen-A
LoRA adapters on the reflection QA set, consolidate them via PCA (or Fisher-weighted PCA)
into a compact basis, train a coordinate generator, and evaluate the synthesized adapter
against the unmodified backbone. The cross-task Gromov-Wasserstein memory tree (section
4.2.2) is out of scope for this single-task build; `src/sigma/memory/single_entry.py`
stands in as a trivial one-leaf "tree" behind the same `route()` interface.

Requires `torch`, `transformers`, `accelerate` (see `requirements.txt`) and real training
compute (GPU); everything below is CPU-runnable for small smoke tests but sized for GPU.

```bash
# 1. Bootstrap M LoRA adapters (shared frozen A, per-adapter trainable B)
python train_bootstrap.py \
    --reflections_path data/hotpotqa_reflections.jsonl \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --output_dir runs/bootstrap \
    --num_adapters 8 --lora_rank 8

# 2. Consolidate into one MemoryEntry (PCA by default; --consolidation_method fisher for
#    the Fisher-weighted variant)
python run_consolidation.py \
    --bootstrap_dir runs/bootstrap \
    --reflections_path data/hotpotqa_reflections.jsonl \
    --output_path runs/memory_entry.pt

# 3. Evaluate the synthesized adapter vs. the unmodified backbone on HotpotQA
python evaluate_sigma.py \
    --memory_entry_path runs/memory_entry.pt \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --limit 200
```

Equivalent `scripts/bootstrap_adapters.sh`, `scripts/run_consolidation.sh`,
`scripts/evaluate_sigma.sh` wrappers (env-var configurable) are provided, mirroring
`MemoryDecoder/scripts/*.sh`.
