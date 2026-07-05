# SIGMA_DEV

Implementation of **SIGMA**, a bootstrap-and-consolidate memory mechanism for LLMs
(see `ideas/sigma proposal v1.pdf` for the full proposal), applied to **HotpotQA** as a
single task. Given a frozen backbone LLM, SIGMA trains many small LoRA adapters on
bootstrapped subsets of a reflection QA dataset, consolidates them into a compact basis
plus a coordinate generator, and at inference time synthesizes a task-specific adapter
on the fly and patches it onto the backbone for that one generation call.

> **Scope of this build:** only the *within-task* mechanism (proposal section 4.2.1) is
> implemented, for one task (HotpotQA). The cross-task Gromov-Wasserstein memory tree
> (section 4.2.2) — which is what would let SIGMA route across *many* tasks — is not
> implemented; `src/sigma/memory/single_entry.py` is a placeholder that always routes to
> the one entry, behind the same `route()` interface a real tree would expose.

## Contents

1. [Setup](#setup)
2. [Pipeline overview](#pipeline-overview)
3. [Step 0 — Generate reflection data](#step-0--generate-reflection-data)
4. [Step 1 — Bootstrap adapters](#step-1--bootstrap-adapters)
5. [Step 2 — Consolidate into a memory entry](#step-2--consolidate-into-a-memory-entry)
6. [Step 3 — Evaluate](#step-3--evaluate)
7. [Comparing against another model (including API models)](#comparing-against-another-model-including-api-models)
8. [Repo layout](#repo-layout)

## Setup

```bash
pip install -r requirements.txt
```

Needs `torch` + `transformers` + `accelerate` for the actual pipeline (steps 1–3), which
in turn need real GPU compute for anything beyond a tiny smoke test — everything is
runnable on CPU, it's just not the intended way to run it.

If you're generating reflections with OpenAI (step 0) or comparing against an OpenAI
model at evaluation time, put your key in a repo-root `.env` file:

```
OPENAI_API_KEY=sk-...
```

(A `.env/` folder layout — `.env/.env` or `.env/local.env` — also works, if you prefer
that structure.)

## Pipeline overview

```
HotpotQA  ──▶  reflection QA data (Q_final)  ──▶  M bootstrapped LoRA adapters
                                                          │
                                                          ▼
                              baseline vs. SIGMA-adapted  ◀──  one consolidated MemoryEntry
                                    EM/F1 on HotpotQA           (basis + coordinate generator)
```

Each stage is a standalone script that reads the previous stage's output from disk, so
you can inspect intermediate artifacts (`runs/bootstrap/`, `runs/memory_entry.pt`) between
steps. `scripts/*.sh` wrap the same commands with env-var overrides if you'd rather not
type full argument lists.

## Step 0 — Generate reflection data

Turns raw HotpotQA questions into the reflection QA set (`Q_final`) the rest of the
pipeline trains on. Two modes:

```bash
# Just export the prompts (no LLM calls) -- useful for inspecting/debugging the data
python generate_hotpotqa_reflections.py --output data/hotpotqa_prompts.jsonl --limit 100

# Actually generate reflections with an OpenAI model
python generate_hotpotqa_reflections.py --mode openai --output data/hotpotqa_reflections.jsonl --limit 100
```

Loads HotpotQA from Hugging Face (`hotpotqa/hotpot_qa`, with the legacy `hotpot_qa` id as
a fallback and `--dataset_name` to override), trying the `distractor` and `fullwiki`
configs in order if `--config` isn't given.

## Step 1 — Bootstrap adapters

Trains `M` LoRA adapters on bootstrapped (with-replacement) subsets of `Q_final`. All `M`
adapters share one frozen, randomly-initialized down-projection `A`; only the
up-projection `B_m` is trained per adapter (eq. 15 in the proposal).

```bash
python train_bootstrap.py \
    --reflections_path data/hotpotqa_reflections.jsonl \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --output_dir runs/bootstrap \
    --num_adapters 8 --lora_rank 8
```

`--model_name_or_path` accepts any local Hugging Face causal LM id/path — it's not
hardcoded to Qwen, that's just a small default for quick iteration.

## Step 2 — Consolidate into a memory entry

Decomposes the `M` adapters into a shared basis (PCA by default; Fisher-weighted PCA via
`--consolidation_method fisher`) and trains the coordinate generator on top of it
(eq. 16–22). Produces one `MemoryEntry` checkpoint.

```bash
python run_consolidation.py \
    --bootstrap_dir runs/bootstrap \
    --reflections_path data/hotpotqa_reflections.jsonl \
    --output_path runs/memory_entry.pt
```

## Step 3 — Evaluate

For each HotpotQA validation question: synthesizes a task-specific adapter from the
memory entry (eq. 23–24), patches it onto the frozen backbone, and compares its answer
against the same backbone with no adapter applied. Scores both with HotpotQA-style
exact-match/F1.

```bash
python evaluate_sigma.py \
    --memory_entry_path runs/memory_entry.pt \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --limit 200
```

## Comparing against another model (including API models)

Bootstrap training and consolidation (steps 1–2) need direct access to weights and
hidden states — to attach LoRA and read context embeddings — so they can only ever run
against a local model. Evaluation (step 3), though, can pull in a **third** set of
predictions purely as a comparison point, via `--baseline_model`:

```bash
# Compare SIGMA-adapted Qwen2.5-0.5B against GPT-4o-mini as an external reference point
python evaluate_sigma.py \
    --memory_entry_path runs/memory_entry.pt \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --baseline_model openai:gpt-4o-mini \
    --limit 200
```

`--baseline_model` takes a `"<provider>:<name>"` spec (`src/sigma/backends/`):

- `openai:<model>` — OpenAI chat completions (needs `OPENAI_API_KEY`, see [Setup](#setup)).
- `hf:<path>` or a bare path — another local Hugging Face model.

This is a comparison point only — it never has the memory attached, since that requires
weights we control. Adding another API provider means adding one backend class in
`src/sigma/backends/` and one branch in `build_backend()`; nothing else in the pipeline
changes, since every backend just exposes `generate(question) -> str`.

## Repo layout

```
generate_hotpotqa_reflections.py   \
train_bootstrap.py                  |  thin root-level CLI wrappers around src/sigma/*
run_consolidation.py                |  (see each file's docstring for what it does)
evaluate_sigma.py                  /

src/sigma/
├── hotpotqa_reflections.py    # step 0: load HotpotQA, build reflection prompts/records
├── reflection_dataset.py      # Q_final loading, bootstrap sampling, answer-masked tokenization
├── adapters/shared_lora.py    # SharedLoRALinear: frozen shared A, trainable per-adapter B
├── train_bootstrap.py         # step 1
├── consolidate/
│   ├── pca.py                 # PCA / Fisher-weighted PCA consolidation (eq. 16-20)
│   └── generator.py           # coordinate generator (eq. 21-22)
├── run_consolidation.py       # step 2
├── memory/
│   ├── entry.py                # MemoryEntry: basis + generator, synthesize_adapter() (eq. 23-24)
│   ├── apply.py                 # patch/unpatch a synthesized adapter onto a live model
│   └── single_entry.py          # single-task stand-in for the (unimplemented) memory tree
├── evaluate_sigma.py          # step 3
├── backends/                  # pluggable comparison backends (local HF model, OpenAI, ...)
└── utils/
    ├── context_embedding.py
    ├── metrics.py              # HotpotQA-style EM/F1
    └── env.py                  # shared .env loading

scripts/*.sh   # env-var-configurable wrappers around the 3 pipeline steps
ideas/         # the SIGMA proposal PDF this implementation follows (gitignored)
MemoryDecoder/ # reference repo this codebase's structure/CLI style is modeled on (gitignored)
```
