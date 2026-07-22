# SIGMA_DEV

Implementation of **SIGMA**, a bootstrap-and-consolidate memory mechanism for LLMs
(see `ideas/sigma proposal v1.pdf` for the full proposal), applied to **HotpotQA**. Given
a frozen backbone LLM, SIGMA trains many small LoRA adapters on bootstrapped subsets of a
reflection QA dataset, consolidates them into a compact basis plus a coordinate
generator, and at inference time synthesizes a task-specific adapter on the fly and
patches it onto the backbone for that one generation call.

Both halves of the proposal are implemented: within-task bootstrap-and-consolidate
(section 4.2.1, steps 0–3 below) and the cross-task memory tree (section 4.2.2, step 4)
that organizes multiple tasks' signatures via Gromov-Wasserstein distance for O(log n)
routing. HotpotQA is still the only real dataset used, but its built-in question-type
metadata (bridge/comparison) doubles as two pseudo-tasks so the tree has more than one
leaf to route between — see step 4.

> **Honesty note on the tree:** the proposal is itself schematic about two things —
> eq. 26's GW-distance formula is written as a proportionality ("≍ ..."), not a closed
> form, and the "growth control" merge policy is described only qualitatively (merge
> when confusable, bound retrieval error) with no concrete algorithm given. `memory/gw.py`
> and `memory/tree.py` fill both gaps with documented, defensible choices — not a
> reproduction of a specific published formula or a proven error bound.

## Contents

1. [Setup](#setup)
2. [Pipeline overview](#pipeline-overview)
3. [Step 0 — Generate reflection data](#step-0--generate-reflection-data)
4. [Step 1 — Bootstrap adapters](#step-1--bootstrap-adapters)
5. [Step 2 — Consolidate into a memory entry](#step-2--consolidate-into-a-memory-entry)
6. [Step 3 — Evaluate a single task](#step-3--evaluate-a-single-task)
7. [Step 4 — Multi-task memory tree](#step-4--multi-task-memory-tree)
8. [Comparing against another model (including API models)](#comparing-against-another-model-including-api-models)
9. [Repo layout](#repo-layout)

## Setup

```bash
pip install -r requirements.txt
```

Needs `torch` + `transformers` + `accelerate` for the actual pipeline (steps 1–4), which
in turn need real GPU compute for anything beyond a tiny smoke test — everything is
runnable on CPU, it's just not the intended way to run it.

If you're generating reflections with OpenAI (step 0) or comparing against an OpenAI
model at evaluation time, put your key in a repo-root `.env` file:

```
OPENAI_API_KEY=sk-...
```

(A `.env/` folder layout — `.env/.env` or `.env/local.env` — also works, if you prefer
that structure.)

Every script in the pipeline (steps 0–4) logs to stdout *and* to a timestamped file
under `logs/` (`logs/<script>_<timestamp>.log`, e.g. `logs/train_bootstrap_20260706_142301.log`),
via `--log_dir` (default `logs`) and the shared `src/sigma/utils/logging_setup.py`
helper. `logs/`, like `runs/`, is gitignored — nothing under it is meant to be committed.

## Pipeline overview

Single task (steps 0-3):

```
HotpotQA  ──▶  reflection QA data (Q_final)  ──▶  M bootstrapped LoRA adapters
                                                          │
                                                          ▼
                              baseline vs. SIGMA-adapted  ◀──  one consolidated MemoryEntry
                                    EM/F1 on HotpotQA           (basis + coordinate generator)
```

Multi-task (step 4 adds a layer on top): run steps 0–2 once per task (each with its own
`--question_type`/`--level` filter and output path) to get N `MemoryEntry` checkpoints,
then `build_memory_tree.py` organizes them into one `MemoryTree` that
`evaluate_sigma.py --memory_tree_path` routes queries through.

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
configs in order if `--config` isn't given. Each record also carries HotpotQA's own
`type` (bridge/comparison) and `level` (easy/medium/hard) metadata, used in step 4 to
carve pseudo-tasks out of one reflections file.

### Other datasets: NarrativeQA, MuSiQue

`generate_hotpotqa_reflections.py` stays HotpotQA-only (unchanged — `evaluate_sigma.py`
still uses its loader for the HotpotQA eval path). `generate_reflections.py` is the same
pipeline generalized to any dataset in `src/sigma/data_sources/`, currently HotpotQA,
**NarrativeQA**, and **MuSiQue** — all normalized to the same schema, so the reflections
JSONL and everything downstream (`reflection_dataset.py`, `train_bootstrap.py`,
`run_consolidation.py`) works identically regardless of source dataset:

```bash
python generate_reflections.py --dataset musique --mode openai \
    --output data/musique_reflections.jsonl --limit 100

python generate_reflections.py --dataset narrativeqa --mode openai \
    --output data/narrativeqa_reflections.jsonl --limit 100
```

Neither NarrativeQA nor MuSiQue is reliably published on Hugging Face, so — matching how
MEMO's own repo actually consumes them — both load from **local files** instead of
`datasets.load_dataset`:

**NarrativeQA** — `--narrativeqa_dir` points at a local checkout of the official repo:

```bash
git clone https://github.com/google-deepmind/narrativeqa

python generate_reflections.py --dataset narrativeqa --mode openai \
    --narrativeqa_dir narrativeqa --output data/narrativeqa_reflections.jsonl --limit 100
```

That clone gives you `documents.csv`, `qaps.csv`, and `third_party/wikipedia/summaries.csv`
directly — no extra download step needed for those three files. We use each story's
Wikipedia plot **summary** as context (not the full book/script text — questions are
written to be answerable from the summary, and full narrative text is often 50k+ words,
impractical for one reflection prompt), so you do **not** need to run the repo's separate
`download_stories.sh`.

**MuSiQue** — `--musique_path` points at a local JSON/JSONL file:

```bash
# See https://github.com/StonyBrookNLP/musique for the actual dataset download link
# (a Google Drive zip at time of writing) -- grab musique_ans_v1.0_train.jsonl or similar.

python generate_reflections.py --dataset musique --mode openai \
    --musique_path musique_ans_v1.0_train.jsonl --output data/musique_reflections.jsonl --limit 100
```

MuSiQue's paragraphs already carry `is_supporting` flags matching HotpotQA's
supporting-facts convention. The loader accepts either the official one-JSON-object-per-line
format or a single JSON array/object, auto-detected.

I can't browse the web from this environment to re-verify those two GitHub URLs still
resolve or that the download links on their READMEs haven't moved — if either 404s,
search for "NarrativeQA deepmind github" / "MuSiQue StonyBrookNLP github" respectively.
Missing `--narrativeqa_dir`/`--musique_path`, or a directory missing the expected files,
raises a clear error telling you exactly what's needed rather than failing silently.

The reflection prompt itself (`reflections.py`) asks for the same six fields as before,
sharpened to match MEMO's actual five-stage synthesis more closely — direct vs. indirect
facts, entity surfacing defined as "describe the entity without naming it," cross-document
synthesis defined as combining evidence across context blocks — but still one LLM call
per example, not MEMO's full multi-call pipeline (their fact-extraction step alone is two
separate call-types, and their verification step is an iterative check/fix loop up to 12
calls per QA pair — a lot more expensive than SIGMA's method needs).

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

**Building one task of several** (for [step 4](#step-4--multi-task-memory-tree)): add
`--question_type bridge` (or `comparison`) and/or `--level easy|medium|hard` to train on
only a filtered slice of `Q_final`, and run this step once per task into a different
`--output_dir` each time. The filter gets saved into `bootstrap_meta.json` and read back
automatically in step 2, so you never have to repeat it.

**No API version.** This step trains LoRA weights directly on the backbone's own
parameters and needs gradients through it, so it can only ever run against a local model
you have the weights for — there's no equivalent for an API model like GPT-4o-mini,
since the API doesn't expose weights, gradients, or a way to attach an adapter.

## Step 2 — Consolidate into a memory entry

Decomposes the `M` adapters into a shared basis (PCA by default; Fisher-weighted PCA via
`--consolidation_method fisher`) and trains the coordinate generator on top of it
(eq. 16–22). Also fits a task **signature** (a shrunk-diagonal Gaussian over the same
per-subset context embeddings) and keeps the raw `(context, alpha)` training pairs —
both unused by the single-task path, but required if this entry later goes into a
`MemoryTree` (step 4). Produces one `MemoryEntry` checkpoint.

```bash
python run_consolidation.py \
    --bootstrap_dir runs/bootstrap \
    --reflections_path data/hotpotqa_reflections.jsonl \
    --output_path runs/memory_entry.pt
```

**No API version**, for the same reason as step 1 — consolidation reads hidden states
and gradients off the local backbone (to compute context embeddings and, for
`--consolidation_method fisher`, a diagonal Fisher estimate), which an API model can't
provide.

## Step 3 — Evaluate a single task

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

## Step 4 — Multi-task memory tree

Once you have **two or more** `MemoryEntry` checkpoints (e.g. one for `--question_type
bridge`, one for `comparison` — see step 1), organize them into a tree and route through
it instead of a single fixed entry:

```bash
# Build the tree from N named tasks (pass --task once per task)
python build_memory_tree.py \
    --task bridge=runs/bridge/memory_entry.pt \
    --task comparison=runs/comparison/memory_entry.pt \
    --output_path runs/memory_tree.pt

# Evaluate through the tree instead of a single entry
python evaluate_sigma.py \
    --memory_tree_path runs/memory_tree.pt \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --limit 200
```

What happens under the hood (`src/sigma/memory/{signature,gw,tree}.py`):

- Each task's **signature** is a diagonal Gaussian fit from its own per-subset context
  embeddings (step 2), shrunk toward the average variance for stability with few samples.
- Tasks are organized into a binary tree by bottom-up clustering on **Gromov-Wasserstein
  distance** between signatures' sorted variance spectra — GW distance rather than plain
  Wasserstein because each task's embeddings live in a different, differently-shaped
  space (its own consolidated adapter), so only the *shape* of the spectrum is
  comparable, not raw coordinates.
- **Routing** descends the tree, at each internal node comparing the query's embedding
  (recomputed under each candidate branch's representative task) via own-space
  Mahalanobis distance, until it reaches a leaf — the exact formula the proposal
  specifies for the final step (eq. 28), just applied at every level on the way down.
- `MemoryTree.consolidate_confusable(threshold)` finds sibling tasks whose GW distance
  is below `threshold` and merges them (concatenates their steering bases, retrains one
  generator on the pooled training pairs) — the "growth control" mechanism, not a proven
  error bound (see the honesty note at the top).

All tasks in one tree must share `--lora_rank` and `--target_modules` (checked with a
clear error at evaluation time if they don't) — routing means temporarily wearing a
candidate task's own frozen `A` to compute its embedding, which only works if every
task's adapter wrapper was sized the same way to begin with.

## Comparing against another model (including API models)

Bootstrap training and consolidation (steps 1–2) need direct access to weights and
hidden states — to attach LoRA and read context embeddings — so they can only ever run
against a local model. Evaluation (steps 3–4), though, can pull in a **third** set of
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
- `hf:<path>` or a bare path — another local Hugging Face model, if you'd rather compare
  against a different/bigger local model instead of an API one.

This is a comparison point only — it never has the memory attached, since that requires
weights we control. Adding another API provider means adding one backend class in
`src/sigma/backends/` and one branch in `build_backend()`; nothing else in the pipeline
changes, since every backend just exposes `generate(question) -> str`.

## Repo layout

```
generate_hotpotqa_reflections.py   \
generate_reflections.py             |
train_bootstrap.py                  |  thin root-level CLI wrappers around src/sigma/*
run_consolidation.py                 \ (see each file's docstring for what it does)
evaluate_sigma.py                    /
build_memory_tree.py                /

src/sigma/
├── hotpotqa_reflections.py    # step 0 (HotpotQA-only path): load HotpotQA, build reflection prompts/records
├── reflections.py             # step 0 (generalized): same pipeline for any data_sources/ dataset
├── data_sources/               # normalized loaders: HotpotQA, NarrativeQA, MuSiQue -> SourceExample
│   ├── base.py                 # SourceExample schema + HF-load-with-fallback helper
│   ├── hotpotqa.py
│   ├── narrativeqa.py
│   └── musique.py
├── reflection_dataset.py      # Q_final loading + type/level filtering, bootstrap sampling,
│                               # answer-masked tokenization
├── adapters/shared_lora.py    # SharedLoRALinear: frozen shared A, trainable per-adapter B
├── train_bootstrap.py         # step 1
├── consolidate/
│   ├── pca.py                 # PCA / Fisher-weighted PCA consolidation (eq. 16-20)
│   └── generator.py           # coordinate generator (eq. 21-22)
├── run_consolidation.py       # step 2
├── memory/
│   ├── entry.py                # MemoryEntry: basis + generator + signature, synthesize_adapter() (eq. 23-24)
│   ├── signature.py             # TaskSignature: shrunk-diagonal Gaussian fit + Mahalanobis
│   ├── gw.py                    # Gromov-Wasserstein distance + barycenter over signatures (eq. 25-27)
│   ├── tree.py                  # MemoryTree: build/insert/route/consolidate_confusable (eq. 28)
│   ├── apply.py                  # patch/unpatch a synthesized adapter (and, for trees, a task's A) onto a live model
│   └── single_entry.py          # single-task stand-in exposing the same route() shape as MemoryTree
├── evaluate_sigma.py          # step 3 / step 4 (single entry or tree)
├── build_memory_tree.py       # step 4: build+save a MemoryTree from N MemoryEntry files
├── backends/                  # pluggable comparison backends (local HF model, OpenAI, ...)
└── utils/
    ├── context_embedding.py
    ├── metrics.py              # HotpotQA-style EM/F1
    ├── env.py                  # shared .env loading
    └── logging_setup.py        # shared stdout+file logging (--log_dir) for every CLI script

scripts/*.sh   # env-var-configurable wrappers around the pipeline steps
ideas/         # the SIGMA proposal PDF this implementation follows (gitignored)
MemoryDecoder/ # reference repo this codebase's structure/CLI style is modeled on (gitignored)
MeMo/          # reference repo for the MEMO method SIGMA builds on (gitignored) -- not
               # code we run directly (different infra: vLLM serving, DeepSpeed SFT,
               # LLM-judge eval); data_sources/ is informed by its dataset conventions
```
