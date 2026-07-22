# SIGMA_DEV — Reproduction Instructions

Implementation of **SIGMA**, a bootstrap-and-consolidate memory mechanism for LLMs (see
`ideas/sigma proposal v1.pdf` for the full proposal). Given a frozen backbone LLM, SIGMA
trains many small LoRA adapters on bootstrapped subsets of a reflection QA dataset,
consolidates them into a compact basis plus a coordinate generator, and at inference time
synthesizes a task-specific adapter on the fly and patches it onto the backbone for that
one generation call.

Both halves of the proposal are implemented: within-task bootstrap-and-consolidate
(section 4.2.1) and the cross-task memory tree (section 4.2.2) that organizes multiple
tasks' signatures via Gromov-Wasserstein distance for O(log n) routing. Supports three
datasets end to end: **HotpotQA**, **NarrativeQA**, **MuSiQue**.

This document is a runbook — follow it top to bottom to reproduce a full run. Design
rationale and implementation details are in [How it works](#how-it-works) at the bottom;
skip there if you want the "why", not just the "how."

## Contents

1. [0. Prerequisites](#0-prerequisites)
2. [1. Download the raw datasets](#1-download-the-raw-datasets)
3. [2. Generate reflection data](#2-generate-reflection-data)
4. [3. Bootstrap adapters](#3-bootstrap-adapters)
5. [4. Consolidate into a memory entry](#4-consolidate-into-a-memory-entry)
6. [5. Evaluate](#5-evaluate)
7. [6. (Optional) Multi-task memory tree](#6-optional-multi-task-memory-tree)
8. [7. (Optional) Compare against another model](#7-optional-compare-against-another-model)
9. [How it works](#how-it-works)
10. [Repo layout](#repo-layout)

## 0. Prerequisites

```bash
pip install -r requirements.txt
```

Needs `torch` + `transformers` + `accelerate` for steps 3–5, and real GPU compute for
anything beyond a tiny smoke test (everything *runs* on CPU, it's just not the intended
way).

Put your OpenAI key in a repo-root `.env` file (needed for step 2, and optionally step 7):

```
OPENAI_API_KEY=sk-...
```

(A `.env/` folder layout — `.env/.env` or `.env/local.env` — also works.)

Every script logs to stdout *and* to a timestamped file under `logs/` automatically
(`--log_dir`, default `logs`) — nothing to set up, just know it's there if a run dies and
you want the transcript.

## 1. Download the raw datasets

**HotpotQA** — nothing to do, loads straight from Hugging Face (`hotpotqa/hotpot_qa`) in
step 2.

**NarrativeQA** — clone the official repo:

```bash
git clone https://github.com/google-deepmind/narrativeqa data/NarrativeQA
```

That gives you `documents.csv`, `qaps.csv`, and `third_party/wikipedia/summaries.csv`
directly — all three files step 2 actually reads. You do **not** need to run the repo's
separate `download_stories.sh`: we use each story's Wikipedia plot *summary* as context
(questions are written to be answerable from it), not the full book/script text, which is
often 50k+ words and impractical for one reflection prompt anyway. See
`data/NarrativeQA/README.md` for more detail.

**MuSiQue** — not on Hugging Face reliably, so get it from the dataset's own repo:

```
https://github.com/StonyBrookNLP/musique
```

Their README has the actual download link (a Google Drive zip at time of writing).
Unzip a split file — e.g. `musique_ans_v1.0_train.jsonl` — into `data/MuSiQue/`. See
`data/MuSiQue/README.md` for more detail.

> I can't browse the web from this environment to re-verify those two GitHub URLs still
> resolve. If either 404s, search "NarrativeQA deepmind github" / "MuSiQue StonyBrookNLP
> github" respectively.

## 2. Generate reflection data

Turns raw questions into the reflection QA set (`Q_final`) the rest of the pipeline
trains on. One LLM call per example asks for six fields — fact extraction (direct facts
stated outright, plus indirect facts derived by combining two or more sentences), a
reasoning reflection (how the facts connect), answer verification, entity surfacing
("describe the entity without naming it"), cross-document synthesis (what combining more
than one context block tells you), and a rewritten, self-contained (question, answer)
pair.

```bash
# HotpotQA
python generate_reflections.py --dataset hotpotqa --mode openai \
    --output data/hotpotqa_reflections.jsonl --limit 100

# NarrativeQA
python generate_reflections.py --dataset narrativeqa --mode openai \
    --narrativeqa_dir data/NarrativeQA \
    --output data/narrativeqa_reflections.jsonl --limit 100

# MuSiQue
python generate_reflections.py --dataset musique --mode openai \
    --musique_path data/MuSiQue/musique_ans_v1.0_train.jsonl \
    --output data/musique_reflections.jsonl --limit 100
```

Drop `--mode openai` (or set it to `--mode prompt`, the default) to export the prompts
without calling an LLM — useful for inspecting/debugging the data before spending API
budget. Each record also carries the source dataset's own `type`/`level` metadata where
available (HotpotQA: bridge/comparison, easy/medium/hard) — used in step 6 to carve
pseudo-tasks out of one reflections file.

`generate_hotpotqa_reflections.py` also exists (the original HotpotQA-only version of
this script) — kept only because `evaluate_sigma.py` reuses its dataset loader for the
HotpotQA eval path in step 5. Use `generate_reflections.py` (above) for generating
training reflections.

## 3. Bootstrap adapters

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

Repeat with `--reflections_path data/narrativeqa_reflections.jsonl`/`musique_...` and a
different `--output_dir` for the other two datasets. `--model_name_or_path` accepts any
local Hugging Face causal LM id/path — it's not hardcoded to Qwen, that's just a small
default for quick iteration.

**Building one task of several** (for [step 6](#6-optional-multi-task-memory-tree)): add
`--question_type bridge` (or `comparison`) and/or `--level easy|medium|hard` to train on
only a filtered slice of `Q_final`, and run this step once per task into a different
`--output_dir` each time. The filter gets saved into `bootstrap_meta.json` and read back
automatically in step 4, so you never have to repeat it.

**No API version.** This step trains LoRA weights directly on the backbone's own
parameters and needs gradients through it, so it can only ever run against a local model
you have the weights for.

## 4. Consolidate into a memory entry

Decomposes the `M` adapters into a shared basis (PCA by default; Fisher-weighted PCA via
`--consolidation_method fisher`) and trains the coordinate generator on top of it
(eq. 16–22). Also fits a task **signature** for later cross-task use (step 6). Produces
one `MemoryEntry` checkpoint.

```bash
python run_consolidation.py \
    --bootstrap_dir runs/bootstrap \
    --reflections_path data/hotpotqa_reflections.jsonl \
    --output_path runs/memory_entry.pt
```

Same "no API version" reason as step 3 — needs hidden states/gradients off the local
backbone.

## 5. Evaluate

For each validation question: synthesizes a task-specific adapter from the memory entry
(eq. 23–24), patches it onto the frozen backbone, and compares its answer against the
same backbone with no adapter applied. Scores both with exact-match/F1.

```bash
python evaluate_sigma.py \
    --memory_entry_path runs/memory_entry.pt \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --limit 200
```

## 6. (Optional) Multi-task memory tree

Once you have **two or more** `MemoryEntry` checkpoints (e.g. one per dataset, or one per
`--question_type` — see step 3), organize them into a tree and route through it instead
of a single fixed entry:

```bash
python build_memory_tree.py \
    --task hotpotqa=runs/hotpotqa/memory_entry.pt \
    --task musique=runs/musique/memory_entry.pt \
    --task narrativeqa=runs/narrativeqa/memory_entry.pt \
    --output_path runs/memory_tree.pt

python evaluate_sigma.py \
    --memory_tree_path runs/memory_tree.pt \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --limit 200
```

All tasks in one tree must share `--lora_rank` and `--target_modules` (checked with a
clear error at evaluation time if they don't) — routing means temporarily wearing a
candidate task's own frozen `A` to compute its embedding, which only works if every
task's adapter wrapper was sized the same way to begin with. See
[How it works](#how-it-works) for what's actually happening under the hood.

## 7. (Optional) Compare against another model

Steps 3–4 need direct access to weights and hidden states, so they can only ever run
against a local model. Evaluation (step 5/6), though, can pull in a **third** set of
predictions purely as a comparison point:

```bash
python evaluate_sigma.py \
    --memory_entry_path runs/memory_entry.pt \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --baseline_model openai:gpt-4o-mini \
    --limit 200
```

`--baseline_model` takes a `"<provider>:<name>"` spec: `openai:<model>` (needs
`OPENAI_API_KEY`) or `hf:<path>`/a bare path for another local Hugging Face model. This
is a comparison point only — it never has the memory attached, since that requires
weights we control.

---

## How it works

<details>
<summary>Why the memory tree's math is a documented approximation, not a literal reproduction of the proposal's equations (click to expand)</summary>

The proposal is itself schematic about two things — eq. 26's GW-distance formula is
written as a proportionality ("≍ ..."), not a closed form, and the "growth control" merge
policy is described only qualitatively (merge when confusable, bound retrieval error)
with no concrete algorithm given. `memory/gw.py` and `memory/tree.py` fill both gaps with
documented, defensible choices — not a reproduction of a specific published formula or a
proven error bound.

- Each task's **signature** is a diagonal Gaussian fit from its own per-subset context
  embeddings (step 4), shrunk toward the average variance for stability with few samples.
- Tasks are organized into a binary tree by bottom-up clustering on **Gromov-Wasserstein
  distance** between signatures' sorted variance spectra — GW distance rather than plain
  Wasserstein because each task's embeddings live in a different, differently-shaped
  space (its own consolidated adapter), so only the *shape* of the spectrum is
  comparable, not raw coordinates.
- **Routing** descends the tree, at each internal node comparing the query's embedding
  (recomputed under each candidate branch's representative task) via own-space
  Mahalanobis distance, until it reaches a leaf — the exact formula the proposal
  specifies for the final step (eq. 28), just applied at every level on the way down.
- `MemoryTree.consolidate_confusable(threshold)` finds sibling tasks whose GW distance is
  below `threshold` and merges them (concatenates their steering bases, retrains one
  generator on the pooled training pairs) — the growth-control mechanism, not a proven
  error bound.

</details>

<details>
<summary>Where the raw dataset files actually come from (click to expand)</summary>

Neither NarrativeQA nor MuSiQue is reliably published on Hugging Face, so both load from
local files (`data_sources/narrativeqa.py`, `data_sources/musique.py`) instead of
`datasets.load_dataset`. MuSiQue's paragraphs already carry `is_supporting` flags
matching HotpotQA's supporting-facts convention; the loader accepts either the official
one-JSON-object-per-line format or a single JSON array/object, auto-detected. Missing
`--narrativeqa_dir`/`--musique_path`, or a directory missing the expected files, raises a
clear error telling you exactly what's needed rather than failing silently.

</details>

## Repo layout

```
generate_hotpotqa_reflections.py   \
generate_reflections.py             |  thin root-level CLI wrappers around src/sigma/*
train_bootstrap.py                  |  (see each file's docstring for what it does)
run_consolidation.py                 \
evaluate_sigma.py                    /
build_memory_tree.py                /

src/sigma/
├── hotpotqa_reflections.py    # legacy HotpotQA-only single-call script (kept for its loader + --mode prompt)
├── reflections.py             # step 2: reflection generation, any data_sources/ dataset
├── data_sources/               # normalized loaders: HotpotQA, NarrativeQA, MuSiQue -> SourceExample
│   ├── base.py                  # SourceExample schema
│   ├── hotpotqa.py
│   ├── narrativeqa.py
│   └── musique.py
├── reflection_dataset.py      # Q_final loading + type/level filtering, bootstrap sampling,
│                               # answer-masked tokenization
├── adapters/shared_lora.py    # SharedLoRALinear: frozen shared A, trainable per-adapter B
├── train_bootstrap.py         # step 3
├── consolidate/
│   ├── pca.py                 # PCA / Fisher-weighted PCA consolidation (eq. 16-20)
│   └── generator.py           # coordinate generator (eq. 21-22)
├── run_consolidation.py       # step 4
├── memory/
│   ├── entry.py                # MemoryEntry: basis + generator + signature, synthesize_adapter() (eq. 23-24)
│   ├── signature.py             # TaskSignature: shrunk-diagonal Gaussian fit + Mahalanobis
│   ├── gw.py                    # Gromov-Wasserstein distance + barycenter over signatures (eq. 25-27)
│   ├── tree.py                  # MemoryTree: build/insert/route/consolidate_confusable (eq. 28)
│   ├── apply.py                  # patch/unpatch a synthesized adapter (and, for trees, a task's A) onto a live model
│   └── single_entry.py          # single-task stand-in exposing the same route() shape as MemoryTree
├── evaluate_sigma.py          # step 5 / step 6 (single entry or tree)
├── build_memory_tree.py       # step 6: build+save a MemoryTree from N MemoryEntry files
├── backends/                  # pluggable comparison backends (local HF model, OpenAI, ...) -- step 7
└── utils/
    ├── context_embedding.py
    ├── metrics.py              # EM/F1
    ├── env.py                  # shared .env loading
    └── logging_setup.py        # shared stdout+file logging (--log_dir) for every CLI script

data/
├── NarrativeQA/README.md      # download instructions (see step 1) -- contents gitignored
└── MuSiQue/README.md          # download instructions (see step 1) -- contents gitignored

scripts/*.sh   # env-var-configurable wrappers around the pipeline steps
ideas/         # the SIGMA proposal PDF this implementation follows (gitignored)
MemoryDecoder/ # reference repo this codebase's structure/CLI style is modeled on (gitignored)
MeMo/          # reference repo for the MEMO method SIGMA builds on (gitignored) -- not
               # code we run directly (different infra: vLLM serving, DeepSpeed SFT,
               # LLM-judge eval); data_sources/ is informed by its dataset conventions
```
