# SIGMA — Setup & Usage Guide

**SIGMA** gives a frozen LLM a trainable "memory" without fine-tuning the whole model.
It works in three stages:

1. **Bootstrap** — train many small LoRA adapters on resampled subsets of a QA dataset.
2. **Consolidate** — compress those adapters into one compact memory (a basis + a small
   generator network).
3. **Synthesize** — at answer time, generate a task-specific adapter on the fly from
   that memory and patch it onto the backbone for one generation call.

Works with **HotpotQA**, **NarrativeQA**, and **MuSiQue**. Full design background is in
`ideas/sigma proposal v1.pdf`; the [How it works](#how-it-works) section at the bottom
covers implementation details if you want them — everything above it is just "how to run
this."

---

## Quick Start (HotpotQA, ~5 commands)

HotpotQA needs no download step, so it's the fastest way to see the whole pipeline run
once. Swap in NarrativeQA/MuSiQue later using the [full steps](#1-install) below.

```bash
# 0. Install
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env    # needed for step 2

# 1. Generate training QA pairs from HotpotQA (small run: 100 examples)
python generate_reflections.py --dataset hotpotqa --mode openai \
    --output data/hotpotqa_reflections.jsonl --limit 100

# 2. Train 8 small LoRA adapters on that data
python train_bootstrap.py \
    --reflections_path data/hotpotqa_reflections.jsonl \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --output_dir runs/bootstrap --num_adapters 8 --lora_rank 8

# 3. Compress those adapters into one memory file
python run_consolidation.py \
    --bootstrap_dir runs/bootstrap \
    --reflections_path data/hotpotqa_reflections.jsonl \
    --output_path runs/memory_entry.pt

# 4. Evaluate: SIGMA-adapted backbone vs. the plain backbone, scored with EM/F1
python evaluate_sigma.py \
    --memory_entry_path runs/memory_entry.pt \
    --model_name_or_path Qwen/Qwen2.5-0.5B --limit 200
```

Step 4 prints something like:

```
Baseline: EM=0.1200 F1=0.1850 (n=200)
SIGMA:    EM=0.1550 F1=0.2210 (n=200)
```

That's the whole loop. Everything below explains each step in more depth and covers
NarrativeQA/MuSiQue, which need one extra data-prep step first.

---

## Contents

1. [1. Install](#1-install)
2. [2. Get the raw datasets](#2-get-the-raw-datasets)
3. [3. Process NarrativeQA/MuSiQue into chunks](#3-process-narrativeqamusique-into-chunks)
4. [4. Generate training QA pairs](#4-generate-training-qa-pairs)
5. [5. Train bootstrap adapters](#5-train-bootstrap-adapters)
6. [6. Consolidate into a memory](#6-consolidate-into-a-memory)
7. [7. Evaluate](#7-evaluate)
8. [8. (Optional) Combine multiple datasets into one memory tree](#8-optional-combine-multiple-datasets-into-one-memory-tree)
9. [9. (Optional) Compare against another model](#9-optional-compare-against-another-model)
10. [How it works](#how-it-works)
11. [Repo layout](#repo-layout)

## 1. Install

```bash
pip install -r requirements.txt
```

Training/consolidating/evaluating (steps 5–7) need `torch` + `transformers` +
`accelerate` and a real GPU to run at a useful scale — everything *works* on CPU too,
just slowly, which is fine for a small smoke test.

Put your OpenAI key in a `.env` file at the repo root (needed for step 4, and
optionally step 9):

```
OPENAI_API_KEY=sk-...
```

Every script also writes its own timestamped log file under `logs/` automatically —
useful if a long run dies partway through and you want to see what happened.

## 2. Get the raw datasets

**HotpotQA** — nothing to do, it loads straight from Hugging Face in step 4.

**NarrativeQA** — clone the official repo:

```bash
git clone https://github.com/google-deepmind/narrativeqa data/NarrativeQA
```

You don't need to run its `download_stories.sh` — SIGMA uses each story's Wikipedia
*summary* instead of the full book text. See `data/NarrativeQA/README.md`.

**MuSiQue** — download a copy from [`StonyBrookNLP/musique`](https://github.com/StonyBrookNLP/musique)
(their README has the current download link) and place a split file under
`data/MuSiQue/`. See `data/MuSiQue/README.md` for details and an alternative source.

> These two GitHub links couldn't be re-verified from this environment (no live
> browsing). If either is dead, search "NarrativeQA deepmind github" / "MuSiQue
> StonyBrookNLP github".

## 3. Process NarrativeQA/MuSiQue into chunks

**Required** for NarrativeQA and MuSiQue before step 4 (HotpotQA skips this — go
straight to step 4). This splits each document into overlapping chunks and writes them
in the format step 4 expects:

```bash
# NarrativeQA (once per split you plan to use)
python process_narrativeqa.py --narrativeqa_dir data/NarrativeQA --split train
python process_narrativeqa.py --narrativeqa_dir data/NarrativeQA --split valid

# MuSiQue
python process_musique.py \
    --musique_path data/MuSiQue/musique_ans_v1.0_train.jsonl \
    --output_dir data/MuSiQue
```

If you skip this, step 4's loaders will fail with a clear error telling you exactly
which command to run.

<details>
<summary>Why this step exists, and why chunking is usually a no-op here (click to expand)</summary>

This mirrors MEMO's own `data_processing_utils/convert_*_to_chunks_jsonl.py` scripts:
split text into overlapping word-count chunks (default 6400 words, 640 overlap) and
write a `{docid, text, url}` corpus file plus a `{query_id, question, answers,
document_id, evidence_docs, ...}` questions file. Since NarrativeQA summaries and
MuSiQue paragraphs are both far shorter than 6400 words, chunking almost always produces
exactly one chunk per document — that's expected, not a bug.

</details>

## 4. Generate training QA pairs

This turns raw documents into the QA pairs (`Q_final`) that steps 5–6 actually train
on. It runs a 5-stage pipeline modeled on MEMO's own reflection synthesis: extract
facts stated directly in each document, extract facts that require combining multiple
sentences, consolidate related facts into richer QA pairs, verify/fix each pair so it
reads correctly on its own, then generate a few "describe this entity without naming
it" and cross-document questions. Details are in
[How it works](#how-the-reflection-pipeline-works).

```bash
# HotpotQA
python generate_reflections.py --dataset hotpotqa --mode openai \
    --output data/hotpotqa_reflections.jsonl --limit 100

# NarrativeQA (needs step 3 run first)
python generate_reflections.py --dataset narrativeqa --mode openai \
    --narrativeqa_dir data/NarrativeQA --split train \
    --output data/narrativeqa_reflections.jsonl --limit 100

# MuSiQue (needs step 3 run first)
python generate_reflections.py --dataset musique --mode openai \
    --musique_dir data/MuSiQue \
    --output data/musique_reflections.jsonl --limit 100
```

**Cost/speed note:** this makes several LLM calls per document (not per question), and
they run one at a time. A `--limit` of 100 questions can mean several minutes and a
couple dollars of API spend depending on the dataset — start smaller (`--limit 10`) to
sanity-check before committing to a big run.

Two alternatives to `--mode openai`:

- `--mode hf --model Qwen/Qwen2.5-7B-Instruct` runs the identical pipeline against a
  local open-source model instead of the OpenAI API (needs a GPU, no per-token cost).
  Must be an **instruction-tuned** model — the prompts are long and structured, and base
  models won't reliably follow them.
- `--mode prompt` (the default) is a free, no-LLM-calls dry run that just shows you the
  first-stage prompt per document, for sanity-checking coverage before spending
  money/GPU time.

## 5. Train bootstrap adapters

Trains several small LoRA adapters on randomly-resampled subsets of the QA pairs from
step 4:

```bash
python train_bootstrap.py \
    --reflections_path data/hotpotqa_reflections.jsonl \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --output_dir runs/bootstrap \
    --num_adapters 8 --lora_rank 8
```

`--model_name_or_path` accepts any local Hugging Face causal LM — it's not tied to
Qwen, that's just a small default for quick iteration. Repeat with a different
`--reflections_path`/`--output_dir` for each dataset you want a memory for.

This step needs direct access to the model's weights and gradients, so it can only run
against a local model — there's no API-only version of this step.

## 6. Consolidate into a memory

Compresses the adapters from step 5 into one compact memory file:

```bash
python run_consolidation.py \
    --bootstrap_dir runs/bootstrap \
    --reflections_path data/hotpotqa_reflections.jsonl \
    --output_path runs/memory_entry.pt
```

Same "local model only" restriction as step 5.

## 7. Evaluate

Compares the SIGMA-adapted backbone against the same backbone with no memory attached,
on held-out questions, scored with exact-match (EM) and F1:

```bash
# HotpotQA (default)
python evaluate_sigma.py \
    --memory_entry_path runs/memory_entry.pt \
    --model_name_or_path Qwen/Qwen2.5-0.5B --limit 200

# NarrativeQA
python evaluate_sigma.py \
    --memory_entry_path runs/narrativeqa/memory_entry.pt \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --dataset narrativeqa --narrativeqa_dir data/NarrativeQA --split validation --limit 200

# MuSiQue -- see the note below before running this one
python evaluate_sigma.py \
    --memory_entry_path runs/musique/memory_entry.pt \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --dataset musique --musique_dir data/MuSiQue/dev --limit 200
```

**MuSiQue-specific note:** unlike the other two datasets, MuSiQue has no `--split`
flag — `process_musique.py` (step 3) always writes the same filenames into whatever
`--output_dir` you give it. So to evaluate on data the model wasn't trained on, run
step 3 **twice** — once for your train file, once for a dev/held-out file — into two
different directories, and point `--musique_dir` at the dev one here:

```bash
python process_musique.py --musique_path data/MuSiQue/musique_ans_v1.0_train.jsonl --output_dir data/MuSiQue/train
python process_musique.py --musique_path data/MuSiQue/musique_ans_v1.0_dev.jsonl   --output_dir data/MuSiQue/dev
# ... then use data/MuSiQue/train for step 4 and data/MuSiQue/dev for step 7
```

Pointing step 4 and step 7 at the same MuSiQue directory trains and evaluates on the
same questions, which will look better than it is.

This is single-shot evaluation — one question in, one answer out. MEMO's own evaluation
harness runs a two-model, multi-turn conversation (a large model asking a small
memory-tuned model sub-questions, across up to 4 protocols); SIGMA has no equivalent
architecture, so this script is the SIGMA-native version of MEMO's single-turn
evaluation specifically, not a port of its multi-turn protocols.

## 8. (Optional) Combine multiple datasets into one memory tree

Once you have two or more memory files (e.g. one per dataset), organize them into a
tree and route between them automatically instead of picking one manually:

```bash
python build_memory_tree.py \
    --task hotpotqa=runs/hotpotqa/memory_entry.pt \
    --task musique=runs/musique/memory_entry.pt \
    --task narrativeqa=runs/narrativeqa/memory_entry.pt \
    --output_path runs/memory_tree.pt

python evaluate_sigma.py \
    --memory_tree_path runs/memory_tree.pt \
    --model_name_or_path Qwen/Qwen2.5-0.5B --limit 200
```

All tasks in one tree must have been trained with the same `--lora_rank` and
`--target_modules` (step 5) — you'll get a clear error at evaluation time if they
don't match. See [How it works](#how-the-memory-tree-works) for what's happening
underneath.

## 9. (Optional) Compare against another model

Evaluation can pull in a third set of predictions purely as a comparison point (steps
5–6 still need a local model, but step 7 can compare against an API model too):

```bash
python evaluate_sigma.py \
    --memory_entry_path runs/memory_entry.pt \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --baseline_model openai:gpt-4o-mini --limit 200
```

`--baseline_model` takes `openai:<model>` (needs `OPENAI_API_KEY`) or a local Hugging
Face path/repo id. It's comparison-only — the memory itself never attaches to it.

---

## How it works

<details>
<summary id="how-the-reflection-pipeline-works">How the reflection pipeline (step 4) works, and how closely it follows MEMO (click to expand)</summary>

`reflection_prompts.py` carries MEMO's own prompts near-verbatim (direct/indirect fact
extraction, consolidation, self-containment check/fix, entity surfacing, cross-document
combination). `reflection_llm.py` calls an OpenAI-compatible client one request at a
time with retries and parses JSON out of the reply — MEMO itself serves these through
vLLM with async "hedging" (racing duplicate requests), which this doesn't replicate.
`reflection_pipeline.py` runs the five stages **document-first**: `build_documents`
dedups every context block across all loaded questions by `(dataset, title)` first, so
each stage runs once per unique document, not once per question. Cross-document
synthesis only combines documents that actually co-occurred as context for the same
original question — never arbitrary pairs from across the whole corpus — and makes one
batched call per group rather than the full pairwise cross product, to keep cost linear
rather than quadratic.

`flatten_to_records` converts the result into the same `source`/`rewritten_qa` record
shape `reflection_dataset.py` expects, tagged with a `source.type` field
(`direct`/`indirect`/`consolidated`/`entity_surfacing`/`crossdoc`) recording which stage
produced it.

</details>

<details>
<summary id="how-the-memory-tree-works">How the memory tree (step 8) works, and where it departs from the proposal (click to expand)</summary>

The proposal itself is schematic here — its Gromov-Wasserstein-distance formula is
written as a proportionality, not a closed form, and its "growth control" merge policy
is described only qualitatively, with no concrete algorithm. `memory/gw.py` and
`memory/tree.py` fill both gaps with documented, defensible choices, not a reproduction
of a specific published formula:

- Each task's **signature** is a diagonal Gaussian fit from its own per-subset context
  embeddings (step 6), shrunk toward the average variance for stability with few
  samples.
- Tasks are organized into a binary tree by clustering on **Gromov-Wasserstein
  distance** between signatures' sorted variance spectra — used instead of plain
  Wasserstein distance because each task's embeddings live in a different,
  differently-shaped space (its own consolidated adapter), so only the *shape* of the
  spectrum is comparable, not raw coordinates.
- **Routing** descends the tree, comparing the query's embedding against each
  candidate branch via Mahalanobis distance, until it reaches a leaf.
- `MemoryTree.consolidate_confusable(threshold)` merges sibling tasks whose distance is
  below `threshold` — a growth-control mechanism, not a proven error bound.

</details>

<details>
<summary>Where NarrativeQA/MuSiQue's raw files actually come from (click to expand)</summary>

Neither is reliably published on Hugging Face, so both are processed from local files
(step 3: `data_sources/process_narrativeqa.py`, `data_sources/process_musique.py`,
mirroring MEMO's own `data_processing_utils/convert_*_to_chunks_jsonl.py`), then loaded
from the resulting chunked JSONL (`data_sources/narrativeqa.py`,
`data_sources/musique.py`) instead of `datasets.load_dataset`. `process_musique.py`
accepts either the official one-JSON-object-per-line format or a single JSON
array/object, auto-detected. A missing `--narrativeqa_dir`/`--musique_dir`, or a
directory missing the chunked files step 3 produces, raises a clear error naming the
exact command to run.

</details>

## Repo layout

```
generate_hotpotqa_reflections.py   \
generate_reflections.py             |
process_narrativeqa.py               |  thin root-level CLI wrappers around src/sigma/*
process_musique.py                   |  (see each file's docstring for what it does)
train_bootstrap.py                   |
run_consolidation.py                 |
evaluate_sigma.py                    |
build_memory_tree.py                /

src/sigma/
├── hotpotqa_reflections.py    # legacy HotpotQA-only single-call script (kept for its loader + --mode prompt)
├── reflections.py             # step 4: reflection generation CLI, any data_sources/ dataset
├── reflection_prompts.py      # MEMO's own prompts, ported near-verbatim
├── reflection_llm.py          # sequential OpenAI-compatible call + JSON/literal response parsing
├── reflection_pipeline.py     # 5-stage document-first orchestration (see "How the reflection
│                               # pipeline works" above)
├── reflection_hf_client.py    # --mode hf: local open-source model, same chat.completions
│                               # .create(...) shape as the OpenAI client
├── data_sources/               # normalized loaders: HotpotQA, NarrativeQA, MuSiQue -> SourceExample
│   ├── base.py                  # SourceExample schema
│   ├── chunking.py               # MEMO's chunk_text word-count sliding-window algorithm
│   ├── process_narrativeqa.py   # step 3: raw NarrativeQA -> chunked corpus/questions JSONL
│   ├── process_musique.py       # step 3: raw MuSiQue -> chunked corpus/questions JSONL
│   ├── hotpotqa.py
│   ├── narrativeqa.py            # reads process_narrativeqa.py's output (required)
│   └── musique.py                # reads process_musique.py's output (required)
├── reflection_dataset.py      # Q_final loading + type/level filtering, bootstrap sampling,
│                               # answer-masked tokenization
├── adapters/shared_lora.py    # SharedLoRALinear: frozen shared A, trainable per-adapter B
├── train_bootstrap.py         # step 5
├── consolidate/
│   ├── pca.py                 # PCA / Fisher-weighted PCA consolidation
│   └── generator.py           # coordinate generator
├── run_consolidation.py       # step 6
├── memory/
│   ├── entry.py                # MemoryEntry: basis + generator + signature, synthesize_adapter()
│   ├── signature.py             # TaskSignature: shrunk-diagonal Gaussian fit + Mahalanobis
│   ├── gw.py                    # Gromov-Wasserstein distance + barycenter over signatures
│   ├── tree.py                  # MemoryTree: build/insert/route/consolidate_confusable
│   ├── apply.py                  # patch/unpatch a synthesized adapter (and, for trees, a task's A) onto a live model
│   └── single_entry.py          # single-task stand-in exposing the same route() shape as MemoryTree
├── evaluate_sigma.py          # step 7 / step 8 (single entry or tree)
├── build_memory_tree.py       # step 8: build+save a MemoryTree from N MemoryEntry files
├── backends/                  # pluggable comparison backends (local HF model, OpenAI, ...) -- step 9
└── utils/
    ├── context_embedding.py
    ├── metrics.py              # EM/F1
    ├── env.py                  # shared .env loading
    └── logging_setup.py        # shared stdout+file logging (--log_dir) for every CLI script

data/
├── NarrativeQA/README.md      # download instructions (see step 2) -- contents gitignored
└── MuSiQue/README.md          # download instructions (see step 2) -- contents gitignored

scripts/*.sh   # env-var-configurable wrappers around the pipeline steps
ideas/         # the SIGMA proposal PDF this implementation follows (gitignored)
MemoryDecoder/ # reference repo this codebase's structure/CLI style is modeled on (gitignored)
MeMo/          # reference repo for the MEMO method SIGMA builds on (gitignored) -- not
               # code we run directly (different infra: vLLM serving, DeepSpeed SFT,
               # LLM-judge eval); data_sources/ is informed by its dataset conventions
```
