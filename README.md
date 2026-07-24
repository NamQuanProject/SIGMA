# SIGMA — Setup & Usage Guide

**SIGMA** gives a frozen LLM a trainable "memory" without fine-tuning the whole model.
It works in three stages:

1. **Bootstrap** — train many small LoRA adapters on resampled subsets of a QA dataset.
2. **Consolidate** — compress those adapters into one compact memory (a basis + a small
   generator network).
3. **Synthesize** — at answer time, generate a task-specific adapter on the fly from
   that memory and patch it onto the backbone for one generation call.

Works with **NarrativeQA** and **MuSiQue**. Full design background is in `ideas/sigma
proposal v1.pdf`; the [How it works](#how-it-works) section at the bottom covers
implementation details if you want them — everything above it is just "how to run this."

---

## Quick Start (MuSiQue, ~6 commands)

MuSiQue's raw data is a single downloaded file, so it's the fastest way to see the whole
pipeline run once. Swap in NarrativeQA later using the [full steps](#1-install) below.

```bash
# 0. Install
pip install -e .
echo "OPENAI_API_KEY=sk-..." > .env    # needed for step 2

# 1. Download musique.json into data/MuSiQue/ -- see step 2 below for the link, then:
python -m sigma.data_process.process_musique --musique_path data/MuSiQue/musique.json --output_dir data/MuSiQue

# 2. Generate training QA pairs (small run: 100 examples)
python -m sigma.reflections --dataset musique --mode openai \
    --corpus_path data/MuSiQue/musique_corpus_chunks.jsonl \
    --qns_path data/MuSiQue/musique_questions_chunks.jsonl \
    --output data/musique_reflections.jsonl --limit 100

# 3. Train 8 small LoRA adapters on that data
python -m sigma.train_bootstrap \
    --reflections_path data/musique_reflections.jsonl \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --output_dir runs/bootstrap --num_adapters 8 --lora_rank 8

# 4. Compress those adapters into one memory file
python -m sigma.run_consolidation \
    --bootstrap_dir runs/bootstrap \
    --reflections_path data/musique_reflections.jsonl \
    --output_path runs/memory_entry.pt

# 5. Evaluate: SIGMA-adapted backbone vs. the plain backbone, scored with EM/F1
python -m sigma.evaluate_sigma \
    --memory_entry_path runs/memory_entry.pt \
    --model_name_or_path Qwen/Qwen2.5-0.5B --dataset musique \
    --corpus_path data/MuSiQue/musique_corpus_chunks.jsonl \
    --qns_path data/MuSiQue/musique_questions_chunks.jsonl --limit 200
```

Step 5 prints something like:

```
Baseline: EM=0.1200 F1=0.1850 (n=200)
SIGMA:    EM=0.1550 F1=0.2210 (n=200)
```

That's the whole loop (this smoke test evaluates on the same questions it trained on --
see step 7 for a real held-out setup). Everything below explains each step in more depth
and covers NarrativeQA too.

---

## Contents

1. [1. Install](#1-install)
2. [2. Get the raw datasets](#2-get-the-raw-datasets)
3. [3. Process into chunks](#3-process-into-chunks)
4. [4. Generate training QA pairs](#4-generate-training-qa-pairs)
5. [5. Train bootstrap adapters](#5-train-bootstrap-adapters)
6. [6. Consolidate into a memory](#6-consolidate-into-a-memory)
7. [7. Evaluate](#7-evaluate)
8. [8. (Optional) Combine multiple datasets into one memory tree](#8-optional-combine-multiple-datasets-into-one-memory-tree)
9. [9. (Optional) Compare against another model](#9-optional-compare-against-another-model)
10. [10. (Optional) Compare against non-SIGMA baselines](#10-optional-compare-against-non-sigma-baselines)
11. [How it works](#how-it-works)
12. [Repo layout](#repo-layout)

## 1. Install

```bash
pip install -e .
```

This is the only install step — `pyproject.toml` pulls in every dependency
(`torch`/`transformers`/`accelerate`/`datasets`/`openai`/`loguru`/...) and installs
`sigma` itself in editable mode, so every command in this README (always run as
`python -m sigma.<module>`, e.g. `python -m sigma.reflections`) works from any directory
without needing `PYTHONPATH` tricks.

Training/consolidating/evaluating (steps 5–7) need a real GPU to run at a useful scale
— everything *works* on CPU too, just slowly, which is fine for a small smoke test.

Put your OpenAI key in a `.env` file at the repo root (needed for step 4, and
optionally step 9/10):

```
OPENAI_API_KEY=sk-...
```

Every script also writes its own timestamped log file under `logs/` automatically —
useful if a long run dies partway through and you want to see what happened.

## 2. Get the raw datasets

**MuSiQue** — this project's copy is `musique.json` from the **HippoRAG_2** dataset on
Hugging Face (a fixed ~1,000-question subset), not the official StonyBrookNLP repo
directly:

**https://huggingface.co/datasets/osunlp/HippoRAG_2/blob/main/musique.json**

Download it into `data/MuSiQue/musique.json`. See `data/MuSiQue/README.md` for the
record shape and an alternative source (the official StonyBrookNLP repo, if you want a
real train/dev split instead of this fixed subset).

**NarrativeQA** — clone the official repo:

```bash
git clone https://github.com/google-deepmind/narrativeqa data/NarrativeQA
```

You don't need to run its `download_stories.sh` — SIGMA uses each story's Wikipedia
*summary* instead of the full book text. See `data/NarrativeQA/README.md`.

## 3. Process into chunks

**Required** before step 4 -- this splits each document into overlapping chunks and
writes them in the format step 4 expects. Both converters live in `data_process/`
(mirroring MEMO's own `data_processing_utils/`):

```bash
# MuSiQue
python -m sigma.data_process.process_musique --musique_path data/MuSiQue/musique.json --output_dir data/MuSiQue

# NarrativeQA (once per split you plan to use)
python -m sigma.data_process.process_narrativeqa --narrativeqa_dir data/NarrativeQA --split train
python -m sigma.data_process.process_narrativeqa --narrativeqa_dir data/NarrativeQA --split valid
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
# MuSiQue (needs step 3 run first)
python -m sigma.reflections --dataset musique --mode openai \
    --corpus_path data/MuSiQue/musique_corpus_chunks.jsonl \
    --qns_path data/MuSiQue/musique_questions_chunks.jsonl \
    --output data/musique_reflections.jsonl --limit 100

# NarrativeQA (needs step 3 run first)
python -m sigma.reflections --dataset narrativeqa --mode openai \
    --corpus_path data/NarrativeQA/narrativeqa_train_corpus_chunks.jsonl \
    --qns_path data/NarrativeQA/narrativeqa_train_questions_chunks.jsonl \
    --output data/narrativeqa_reflections.jsonl --limit 100
```

`--corpus_path`/`--qns_path` are two explicit file paths, matching MEMO's own
`data_synthesis_pipeline/*_datasynth_pipeline.sh` scripts exactly (they take the same
two flags) — not a directory with an implied filename. `--limit` also matches MEMO's own
loaders here: it keeps the first N **in file order**, not a random sample (for
NarrativeQA this counts unique source documents, since MEMO's own `nqa_data_utils.py`
subsets by document, not by question).

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
python -m sigma.train_bootstrap \
    --reflections_path data/musique_reflections.jsonl \
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
python -m sigma.run_consolidation \
    --bootstrap_dir runs/bootstrap \
    --reflections_path data/musique_reflections.jsonl \
    --output_path runs/memory_entry.pt
```

Same "local model only" restriction as step 5.

## 7. Evaluate

Compares the SIGMA-adapted backbone against the same backbone with no memory attached,
on held-out questions, scored with exact-match (EM) and F1:

```bash
# MuSiQue -- see the note below before running this one
python -m sigma.evaluate_sigma \
    --memory_entry_path runs/musique/memory_entry.pt \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --dataset musique \
    --corpus_path data/MuSiQue/dev/musique_corpus_chunks.jsonl \
    --qns_path data/MuSiQue/dev/musique_questions_chunks.jsonl --limit 200

# NarrativeQA
python -m sigma.evaluate_sigma \
    --memory_entry_path runs/narrativeqa/memory_entry.pt \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --dataset narrativeqa \
    --corpus_path data/NarrativeQA/narrativeqa_valid_corpus_chunks.jsonl \
    --qns_path data/NarrativeQA/narrativeqa_valid_questions_chunks.jsonl --limit 200
```

`--corpus_path`/`--qns_path` (same two explicit file paths as step 4, matching MEMO's
own convention) let you point evaluation at a *different* chunked file than the one you
trained on. This matters most for MuSiQue: this project's own `musique.json` (see step
2) is a single fixed ~1,000-question subset, not a pre-split train/dev pair, so
`python -m sigma.data_process.process_musique` always writes the same filenames into whatever `--output_dir`
you give it. For a genuine held-out evaluation, get a separate train/dev pair from the
official StonyBrookNLP repo (see `data/MuSiQue/README.md`) and run step 3 **twice** —
once per file, into two different directories:

```bash
python -m sigma.data_process.process_musique --musique_path data/MuSiQue/musique_ans_v1.0_train.jsonl --output_dir data/MuSiQue/train
python -m sigma.data_process.process_musique --musique_path data/MuSiQue/musique_ans_v1.0_dev.jsonl   --output_dir data/MuSiQue/dev
# ... then use data/MuSiQue/train's files for step 4 and data/MuSiQue/dev's for step 7
```

Pointing step 4 and step 7 at the same chunked files trains and evaluates on the
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
python -m sigma.build_memory_tree \
    --task musique=runs/musique/memory_entry.pt \
    --task narrativeqa=runs/narrativeqa/memory_entry.pt \
    --output_path runs/memory_tree.pt

python -m sigma.evaluate_sigma \
    --memory_tree_path runs/memory_tree.pt \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --dataset musique \
    --corpus_path data/MuSiQue/dev/musique_corpus_chunks.jsonl \
    --qns_path data/MuSiQue/dev/musique_questions_chunks.jsonl --limit 200
```

All tasks in one tree must have been trained with the same `--lora_rank` and
`--target_modules` (step 5) — you'll get a clear error at evaluation time if they
don't match. See [How it works](#how-the-memory-tree-works) for what's happening
underneath.

## 9. (Optional) Compare against another model

Evaluation can pull in a third set of predictions purely as a comparison point (steps
5–6 still need a local model, but step 7 can compare against an API model too):

```bash
python -m sigma.evaluate_sigma \
    --memory_entry_path runs/memory_entry.pt \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --dataset musique \
    --corpus_path data/MuSiQue/musique_corpus_chunks.jsonl \
    --qns_path data/MuSiQue/musique_questions_chunks.jsonl \
    --baseline_model openai:gpt-4o-mini --limit 200
```

`--baseline_model` takes `openai:<model>` (needs `OPENAI_API_KEY`) or a local Hugging
Face path/repo id. It's comparison-only — the memory itself never attaches to it.

## 10. (Optional) Compare against non-SIGMA baselines

`baselines/` has two standalone scripts for comparing SIGMA against retrieval-based
approaches instead of a bare backbone — an oracle-context in-context-learning baseline
and a BM25 retrieval baseline, both scored with the same EM/F1 metric and readable on
the same held-out examples (matching `--dataset`/`--corpus_path`/`--qns_path`/`--limit`):

```bash
python -m baselines.icl.run_icl_baseline \
    --dataset musique --corpus_path data/MuSiQue/dev/musique_corpus_chunks.jsonl \
    --qns_path data/MuSiQue/dev/musique_questions_chunks.jsonl \
    --model openai:gpt-4o-mini --limit 100

python -m baselines.bm25.run_bm25_baseline \
    --dataset musique --corpus_path data/MuSiQue/dev/musique_corpus_chunks.jsonl \
    --qns_path data/MuSiQue/dev/musique_questions_chunks.jsonl \
    --model openai:gpt-4o-mini --top_k 3 --limit 100
```

See `baselines/README.md` for what each one does, how it maps to
[MeMo's own baselines](MeMo/baselines), and why the heavier ones (cartridges, HippoRAG2,
NV-Embed) aren't ported.

---

## How it works

<details>
<summary id="how-the-reflection-pipeline-works">How the reflection pipeline (step 4) works, and how closely it follows MEMO (click to expand)</summary>

`reflection/prompts.py` carries MEMO's own prompts near-verbatim (direct/indirect fact
extraction, consolidation, self-containment check/fix, entity surfacing, cross-document
combination). `reflection/llm.py` calls an OpenAI-compatible client one request at a
time with retries and parses JSON out of the reply — MEMO itself serves these through
vLLM with async "hedging" (racing duplicate requests), which this doesn't replicate.
`reflection/pipeline.py` runs the five stages **document-first**: `build_documents`
dedups every context block across all loaded questions by `(dataset, title)` first, so
each stage runs once per unique document, not once per question. Cross-document
synthesis only combines documents that actually co-occurred as context for the same
original question — never arbitrary pairs from across the whole corpus — and makes one
batched call per group rather than the full pairwise cross product, to keep cost linear
rather than quadratic.

`flatten_to_records` converts the result into the same `source`/`rewritten_qa` record
shape `reflection/dataset.py` expects, tagged with a `source.type` field
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

Neither is reliably published on Hugging Face in a form this pipeline can read directly,
so both are processed from local files first (step 3: `data_process/process_narrativeqa.py`,
`data_process/process_musique.py`, mirroring MEMO's own
`data_processing_utils/convert_*_to_chunks_jsonl.py`), then loaded from the resulting
chunked JSONL (`data_sources/narrativeqa.py`, `data_sources/musique.py`) instead of
`datasets.load_dataset`. `process_musique.py` accepts either the official
one-JSON-object-per-line format or a single JSON array/object (this project's own
`musique.json`, from HippoRAG_2 -- see step 2), auto-detected. A missing
`--corpus_path`/`--qns_path`, or a file missing the chunked content step 3 produces,
raises a clear error naming the exact command to run.

**Loading convention matches MEMO exactly, not just the chunk file schema:**
`data_sources/musique.py`/`narrativeqa.py` take `--corpus_path`/`--qns_path` as two
explicit file paths (not a directory with an implied filename) and load them the same
way MEMO's own `data_synthesis_pipeline/musique_data_utils.py`/`nqa_data_utils.py` do --
`--limit` keeps the first N **in file order** (no shuffling, matching MEMO's loaders
exactly), and the corpus is filtered down to only chunks actually referenced by the
loaded questions' evidence/gold docs before anything else happens. The one MEMO behavior
intentionally *not* ported is `nqa_data_utils.py`'s `SUBSET_MAP` -- three named subset
sizes (10/5_1/5_2 documents) selected via a hardcoded doc-ID list tied to MEMO's own
specific corpus build, which isn't reproducible without their exact chunk IDs; every
`--limit` here uses the general "first N unique source documents in file order" rule
instead.

</details>

## Repo layout

```
pyproject.toml   # packaging: deps + the sigma-* console scripts used throughout this README

src/sigma/
├── data_process/               # step 3: raw dataset -> MEMO-shaped chunked JSONL, mirroring
│   │                             # MEMO's own data_processing_utils/ as a distinct concern
│   │                             # from data_sources/'s loaders below
│   ├── chunking.py               # MEMO's chunk_text word-count sliding-window algorithm
│   ├── process_musique.py       # raw MuSiQue (musique.json) -> chunked corpus/questions JSONL
│   └── process_narrativeqa.py   # raw NarrativeQA (documents/qaps/summaries.csv) -> same
├── data_sources/               # normalized loaders: NarrativeQA, MuSiQue -> SourceExample
│   ├── base.py                  # SourceExample schema
│   ├── narrativeqa.py            # reads data_process/process_narrativeqa.py's output (required)
│   └── musique.py                # reads data_process/process_musique.py's output (required)
├── reflections.py             # step 4: reflection generation CLI, any data_sources/ dataset
├── reflection/                  # supporting modules for step 4 (grouped like adapters/,
│                                 # consolidate/, memory/ below)
│   ├── prompts.py                 # MEMO's own prompts, ported near-verbatim
│   ├── llm.py                     # sequential OpenAI-compatible call + JSON/literal response parsing
│   ├── pipeline.py                 # 5-stage document-first orchestration (see "How the
│   │                                # reflection pipeline works" above)
│   ├── hf_client.py                # --mode hf: local open-source model, same chat.completions
│   │                                # .create(...) shape as the OpenAI client
│   └── dataset.py                  # Q_final loading + type/level filtering, bootstrap sampling,
│                                    # answer-masked tokenization
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
├── backends/                  # pluggable comparison backends (local HF model, OpenAI, ...) --
│                               # step 9, and reused by baselines/ (step 10)
└── utils/
    ├── context_embedding.py
    ├── metrics.py              # EM/F1
    ├── env.py                  # shared .env loading
    └── logging_setup.py        # shared stdout+file logging (--log_dir) for every CLI script

baselines/       # step 10: non-SIGMA comparison baselines (oracle ICL, BM25) -- see baselines/README.md
├── _common.py     # shared dataset-args/example-loading/prompt-rendering, reusing data_sources/
├── icl/run_icl_baseline.py
└── bm25/{bm25_utils.py, run_bm25_baseline.py}

data/
├── NarrativeQA/README.md      # download instructions (see step 2) -- contents gitignored
└── MuSiQue/README.md          # download instructions (see step 2) -- contents gitignored

scripts/*.sh   # env-var-configurable wrappers around the sigma-* commands above
ideas/         # the SIGMA proposal PDF this implementation follows (gitignored)
MemoryDecoder/ # reference repo this codebase's structure/CLI style is modeled on (gitignored)
MeMo/          # reference repo for the MEMO method SIGMA builds on (gitignored) -- not
               # code we run directly (different infra: vLLM serving, DeepSpeed SFT,
               # LLM-judge eval); data_process/, data_sources/, and baselines/ are informed
               # by its dataset/baseline conventions
```
