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
3. [2. Process/chunk NarrativeQA and MuSiQue](#2-processchunk-narrativeqa-and-musique)
4. [3. Generate reflection data](#3-generate-reflection-data)
5. [4. Bootstrap adapters](#4-bootstrap-adapters)
6. [5. Consolidate into a memory entry](#5-consolidate-into-a-memory-entry)
7. [6. Evaluate](#6-evaluate)
8. [7. (Optional) Multi-task memory tree](#7-optional-multi-task-memory-tree)
9. [8. (Optional) Compare against another model](#8-optional-compare-against-another-model)
10. [How it works](#how-it-works)
11. [Repo layout](#repo-layout)

## 0. Prerequisites

```bash
pip install -r requirements.txt
```

Needs `torch` + `transformers` + `accelerate` for steps 4–6, and real GPU compute for
anything beyond a tiny smoke test (everything *runs* on CPU, it's just not the intended
way).

Put your OpenAI key in a repo-root `.env` file (needed for step 3, and optionally step 8):

```
OPENAI_API_KEY=sk-...
```

(A `.env/` folder layout — `.env/.env` or `.env/local.env` — also works.)

Every script logs to stdout *and* to a timestamped file under `logs/` automatically
(`--log_dir`, default `logs`) — nothing to set up, just know it's there if a run dies and
you want the transcript.

## 1. Download the raw datasets

**HotpotQA** — nothing to do, loads straight from Hugging Face (`hotpotqa/hotpot_qa`) in
step 3.

**NarrativeQA** — clone the official repo:

```bash
git clone https://github.com/google-deepmind/narrativeqa data/NarrativeQA
```

That gives you `documents.csv`, `qaps.csv`, and `third_party/wikipedia/summaries.csv`
directly — all three files step 3 actually reads (after step 2 chunks them). You do **not** need to run the repo's
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

## 2. Process/chunk NarrativeQA and MuSiQue

Mandatory before step 3 for these two datasets. Mirrors MEMO's own
`data_processing_utils/convert_*_to_chunks_jsonl.py` scripts: split each document's text
into overlapping word-count chunks (default 6400 words, 640 overlap — MEMO's own
defaults), and write a MEMO-shaped `corpus.jsonl` (`{docid, text, url}`) +
`questions.jsonl` (`{query_id, question, answers, document_id, evidence_docs, ...}`) pair.
HotpotQA needs no such step — it loads straight from Hugging Face in step 3.

```bash
# NarrativeQA (once per split you plan to use)
python process_narrativeqa.py --narrativeqa_dir data/NarrativeQA --split train
python process_narrativeqa.py --narrativeqa_dir data/NarrativeQA --split valid

# MuSiQue
python process_musique.py \
    --musique_path data/MuSiQue/musique_ans_v1.0_train.jsonl \
    --output_dir data/MuSiQue
```

This writes `narrativeqa_<split>_corpus_chunks.jsonl`/`_questions_chunks.jsonl` next to
the raw NarrativeQA checkout, and `musique_corpus_chunks.jsonl`/`_questions_chunks.jsonl`
into `data/MuSiQue/`. Both are gitignored (regenerate them locally; see each dataset's
`data/*/README.md`). Step 3's loaders require these files to already exist and raise a
clear error naming the exact command to run if they're missing.

Since NarrativeQA's summaries and MuSiQue's paragraphs are both well under 6400 words,
chunking is almost always a no-op (one chunk per document/paragraph) — expected, not a
bug; it's still MEMO's real algorithm, faithfully applied to shorter input than MEMO's
own full-length documents.

## 3. Generate reflection data

Turns the processed corpus into the reflection QA set (`Q_final`) the rest of the
pipeline trains on, using a **document-first, multi-stage pipeline aligned with MEMO's
own reflection synthesis** (`reflection_pipeline.py`, prompts ported near-verbatim from
MEMO's `data_synthesis_pipeline/`) rather than one call per question:

1. **Fact extraction** — two calls per unique document: direct facts (stated outright)
   and indirect facts (require combining two or more statements — pronoun resolution,
   possessive bidirectional extraction, calculated attributes, action chains).
2. **Consolidation** — combines related facts (same entity, same time period, cause and
   effect, ...) into richer multi-fact QA pairs.
3. **Self-containment check/fix** — every QA pair from steps 1–2 is checked for
   self-containment (no pronouns, no relative dates, no document references) and, if it
   fails, rewritten against the source document — capped at `--max_fix_retries` attempts
   per pair (default 1; this stage is already O(#qa_pairs) LLM calls).
4. **Entity surfacing** — "describe the entity through its facts without naming it"
   QA pairs, plus relationship-traversal questions between named entities.
5. **Cross-document synthesis** — for documents that co-occurred in the same source
   example, one anchor-vs-batch call surfaces converging clues (same entity, different
   documents) or parallel properties (different entities, shared trait).

Documents are deduplicated by `(dataset, title)` across all loaded examples first, so a
context block referenced by multiple questions is only processed once.

```bash
# HotpotQA
python generate_reflections.py --dataset hotpotqa --mode openai \
    --output data/hotpotqa_reflections.jsonl --limit 100

# NarrativeQA (needs step 2 run first)
python generate_reflections.py --dataset narrativeqa --mode openai \
    --narrativeqa_dir data/NarrativeQA --split train \
    --output data/narrativeqa_reflections.jsonl --limit 100

# MuSiQue (needs step 2 run first)
python generate_reflections.py --dataset musique --mode openai \
    --musique_dir data/MuSiQue \
    --output data/musique_reflections.jsonl --limit 100
```

Drop `--mode openai` (or set it to `--mode prompt`, the default) for a cheap, offline
dry-run that only exports each document's stage-1 prompt with no LLM calls — useful for
inspecting/debugging coverage and token counts before spending API budget; it doesn't run
consolidation/self-containment/entity-surfacing/cross-doc, since those all depend on
stage 1's actual output.

Every output record carries a `source.type` field set to the pipeline stage that produced
it (`direct`/`indirect`/`consolidated`/`entity_surfacing`/`crossdoc`) — usable as a
`--question_type`-style filter in step 4 if you want to train on only one stage's output.

`generate_hotpotqa_reflections.py` also exists (the original HotpotQA-only single-call
script) — kept only because `evaluate_sigma.py` reuses its dataset loader for the
HotpotQA eval path in step 6. Use `generate_reflections.py` (above) for generating
training reflections.

## 4. Bootstrap adapters

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

**Building one task of several** (for [step 7](#7-optional-multi-task-memory-tree)): add
`--question_type bridge` (or `comparison`) and/or `--level easy|medium|hard` to train on
only a filtered slice of `Q_final`, and run this step once per task into a different
`--output_dir` each time. The filter gets saved into `bootstrap_meta.json` and read back
automatically in step 5, so you never have to repeat it.

**No API version.** This step trains LoRA weights directly on the backbone's own
parameters and needs gradients through it, so it can only ever run against a local model
you have the weights for.

## 5. Consolidate into a memory entry

Decomposes the `M` adapters into a shared basis (PCA by default; Fisher-weighted PCA via
`--consolidation_method fisher`) and trains the coordinate generator on top of it
(eq. 16–22). Also fits a task **signature** for later cross-task use (step 7). Produces
one `MemoryEntry` checkpoint.

```bash
python run_consolidation.py \
    --bootstrap_dir runs/bootstrap \
    --reflections_path data/hotpotqa_reflections.jsonl \
    --output_path runs/memory_entry.pt
```

Same "no API version" reason as step 4 — needs hidden states/gradients off the local
backbone.

## 6. Evaluate

For each validation question: synthesizes a task-specific adapter from the memory entry
(eq. 23–24), patches it onto the frozen backbone, and compares its answer against the
same backbone with no adapter applied. Scores both with exact-match/F1.

```bash
python evaluate_sigma.py \
    --memory_entry_path runs/memory_entry.pt \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --limit 200
```

## 7. (Optional) Multi-task memory tree

Once you have **two or more** `MemoryEntry` checkpoints (e.g. one per dataset, or one per
`--question_type` — see step 4), organize them into a tree and route through it instead
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

## 8. (Optional) Compare against another model

Steps 3–4 need direct access to weights and hidden states, so they can only ever run
against a local model. Evaluation (step 6/7), though, can pull in a **third** set of
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
  embeddings (step 5), shrunk toward the average variance for stability with few samples.
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

Neither NarrativeQA nor MuSiQue is reliably published on Hugging Face, so both are
processed from local files first (step 2: `data_sources/process_narrativeqa.py`,
`data_sources/process_musique.py`, mirroring MEMO's own
`data_processing_utils/convert_*_to_chunks_jsonl.py`), then loaded from the resulting
chunked JSONL (`data_sources/narrativeqa.py`, `data_sources/musique.py`) instead of
`datasets.load_dataset`. MuSiQue's paragraphs already carry `is_supporting` flags
matching HotpotQA's supporting-facts convention; `process_musique.py` accepts either the
official one-JSON-object-per-line format or a single JSON array/object, auto-detected.
Missing `--narrativeqa_dir`/`--musique_dir`, or a directory missing the chunked files
step 2 produces, raises a clear error naming the exact command to run rather than failing
silently.

</details>

<details>
<summary>How the reflection pipeline maps onto MEMO's own synthesis stages (click to expand)</summary>

`reflection_prompts.py` carries MEMO's prompts near-verbatim (direct/indirect fact
extraction, consolidation, self-containment check/fix, entity surfacing, cross-document
anchor combination); `reflection_llm.py` calls an OpenAI-compatible client sequentially
with retries and parses JSON/Python-literal responses out of the reply (MEMO itself
serves these through vLLM with async "hedging" — racing duplicate requests — which this
does not replicate); `reflection_pipeline.py` orchestrates the five stages document-first
(`build_documents` dedups context blocks by `(dataset, title)` across every loaded
example first, then each stage in `reflections.py`'s `run_pipeline` runs once per unique
document, not once per question) and `flatten_to_records` converts the result into the
`source`/`rewritten_qa` record shape `reflection_dataset.py` already expects.

Cross-document synthesis (stage 5) only combines documents that co-occurred as context
in the same source example (`build_documents`'s `groups` return value) — never arbitrary
document pairs from across the whole corpus — and runs one anchor-vs-batch call per
group rather than the full pairwise cross product, to keep cost linear in the number of
examples.

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
├── reflections.py             # step 3: reflection generation CLI, any data_sources/ dataset
├── reflection_prompts.py      # MEMO's own prompts, ported near-verbatim
├── reflection_llm.py          # sequential OpenAI-compatible call + JSON/literal response parsing
├── reflection_pipeline.py     # 5-stage document-first orchestration (see "How the reflection
│                               # pipeline maps onto MEMO's own synthesis stages" above)
├── data_sources/               # normalized loaders: HotpotQA, NarrativeQA, MuSiQue -> SourceExample
│   ├── base.py                  # SourceExample schema
│   ├── chunking.py               # MEMO's chunk_text word-count sliding-window algorithm
│   ├── process_narrativeqa.py   # step 2: raw NarrativeQA -> chunked corpus/questions JSONL
│   ├── process_musique.py       # step 2: raw MuSiQue -> chunked corpus/questions JSONL
│   ├── hotpotqa.py
│   ├── narrativeqa.py            # reads process_narrativeqa.py's output (mandatory)
│   └── musique.py                # reads process_musique.py's output (mandatory)
├── reflection_dataset.py      # Q_final loading + type/level filtering, bootstrap sampling,
│                               # answer-masked tokenization
├── adapters/shared_lora.py    # SharedLoRALinear: frozen shared A, trainable per-adapter B
├── train_bootstrap.py         # step 4
├── consolidate/
│   ├── pca.py                 # PCA / Fisher-weighted PCA consolidation (eq. 16-20)
│   └── generator.py           # coordinate generator (eq. 21-22)
├── run_consolidation.py       # step 5
├── memory/
│   ├── entry.py                # MemoryEntry: basis + generator + signature, synthesize_adapter() (eq. 23-24)
│   ├── signature.py             # TaskSignature: shrunk-diagonal Gaussian fit + Mahalanobis
│   ├── gw.py                    # Gromov-Wasserstein distance + barycenter over signatures (eq. 25-27)
│   ├── tree.py                  # MemoryTree: build/insert/route/consolidate_confusable (eq. 28)
│   ├── apply.py                  # patch/unpatch a synthesized adapter (and, for trees, a task's A) onto a live model
│   └── single_entry.py          # single-task stand-in exposing the same route() shape as MemoryTree
├── evaluate_sigma.py          # step 6 / step 7 (single entry or tree)
├── build_memory_tree.py       # step 7: build+save a MemoryTree from N MemoryEntry files
├── backends/                  # pluggable comparison backends (local HF model, OpenAI, ...) -- step 8
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
