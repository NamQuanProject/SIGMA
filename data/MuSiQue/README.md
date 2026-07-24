# MuSiQue — download & setup

MuSiQue isn't reliably published on Hugging Face, so it's loaded from a local file
instead. Everything except this file is gitignored (see `.gitignore`) — you need to
fetch the data yourself.

## 1. Download

Get a MuSiQue JSON/JSONL file that looks like this (one record per question, its
supporting + distractor paragraphs inline):

```json
{"id": "2hop__13548_13529", "question": "...", "answer": "...", "answer_aliases": [],
 "paragraphs": [{"idx": 0, "title": "...", "paragraph_text": "...", "is_supporting": false}, ...]}
```

Two known sources for this shape:

- The official repo, [`StonyBrookNLP/musique`](https://github.com/StonyBrookNLP/musique)
  — its README has the actual download link (a Google Drive zip at time of writing).
  Unzip a split file, e.g. `musique_ans_v1.0_train.jsonl`.
- A pre-merged `musique.json` from the HippoRAG2 dataset release, if you already have
  one lying around — same record shape, just packaged as one JSON array instead of
  JSONL.

Either format works — `process_musique.py` (step 2) auto-detects JSON array vs.
one-object-per-line.

> These links couldn't be re-verified from this environment (no live web browsing at
> the time this file was written). If either 404s, search "MuSiQue StonyBrookNLP
> github" / "HippoRAG2 musique dataset".

## 2. Process into chunks

From the repo root (after `pip install -e .` — see the repo-root README), once per file
you have (e.g. once for train, once for dev — see below for why that matters):

```bash
sigma-process-musique \
    --musique_path data/MuSiQue/<your_downloaded_file>.json \
    --output_dir data/MuSiQue
```

This writes `musique_corpus_chunks.jsonl` and `musique_questions_chunks.jsonl` into
`--output_dir`.

**If you plan to both train and evaluate on MuSiQue**, run this step twice into two
*different* output directories, once per raw file (e.g. train and dev):

```bash
sigma-process-musique --musique_path data/MuSiQue/musique_ans_v1.0_train.jsonl --output_dir data/MuSiQue/train
sigma-process-musique --musique_path data/MuSiQue/musique_ans_v1.0_dev.jsonl   --output_dir data/MuSiQue/dev
```

There's no `--split` flag anywhere in this pipeline for MuSiQue — the raw file you feed
`sigma-process-musique` *is* the split. Pointing both training and evaluation at the same
processed directory means testing on the exact same questions you trained on.

`src/sigma/data_sources/musique.py` reads the two chunked files directly via
`--corpus_path`/`--qns_path` (two explicit file paths, matching MEMO's own
`data_synthesis_pipeline/musique_data_utils.py` convention), e.g.:

```bash
sigma-reflections --dataset musique --mode openai \
    --corpus_path data/MuSiQue/train/musique_corpus_chunks.jsonl \
    --qns_path data/MuSiQue/train/musique_questions_chunks.jsonl \
    --output data/musique_reflections.jsonl --limit 100
```

See the repo-root `README.md` for the full pipeline this feeds into.
