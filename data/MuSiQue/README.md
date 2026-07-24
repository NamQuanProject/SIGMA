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

From the repo root, once per file you have (e.g. once for train, once for dev — see
below for why that matters):

```bash
python process_musique.py \
    --musique_path data/MuSiQue/<your_downloaded_file>.json \
    --output_dir data/MuSiQue
```

This writes `musique_corpus_chunks.jsonl` and `musique_questions_chunks.jsonl` into
`--output_dir` — the format `src/sigma/data_sources/musique.py` actually reads.

**If you plan to both train and evaluate on MuSiQue**, run this step twice into two
*different* output directories, once per raw file (e.g. train and dev):

```bash
python process_musique.py --musique_path data/MuSiQue/musique_ans_v1.0_train.jsonl --output_dir data/MuSiQue/train
python process_musique.py --musique_path data/MuSiQue/musique_ans_v1.0_dev.jsonl   --output_dir data/MuSiQue/dev
```

There's no `--split` flag anywhere in this pipeline for MuSiQue — the raw file you feed
`process_musique.py` *is* the split. Pointing both training and evaluation at the same
processed directory means testing on the exact same questions you trained on.

See the repo-root `README.md` for the full pipeline this feeds into.
