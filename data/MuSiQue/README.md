# MuSiQue — download & setup

MuSiQue isn't reliably published on Hugging Face, so it's loaded from a local file
instead. Everything except this file is gitignored (see `.gitignore`) — you need to
fetch the data yourself.

## 1. Download

This project's copy of MuSiQue is `musique.json` from the **HippoRAG_2** dataset on
Hugging Face — **not** the official StonyBrookNLP repo directly:

**https://huggingface.co/datasets/osunlp/HippoRAG_2/blob/main/musique.json**

Download it into `data/MuSiQue/musique.json`. It's a single JSON array, one record per
question, with its supporting + distractor paragraphs inline:

```json
[{"id": "2hop__13548_13529", "question": "...", "answer": "...", "answer_aliases": [],
  "paragraphs": [{"idx": 0, "title": "...", "paragraph_text": "...", "is_supporting": false}, ...]}, ...]
```

(The official [`StonyBrookNLP/musique`](https://github.com/StonyBrookNLP/musique) repo
distributes the same record shape too, just as one-JSON-object-per-line files instead of
a single array — `process_musique.py` (step 2) auto-detects either format, so either
source works if you already have one, but the HippoRAG_2 link above is the one this
project's own `data/MuSiQue/musique.json` actually came from.)

## 2. Process into chunks

From the repo root (after `pip install -e .` — see the repo-root README):

```bash
python -m sigma.data_process.process_musique \
    --musique_path data/MuSiQue/musique.json \
    --output_dir data/MuSiQue
```

This writes `musique_corpus_chunks.jsonl` and `musique_questions_chunks.jsonl` into
`--output_dir`.

**Note on train/dev:** the HippoRAG_2 `musique.json` above is a single, fixed ~1,000
question subset of MuSiQue (a common evaluation-oriented sample used by several RAG
papers) — it isn't pre-split into train/dev files the way the official StonyBrookNLP
release is. Using the same processed file for both reflection generation (step 4 in the
repo-root README) and evaluation (step 7) means testing on the exact same questions you
trained on, which will look better than it is. If you want a genuine held-out split,
either download the official StonyBrookNLP train/dev JSONL files instead (same record
shape, see step 1) and run this step once per file into separate `--output_dir`s, or
split `musique.json`'s 1,000 records yourself before running this step.

`src/sigma/data_sources/musique.py` reads the two chunked files directly via
`--corpus_path`/`--qns_path` (two explicit file paths, matching MEMO's own
`data_synthesis_pipeline/musique_data_utils.py` convention), e.g.:

```bash
python -m sigma.reflections --dataset musique --mode openai \
    --corpus_path data/MuSiQue/musique_corpus_chunks.jsonl \
    --qns_path data/MuSiQue/musique_questions_chunks.jsonl \
    --output data/musique_reflections.jsonl --limit 100
```

See the repo-root `README.md` for the full pipeline this feeds into.
