# Baselines

Non-SIGMA comparison points, analogous to [`MeMo/baselines/`](../MeMo/baselines), scoped
to what's actually runnable here (no vLLM serving, no DeepSpeed, no Pyserini/Elasticsearch
index, no LLM-judge). Every baseline scores with the same exact-match/F1 metric SIGMA
itself uses (`sigma.utils.metrics`), and reads the same `--dataset`/`--corpus_path`/
`--qns_path`/`--limit` CLI surface as `evaluate_sigma.py`'s `data_sources` loaders, so a
baseline run and a SIGMA run drawn with matching flags are evaluated on the *same*
held-out examples.

Requires `pip install -e .` from the repo root first (see the main [README](../README.md)).

| Here | MeMo equivalent | What it does |
|---|---|---|
| `icl/run_icl_baseline.py` | `baselines/icl/` | Dumps every context block a question has straight into the prompt (oracle context), no retrieval, no memory. An upper bound on "just paste everything in." |
| `bm25/run_bm25_baseline.py` | `baselines/bm25/` | Ranks each question's own candidate context blocks with BM25, keeps the top `--top_k`, answers from only those. |
| `evaluate_sigma.py --baseline_model` (in the main pipeline, not here) | `single_turn_baseline/` | The plain backbone with **no** context and **no** memory -- the floor everything else should beat. Already built into the main eval script since it needs no extra infrastructure. |

Not ported: MeMo's `cartridges/`, `hipporag2/`, `nv_embed/` baselines depend on
infrastructure this repo doesn't set up (a training-time cartridge/KV-cache format,
OpenIE + a graph index, a dedicated embedding-model server). `evaluation_pipeline/`'s
multi-turn LM+SM protocols are architecturally incompatible with SIGMA's single-adapter
design -- see the "How it works" note in the main README for why.

## Usage

Both scripts take the same dataset flags as `evaluate_sigma.py` (`--dataset`,
`--corpus_path`/`--qns_path` for the two datasets that need chunked local data first --
see the main README's steps 2-3, `--limit`) plus `--model`, a spec of the form
`hf:<local model path>` or `openai:<model>` (needs `OPENAI_API_KEY`).
`--corpus_path`/`--qns_path` are two explicit file paths, matching MEMO's own
`--corpus_path`/`--qns_path` CLI convention exactly (see the main README's note on this).

```bash
# Oracle-context ICL baseline
python -m baselines.icl.run_icl_baseline \
    --dataset musique --corpus_path data/MuSiQue/dev/musique_corpus_chunks.jsonl \
    --qns_path data/MuSiQue/dev/musique_questions_chunks.jsonl \
    --model openai:gpt-4o-mini --limit 100

# BM25 retrieval baseline (top 3 chunks per question)
python -m baselines.bm25.run_bm25_baseline \
    --dataset musique --corpus_path data/MuSiQue/dev/musique_corpus_chunks.jsonl \
    --qns_path data/MuSiQue/dev/musique_questions_chunks.jsonl \
    --model openai:gpt-4o-mini --top_k 3 --limit 100
```

Both print `EM=... F1=... (n=...)` at the end, in the same format
`evaluate_sigma.py` uses, so you can put all three numbers (baseline backbone, a
retrieval baseline, SIGMA) side by side.
