"""BM25 retrieval baseline.

Ranks each question's own candidate context blocks (SourceExample.context -- MuSiQue's
evidence + distractor paragraphs, HotpotQA's 10 distractor-setting paragraphs, or
NarrativeQA's summary) by BM25 against the question, keeps the top ``--top_k``, and
answers from only those. This is the SIGMA-scale analogue of MeMo/baselines/bm25
(retrieve, then answer) -- ranking within each question's own candidate pool rather than
a corpus-wide index, without MEMO's Pyserini/Elasticsearch stack.

Run with `pip install -e .` already done (see README):

    python -m baselines.bm25.run_bm25_baseline \\
        --dataset musique --corpus_path data/MuSiQue/dev/musique_corpus_chunks.jsonl \\
        --qns_path data/MuSiQue/dev/musique_questions_chunks.jsonl \\
        --model openai:gpt-4o-mini --top_k 3 --limit 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm

from sigma.backends import build_backend
from sigma.utils.env import load_environment
from sigma.utils.logging_setup import setup_logging
from sigma.utils.metrics import aggregate_scores

from .._common import add_dataset_args, build_answer_prompt, load_examples, render_context
from .bm25_utils import bm25_rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BM25 retrieval baseline (retrieve top_k, then answer)")
    add_dataset_args(parser)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="'hf:<path>' or 'openai:<model>' (bare strings default to 'hf'), e.g. openai:gpt-4o-mini",
    )
    parser.add_argument("--top_k", type=int, default=3, help="Context blocks to keep per question after ranking")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--log_dir", type=Path, default=Path("logs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("bm25_baseline", log_dir=args.log_dir)
    load_environment()

    examples = load_examples(args)
    logger.info(f"Evaluating BM25 baseline (top_k={args.top_k}) on {len(examples)} {args.dataset} examples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = build_backend(args.model, device=device, max_new_tokens=args.max_new_tokens)

    predictions: list[str] = []
    golds: list[str] = []
    for example in tqdm(examples, desc="BM25 baseline"):
        blocks = example.context
        block_texts = [" ".join(str(s) for s in (b.get("sentences") or [])) for b in blocks]
        ranking = bm25_rank(example.question, block_texts)
        top_blocks = [blocks[i] for i in ranking[: args.top_k]]

        context_text = render_context(top_blocks)
        prompt = build_answer_prompt(context_text, example.question)
        predictions.append(backend.generate_raw(prompt))
        golds.append(example.answer)

    scores = aggregate_scores(predictions, golds)
    logger.info(
        f"BM25 baseline ({args.model}, top_k={args.top_k}): EM={scores['em']:.4f} "
        f"F1={scores['f1']:.4f} (n={int(scores['n'])})"
    )


if __name__ == "__main__":
    main()
