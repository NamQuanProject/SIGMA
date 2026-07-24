"""Oracle-context in-context-learning baseline.

Dumps *every* context block a question has (SourceExample.context -- MuSiQue's
evidence + distractor paragraphs, HotpotQA's 10 distractor-setting paragraphs, or
NarrativeQA's summary) directly into the prompt, then answers -- no retrieval, no
memory, no adapter. This is the SIGMA-scale analogue of MeMo/baselines/icl's oracle
in-context baseline: an upper bound on what "just paste all the context in" gets you,
without MEMO's vLLM serving stack.

Run with `pip install -e .` already done (see README):

    python -m baselines.icl.run_icl_baseline \\
        --dataset musique --corpus_path data/MuSiQue/dev/musique_corpus_chunks.jsonl \\
        --qns_path data/MuSiQue/dev/musique_questions_chunks.jsonl \\
        --model openai:gpt-4o-mini --limit 100
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle-context ICL baseline (no retrieval, no memory)")
    add_dataset_args(parser)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="'hf:<path>' or 'openai:<model>' (bare strings default to 'hf'), e.g. openai:gpt-4o-mini",
    )
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--log_dir", type=Path, default=Path("logs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("icl_baseline", log_dir=args.log_dir)
    load_environment()

    examples = load_examples(args)
    logger.info(f"Evaluating ICL baseline on {len(examples)} {args.dataset} examples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = build_backend(args.model, device=device, max_new_tokens=args.max_new_tokens)

    predictions: list[str] = []
    golds: list[str] = []
    for example in tqdm(examples, desc="ICL baseline"):
        context_text = render_context(example.context)
        prompt = build_answer_prompt(context_text, example.question)
        predictions.append(backend.generate_raw(prompt))
        golds.append(example.answer)

    scores = aggregate_scores(predictions, golds)
    logger.info(f"ICL baseline ({args.model}): EM={scores['em']:.4f} F1={scores['f1']:.4f} (n={int(scores['n'])})")


if __name__ == "__main__":
    main()
