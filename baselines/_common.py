"""Shared CLI-args / example-loading / prompt-rendering helpers for baselines/icl and
baselines/bm25.

Reuses ``sigma.data_sources.LOADERS``/``build_loader_kwargs`` -- the same dispatch
``evaluate_sigma.py`` uses -- so a baseline run and a SIGMA run drawn with the same
``--dataset``/``--split``/``--limit``/``--seed`` see the *same* held-out examples and are
directly comparable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sigma.data_sources import LOADERS, SourceExample, build_loader_kwargs

__all__ = ["add_dataset_args", "load_examples", "render_context", "build_answer_prompt"]


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=sorted(LOADERS.keys()), default="hotpotqa")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument(
        "--dataset_name", type=str, default=None, help="Override the HF dataset repo id (--dataset hotpotqa only)"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="HF dataset config, e.g. distractor/fullwiki (--dataset hotpotqa only)"
    )
    parser.add_argument(
        "--streaming", action="store_true", help="Use streaming dataset access (--dataset hotpotqa only)"
    )
    parser.add_argument(
        "--corpus_path",
        type=Path,
        default=None,
        help="Chunked corpus JSONL (sigma-process-narrativeqa/sigma-process-musique output). "
        "Required for --dataset narrativeqa/musique -- matches MEMO's own --corpus_path convention.",
    )
    parser.add_argument(
        "--qns_path",
        type=Path,
        default=None,
        help="Chunked questions JSONL (sigma-process-narrativeqa/sigma-process-musique output). "
        "Required for --dataset narrativeqa/musique -- matches MEMO's own --qns_path convention.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="For narrativeqa/musique this is the first N in file order (matching MEMO), not a "
        "random sample -- for narrativeqa it counts unique source documents.",
    )
    parser.add_argument("--seed", type=int, default=42)


def load_examples(args: argparse.Namespace) -> list[SourceExample]:
    loader = LOADERS[args.dataset]
    return list(loader(**build_loader_kwargs(args)))


def render_context(blocks: list[dict]) -> str:
    """Render a list of ``{"title", "sentences"}`` context blocks (SourceExample.context's
    shape) as plain text, one block per paragraph.
    """

    parts = []
    for block in blocks:
        title = str(block.get("title") or "").strip()
        text = " ".join(str(s) for s in (block.get("sentences") or [])).strip()
        if not text:
            continue
        parts.append(f"{title}: {text}" if title else text)
    return "\n\n".join(parts)


def build_answer_prompt(context_text: str, question: str) -> str:
    if not context_text.strip():
        return f"Question: {question}\nAnswer as briefly as possible, a few words at most:\nAnswer:"
    return (
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer the question as briefly as possible using only the context above: a few "
        "words at most, no explanation, no restating the question.\nAnswer:"
    )
