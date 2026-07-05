"""Evaluate a SIGMA memory entry against the unmodified backbone on HotpotQA.

For each validation question: compute the context embedding under the fundamentals-only
adapter, route (trivially, single entry -- see ``memory/single_entry.py``), synthesize a
task-specific adapter (eq. 23-24), patch it onto the frozen backbone, generate an answer,
then compare against the same backbone with no adapter applied (baseline).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from .hotpotqa_reflections import load_hotpotqa_examples
from .memory.apply import apply_adapter, attach_memory
from .memory.entry import MemoryEntry
from .memory.single_entry import SingleEntryMemory
from .reflection_dataset import build_prompt
from .utils.context_embedding import compute_context_embedding
from .utils.metrics import aggregate_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SIGMA memory entry on HotpotQA")
    parser.add_argument("--memory_entry_path", type=Path, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=1, help="alpha ensembling samples, eq. 24")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def generate_answer(model, tokenizer, question: str, *, max_new_tokens: int) -> str:
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    entry = MemoryEntry.load(args.memory_entry_path)
    memory = SingleEntryMemory(entry)
    adapters = attach_memory(model, entry)

    # Fundamentals-only adapter (b̄, no steering) -- matches how context embeddings were
    # computed during consolidation (run_consolidation.py), so the generator sees inputs
    # from the same distribution it was trained on.
    fundamentals = {name: basis.mean.t() for name, basis in entry.layer_bases.items()}

    examples = list(
        load_hotpotqa_examples(split=args.split, config=args.config, limit=args.limit, seed=args.seed)
    )
    logger.info(f"Evaluating on {len(examples)} HotpotQA examples")

    baseline_predictions: list[str] = []
    sigma_predictions: list[str] = []
    golds: list[str] = []

    for i, example in enumerate(examples):
        golds.append(example.answer)

        with apply_adapter(adapters, None):
            baseline_predictions.append(
                generate_answer(model, tokenizer, example.question, max_new_tokens=args.max_new_tokens)
            )

        with apply_adapter(adapters, fundamentals):
            context = compute_context_embedding(model, tokenizer, [build_prompt(example.question)])

        routed_entry = memory.route(context)
        b_prime = routed_entry.synthesize_adapter(context, num_samples=args.num_samples)

        with apply_adapter(adapters, b_prime):
            sigma_predictions.append(
                generate_answer(model, tokenizer, example.question, max_new_tokens=args.max_new_tokens)
            )

        if (i + 1) % 20 == 0:
            logger.info(f"...{i + 1}/{len(examples)}")

    baseline_scores = aggregate_scores(baseline_predictions, golds)
    sigma_scores = aggregate_scores(sigma_predictions, golds)

    logger.info(f"Baseline: EM={baseline_scores['em']:.4f} F1={baseline_scores['f1']:.4f} (n={int(baseline_scores['n'])})")
    logger.info(f"SIGMA:    EM={sigma_scores['em']:.4f} F1={sigma_scores['f1']:.4f} (n={int(sigma_scores['n'])})")


if __name__ == "__main__":
    main()
