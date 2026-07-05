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
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .backends import build_backend
from .hotpotqa_reflections import load_hotpotqa_examples
from .memory.apply import apply_adapter, attach_memory
from .memory.entry import MemoryEntry
from .memory.single_entry import SingleEntryMemory
from .reflection_dataset import build_prompt
from .utils.context_embedding import compute_context_embedding
from .utils.env import load_environment
from .utils.metrics import aggregate_scores

DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SIGMA memory entry on HotpotQA")
    parser.add_argument("--memory_entry_path", type=Path, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--dataset_name", type=str, default=None, help="Override the HF HotpotQA repo id")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=1, help="alpha ensembling samples, eq. 24")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dtype",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="auto",
        help="auto = bf16 on CUDA, fp32 on CPU (matches typical bootstrap-training precision)",
    )
    parser.add_argument(
        "--empty_cache_every",
        type=int,
        default=20,
        help="Call torch.cuda.empty_cache() every N examples to counter CUDA allocator "
        "fragmentation from generate()-in-a-loop with variable-length prompts (0 disables)",
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default=None,
        help="Optional extra comparison model spec, e.g. 'openai:gpt-4o-mini' or a local HF "
        "path/repo id (bare strings default to a local HF model). Evaluated as a third set "
        "of predictions alongside the unmodified local backbone and the SIGMA-adapted one. "
        "Note: this can only ever be a comparison point -- the memory itself can only attach "
        "to a local model whose weights/hidden states we control.",
    )
    return parser.parse_args()


def generate_answer(model, tokenizer, question: str, *, max_new_tokens: int) -> str:
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
    with torch.inference_mode():
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
    load_environment()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "auto":
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        dtype = DTYPE_MAP[args.dtype]
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)} -- using device={device}, dtype={dtype}")
    else:
        logger.warning(f"CUDA not available -- running on {device} (this will be slow)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype)
    model.resize_token_embeddings(len(tokenizer))

    entry = MemoryEntry.load(args.memory_entry_path)
    memory = SingleEntryMemory(entry)
    adapters = attach_memory(model, entry)

    # attach_memory adds SharedLoRALinear wrappers (frozen base + shared A + zero-init B)
    # -- move the *whole* patched model to the target device/dtype in one shot, same order
    # train_bootstrap.py uses (attach adapters, then place on device).
    model = model.to(device)
    model.eval()

    # Fundamentals-only adapter (b̄, no steering) -- matches how context embeddings were
    # computed during consolidation (run_consolidation.py), so the generator sees inputs
    # from the same distribution it was trained on.
    fundamentals = {name: basis.mean.t() for name, basis in entry.layer_bases.items()}

    examples = list(
        load_hotpotqa_examples(
            split=args.split,
            dataset_name=args.dataset_name,
            config=args.config,
            limit=args.limit,
            seed=args.seed,
        )
    )
    logger.info(f"Evaluating on {len(examples)} HotpotQA examples")

    external_backend = None
    if args.baseline_model:
        logger.info(f"Building external comparison backend: {args.baseline_model!r}")
        external_backend = build_backend(
            args.baseline_model, device=device, dtype=dtype, max_new_tokens=args.max_new_tokens
        )

    baseline_predictions: list[str] = []
    sigma_predictions: list[str] = []
    external_predictions: list[str] = []
    golds: list[str] = []

    progress = tqdm(examples, desc="Evaluating")
    for i, example in enumerate(progress):
        golds.append(example.answer)

        with apply_adapter(adapters, None):
            baseline_predictions.append(
                generate_answer(model, tokenizer, example.question, max_new_tokens=args.max_new_tokens)
            )

        with apply_adapter(adapters, fundamentals):
            context = compute_context_embedding(model, tokenizer, [build_prompt(example.question)])
        # entry.layer_bases / entry.generator are kept on CPU (they're tiny -- a couple of
        # small matrices and an MLP -- no need to ever move them to the GPU); the context
        # embedding comes back on the backbone's device, so bring it back to CPU before
        # handing it to the generator/basis math in synthesize_adapter.
        context = context.cpu()

        routed_entry = memory.route(context)
        b_prime = routed_entry.synthesize_adapter(context, num_samples=args.num_samples)

        with apply_adapter(adapters, b_prime):
            sigma_predictions.append(
                generate_answer(model, tokenizer, example.question, max_new_tokens=args.max_new_tokens)
            )

        if external_backend is not None:
            external_predictions.append(external_backend.generate(example.question))

        # model.generate() in a tight loop with variable-length prompts (every HotpotQA
        # question is a different length) is a known way to fragment PyTorch's CUDA
        # caching allocator, causing generation to gradually get slower over hundreds of
        # calls -- periodically releasing cached (but unused) blocks keeps that in check.
        if args.empty_cache_every and torch.cuda.is_available() and (i + 1) % args.empty_cache_every == 0:
            torch.cuda.empty_cache()

    baseline_scores = aggregate_scores(baseline_predictions, golds)
    sigma_scores = aggregate_scores(sigma_predictions, golds)

    logger.info(f"Baseline: EM={baseline_scores['em']:.4f} F1={baseline_scores['f1']:.4f} (n={int(baseline_scores['n'])})")
    logger.info(f"SIGMA:    EM={sigma_scores['em']:.4f} F1={sigma_scores['f1']:.4f} (n={int(sigma_scores['n'])})")

    if external_backend is not None:
        external_scores = aggregate_scores(external_predictions, golds)
        logger.info(
            f"External ({args.baseline_model}): EM={external_scores['em']:.4f} "
            f"F1={external_scores['f1']:.4f} (n={int(external_scores['n'])})"
        )


if __name__ == "__main__":
    main()
