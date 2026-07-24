"""Evaluate a SIGMA memory against the unmodified backbone on any of the supported
source datasets (NarrativeQA, MuSiQue -- ``data_sources/LOADERS``, the same loaders
``reflections.py`` uses to generate the reflections a memory was trained on).

Works with either a single-task ``MemoryEntry`` (``--memory_entry_path``) or a
multi-task ``MemoryTree`` (``--memory_tree_path``, see ``build_memory_tree.py``) --
exactly one is required. Both expose the same ``route(context_fn) -> (entry, context)``
shape (``memory/single_entry.py``, ``memory/tree.py``), so the routing/synthesis/eval
logic below doesn't need to know which one it has.

For each validation question in the chosen dataset: route to a task (trivial for a
single entry; GW-distance descent + own-space Mahalanobis for a tree, eq. 25-28),
synthesize a task-specific adapter from the routed entry (eq. 23-24), patch it onto the
frozen backbone, generate an answer, then compare against the same backbone with no
adapter applied (baseline). This is single-shot (one question in, one answer out, scored
with EM/F1) -- there is no multi-turn sub-question loop or judge model here, unlike
MEMO's own evaluation harness (``MeMo/evaluation_pipeline/``), which is built around a
two-model (LM + memory-tuned SM) conversation. SIGMA has no equivalent architecture (one
backbone, one synthesized adapter, one generation call), so this script is the
SIGMA-native analogue of MEMO's single-turn/closed-book paradigms specifically, not a
port of its structured/unstructured multi-turn protocols.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .backends import build_backend
from .data_sources import LOADERS, build_loader_kwargs
from .memory.apply import apply_adapter, apply_entry, attach_memory, attach_memory_tree
from .memory.entry import MemoryEntry
from .memory.single_entry import SingleEntryMemory
from .memory.tree import MemoryTree
from .reflection.dataset import build_prompt
from .utils.context_embedding import compute_context_embedding
from .utils.env import load_environment
from .utils.logging_setup import setup_logging
from .utils.metrics import aggregate_scores

DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a SIGMA memory on a source dataset (NarrativeQA, MuSiQue)."
    )
    memory_group = parser.add_mutually_exclusive_group(required=True)
    memory_group.add_argument("--memory_entry_path", type=Path, default=None, help="A single-task MemoryEntry")
    memory_group.add_argument(
        "--memory_tree_path", type=Path, default=None, help="A multi-task MemoryTree (build_memory_tree.py)"
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset", choices=sorted(LOADERS.keys()), required=True)
    parser.add_argument(
        "--corpus_path",
        type=Path,
        required=True,
        help="Chunked corpus JSONL (produced by python -m sigma.data_process.process_narrativeqa or python -m sigma.data_process.process_musique, "
        "pointed at a held-out file -- see the README). Matches MEMO's own --corpus_path convention.",
    )
    parser.add_argument(
        "--qns_path",
        type=Path,
        required=True,
        help="Chunked questions JSONL (produced by python -m sigma.data_process.process_narrativeqa or python -m sigma.data_process.process_musique). "
        "Matches MEMO's own --qns_path convention.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="The first N in file order (matching MEMO's own loaders), not a random sample -- "
        "for narrativeqa this counts unique source documents. Default: no limit, evaluate on "
        "every row in --qns_path (this runs 2+ generate() calls per example, so start small "
        "to sanity-check timing before running unlimited).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=1, help="alpha ensembling samples, eq. 24")
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
    parser.add_argument("--log_dir", type=Path, default=Path("logs"), help="Where to write this run's log file")
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
    setup_logging("evaluate_sigma", log_dir=args.log_dir)
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

    if args.memory_tree_path is not None:
        memory = MemoryTree.load(args.memory_tree_path)
        adapters = attach_memory_tree(model, memory)
        logger.info(f"Loaded MemoryTree with tasks: {[leaf.name for leaf in memory.leaves()]}")
    else:
        entry = MemoryEntry.load(args.memory_entry_path)
        memory = SingleEntryMemory(entry)
        adapters = attach_memory(model, entry)

    # attach_memory/attach_memory_tree add SharedLoRALinear wrappers (frozen base +
    # shared A + zero-init B) -- move the *whole* patched model to the target
    # device/dtype in one shot, same order train_bootstrap.py uses (attach adapters,
    # then place on device).
    model = model.to(device)
    model.eval()

    loader = LOADERS[args.dataset]
    loader_kwargs = build_loader_kwargs(args)
    examples = list(loader(**loader_kwargs))
    logger.info(f"Evaluating on {len(examples)} {args.dataset} examples")

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

        def context_fn(entry: MemoryEntry, question: str = example.question) -> torch.Tensor:
            # Fundamentals-only adapter (b̄, no steering) for *this* entry -- matches how
            # context embeddings were computed during its consolidation
            # (run_consolidation.py), so its generator sees inputs from the same
            # distribution it was trained on. Different tree entries have different A/basis,
            # hence apply_entry (which swaps both) rather than apply_adapter (B only).
            fundamentals = {name: basis.mean.t() for name, basis in entry.layer_bases.items()}
            with apply_entry(adapters, entry, fundamentals):
                embedding = compute_context_embedding(model, tokenizer, [build_prompt(question)])
            # entry.layer_bases / entry.generator are kept on CPU (tiny compared to the
            # backbone) -- bring the embedding back before any generator/basis math.
            return embedding.cpu()

        routed_entry, context = memory.route(context_fn)
        b_prime = routed_entry.synthesize_adapter(context, num_samples=args.num_samples)

        with apply_entry(adapters, routed_entry, b_prime):
            sigma_predictions.append(
                generate_answer(model, tokenizer, example.question, max_new_tokens=args.max_new_tokens)
            )

        if external_backend is not None:
            external_predictions.append(external_backend.generate(example.question))

        # model.generate() in a tight loop with variable-length prompts (every question
        # is a different length) is a known way to fragment PyTorch's CUDA
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
