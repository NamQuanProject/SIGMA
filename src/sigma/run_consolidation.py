"""Bootstrap-and-consolidate, stage 2: PCA/Fisher-PCA consolidation of the M bootstrapped
adapters (eq. 16-20), then training the coordinate generator (eq. 21-22) on the resulting
{(context_embedding_m, alpha_m)} supervised set. Saves one ``MemoryEntry``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapters.shared_lora import attach_shared_lora
from .consolidate.generator import CoordinateGenerator, train_generator
from .consolidate.pca import compute_diagonal_fisher, consolidate_cloud, fisher_weighted_consolidate
from .memory.entry import CoordinateLayout, MemoryEntry
from .reflection_dataset import build_prompt, bootstrap_subsets, load_qa_examples
from .utils.context_embedding import compute_context_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate bootstrapped adapters into a MemoryEntry")
    parser.add_argument("--bootstrap_dir", type=Path, required=True)
    parser.add_argument(
        "--reflections_path",
        type=Path,
        required=True,
        help="Same file used for train_bootstrap.py, to recompute identical subsets + context embeddings",
    )
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--consolidation_method", choices=["pca", "fisher"], default="pca")
    parser.add_argument("--explained_variance_threshold", type=float, default=0.9)
    parser.add_argument("--fisher_holdout_fraction", type=float, default=0.1)
    parser.add_argument("--generator_hidden_dim", type=int, default=256)
    parser.add_argument("--generator_epochs", type=int, default=200)
    parser.add_argument("--generator_lr", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    meta = json.loads((args.bootstrap_dir / "bootstrap_meta.json").read_text())
    shared_A = torch.load(args.bootstrap_dir / "shared_A.pt", weights_only=True)
    b_matrices = {
        str(m): torch.load(args.bootstrap_dir / f"adapter_{m}.pt", weights_only=True)
        for m in range(meta["num_adapters"])
    }
    logger.info(f"Loaded {len(b_matrices)} bootstrapped adapters from {args.bootstrap_dir}")

    tokenizer = AutoTokenizer.from_pretrained(meta["model_name_or_path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(meta["model_name_or_path"])
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    target_modules = [f"^{re.escape(name)}$" for name in meta["layer_names"]]
    adapters = attach_shared_lora(model, rank=meta["lora_rank"], target_modules=target_modules, seed=meta["seed"])
    for name, adapter in adapters.items():
        adapter.A.copy_(shared_A[name].to(adapter.A.dtype))

    examples = load_qa_examples(args.reflections_path)
    subsets = bootstrap_subsets(
        examples, num_subsets=meta["num_adapters"], subset_size=meta.get("bootstrap_size"), seed=meta["seed"]
    )

    if args.consolidation_method == "pca":
        layer_bases = consolidate_cloud(b_matrices, explained_variance_threshold=args.explained_variance_threshold)
    else:
        mean_b = {
            name: torch.stack([b_matrices[str(m)][name] for m in range(meta["num_adapters"])]).mean(dim=0)
            for name in meta["layer_names"]
        }
        holdout_size = max(1, int(len(examples) * args.fisher_holdout_fraction))
        holdout_examples = examples[:holdout_size]
        logger.info(f"Estimating Fisher on {len(holdout_examples)} held-out examples")
        fisher = compute_diagonal_fisher(model, adapters, mean_b, holdout_examples, tokenizer)
        layer_bases = fisher_weighted_consolidate(
            b_matrices, fisher, explained_variance_threshold=args.explained_variance_threshold
        )

    layout = CoordinateLayout.from_layer_bases(layer_bases)
    logger.info(f"Consolidated basis dims per layer: { {k: v.basis_dims for k, v in layer_bases.items()} }")

    # Set adapters to fundamentals-only (mean, no steering) to compute each bootstrap
    # subset's context embedding -- this is the "double duty" embedding of eq. before 21.
    for name, adapter in adapters.items():
        adapter.B.data.copy_(layer_bases[name].mean.t().to(adapter.B.dtype))

    contexts, targets = [], []
    for m, subset in enumerate(subsets):
        prompts = [build_prompt(ex.question) for ex in subset]
        embedding = compute_context_embedding(model, tokenizer, prompts).mean(dim=0)
        contexts.append(embedding.cpu())
        targets.append(layout.flatten(layer_bases, adapter_index=m))

    contexts_tensor = torch.stack(contexts)
    targets_tensor = torch.stack(targets)
    logger.info(
        f"Training coordinate generator: context_dim={contexts_tensor.shape[1]} alpha_dim={targets_tensor.shape[1]}"
    )

    generator = CoordinateGenerator(
        context_dim=contexts_tensor.shape[1],
        alpha_dim=targets_tensor.shape[1],
        hidden_dim=args.generator_hidden_dim,
    )
    train_generator(
        generator, contexts_tensor, targets_tensor, num_epochs=args.generator_epochs, learning_rate=args.generator_lr
    )

    entry = MemoryEntry(shared_A=shared_A, layer_bases=layer_bases, layout=layout, generator=generator)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    entry.save(args.output_path)
    logger.info(f"Saved MemoryEntry to {args.output_path}")


if __name__ == "__main__":
    main()
