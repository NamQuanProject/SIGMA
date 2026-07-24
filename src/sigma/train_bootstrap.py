"""Bootstrap-and-consolidate, stage 1: train M LoRA adapters (shared frozen A, per-adapter
trainable B) on bootstrapped subsets of Q_final (SIGMA proposal, section 4.2.1, eq. 15).

Mirrors MemoryDecoder/train_memdec.py's CLI style (argparse + loguru + accelerate), but
trains M independent adapters in one run instead of a single KNN-distillation model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from accelerate import Accelerator
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapters.shared_lora import attach_shared_lora, collect_B_matrices, reset_all_B, trainable_parameters
from .reflection_dataset import AnswerMaskedDataset, bootstrap_subsets, collate_answer_masked, load_qa_examples
from .utils.logging_setup import setup_logging

DEFAULT_TARGET_MODULES = (r"q_proj$", r"v_proj$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train bootstrapped shared-A LoRA adapters on Q_final")
    parser.add_argument("--reflections_path", type=Path, required=True)
    parser.add_argument(
        "--question_type",
        type=str,
        default=None,
        help="Keep only examples with this HotpotQA question type ('bridge'/'comparison'), "
        "to build one task of a multi-task MemoryTree from a single reflections file",
    )
    parser.add_argument(
        "--level",
        type=str,
        default=None,
        help="Keep only examples with this HotpotQA difficulty level ('easy'/'medium'/'hard')",
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_adapters", type=int, default=8, help="M: number of bootstrapped adapters")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--target_modules", type=str, nargs="+", default=list(DEFAULT_TARGET_MODULES))
    parser.add_argument("--bootstrap_size", type=int, default=None, help="Defaults to len(Q_final)")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Kept small by default since bootstrapped adapters retrain from scratch M times; raise it if GPU memory allows",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--log_dir", type=Path, default=Path("logs"), help="Where to write this run's log file")
    return parser.parse_args()


def train_one_adapter(
    model,
    adapters,
    dataset,
    *,
    accelerator: Accelerator,
    tokenizer,
    args: argparse.Namespace,
    adapter_index: int,
) -> float:
    reset_all_B(adapters)
    params = list(trainable_parameters(adapters))
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate(batch):
        return collate_answer_masked(batch, pad_token_id=pad_token_id)

    dataloader = DataLoader(
        dataset, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=collate
    )
    optimizer, dataloader = accelerator.prepare(optimizer, dataloader)

    model.train()
    last_loss = 0.0
    for epoch in range(args.num_train_epochs):
        progress = tqdm(
            dataloader,
            desc=f"adapter {adapter_index} | epoch {epoch}",
            disable=not accelerator.is_local_main_process,
            leave=False,
        )
        for batch in progress:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            accelerator.backward(outputs.loss)
            accelerator.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            last_loss = outputs.loss.item()
            progress.set_postfix(loss=f"{last_loss:.4f}")
        logger.info(f"  epoch {epoch}: loss={last_loss:.4f}")
    model.eval()

    # accelerator.prepare() above registers this adapter's optimizer/dataloader with the
    # Accelerator instance, which persists across the whole M-adapter loop in main() --
    # without releasing them here, accelerate keeps every previous adapter's optimizer
    # state (momentum/variance buffers, roughly 2x param count for Adam) alive for the
    # rest of the run, so memory grows monotonically adapter over adapter until it's
    # exhausted. free_memory() is accelerate's documented way to drop those references
    # "between two trainings with different models/optimizers" -- exactly this loop.
    accelerator.free_memory(optimizer, dataloader)
    return last_loss


def main() -> None:
    args = parse_args()
    setup_logging("train_bootstrap", log_dir=args.log_dir)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        logger.info(
            f"CUDA available: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)} "
            f"-- accelerator will use device={accelerator.device}"
        )
    else:
        logger.warning(f"CUDA not available to this Python/torch install -- training will run on {accelerator.device}")

    logger.info(f"Loading Q_final from {args.reflections_path}")
    examples = load_qa_examples(args.reflections_path, type_filter=args.question_type, level_filter=args.level)
    if not examples:
        raise ValueError(
            f"No examples left after filtering (question_type={args.question_type!r}, level={args.level!r}) -- "
            "check the reflections file actually has that metadata (regenerate with the updated "
            "hotpotqa_reflections.py if it predates type/level support)"
        )
    logger.info(f"Loaded {len(examples)} QA examples")

    subsets = bootstrap_subsets(
        examples, num_subsets=args.num_adapters, subset_size=args.bootstrap_size, seed=args.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    adapters = attach_shared_lora(model, rank=args.lora_rank, target_modules=args.target_modules, seed=args.seed)
    logger.info(f"Attached shared-A LoRA to {len(adapters)} layers: {list(adapters)}")

    model = accelerator.prepare(model)
    model_device = next(model.parameters()).device
    logger.info(f"Model parameters are on device: {model_device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    shared_A = {name: adapter.A.detach().cpu() for name, adapter in adapters.items()}
    torch.save(shared_A, args.output_dir / "shared_A.pt")

    adapter_progress = tqdm(
        list(enumerate(subsets)), desc="Bootstrapping adapters", disable=not accelerator.is_local_main_process
    )
    for m, subset in adapter_progress:
        adapter_progress.set_description(f"Bootstrapping adapter {m + 1}/{len(subsets)}")
        dataset = AnswerMaskedDataset(subset, tokenizer, max_length=args.max_length)
        last_loss = train_one_adapter(
            model, adapters, dataset, accelerator=accelerator, tokenizer=tokenizer, args=args, adapter_index=m
        )
        adapter_progress.set_postfix(loss=f"{last_loss:.4f}")

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 2**30
            reserved = torch.cuda.memory_reserved() / 2**30
            logger.info(f"  GPU memory: {allocated:.2f} GiB allocated / {reserved:.2f} GiB reserved")

        b_matrices = collect_B_matrices(adapters)
        b_matrices_cpu = {name: tensor.cpu() for name, tensor in b_matrices.items()}
        torch.save(b_matrices_cpu, args.output_dir / f"adapter_{m}.pt")

    meta = {
        "model_name_or_path": args.model_name_or_path,
        "lora_rank": args.lora_rank,
        "target_modules": args.target_modules,
        "num_adapters": args.num_adapters,
        "bootstrap_size": args.bootstrap_size,
        "seed": args.seed,
        "layer_names": list(adapters),
        # Persisted so run_consolidation.py reconstructs the *identical* filtered
        # subsets automatically, without the user having to pass matching flags twice.
        "question_type": args.question_type,
        "level": args.level,
    }
    with (args.output_dir / "bootstrap_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved {args.num_adapters} adapters + shared A to {args.output_dir}")


if __name__ == "__main__":
    main()
