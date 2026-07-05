"""Build a cross-task memory tree (proposal section 4.2.2) from N already-consolidated
``MemoryEntry`` checkpoints, one per task (see ``run_consolidation.py``).

Every entry must have a signature attached (the default in ``run_consolidation.py``) and
share the same rank/target layers (i.e. every task was bootstrapped with the same
``--lora_rank``/``--target_modules``) -- see ``memory/apply.py:attach_memory_tree`` for
why the latter is required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from .memory.entry import MemoryEntry
from .memory.tree import MemoryTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a MemoryTree from N consolidated MemoryEntry files")
    parser.add_argument(
        "--task",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="One task's name and MemoryEntry path, e.g. --task bridge=runs/bridge/memory_entry.pt "
        "-- pass this flag once per task (at least 2 needed for a tree to be interesting, "
        "though 1 is allowed)",
    )
    parser.add_argument("--output_path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    entries: dict[str, MemoryEntry] = {}
    for item in args.task:
        if "=" not in item:
            raise ValueError(f"--task must be NAME=PATH, got {item!r}")
        name, path_str = item.split("=", 1)
        name = name.strip()
        if name in entries:
            raise ValueError(f"Duplicate task name {name!r}")
        logger.info(f"Loading task {name!r} from {path_str}")
        entries[name] = MemoryEntry.load(Path(path_str.strip()))

    tree = MemoryTree.build(entries)
    logger.info(f"Built tree over {len(entries)} task(s): {[leaf.name for leaf in tree.leaves()]}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.save(args.output_path)
    logger.info(f"Saved MemoryTree to {args.output_path}")


if __name__ == "__main__":
    main()
