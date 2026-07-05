"""CLI wrapper for building a SIGMA cross-task memory tree."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sigma.build_memory_tree import main

if __name__ == "__main__":
    main()
