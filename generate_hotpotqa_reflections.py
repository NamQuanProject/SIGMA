"""CLI wrapper for HotpotQA reflection generation."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sigma.hotpotqa_reflections import main


if __name__ == "__main__":
    main()
