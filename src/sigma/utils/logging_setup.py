"""Shared loguru setup: every CLI script logs to stdout *and* to a timestamped file
under ``--log_dir``, so a run's full output survives after the terminal is gone without
each script having to reimplement this.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger


def setup_logging(name: str, log_dir: Path | str = "logs", level: str = "INFO") -> Path:
    """Reset loguru's sinks to stdout + ``log_dir/<name>_<timestamp>.log``.

    Returns the log file path. Safe to call once per process (each CLI's ``main()``
    calls it right after parsing args) -- ``logger.remove()`` clears any prior sinks
    first, so repeated calls (e.g. in tests) don't stack duplicate handlers.
    """

    logger.remove()
    logger.add(sys.stdout, level=level)

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{name}_{timestamp}.log"
    logger.add(log_path, level=level)
    logger.info(f"Logging to {log_path}")
    return log_path
