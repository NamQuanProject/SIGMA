"""Shared .env loading, used by anything that needs API keys (OpenAI reflections
generation, OpenAI evaluation backend, ...).
"""

from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency guard
    load_dotenv = None


def load_environment() -> None:
    """Load environment variables from local .env files if available.

    Supports both a plain repo-root `.env` file and a `.env/` folder layout.
    """

    if load_dotenv is None:
        return

    repo_root = Path(__file__).resolve().parents[3]
    env_candidates = [
        repo_root / ".env",
        repo_root / ".env.local",
        repo_root / ".env" / ".env",
        repo_root / ".env" / "local.env",
    ]
    for candidate in env_candidates:
        if candidate.is_file():
            load_dotenv(candidate, override=False)
