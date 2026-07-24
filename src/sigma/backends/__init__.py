"""Pluggable answer-generation backends, used both for comparison baselines in
evaluate_sigma.py and for the retrieval baselines in ``baselines/``.

A "model spec" is a string ``"<provider>:<name>"``; a bare string with no ``:`` defaults
to the ``hf`` provider (a local Hugging Face model path). Currently supported providers:

- ``hf:<model_name_or_path>`` -- a local Hugging Face causal LM (no memory attached).
- ``openai:<model>`` -- an OpenAI chat-completions model, e.g. ``openai:gpt-4o-mini``.

Adding another API provider (Anthropic, etc.) means adding one more backend class here
and one more branch in ``build_backend`` -- nothing else in the pipeline needs to change,
since every backend exposes the same ``generate(question) -> str`` method. Both backends
also expose ``generate_raw(prompt) -> str``, which skips the question-only prompt
template ``evaluate_sigma.py`` uses -- that's what ``baselines/`` calls, since its
prompts already embed retrieved context alongside the question.
"""

from __future__ import annotations

from typing import Protocol

import torch

from .hf_backend import HFAnswerBackend
from .openai_backend import OpenAIAnswerBackend

__all__ = ["AnswerBackend", "HFAnswerBackend", "OpenAIAnswerBackend", "build_backend", "parse_model_spec"]


class AnswerBackend(Protocol):
    def generate(self, question: str) -> str: ...
    def generate_raw(self, prompt: str) -> str: ...


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Split a "provider:name" spec; bare strings default to the "hf" provider."""

    if ":" in spec:
        provider, name = spec.split(":", 1)
        if provider in ("hf", "openai"):
            return provider, name
    return "hf", spec


def build_backend(
    spec: str,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    max_new_tokens: int = 16,
) -> AnswerBackend:
    provider, name = parse_model_spec(spec)

    if provider == "openai":
        return OpenAIAnswerBackend(name)
    if provider == "hf":
        return HFAnswerBackend(name, device=device, dtype=dtype, max_new_tokens=max_new_tokens)
    raise ValueError(f"Unknown backend provider {provider!r} (expected 'hf' or 'openai')")
