"""OpenAI chat-completions backend, used to get a comparison baseline in
`evaluate_sigma.py` from an API model instead of a locally-hosted one.

Note this can only ever be a *comparison point*, not something the SIGMA memory attaches
to -- bootstrap-and-consolidate needs direct access to weights/hidden states (to attach
LoRA and read context embeddings), which an API model doesn't expose.
"""

from __future__ import annotations

import os
import time

from loguru import logger

from ..utils.env import load_environment

DEFAULT_SYSTEM_PROMPT = (
    "Answer the question as briefly as possible: a few words at most, no explanation, "
    "no restating the question."
)


class OpenAIAnswerBackend:
    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_retries: int = 3,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        load_environment()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for the 'openai:' backend -- set it in the repo root .env"
            )

        from openai import OpenAI  # local import: only needed for this backend

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.system_prompt = system_prompt

    def generate(self, question: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, temperature=self.temperature, messages=messages
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:  # pragma: no cover - network/API failure path
                last_error = exc
                logger.warning(f"OpenAI call failed (attempt {attempt + 1}/{self.max_retries}): {exc}")
                time.sleep(2**attempt)
        raise RuntimeError(f"OpenAI backend failed after {self.max_retries} attempts") from last_error
