"""LLM calling helpers for the MEMO-aligned reflection pipeline (``reflection_pipeline.py``).

Ported from the deleted ``memo_pipeline/llm.py``. MEMO itself serves its prompts through
vLLM with async "hedging" (racing duplicate requests and keeping whichever returns
first); we call a single OpenAI-compatible client sequentially with retries instead --
same prompts and JSON-parsing contract, simpler execution model.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from loguru import logger


def parse_content_string(text: str) -> Any:
    """Parse a JSON/Python-literal object out of an LLM response that may be wrapped in
    markdown code fences, preceded by preamble, or use single quotes instead of double
    quotes. Returns ``{}`` if nothing parseable is found.
    """

    if not text:
        return {}

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
    candidate = match.group(1) if match else cleaned

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(candidate)
    except (ValueError, SyntaxError):
        pass

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse LLM response as JSON/literal: {text[:200]!r}")
        return {}


def call_llm_json(
    client: Any,
    *,
    model: str,
    prompt: str,
    temperature: float = 1.1,
    max_retries: int = 3,
) -> Any:
    """Call an OpenAI-compatible chat completion endpoint and parse the response as
    JSON, retrying on transport errors or unparseable responses. Returns ``{}`` if every
    attempt fails, so callers can treat that as "no facts extracted" rather than crashing
    the whole pipeline over one bad chunk.
    """

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
            parsed = parse_content_string(content)
            if parsed:
                return parsed
            logger.warning(f"Attempt {attempt}/{max_retries}: empty/unparseable JSON, retrying")
        except Exception as exc:  # noqa: BLE001 - any transport/SDK error should trigger a retry
            last_error = exc
            logger.warning(f"Attempt {attempt}/{max_retries} failed: {exc}")

    if last_error is not None:
        logger.error(f"All {max_retries} attempts failed, last error: {last_error}")
    return {}
