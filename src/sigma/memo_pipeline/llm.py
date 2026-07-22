"""Synchronous OpenAI calling + robust JSON parsing.

``parse_content_string`` mirrors MEMO's identical helper (repeated verbatim across every
one of their ``data_synthesis_pipeline`` scripts): strip code fences, regex-extract the
first ``{...}``/``[...]`` block, then try ``ast.literal_eval`` before ``json.loads`` (more
forgiving of single-quoted, Python-dict-ish LLM output than ``json.loads`` alone).

Not ported: MEMO's async "hedging" (N parallel duplicate requests racing each other,
first one wins) -- that's reliability engineering against a flaky self-hosted vLLM
server. We call the OpenAI API directly, so a plain sequential retry-with-backoff serves
the same purpose without the complexity.
"""

from __future__ import annotations

import ast
import json
import time
from typing import Any

from loguru import logger


def parse_content_string(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].lstrip()

    obj_start, obj_end = stripped.find("{"), stripped.rfind("}")
    arr_start, arr_end = stripped.find("["), stripped.rfind("]")
    if obj_start != -1 and (arr_start == -1 or obj_start < arr_start):
        start, end = obj_start, obj_end
    else:
        start, end = arr_start, arr_end
    if start != -1 and end != -1 and end > start:
        stripped = stripped[start : end + 1]

    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return json.loads(stripped)


def call_llm_json(
    client: Any,
    *,
    model: str,
    prompt: str,
    temperature: float = 1.1,
    max_retries: int = 3,
) -> Any:
    """One single-turn chat call, parsed as JSON. Returns ``{}`` (never raises) after
    exhausting retries, so one bad document/QA pair can't take down a whole pipeline run
    -- callers should treat an empty/falsy result as "this call produced nothing usable."
    """

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content or "{}"
            return parse_content_string(content)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: API errors, parse errors, etc.
            last_error = exc
            logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {exc}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)

    logger.error(f"LLM call failed after {max_retries} attempts, giving up on this item: {last_error}")
    return {}
