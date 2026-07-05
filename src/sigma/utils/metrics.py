"""HotpotQA-style exact-match / F1 scoring, following the standard normalization rules
used by the official HotpotQA and SQuAD evaluation scripts.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def aggregate_scores(predictions: list[str], golds: list[str]) -> dict[str, float]:
    em_scores = [exact_match_score(p, g) for p, g in zip(predictions, golds)]
    f1_scores = [f1_score(p, g) for p, g in zip(predictions, golds)]
    n = max(len(em_scores), 1)
    return {"em": sum(em_scores) / n, "f1": sum(f1_scores) / n, "n": float(len(em_scores))}
