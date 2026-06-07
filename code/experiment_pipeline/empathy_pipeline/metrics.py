from __future__ import annotations

import ast
import math
import re
from difflib import SequenceMatcher
from typing import Any

import numpy as np


def normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_facts(facts: str | list[str] | None) -> list[str]:
    if facts is None:
        return []
    if isinstance(facts, list):
        return [normalize_text(x) for x in facts if normalize_text(x)]
    text = facts.strip()
    if not text or text in {"[]", "No factual assertions to preserve."}:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [normalize_text(x) for x in parsed if normalize_text(x)]
    except (SyntaxError, ValueError):
        pass

    bracketed = re.findall(r"\[([^\[\]]+)\]", text)
    if bracketed:
        return [normalize_text(x.strip("\"' ")) for x in bracketed if normalize_text(x)]
    lines = [line.strip("-* \t") for line in text.splitlines()]
    return [normalize_text(line) for line in lines if normalize_text(line)]


def exact_fact_coverage(response: str, facts: list[str]) -> float:
    if not facts:
        return math.nan
    response_norm = normalize_text(response).lower()
    hits = sum(1 for fact in facts if normalize_text(fact).lower() in response_norm)
    return hits / len(facts)


def fuzzy_fact_coverage(response: str, facts: list[str], threshold: float = 0.82) -> float:
    if not facts:
        return math.nan
    response_norm = normalize_text(response).lower()
    hits = 0
    for fact in facts:
        fact_norm = normalize_text(fact).lower()
        if not fact_norm:
            continue
        if fact_norm in response_norm:
            hits += 1
            continue
        if SequenceMatcher(None, fact_norm, response_norm).ratio() >= threshold:
            hits += 1
    return hits / len(facts)


def length_ratio(candidate: str, source: str) -> float:
    source_len = max(1, len(normalize_text(source)))
    return len(normalize_text(candidate)) / source_len


def repetition_rate(text: str, ngram: int = 4) -> float:
    tokens = normalize_text(text).split()
    if len(tokens) < ngram:
        return 0.0
    grams = [tuple(tokens[i : i + ngram]) for i in range(len(tokens) - ngram + 1)]
    return 1.0 - (len(set(grams)) / max(1, len(grams)))


def mean_or_nan(values: list[float]) -> float:
    arr = np.array([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return math.nan
    return float(arr.mean())


def bootstrap_ci(values: list[float], seed: int = 42, rounds: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    arr = np.array([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(rounds):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(sample.mean())
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return low, high


def add_local_metrics(record: dict[str, Any], candidate_field: str = "final_response") -> dict[str, Any]:
    candidate = record.get(candidate_field) or record.get("response") or record.get("raw_response") or ""
    raw = record.get("raw_response") or ""
    facts = parse_facts(record.get("extracted_facts"))
    return {
        "id": record.get("id"),
        "dataset": record.get("dataset"),
        "split": record.get("split"),
        "language": record.get("language"),
        "fact_exact_coverage": exact_fact_coverage(candidate, facts),
        "fact_fuzzy_coverage": fuzzy_fact_coverage(candidate, facts),
        "length_ratio_to_raw": length_ratio(candidate, raw),
        "repetition_4gram": repetition_rate(candidate, 4),
        "empty": 1.0 if not normalize_text(candidate) else 0.0,
    }
