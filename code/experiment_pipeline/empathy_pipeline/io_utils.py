from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}") from exc
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def stable_id(*parts: object, prefix: str | None = None, length: int = 16) -> str:
    raw = "||".join(str(part) for part in parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}-{digest}" if prefix else digest


def sample_records(
    records: list[dict[str, Any]],
    max_records: int | None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if max_records is None or max_records >= len(records):
        return records
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(records)), max_records))
    return [records[i] for i in idxs]


def split_records(
    records: list[dict[str, Any]],
    validation_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * validation_ratio)))
    return shuffled[n_val:], shuffled[:n_val]


def parse_key_value_arg(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got: {value}")
        key, path = value.split("=", 1)
        key = key.strip()
        path = path.strip()
        if not key or not path:
            raise ValueError(f"Expected NAME=PATH, got: {value}")
        parsed[key] = path
    return parsed

