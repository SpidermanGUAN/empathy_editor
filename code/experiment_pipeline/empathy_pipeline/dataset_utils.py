from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import snapshot_download

from .io_utils import stable_id
from .prompts import history_to_text


def normalize_role(role: str) -> str:
    role = str(role).strip().lower()
    if role in {"assistant", "bot", "chatbot", "system"}:
        return "assistant"
    if role in {"user", "person", "client", "speaker", "human"}:
        return "user"
    return role


def canonical_turn_records(
    *,
    dataset: str,
    split: str,
    conversation_id: str,
    messages: list[dict[str, str]],
    language: str,
    topic: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    clean_messages = [
        {"role": normalize_role(m["role"]), "content": str(m["content"]).strip()}
        for m in messages
        if str(m.get("content", "")).strip()
    ]
    records: list[dict[str, Any]] = []
    history: list[dict[str, str]] = []
    user_turn = 0

    for idx, message in enumerate(clean_messages):
        if message["role"] != "user":
            history.append(message)
            continue

        user_turn += 1
        current_history = history + [message]
        reference_response = None
        if idx + 1 < len(clean_messages) and clean_messages[idx + 1]["role"] == "assistant":
            reference_response = clean_messages[idx + 1]["content"]

        record_id = stable_id(dataset, split, conversation_id, user_turn, message["content"], prefix=dataset)
        records.append(
            {
                "id": record_id,
                "dataset": dataset,
                "split": split,
                "language": language,
                "conversation_id": str(conversation_id),
                "turn_id": user_turn,
                "topic": topic,
                "user_utterance": message["content"],
                "history": current_history,
                "history_text": history_to_text(current_history),
                "reference_response": reference_response,
                "metadata": metadata or {},
            }
        )
        history = current_history

    return records


def load_soulchat(repo_id: str, max_per_split: int | None = None) -> dict[str, list[dict[str, Any]]]:
    from datasets import load_dataset

    ds = load_dataset(repo_id)
    out: dict[str, list[dict[str, Any]]] = {}
    for split in ds.keys():
        records: list[dict[str, Any]] = []
        for idx, conv in enumerate(ds[split]):
            if max_per_split and idx >= max_per_split:
                break
            messages = conv.get("messages") or []
            records.extend(
                canonical_turn_records(
                    dataset="soulchat",
                    split=split,
                    conversation_id=str(conv.get("id", idx)),
                    messages=messages,
                    language="zh",
                    topic=conv.get("topic"),
                    metadata={"source_repo": repo_id},
                )
            )
        out[split] = records
    return out


def _ed_conversations_from_sharegpt(rows: Iterable[dict[str, Any]]) -> Iterable[tuple[str, str | None, list[dict[str, str]], dict[str, Any]]]:
    for idx, row in enumerate(rows):
        convs = row.get("conversations") or row.get("messages")
        if not convs:
            continue
        messages = []
        for turn in convs:
            role = turn.get("role") or turn.get("from") or turn.get("speaker") or "user"
            content = turn.get("content") or turn.get("value") or turn.get("text") or ""
            messages.append({"role": normalize_role(role), "content": content})
        conv_id = str(row.get("id") or row.get("conv_id") or idx)
        topic = row.get("emotion") or row.get("context") or row.get("situation")
        yield conv_id, topic, messages, {"source_format": "sharegpt"}


def _ed_conversations_from_standard_rows(rows: Iterable[dict[str, Any]]) -> Iterable[tuple[str, str | None, list[dict[str, str]], dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        conv_id = str(row.get("conv_id") or row.get("conversation_id") or idx)
        grouped[conv_id].append(row)

    for conv_id, conv_rows in grouped.items():
        conv_rows = sorted(conv_rows, key=lambda r: int(r.get("utterance_idx", r.get("turn_id", 0))))
        messages = []
        for i, row in enumerate(conv_rows):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": row.get("utterance", "")})
        first = conv_rows[0]
        topic = first.get("context") or first.get("prompt")
        yield conv_id, topic, messages, {"source_format": "facebook_standard", "prompt": first.get("prompt")}


def load_empathetic_dialogues(repo_id: str, max_per_split: int | None = None) -> dict[str, list[dict[str, Any]]]:
    from datasets import load_dataset

    ds = load_dataset(repo_id)
    out: dict[str, list[dict[str, Any]]] = {}
    for split in ds.keys():
        rows = list(ds[split])
        if max_per_split:
            rows = rows[:max_per_split]
        if rows and ("conversations" in rows[0] or "messages" in rows[0]):
            iterator = _ed_conversations_from_sharegpt(rows)
        else:
            iterator = _ed_conversations_from_standard_rows(rows)

        records: list[dict[str, Any]] = []
        for conv_id, topic, messages, metadata in iterator:
            metadata["source_repo"] = repo_id
            records.extend(
                canonical_turn_records(
                    dataset="empathetic_dialogues",
                    split=split,
                    conversation_id=conv_id,
                    messages=messages,
                    language="en",
                    topic=topic,
                    metadata=metadata,
                )
            )
        out[split] = records
    return out


def _read_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_personas(persona_ids: list[str], seed: int = 42) -> dict[str, set[str]]:
    rng = random.Random(seed)
    shuffled = list(persona_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.10))
    return {
        "train": set(shuffled[:n_train]),
        "validation": set(shuffled[n_train : n_train + n_val]),
        "test": set(shuffled[n_train + n_val :]),
    }


def load_tide(
    repo_id: str,
    token: str | None = None,
    cache_dir: str | None = None,
    seed: int = 42,
    max_per_split: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    root = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            cache_dir=cache_dir,
        )
    )
    conversation_files = sorted((root / "conversations").glob("*_conversation*.json"))
    if not conversation_files:
        conversation_files = sorted(root.glob("**/*conversation*.json"))
    if not conversation_files:
        raise FileNotFoundError(f"No TIDE conversation files found after downloading {repo_id} to {root}")

    persona_ids = [path.name.split("_")[0] for path in conversation_files]
    split_map = split_personas(persona_ids, seed=seed)
    out = {"train": [], "validation": [], "test": []}

    for conv_path in conversation_files:
        persona_id = conv_path.name.split("_")[0]
        split = next(name for name, ids in split_map.items() if persona_id in ids)
        if max_per_split and len(out[split]) >= max_per_split:
            continue

        metadata_path = root / "metadata" / f"{persona_id}_metadata.json"
        metadata = _read_json_file(metadata_path) if metadata_path.exists() else {}
        conversation_obj = _read_json_file(conv_path)
        conversations = conversation_obj.get("conversations", conversation_obj)
        for turn_key, pair in conversations.items():
            user_text = pair.get("person") or pair.get("client") or pair.get("user_input") or pair.get("user") or ""
            assistant_text = pair.get("chatbot") or pair.get("assistant") or pair.get("reference_response") or pair.get("response") or ""
            if not user_text or not assistant_text:
                continue
            messages = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
            out[split].extend(
                canonical_turn_records(
                    dataset="tide",
                    split=split,
                    conversation_id=f"{persona_id}-{turn_key}",
                    messages=messages,
                    language="en",
                    topic="PTSD support",
                    metadata={"persona_id": persona_id, **metadata, "source_repo": repo_id},
                )
            )
    return out


def load_tide_all_records(
    repo_id: str,
    token: str | None = None,
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    root = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            cache_dir=cache_dir,
        )
    )
    conversation_files = sorted((root / "conversations").glob("*_conversation*.json"))
    if not conversation_files:
        conversation_files = sorted(root.glob("**/*conversation*.json"))
    if not conversation_files:
        raise FileNotFoundError(f"No TIDE conversation files found after downloading {repo_id} to {root}")

    records: list[dict[str, Any]] = []
    for conv_path in conversation_files:
        persona_id = conv_path.name.split("_")[0]
        metadata_path = root / "metadata" / f"{persona_id}_metadata.json"
        metadata = _read_json_file(metadata_path) if metadata_path.exists() else {}
        conversation_obj = _read_json_file(conv_path)
        conversations = conversation_obj.get("conversations", conversation_obj)
        for turn_key, pair in conversations.items():
            user_text = pair.get("person") or pair.get("client") or pair.get("user_input") or pair.get("user") or ""
            assistant_text = pair.get("chatbot") or pair.get("assistant") or pair.get("reference_response") or pair.get("response") or ""
            if not user_text or not assistant_text:
                continue
            messages = [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
            records.extend(
                canonical_turn_records(
                    dataset="tide",
                    split="all",
                    conversation_id=f"{persona_id}-{turn_key}",
                    messages=messages,
                    language="en",
                    topic="PTSD support",
                    metadata={"persona_id": persona_id, **metadata, "source_repo": repo_id},
                )
            )
    return records
