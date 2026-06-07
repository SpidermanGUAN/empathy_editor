#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any

from empathy_pipeline.io_utils import read_jsonl, split_records, write_json, write_jsonl
from empathy_pipeline.prompts import PROFILE_SYSTEM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Agent 1 SFT JSONL from existing SFT data or teacher responses.")
    parser.add_argument("--existing_sft_jsonl", default=None, help="JSONL already containing chat messages with assistant labels.")
    parser.add_argument("--requests_jsonl", default=None, help="Teacher request JSONL from 01_build_agent1_teacher_requests.py.")
    parser.add_argument("--teacher_outputs_jsonl", default=None, help="Teacher outputs keyed by custom_id.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--validation_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _extract_teacher_text(row: dict[str, Any]) -> str | None:
    if "assistant" in row:
        return str(row["assistant"])
    if "output_text" in row:
        return str(row["output_text"])
    if "response" in row:
        response = row["response"]
        if isinstance(response, str):
            return response
        body = response.get("body", {}) if isinstance(response, dict) else {}
        choices = body.get("choices") or []
        if choices:
            message = choices[0].get("message", {})
            return message.get("content")
    if "messages" in row and row["messages"] and row["messages"][-1].get("role") == "assistant":
        return row["messages"][-1].get("content")
    return None


def _validate_sft_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            continue
        if messages[0].get("role") != "system":
            messages = [{"role": "system", "content": PROFILE_SYSTEM}] + messages
        if messages[-1].get("role") != "assistant":
            continue
        valid.append({"custom_id": row.get("custom_id"), "messages": messages})
    return valid


def _build_from_teacher_outputs(requests_path: str, outputs_path: str) -> list[dict[str, Any]]:
    requests = {row["custom_id"]: row for row in read_jsonl(requests_path)}
    outputs = read_jsonl(outputs_path)
    rows = []
    for output in outputs:
        custom_id = output.get("custom_id")
        if custom_id not in requests:
            continue
        teacher_text = _extract_teacher_text(output)
        if not teacher_text:
            continue
        req = requests[custom_id]
        messages = list(req["messages"]) + [{"role": "assistant", "content": teacher_text.strip()}]
        rows.append({"custom_id": custom_id, "messages": messages})
    return rows


def main() -> None:
    args = parse_args()
    if args.existing_sft_jsonl:
        rows = _validate_sft_rows(read_jsonl(args.existing_sft_jsonl))
    else:
        if not args.requests_jsonl or not args.teacher_outputs_jsonl:
            raise ValueError("Provide either --existing_sft_jsonl or both --requests_jsonl and --teacher_outputs_jsonl.")
        rows = _build_from_teacher_outputs(args.requests_jsonl, args.teacher_outputs_jsonl)

    train, validation = split_records(rows, validation_ratio=args.validation_ratio, seed=args.seed)
    write_jsonl(train, f"{args.out_dir}/train.jsonl")
    write_jsonl(validation, f"{args.out_dir}/validation.jsonl")
    write_json({"train": len(train), "validation": len(validation)}, f"{args.out_dir}/counts.json")
    print(f"Prepared Agent 1 SFT data: train={len(train)} validation={len(validation)}")


if __name__ == "__main__":
    main()

