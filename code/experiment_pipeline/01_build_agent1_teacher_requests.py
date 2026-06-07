#!/usr/bin/env python3
from __future__ import annotations

import argparse

from empathy_pipeline.io_utils import read_jsonl, sample_records, write_jsonl
from empathy_pipeline.prompts import PROFILE_SYSTEM, build_profile_user_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build teacher-model request JSONL for Agent 1 profile distillation.")
    parser.add_argument("--input", required=True, help="Canonical dataset JSONL.")
    parser.add_argument("--output", required=True, help="Output request JSONL.")
    parser.add_argument("--teacher_model", default="teacher-model-name")
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = sample_records(read_jsonl(args.input), args.max_records, seed=args.seed)
    requests = []
    for record in records:
        user_prompt = build_profile_user_prompt(record)
        requests.append(
            {
                "custom_id": record["id"],
                "metadata": {
                    "dataset": record["dataset"],
                    "split": record["split"],
                    "conversation_id": record["conversation_id"],
                    "turn_id": record["turn_id"],
                },
                "messages": [
                    {"role": "system", "content": PROFILE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                "model": args.teacher_model,
                "temperature": args.temperature,
            }
        )
    write_jsonl(requests, args.output)
    print(f"Wrote {len(requests)} teacher requests to {args.output}")


if __name__ == "__main__":
    main()

