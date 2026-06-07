#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from empathy_pipeline.dataset_utils import load_soulchat
from empathy_pipeline.io_utils import ensure_dir, read_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align existing SoulChat raw/edited JSON response lists with canonical turn-level records."
    )
    parser.add_argument("--soulchat_repo", default="Spiderman01/soulchat_split_raw")
    parser.add_argument("--split", choices=["test", "human_validation"], required=True)
    parser.add_argument("--raw_json", required=True, help="JSON list of raw responses in turn order.")
    parser.add_argument("--edited_json", default=None, help="Optional JSON list of edited responses in turn order.")
    parser.add_argument("--facts_json", default=None, help="Optional JSON list of extracted facts in turn order.")
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--raw_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--editor_name", default="EmpathyEditor")
    parser.add_argument("--max_records", type=int, default=None)
    return parser.parse_args()


def load_response_list(path: str) -> list[str]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return [str(x) for x in data]


def main() -> None:
    args = parse_args()
    splits = load_soulchat(args.soulchat_repo)
    records = splits[args.split]
    raw_responses = load_response_list(args.raw_json)
    edited_responses = load_response_list(args.edited_json) if args.edited_json else None
    facts = load_response_list(args.facts_json) if args.facts_json else None

    if args.max_records is not None:
        records = records[: args.max_records]
        raw_responses = raw_responses[: args.max_records]
        if edited_responses is not None:
            edited_responses = edited_responses[: args.max_records]
        if facts is not None:
            facts = facts[: args.max_records]

    if len(records) != len(raw_responses):
        raise ValueError(
            f"Turn count mismatch for {args.split}: records={len(records)} raw_responses={len(raw_responses)}. "
            "Check that the JSON file was generated from the same split and turn extraction order."
        )
    if edited_responses is not None and len(records) != len(edited_responses):
        raise ValueError(
            f"Turn count mismatch for {args.split}: records={len(records)} edited_responses={len(edited_responses)}."
        )
    if facts is not None and len(records) != len(facts):
        raise ValueError(f"Turn count mismatch for {args.split}: records={len(records)} facts={len(facts)}.")

    out = []
    for i, record in enumerate(records):
        enriched = dict(record)
        enriched["raw_response"] = raw_responses[i]
        enriched["raw_model"] = args.raw_model
        if edited_responses is not None:
            enriched["final_response"] = edited_responses[i]
            enriched["variant"] = args.editor_name
        if facts is not None:
            enriched["extracted_facts"] = facts[i]
        out.append(enriched)

    write_jsonl(out, args.out_jsonl)
    meta = {
        "split": args.split,
        "records": len(out),
        "raw_json": args.raw_json,
        "edited_json": args.edited_json,
        "facts_json": args.facts_json,
    }
    out_path = Path(args.out_jsonl)
    meta_path = ensure_dir(out_path.parent if str(out_path.parent) else ".") / (out_path.name + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(out)} aligned rows to {args.out_jsonl}")
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
