#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from empathy_pipeline.dataset_utils import load_empathetic_dialogues, load_soulchat, load_tide
from empathy_pipeline.io_utils import write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare canonical JSONL datasets for EmpathyEditor experiments.")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--include", nargs="+", default=["soulchat", "empathetic_dialogues", "tide"])
    parser.add_argument("--soulchat_repo", default="Spiderman01/soulchat_split_raw")
    parser.add_argument("--ed_repo", default="Estwld/empathetic_dialogues_llm")
    parser.add_argument("--tide_repo", default="yenopoya/TIDE")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_per_split", type=int, default=None)
    return parser.parse_args()


def write_dataset(name: str, splits: dict[str, list[dict]], out_dir: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split, records in splits.items():
        path = f"{out_dir}/{name}_{split}.jsonl"
        write_jsonl(records, path)
        counts[f"{name}_{split}"] = len(records)
    return counts


def main() -> None:
    args = parse_args()
    include = set(args.include)
    counts: dict[str, int] = {}

    if "soulchat" in include:
        soulchat = load_soulchat(args.soulchat_repo, max_per_split=args.max_per_split)
        counts.update(write_dataset("soulchat", soulchat, args.out_dir))

    if "empathetic_dialogues" in include or "ed" in include:
        ed = load_empathetic_dialogues(args.ed_repo, max_per_split=args.max_per_split)
        counts.update(write_dataset("empathetic_dialogues", ed, args.out_dir))

    if "tide" in include:
        tide = load_tide(
            args.tide_repo,
            token=args.hf_token,
            cache_dir=args.cache_dir,
            seed=args.seed,
            max_per_split=args.max_per_split,
        )
        counts.update(write_dataset("tide", tide, args.out_dir))

    write_json(counts, f"{args.out_dir}/dataset_counts.json")
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
