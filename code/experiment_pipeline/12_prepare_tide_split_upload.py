#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random

from datasets import Dataset, DatasetDict

from empathy_pipeline.dataset_utils import load_tide_all_records
from empathy_pipeline.io_utils import write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download TIDE, create 100-row human validation plus test split, and optionally push to HF.")
    parser.add_argument("--tide_repo", default="yenopoya/TIDE")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--human_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push_repo", default=None, help="Optional HF dataset repo id, e.g. Spiderman01/tide_split_raw.")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--cache_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_tide_all_records(args.tide_repo, token=args.hf_token, cache_dir=args.cache_dir)
    rng = random.Random(args.seed)
    records = list(records)
    rng.shuffle(records)

    human = records[: args.human_size]
    test = records[args.human_size :]
    for record in human:
        record["split"] = "human_validation"
    for record in test:
        record["split"] = "test"

    write_jsonl(human, f"{args.out_dir}/tide_human_validation.jsonl")
    write_jsonl(test, f"{args.out_dir}/tide_test.jsonl")
    write_json(
        {
            "source_repo": args.tide_repo,
            "human_validation": len(human),
            "test": len(test),
            "seed": args.seed,
        },
        f"{args.out_dir}/tide_split_counts.json",
    )

    if args.push_repo:
        ds = DatasetDict(
            {
                "human_validation": Dataset.from_list(human),
                "test": Dataset.from_list(test),
            }
        )
        ds.push_to_hub(args.push_repo, token=args.hf_token, private=args.private)
        print(f"Pushed TIDE split to https://huggingface.co/datasets/{args.push_repo}")

    print(f"TIDE split complete: human_validation={len(human)} test={len(test)}")


if __name__ == "__main__":
    main()

