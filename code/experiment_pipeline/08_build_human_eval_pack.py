#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random

import pandas as pd

from empathy_pipeline.io_utils import ensure_dir, parse_key_value_arg, read_jsonl
from empathy_pipeline.prompts import history_to_text


GUIDELINES = """# Human Evaluation Guidelines

Rate each response independently on a 1-5 scale.

Empathy:
1 = dismissive or emotionally inappropriate; 3 = somewhat supportive but generic; 5 = clearly understands and validates the user's emotional state.

Factual accuracy/preservation:
1 = adds or changes important facts; 3 = mostly preserves content with minor unsupported wording; 5 = preserves the provided facts and does not invent claims.

Helpfulness:
1 = not useful or potentially harmful; 3 = somewhat helpful; 5 = concretely supportive and appropriate for the user's situation.

Fluency/coherence:
1 = hard to understand; 3 = understandable but awkward; 5 = natural, coherent, and easy to read.

Preference:
Choose A, B, or Tie according to which response is better overall for this user in this context.

Do not reward a response just because it is longer. Penalize unsupported diagnosis, exaggerated reassurance, moral judgment, or advice that is not grounded in the conversation.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build randomized human-evaluation CSV/XLSX packs.")
    parser.add_argument("--system", action="append", required=True, help="NAME=path.jsonl. Can be repeated.")
    parser.add_argument("--target_system", required=True, help="System compared against every other system.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_items", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default=None, help="Optional dataset filter.")
    return parser.parse_args()


def candidate(record: dict) -> str:
    return record.get("final_response") or record.get("response") or record.get("raw_response") or ""


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    systems = parse_key_value_arg(args.system)
    if args.target_system not in systems:
        raise ValueError(f"--target_system {args.target_system} not found in --system values.")

    loaded = {name: {row["id"]: row for row in read_jsonl(path)} for name, path in systems.items()}
    target_ids = set(loaded[args.target_system])
    rows = []
    key_rows = []

    for baseline_name, baseline_records in loaded.items():
        if baseline_name == args.target_system:
            continue
        common_ids = sorted(target_ids & set(baseline_records))
        if args.dataset:
            common_ids = [rid for rid in common_ids if loaded[args.target_system][rid].get("dataset") == args.dataset]
        if args.max_items and len(common_ids) > args.max_items:
            common_ids = sorted(rng.sample(common_ids, args.max_items))

        for rid in common_ids:
            target = loaded[args.target_system][rid]
            baseline = baseline_records[rid]
            target_side = rng.choice(["A", "B"])
            if target_side == "A":
                response_a, response_b = candidate(target), candidate(baseline)
                system_a, system_b = args.target_system, baseline_name
            else:
                response_a, response_b = candidate(baseline), candidate(target)
                system_a, system_b = baseline_name, args.target_system

            item_id = f"{args.target_system}_vs_{baseline_name}_{rid}"
            rows.append(
                {
                    "item_id": item_id,
                    "dataset": target.get("dataset"),
                    "language": target.get("language"),
                    "conversation_id": target.get("conversation_id"),
                    "turn_id": target.get("turn_id"),
                    "history": target.get("history_text") or history_to_text(target.get("history", [])),
                    "user_utterance": target.get("user_utterance"),
                    "raw_response": target.get("raw_response"),
                    "response_a": response_a,
                    "response_b": response_b,
                    "empathy_a": "",
                    "empathy_b": "",
                    "fact_a": "",
                    "fact_b": "",
                    "helpfulness_a": "",
                    "helpfulness_b": "",
                    "fluency_a": "",
                    "fluency_b": "",
                    "preference": "",
                    "comments": "",
                }
            )
            key_rows.append(
                {
                    "item_id": item_id,
                    "record_id": rid,
                    "system_a": system_a,
                    "system_b": system_b,
                    "target_system": args.target_system,
                    "baseline_system": baseline_name,
                }
            )

    rng.shuffle(rows)
    out_dir = ensure_dir(args.out_dir)
    items_path = out_dir / "human_eval_items.csv"
    key_path = out_dir / "human_eval_key.csv"
    pd.DataFrame(rows).to_csv(items_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(key_rows).to_csv(key_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_excel(out_dir / "human_eval_items.xlsx", index=False)
    with (out_dir / "annotation_guidelines.md").open("w", encoding="utf-8") as f:
        f.write(GUIDELINES)
    print(f"Wrote {len(rows)} blinded items to {items_path}")
    print(f"Wrote key to {key_path}")


if __name__ == "__main__":
    main()

