#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd

from empathy_pipeline.io_utils import ensure_dir


DIMENSIONS = ["empathy", "fact", "helpfulness", "fluency"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze completed human-evaluation forms.")
    parser.add_argument("--annotations", nargs="+", required=True, help="CSV files or globs, one per annotator.")
    parser.add_argument("--key", required=True, help="human_eval_key.csv from 08_build_human_eval_pack.py.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--bootstrap_rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def expand_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])
    return sorted(set(paths))


def bootstrap_ci(values: np.ndarray, seed: int, rounds: int) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=values.size, replace=True).mean() for _ in range(rounds)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def fleiss_kappa(label_lists: list[list[int]], labels: list[int]) -> float:
    rows = []
    for item_labels in label_lists:
        counts = [item_labels.count(label) for label in labels]
        if sum(counts) > 1:
            rows.append(counts)
    if not rows:
        return math.nan
    matrix = np.array(rows, dtype=float)
    n_items, n_cats = matrix.shape
    n_raters = matrix.sum(axis=1)
    if np.any(n_raters < 2):
        return math.nan
    p = matrix.sum(axis=0) / matrix.sum()
    p_i = ((matrix * matrix).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_e = (p * p).sum()
    if p_e == 1:
        return math.nan
    return float((p_bar - p_e) / (1 - p_e))


def load_annotations(paths: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        annotator = Path(path).stem
        df = pd.read_csv(path)
        df["annotator"] = annotator
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def pointwise_scores(annotations: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    merged = annotations.merge(key, on="item_id", how="inner")
    rows = []
    for _, row in merged.iterrows():
        for side in ["a", "b"]:
            system = row[f"system_{side}"]
            for dim in DIMENSIONS:
                rows.append(
                    {
                        "item_id": row["item_id"],
                        "record_id": row["record_id"],
                        "annotator": row["annotator"],
                        "system": system,
                        "dimension": dim,
                        "score": pd.to_numeric(row.get(f"{dim}_{side}"), errors="coerce"),
                    }
                )
    return pd.DataFrame(rows)


def summarize_scores(scores: pd.DataFrame, seed: int, rounds: int) -> pd.DataFrame:
    rows = []
    for (system, dimension), group in scores.groupby(["system", "dimension"]):
        values = group["score"].to_numpy(dtype=float)
        low, high = bootstrap_ci(values, seed=seed, rounds=rounds)
        rows.append(
            {
                "system": system,
                "dimension": dimension,
                "n_ratings": int(np.sum(~np.isnan(values))),
                "mean": float(np.nanmean(values)) if np.sum(~np.isnan(values)) else math.nan,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def preference_summary(annotations: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    merged = annotations.merge(key, on="item_id", how="inner")
    rows = []
    for _, row in merged.iterrows():
        pref = str(row.get("preference", "")).strip().upper()
        if pref not in {"A", "B", "TIE"}:
            continue
        if pref == "TIE":
            winner = "tie"
        elif pref == "A":
            winner = row["system_a"]
        else:
            winner = row["system_b"]
        rows.append(
            {
                "target_system": row["target_system"],
                "baseline_system": row["baseline_system"],
                "winner": winner,
            }
        )
    pref_df = pd.DataFrame(rows)
    if pref_df.empty:
        return pref_df
    summary_rows = []
    for (target, baseline), group in pref_df.groupby(["target_system", "baseline_system"]):
        total = len(group)
        target_wins = int((group["winner"] == target).sum())
        baseline_wins = int((group["winner"] == baseline).sum())
        ties = int((group["winner"] == "tie").sum())
        summary_rows.append(
            {
                "target_system": target,
                "baseline_system": baseline,
                "n": total,
                "target_win_rate": target_wins / total,
                "baseline_win_rate": baseline_wins / total,
                "tie_rate": ties / total,
                "target_wins": target_wins,
                "baseline_wins": baseline_wins,
                "ties": ties,
            }
        )
    return pd.DataFrame(summary_rows)


def agreement(scores: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension in DIMENSIONS:
        labels_by_item = []
        dim_scores = scores[scores["dimension"] == dimension]
        for _, group in dim_scores.groupby(["item_id", "system"]):
            labels = [int(x) for x in group["score"].dropna().round().clip(1, 5).tolist()]
            if len(labels) > 1:
                labels_by_item.append(labels)
        rows.append({"target": dimension, "fleiss_kappa": fleiss_kappa(labels_by_item, [1, 2, 3, 4, 5])})

    pref_labels_by_item = []
    for _, group in annotations.groupby("item_id"):
        labels = []
        for pref in group["preference"].dropna().astype(str).str.upper():
            if pref in {"A", "B", "TIE"}:
                labels.append({"A": 1, "B": 2, "TIE": 3}[pref])
        if len(labels) > 1:
            pref_labels_by_item.append(labels)
    rows.append({"target": "preference", "fleiss_kappa": fleiss_kappa(pref_labels_by_item, [1, 2, 3])})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    paths = expand_paths(args.annotations)
    annotations = load_annotations(paths)
    key = pd.read_csv(args.key)
    scores = pointwise_scores(annotations, key)
    score_summary = summarize_scores(scores, args.seed, args.bootstrap_rounds)
    pref = preference_summary(annotations, key)
    agree = agreement(scores, annotations)

    out_dir = ensure_dir(args.out_dir)
    scores.to_csv(out_dir / "human_scores_long.csv", index=False)
    score_summary.to_csv(out_dir / "human_metric_summary.csv", index=False)
    pref.to_csv(out_dir / "pairwise_preferences.csv", index=False)
    agree.to_csv(out_dir / "agreement.csv", index=False)
    print(f"Wrote human-evaluation analysis to {out_dir}")


if __name__ == "__main__":
    main()

