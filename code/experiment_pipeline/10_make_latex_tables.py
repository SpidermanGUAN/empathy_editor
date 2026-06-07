#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from empathy_pipeline.io_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create LaTeX-ready tables from automatic and human metric summaries.")
    parser.add_argument("--automatic", default=None, help="results/automatic/summary.csv")
    parser.add_argument("--human", default=None, help="results/human/.../human_metric_summary.csv")
    parser.add_argument("--preferences", default=None, help="results/human/.../pairwise_preferences.csv")
    parser.add_argument("--out_dir", required=True)
    return parser.parse_args()


def fmt(x: object, digits: int = 3) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return ""
    if pd.isna(value):
        return ""
    return f"{value:.{digits}f}"


def automatic_table(path: str) -> str:
    df = pd.read_csv(path)
    keep = [
        "system",
        "bertscore_f1_ref",
        "rougeL_ref",
        "bertscore_f1_raw",
        "fact_fuzzy_coverage",
        "length_ratio_to_raw",
        "repetition_4gram",
    ]
    df = df[[col for col in keep if col in df.columns]]
    lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "System & BERTScore Ref & ROUGE-L & BERTScore Raw & Fact Cov. & Len. Ratio & Rep. \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['system']} & {fmt(row.get('bertscore_f1_ref'))} & {fmt(row.get('rougeL_ref'))} & "
            f"{fmt(row.get('bertscore_f1_raw'))} & {fmt(row.get('fact_fuzzy_coverage'))} & "
            f"{fmt(row.get('length_ratio_to_raw'))} & {fmt(row.get('repetition_4gram'))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def human_table(path: str) -> str:
    df = pd.read_csv(path)
    pivot = df.pivot_table(index="system", columns="dimension", values="mean", aggfunc="mean").reset_index()
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "System & Empathy & Fact. & Help. & Fluency \\\\",
        "\\midrule",
    ]
    for _, row in pivot.iterrows():
        lines.append(
            f"{row['system']} & {fmt(row.get('empathy'), 2)} & {fmt(row.get('fact'), 2)} & "
            f"{fmt(row.get('helpfulness'), 2)} & {fmt(row.get('fluency'), 2)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def preference_table(path: str) -> str:
    df = pd.read_csv(path)
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Comparison & Win & Loss & Tie & N \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        comparison = f"{row['target_system']} vs. {row['baseline_system']}"
        lines.append(
            f"{comparison} & {fmt(row.get('target_win_rate'), 2)} & {fmt(row.get('baseline_win_rate'), 2)} & "
            f"{fmt(row.get('tie_rate'), 2)} & {int(row.get('n', 0))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    if args.automatic:
        Path(out_dir / "automatic_main_table.tex").write_text(automatic_table(args.automatic), encoding="utf-8")
    if args.human:
        Path(out_dir / "human_rating_table.tex").write_text(human_table(args.human), encoding="utf-8")
    if args.preferences:
        Path(out_dir / "human_preference_table.tex").write_text(preference_table(args.preferences), encoding="utf-8")
    print(f"Wrote LaTeX tables to {out_dir}")


if __name__ == "__main__":
    main()

