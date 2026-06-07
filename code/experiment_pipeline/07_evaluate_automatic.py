#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math

import pandas as pd
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

from empathy_pipeline.io_utils import ensure_dir, parse_key_value_arg, read_jsonl
from empathy_pipeline.metrics import add_local_metrics, bootstrap_ci, mean_or_nan, normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute automatic metrics for one or more system output files.")
    parser.add_argument("--system", action="append", required=True, help="NAME=path.jsonl. Can be repeated.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--candidate_field", default=None, help="Defaults to final_response, response, then raw_response.")
    parser.add_argument("--bootstrap_rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def candidate_text(record: dict, preferred_field: str | None = None) -> str:
    if preferred_field and record.get(preferred_field):
        return record[preferred_field]
    return record.get("final_response") or record.get("response") or record.get("raw_response") or ""


def add_rouge(rows: list[dict]) -> None:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    for row in rows:
        cand = row.get("candidate", "")
        ref = row.get("reference_response", "")
        if not normalize_text(cand) or not normalize_text(ref):
            row["rougeL_ref"] = math.nan
        else:
            row["rougeL_ref"] = scorer.score(ref, cand)["rougeL"].fmeasure


def add_bleu_summary(rows: list[dict]) -> float:
    pairs = [(row.get("candidate", ""), row.get("reference_response", "")) for row in rows]
    pairs = [(c, r) for c, r in pairs if normalize_text(c) and normalize_text(r)]
    if not pairs:
        return math.nan
    bleu = BLEU(effective_order=True)
    return float(bleu.corpus_score([c for c, _ in pairs], [[r for _, r in pairs]]).score)


def add_bertscore(rows: list[dict]) -> None:
    for row in rows:
        row["bertscore_f1_ref"] = math.nan
        row["bertscore_f1_raw"] = math.nan
    try:
        from bert_score import score
    except ImportError:
        return

    for lang in ["en", "zh"]:
        idxs = [
            i
            for i, row in enumerate(rows)
            if row.get("language") == lang
            and normalize_text(row.get("candidate"))
            and normalize_text(row.get("reference_response"))
        ]
        if idxs:
            cands = [rows[i]["candidate"] for i in idxs]
            refs = [rows[i]["reference_response"] for i in idxs]
            _, _, f1 = score(cands, refs, lang=lang, verbose=False, rescale_with_baseline=False)
            for i, value in zip(idxs, f1.tolist()):
                rows[i]["bertscore_f1_ref"] = float(value)

        raw_idxs = [
            i
            for i, row in enumerate(rows)
            if row.get("language") == lang
            and normalize_text(row.get("candidate"))
            and normalize_text(row.get("raw_response"))
        ]
        if raw_idxs:
            cands = [rows[i]["candidate"] for i in raw_idxs]
            refs = [rows[i]["raw_response"] for i in raw_idxs]
            _, _, f1 = score(cands, refs, lang=lang, verbose=False, rescale_with_baseline=False)
            for i, value in zip(raw_idxs, f1.tolist()):
                rows[i]["bertscore_f1_raw"] = float(value)


def summarize(system: str, rows: list[dict], bootstrap_rounds: int, seed: int) -> dict:
    metric_names = [
        "bertscore_f1_ref",
        "rougeL_ref",
        "bertscore_f1_raw",
        "fact_exact_coverage",
        "fact_fuzzy_coverage",
        "length_ratio_to_raw",
        "repetition_4gram",
        "empty",
    ]
    summary = {"system": system, "n": len(rows), "bleu_ref": add_bleu_summary(rows)}
    for metric in metric_names:
        values = [float(row.get(metric, math.nan)) for row in rows]
        mean_value = mean_or_nan(values)
        low, high = bootstrap_ci(values, seed=seed, rounds=bootstrap_rounds)
        summary[metric] = mean_value
        summary[f"{metric}_ci_low"] = low
        summary[f"{metric}_ci_high"] = high
    return summary


def main() -> None:
    args = parse_args()
    systems = parse_key_value_arg(args.system)
    all_records = []

    facts_by_id = {}
    for path in systems.values():
        for record in read_jsonl(path):
            if record.get("extracted_facts"):
                facts_by_id[record["id"]] = record["extracted_facts"]

    summaries = []
    for name, path in systems.items():
        records = read_jsonl(path)
        rows = []
        for record in records:
            metric_record = dict(record)
            if not metric_record.get("extracted_facts") and record.get("id") in facts_by_id:
                metric_record["extracted_facts"] = facts_by_id[record["id"]]
            if not metric_record.get("extracted_facts") and candidate_text(metric_record, args.candidate_field) == metric_record.get("raw_response"):
                metric_record["extracted_facts"] = []

            row = add_local_metrics(metric_record, candidate_field=args.candidate_field or "final_response")
            row["system"] = name
            row["candidate"] = candidate_text(record, args.candidate_field)
            row["reference_response"] = record.get("reference_response")
            row["raw_response"] = record.get("raw_response")
            if candidate_text(record, args.candidate_field) == record.get("raw_response") and math.isnan(row["fact_exact_coverage"]):
                row["fact_exact_coverage"] = 1.0
                row["fact_fuzzy_coverage"] = 1.0
            rows.append(row)

        add_rouge(rows)
        add_bertscore(rows)
        all_records.extend(rows)
        summaries.append(summarize(name, rows, args.bootstrap_rounds, args.seed))

    out_dir = ensure_dir(args.out_dir)
    pd.DataFrame(all_records).to_csv(out_dir / "by_record.csv", index=False)
    pd.DataFrame(summaries).to_csv(out_dir / "summary.csv", index=False)
    print(f"Wrote automatic metrics to {out_dir / 'by_record.csv'} and {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
