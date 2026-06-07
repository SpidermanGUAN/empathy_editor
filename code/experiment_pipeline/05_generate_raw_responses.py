#!/usr/bin/env python3
from __future__ import annotations

import argparse

from empathy_pipeline.io_utils import read_jsonl, sample_records, write_jsonl
from empathy_pipeline.prompts import raw_response_system
from empathy_pipeline.vllm_utils import VLLMChatGenerator, clear_gpu_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw backbone responses for canonical turn records.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--generation_batch_size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = sample_records(read_jsonl(args.input), args.max_records, seed=args.seed)
    messages_list = []
    for record in records:
        messages = [{"role": "system", "content": raw_response_system(record.get("language", "en"))}]
        messages.extend(record["history"])
        messages_list.append(messages)

    generator = VLLMChatGenerator(
        model=args.model,
        tokenizer_name=args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    responses = []
    try:
        for start in range(0, len(messages_list), max(1, args.generation_batch_size)):
            batch = messages_list[start : start + args.generation_batch_size]
            responses.extend(
                generator.generate(
                    batch,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    min_p=args.min_p,
                    max_tokens=args.max_tokens,
                )
            )
    finally:
        del generator
        clear_gpu_memory()
    out = []
    for record, response in zip(records, responses):
        enriched = dict(record)
        enriched["raw_response"] = response
        enriched["raw_model"] = args.model
        out.append(enriched)
    write_jsonl(out, args.output)
    print(f"Wrote {len(out)} raw responses to {args.output}")


if __name__ == "__main__":
    main()
