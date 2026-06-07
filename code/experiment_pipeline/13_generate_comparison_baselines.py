#!/usr/bin/env python3
from __future__ import annotations

import argparse

from empathy_pipeline.io_utils import parse_key_value_arg, read_jsonl, write_jsonl
from empathy_pipeline.prompts import raw_response_system
from empathy_pipeline.vllm_utils import VLLMChatGenerator, clear_gpu_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate direct-response comparison baselines with vLLM.")
    parser.add_argument("--input", required=True, help="Canonical JSONL turn records.")
    parser.add_argument("--system", action="append", required=True, help="NAME=MODEL_ID. Can be repeated.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--generation_batch_size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    if args.max_records is not None:
        records = records[: args.max_records]
    systems = parse_key_value_arg(args.system)

    for name, model in systems.items():
        messages_list = []
        for record in records:
            messages = [{"role": "system", "content": raw_response_system(record.get("language", "en"))}]
            messages.extend(record["history"])
            messages_list.append(messages)

        generator = VLLMChatGenerator(
            model=model,
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
                        max_tokens=args.max_tokens,
                    )
                )
        finally:
            del generator
            clear_gpu_memory()
        out = []
        for record, response in zip(records, responses):
            enriched = dict(record)
            enriched["final_response"] = response
            enriched["system_name"] = name
            enriched["model"] = model
            out.append(enriched)
        path = f"{args.out_dir}/{name}.jsonl"
        write_jsonl(out, path)
        print(f"Wrote {len(out)} responses for {name} to {path}")


if __name__ == "__main__":
    main()
