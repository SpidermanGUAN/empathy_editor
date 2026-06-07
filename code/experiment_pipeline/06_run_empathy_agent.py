#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict

from empathy_pipeline.io_utils import read_jsonl, sample_records, write_jsonl
from empathy_pipeline.prompts import (
    FACT_SYSTEM,
    FUSION_NO_FACT_SYSTEM,
    FUSION_SYSTEM,
    PROFILE_UPDATE_SYSTEM,
    build_fact_user_prompt,
    build_fusion_user_prompt,
    build_profile_user_prompt,
    chat_messages,
)
from empathy_pipeline.vllm_utils import VLLMChatGenerator, clear_gpu_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EmpathyEditor full system or ablation variants.")
    parser.add_argument("--input", required=True, help="JSONL with raw_response.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=["full", "no_profile", "no_fact", "no_tom_map", "fusion_only"], default="full")
    parser.add_argument("--agent1_model", default="Spiderman01/finetuned_Qwen_35_08B_empathy")
    parser.add_argument("--agent2_model", default="unsloth/Qwen3.5-0.8B")
    parser.add_argument("--fusion_model", default="unsloth/Qwen3.5-0.8B")
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens_profile", type=int, default=768)
    parser.add_argument("--max_tokens_fact", type=int, default=384)
    parser.add_argument("--max_tokens_fusion", type=int, default=512)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_profile_chars", type=int, default=5000)
    parser.add_argument("--generation_batch_size", type=int, default=1024)
    return parser.parse_args()


def _run_generator(generator: VLLMChatGenerator, messages_list: list[list[dict[str, str]]], max_tokens: int, args: argparse.Namespace) -> list[str]:
    if not messages_list:
        return []
    outputs: list[str] = []
    batch_size = max(1, args.generation_batch_size)
    for start in range(0, len(messages_list), batch_size):
        batch = messages_list[start : start + batch_size]
        outputs.extend(
            generator.generate(
                batch,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=max_tokens,
            )
        )
    return outputs


def _load_generator(model: str, args: argparse.Namespace) -> VLLMChatGenerator:
    return VLLMChatGenerator(
        model=model,
        tensor_parallel_size=args.tensor_parallel_size,
    )


def _append_profile(previous: str, new_profile: str, max_chars: int) -> str:
    combined = (previous + "\n\n" + new_profile).strip() if previous else new_profile.strip()
    if len(combined) <= max_chars:
        return combined
    return combined[-max_chars:]


def main() -> None:
    args = parse_args()
    records = sample_records(read_jsonl(args.input), args.max_records, seed=args.seed)
    records = sorted(records, key=lambda r: (r.get("dataset", ""), r.get("conversation_id", ""), int(r.get("turn_id", 0))))

    for record in records:
        if not record.get("raw_response"):
            raise ValueError(f"Record {record.get('id')} has no raw_response.")

    profiles: dict[str, str] = {record["id"]: "" for record in records}
    if args.variant not in {"no_profile", "fusion_only"}:
        profile_generator = _load_generator(args.agent1_model, args)
        if args.variant == "no_tom_map":
            profile_messages = [
                chat_messages(
                    PROFILE_UPDATE_SYSTEM,
                    build_profile_user_prompt(record, previous_profile=None, current_turn_only=True),
                )
                for record in records
            ]
            profile_outputs = _run_generator(profile_generator, profile_messages, args.max_tokens_profile, args)
            profiles = {record["id"]: profile for record, profile in zip(records, profile_outputs)}
        else:
            profile_by_conversation: dict[str, str] = defaultdict(str)
            turn_ids = sorted({int(record.get("turn_id", 0)) for record in records})
            for turn_id in turn_ids:
                batch = [record for record in records if int(record.get("turn_id", 0)) == turn_id]
                profile_messages = []
                for record in batch:
                    conversation_key = f"{record.get('dataset')}::{record.get('conversation_id')}"
                    previous_profile = profile_by_conversation[conversation_key]
                    profile_messages.append(
                        chat_messages(
                            PROFILE_UPDATE_SYSTEM,
                            build_profile_user_prompt(record, previous_profile=previous_profile or None),
                        )
                    )
                profile_outputs = _run_generator(profile_generator, profile_messages, args.max_tokens_profile, args)
                for record, profile in zip(batch, profile_outputs):
                    profiles[record["id"]] = profile
                    conversation_key = f"{record.get('dataset')}::{record.get('conversation_id')}"
                    profile_by_conversation[conversation_key] = _append_profile(
                        profile_by_conversation[conversation_key],
                        profile,
                        args.max_profile_chars,
                    )
        del profile_generator
        clear_gpu_memory()

    facts_by_id: dict[str, str] = {record["id"]: "[]" for record in records}
    shared_fact_fusion_generator = None
    if args.variant not in {"no_fact", "fusion_only"}:
        fact_generator = _load_generator(args.agent2_model, args)
        fact_messages = [chat_messages(FACT_SYSTEM, build_fact_user_prompt(record["raw_response"])) for record in records]
        fact_outputs = _run_generator(fact_generator, fact_messages, args.max_tokens_fact, args)
        facts_by_id = {record["id"]: facts for record, facts in zip(records, fact_outputs)}
        if args.agent2_model == args.fusion_model:
            shared_fact_fusion_generator = fact_generator
        else:
            del fact_generator
            clear_gpu_memory()

    fusion_messages = []
    for record in records:
        profile = profiles.get(record["id"], "")
        facts = facts_by_id.get(record["id"], "[]")
        if args.variant == "fusion_only":
            profile_for_fusion = None
            facts_for_fusion = None
            fusion_system = FUSION_NO_FACT_SYSTEM
        elif args.variant == "no_profile":
            profile_for_fusion = None
            facts_for_fusion = facts
            fusion_system = FUSION_SYSTEM
        elif args.variant == "no_fact":
            profile_for_fusion = profile
            facts_for_fusion = None
            fusion_system = FUSION_NO_FACT_SYSTEM
        else:
            profile_for_fusion = profile
            facts_for_fusion = facts
            fusion_system = FUSION_SYSTEM
        fusion_messages.append(
            chat_messages(
                fusion_system,
                build_fusion_user_prompt(
                    record,
                    raw_response=record["raw_response"],
                    profile=profile_for_fusion,
                    facts=facts_for_fusion,
                ),
            )
        )

    fusion_generator = shared_fact_fusion_generator or _load_generator(args.fusion_model, args)
    final_outputs = _run_generator(fusion_generator, fusion_messages, args.max_tokens_fusion, args)
    del fusion_generator
    clear_gpu_memory()

    out_records: list[dict] = []
    for record, final_response in zip(records, final_outputs):
        profile = profiles.get(record["id"], "")
        facts = facts_by_id.get(record["id"], "[]")
        enriched = dict(record)
        enriched.update(
            {
                "variant": args.variant,
                "agent1_model": args.agent1_model if profile else None,
                "agent2_model": args.agent2_model if args.variant not in {"no_fact", "fusion_only"} else None,
                "fusion_model": args.fusion_model,
                "profile": profile,
                "extracted_facts": facts,
                "final_response": final_response,
            }
        )
        out_records.append(enriched)

    write_jsonl(out_records, args.output)
    print(f"Wrote {len(out_records)} {args.variant} outputs to {args.output}")


if __name__ == "__main__":
    main()
