#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from empathy_pipeline.io_utils import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Agent 1 profile model with LoRA.")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--eval_jsonl", default=None)
    parser.add_argument("--model_name", default="unsloth/Qwen3.5-0.8B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_merged_16bit", action="store_true")
    parser.add_argument("--push_to_hub", default=None)
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    return parser.parse_args()


def rows_to_text_dataset(rows: list[dict], tokenizer):
    from datasets import Dataset

    examples = []
    for row in rows:
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        examples.append({"text": text})
    return Dataset.from_list(examples)


def main() -> None:
    args = parse_args()

    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel, is_bfloat16_supported

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        dtype=None,
        token=args.hf_token,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    train_ds = rows_to_text_dataset(read_jsonl(args.train_jsonl), tokenizer)
    eval_ds = rows_to_text_dataset(read_jsonl(args.eval_jsonl), tokenizer) if args.eval_jsonl else None

    training_config = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",
        "weight_decay": 0.005,
        "bf16": is_bfloat16_supported(),
        "fp16": not is_bfloat16_supported(),
        "logging_steps": 10,
        "save_steps": 500,
        "report_to": "none",
        "seed": 3407,
    }
    if eval_ds is not None:
        training_config["eval_steps"] = 500

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=SFTConfig(**training_config),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.save_merged_16bit:
        model.save_pretrained_merged(f"{args.output_dir}_merged_16bit", tokenizer, save_method="merged_16bit")
    if args.push_to_hub:
        model.push_to_hub_merged(args.push_to_hub, tokenizer, token=args.hf_token, save_method="merged_16bit")


if __name__ == "__main__":
    main()
