#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a direct SoulChat response baseline.")
    parser.add_argument("--repo_id", default="Spiderman01/soulchat_split_raw")
    parser.add_argument("--model_name", default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=1536)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--save_merged_16bit", action="store_true")
    parser.add_argument("--push_to_hub", default=None)
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel, is_bfloat16_supported

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
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

    dataset = load_dataset(args.repo_id, token=args.hf_token)
    train_data = dataset["train"]
    if args.max_train_samples:
        train_data = train_data.select(range(min(args.max_train_samples, len(train_data))))

    def formatting_func(examples):
        texts = []
        batch_messages = examples["messages"]
        for messages in batch_messages:
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        return texts

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        formatting_func=formatting_func,
        max_seq_length=args.max_seq_length,
        packing=False,
        args=SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            weight_decay=0.01,
            bf16=is_bfloat16_supported(),
            fp16=not is_bfloat16_supported(),
            logging_steps=20,
            save_steps=1000,
            report_to="none",
            seed=3407,
        ),
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

