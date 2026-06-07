# EmpathyEditor Experiment Pipeline

This folder contains script versions of the remaining experiments. The scripts are designed for a GPU machine and are not run automatically.

## Recommended Order

```bash
python code/experiment_pipeline/00_prepare_datasets.py --out_dir data/processed
python code/experiment_pipeline/01_build_agent1_teacher_requests.py --input data/processed/empathetic_dialogues_train.jsonl --output data/teacher_requests/ed_train_requests.jsonl
python code/experiment_pipeline/02_prepare_agent1_sft.py --existing_sft_jsonl Agent1_finetune_data/qwen35_08b_sft_dataset.jsonl --out_dir data/agent1_sft
python code/experiment_pipeline/03_train_agent1_sft.py --train_jsonl data/agent1_sft/train.jsonl --eval_jsonl data/agent1_sft/validation.jsonl --output_dir outputs/agent1_qwen35_08b
python code/experiment_pipeline/04_train_soulchat_sft.py --output_dir outputs/qwen25_7b_soulchat
python code/experiment_pipeline/05_generate_raw_responses.py --input data/processed/soulchat_test.jsonl --output outputs/raw/soulchat_test_qwen25_7b.jsonl --model Qwen/Qwen2.5-7B-Instruct
python code/experiment_pipeline/06_run_empathy_agent.py --input outputs/raw/soulchat_test_qwen25_7b.jsonl --output outputs/agent/soulchat_test_full.jsonl --variant full --agent1_model outputs/agent1_qwen35_08b
python code/experiment_pipeline/07_evaluate_automatic.py --system full=outputs/agent/soulchat_test_full.jsonl --system raw=outputs/raw/soulchat_test_qwen25_7b.jsonl --out_dir results/automatic
python code/experiment_pipeline/08_build_human_eval_pack.py --system full=outputs/agent/soulchat_human_full.jsonl --system raw=outputs/raw/soulchat_human_qwen25_7b.jsonl --target_system full --out_dir human_eval/soulchat
python code/experiment_pipeline/09_analyze_human_eval.py --annotations human_eval/soulchat/annotator_*.csv --key human_eval/soulchat/human_eval_key.csv --out_dir results/human/soulchat
python code/experiment_pipeline/10_make_latex_tables.py --automatic results/automatic/summary.csv --human results/human/soulchat/human_metric_summary.csv --out_dir results/tables
```

## Current Requested SoulChat/TIDE Run

Use `code/EmpathyEditor_Colab_Run_All.ipynb` on the Colab GPU runtime for the full vLLM run.

SoulChat alignment:

```bash
PYTHONPATH=code/experiment_pipeline python code/experiment_pipeline/11_align_existing_soulchat_outputs.py \
  --split test \
  --raw_json validation/qwen25_7B_soulchat_1w.json \
  --out_jsonl outputs/aligned/soulchat_test_raw.jsonl
```

This aligns the existing raw response file for 10k SoulChat test conversations to 59,261 turn-level rows.

Run full EmpathyEditor:

```bash
PYTHONPATH=code/experiment_pipeline python code/experiment_pipeline/06_run_empathy_agent.py \
  --input outputs/aligned/soulchat_test_raw.jsonl \
  --output outputs/agent/soulchat_test_full.jsonl \
  --variant full \
  --agent1_model Spiderman01/finetuned_Qwen_35_08B_empathy \
  --agent2_model unsloth/Qwen3.5-0.8B \
  --fusion_model unsloth/Qwen3.5-0.8B
```

TIDE split and upload:

```bash
PYTHONPATH=code/experiment_pipeline python code/experiment_pipeline/12_prepare_tide_split_upload.py \
  --out_dir data/processed \
  --human_size 100 \
  --push_repo Spiderman01/tide_split_raw
```

Set `HF_TOKEN` in the runtime environment before pushing to Hugging Face.

## Notes

- TIDE access is gated. Accept the Hugging Face dataset terms first, then set `HF_TOKEN`.
- Do not store tokens in notebooks or scripts.
- The scripts use JSONL artifacts so each stage can be inspected and rerun independently.
- Run ablations by changing `--variant` in `06_run_empathy_agent.py`:
  - `full`
  - `no_profile`
  - `no_fact`
  - `no_tom_map`
  - `fusion_only`
