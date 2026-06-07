# Remaining Experiment Plan for EmpathyEditor

## Current Progress

The project already has these working pieces:

1. SoulChat has been split into `train`, `test`, and `human_validation` in the Hugging Face dataset `Spiderman01/soulchat_split_raw`.
2. `code/Run_EmpathyAgent_on_SoulChat.ipynb` runs a first three-agent pipeline on SoulChat human validation:
   - raw response generation with `Qwen/Qwen2.5-7B-Instruct`;
   - Agent 1 profile/ToM generation with `Spiderman01/finetuned_Qwen_35_08B_empathy`;
   - Agent 2 fact extraction with `unsloth/Qwen3.5-0.8B`;
   - Agent 3 fusion/editing with `unsloth/Qwen3.5-0.8B`;
   - saved artifacts under `validation/EmpathyAgent_100_Soulchat_middle_stage/`.
3. `code/Qwen2.5_7B_finetune_on_Soulchat.ipynb` is a working SoulChat SFT baseline notebook.
4. Agent 1 distillation data already exists in `Agent1_finetune_data/qwen35_08b_sft_dataset.jsonl`, with 19k+ profile-generation examples.

Important current gaps:

1. The current SoulChat run is pilot scale: 100 conversations, 597 turn-level examples.
2. Automatic metrics and human-rating analysis are not implemented yet.
3. Full-system English evaluation on TIDE has not been run.
4. EmpatheticDialogues is currently only used for Agent 1 training/data construction; it should also be used as a held-out multi-turn generalization evaluation, not as the only main benchmark.
5. The LaTeX draft still had unrelated privacy-benchmark text after the empathy-agent sections; this should be removed before submission.
6. The notebooks contain hard-coded Hugging Face tokens. Rotate those tokens and use environment variables only.

## Core Study Logic

The strongest AAAI framing is:

**Can a lightweight plug-and-play multi-agent editor improve empathy while preserving the factual content of an existing dialogue system response?**

This means the paper should not only show that the final response is empathetic. It must show the tradeoff:

1. Empathy improves over the raw backbone response.
2. Factual preservation does not degrade compared with direct generation or full SFT baselines.
3. Agent 1's ToM/profile map improves multi-turn consistency.
4. Agent 2's fact extraction prevents unsupported rewriting.
5. The approach works in English trauma-informed dialogue and Chinese multi-turn empathy dialogue.

## Dataset Roles

### TIDE

Use TIDE as the main English, trauma-informed, clinically relevant benchmark. It is two-turn, so it is not ideal for proving the ToM map, but it is ideal for trauma-informed empathy quality and safety-sensitive human evaluation.

Required split:

1. Split by persona, not by dialogue, to prevent persona leakage.
2. Use 70/10/20 train/dev/test persona partitions if you train any component on TIDE.
3. For the main cross-domain claim, keep Agent 1 trained on EmpatheticDialogues only and evaluate on TIDE test.
4. Optionally report an ED+TIDE-train adapted Agent 1 in a secondary table.

### SoulChat

Use SoulChat as the main Chinese multi-turn and cross-lingual benchmark.

Required experiments:

1. Full SoulChat `test` split automatic evaluation.
2. Full SoulChat `human_validation` split human evaluation.
3. Compare:
   - raw Qwen2.5-7B-Instruct response;
   - Qwen2.5-7B SoulChat SFT response;
   - full EmpathyEditor;
   - ablations.

### EmpatheticDialogues

Yes, use ED for whole-system evaluation, but position it carefully:

1. Use ED train for Agent 1 distillation/SFT only.
2. Use ED validation/test for multi-turn ToM-map evaluation.
3. Do not make ED the main clinical/trauma claim.
4. Use ED to answer: does a running profile improve empathy consistency across turns?

## Required Experiments

### E1. Agent 1 Profile Quality

Goal: show that the small profile model produces useful ToM/profile outputs.

Datasets:

1. ED validation/test.
2. TIDE test as out-of-domain trauma benchmark.
3. SoulChat human validation as cross-lingual stress test.

Models:

1. Qwen3.5-0.8B zero-shot profile prompting.
2. Qwen3.5-0.8B Agent 1 SFT.
3. Optional Qwen3.5-7B zero-shot profile prompting.
4. Teacher model outputs as upper reference where available.

Metrics:

1. XML/tag validity.
2. Section completeness for `<ToM>`, `<Strategy>`, `<Style>`, `<Context>`.
3. Semantic similarity to teacher profile on held-out ED.
4. Human rating on ToM usefulness, emotional-state accuracy, and response-strategy appropriateness.

### E2. Full-System Main Evaluation

Goal: show that EmpathyEditor improves a raw response.

Datasets:

1. TIDE test.
2. SoulChat test.
3. ED test for multi-turn generalization.

Systems:

1. Raw backbone: Qwen2.5-7B-Instruct.
2. Direct small model: Qwen3.5-0.8B prompt-only.
3. Larger direct model: Qwen2.5-7B-Instruct prompt-only.
4. SoulChat SFT baseline on SoulChat.
5. External empathy/counseling models where runnable: SoulChat, EmoLLM, MINT-empathy-Qwen3-4B, CounseLLM, and any released TIDE fine-tuned SLM.
6. TIDE/ED SFT baseline if feasible.
7. EmpathyEditor full system.

Automatic metrics:

1. Empathy reference similarity: BERTScore, ROUGE-L, optional sentence embedding similarity.
2. Factual preservation: fact exact coverage, fact fuzzy coverage, BERTScore to raw response, optional NLI/LLM fact judge.
3. Fluency: length ratio, repetition rate, optional perplexity.
4. Safety red flags: crisis-advice disclaimer overuse, diagnosis overclaiming, unsupported medical claims.

Human metrics:

1. Empathy: 1 to 5.
2. Factual accuracy/preservation: 1 to 5.
3. Helpfulness: 1 to 5.
4. Fluency/coherence: 1 to 5.
5. Pairwise preference: A better, B better, tie.

### E3. System Ablation

Required ablations:

1. Full system.
2. No Agent 1 profile: fusion edits raw response using facts only.
3. No Fact Agent: fusion uses profile and raw response but no extracted facts.
4. No ToM map: Agent 1 sees only the current user turn, not accumulated profile/history.
5. Agent 1 zero-shot: replace SFT Agent 1 with base Qwen3.5-0.8B.
6. Direct SFT response model: compare against changing the backbone itself.

Expected interpretation:

1. Removing Agent 1 should mainly reduce empathy and personalization.
2. Removing the ToM map should reduce multi-turn consistency.
3. Removing Agent 2 should mainly reduce factual preservation.
4. Full SFT should improve empathy but may not preserve factual content from an arbitrary external backbone, which supports the plug-and-play motivation.

### E4. Human Validation

Minimum recommended sample:

1. TIDE: 150 examples.
2. SoulChat human validation: all 597 turn-level examples if budget allows, otherwise 150 stratified by conversation/topic/turn.
3. ED: 150 examples, stratified by turn depth.

Annotators:

1. At least 3 annotators.
2. Prefer psychology, counseling, or mental-health communication background.
3. Use randomized, anonymized A/B displays.

Report:

1. Mean and 95% bootstrap confidence intervals by system and dimension.
2. Pairwise win/loss/tie.
3. Fleiss' kappa or Krippendorff's alpha for agreement.
4. Qualitative error categories with examples:
   - unsupported added claim;
   - excessive verbosity;
   - false or generic empathy;
   - missed emotion;
   - fact dropped;
   - unsafe advice.

## Execution Order

1. Prepare canonical datasets:
   - `00_prepare_datasets.py`
2. Build/validate Agent 1 teacher/SFT data:
   - `01_build_agent1_teacher_requests.py`
   - `02_prepare_agent1_sft.py`
3. Train Agent 1:
   - `03_train_agent1_sft.py`
4. Train direct response baselines:
   - `04_train_soulchat_sft.py`
5. Generate raw backbone responses:
   - `05_generate_raw_responses.py`
6. Run full system and ablations:
   - `06_run_empathy_agent.py`
7. Compute automatic metrics:
   - `07_evaluate_automatic.py`
8. Build human-evaluation packs:
   - `08_build_human_eval_pack.py`
9. Analyze human annotations:
   - `09_analyze_human_eval.py`
10. Produce LaTeX-ready result tables:
   - `10_make_latex_tables.py`

## Key Implementation Notes

1. Always split TIDE by persona, not dialogue.
2. Always keep train/dev/test split identity in output filenames.
3. Keep raw response generation fixed across all editing systems.
4. Never evaluate factual preservation only against the gold empathetic response; evaluate preservation against the raw response and extracted facts.
5. For SoulChat, use the `human_validation` split only for human evaluation and pilot debugging.
6. For ED, prevent train/test contamination because Agent 1 was built from ED-derived profile data.
7. Use environment variables for tokens:
   - `HF_TOKEN`
   - optional teacher API key if you use a hosted teacher model.
