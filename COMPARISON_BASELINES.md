# Comparison Baselines for EmpathyEditor

## Baseline Categories

### 1. Editor Input Baseline

**Raw backbone: `Qwen/Qwen2.5-7B-Instruct`**

This is not a competing empathy-specialized system. It is the factual/raw response that EmpathyEditor edits. Report it because it answers the core question: how much does the editor improve empathy over the original response, and how much factual content is preserved?

Use for:

1. SoulChat human validation: existing `validation/qwen25_7B_soulchat_human_validation_100.json`.
2. SoulChat test: existing `validation/qwen25_7B_soulchat_1w.json`, aligned to 59,261 turn-level rows from 10k test conversations.
3. TIDE human validation and test: generate with vLLM.

### 2. Direct Prompting Baselines

**Direct prompted SLM: `Qwen3.5-0.8B`**

This tests whether the small model can directly generate empathetic responses without the editor decomposition. It is a size-controlled lower baseline for the agents.

**Direct prompted LLM: `Qwen/Qwen2.5-7B-Instruct`**

This tests whether a strong general 7B model can solve the task with an empathy prompt alone. It is the most important direct-generation baseline because EmpathyEditor edits responses from this same model.

### 3. In-Domain Fine-Tuning Baseline

**SoulChat-SFT Qwen2.5-7B-Instruct**

This is your already implemented model from `code/Qwen2.5_7B_finetune_on_Soulchat.ipynb`. It is the strongest in-domain Chinese baseline because it directly trains the response model on SoulChat.

Report this on:

1. SoulChat human validation.
2. SoulChat test.

Interpretation:

- If SoulChat-SFT has strong empathy but lower factual preservation relative to a fixed raw response, it supports the plug-and-play editing motivation.
- If SoulChat-SFT is strongest on SoulChat, EmpathyEditor can still be valuable if it is competitive while requiring no full response-model retraining.

### 4. External Empathy / Counseling LLM Baselines

These are the baselines that make the comparison credible against current empathy-oriented systems.

#### Chinese / Mental-Health-Oriented

1. **SoulChat (`scutcyr/SoulChat`)**
   - Chinese mental-health dialogue model focused on empathy and listening.
   - Use if the model can be loaded reliably on the GPU environment.

2. **EmoLLM family**
   - Chinese mental-health/emotional-support model family.
   - Use the most runnable 7B checkpoint available in your environment, such as an InternLM2.5-7B-chat-based checkpoint or a ModelScope/OpenXLab release.

3. **Psyche-R1 / Empathy-R1-style systems**
   - Include as related current systems in the paper.
   - Run only if public weights and inference format are available; otherwise report them as literature baselines, not direct experimental baselines.

#### English / Empathetic Dialogue

1. **MINT-empathy-Qwen3-4B (`hongli-zhan/MINT-empathy-Qwen3-4B`)**
   - vLLM-ready empathy model for multi-turn empathic dialogue.
   - Good direct-generation baseline for ED and TIDE.

2. **CounseLLM (`Wothmag07/counseLLM`)**
   - Llama-3.1-8B-based counseling support model.
   - Useful as an English counseling/empathy model baseline if GPU memory allows.

3. **TIDE fine-tuned SLMs**
   - If the TIDE paper releases checkpoints or training recipes, reproduce one small fine-tuned TIDE model.
   - Otherwise compare against the reported TIDE literature results qualitatively and run your own Qwen2.5-7B TIDE prompt baseline.

4. **Frontier closed-source model upper bound**
   - Optional, budget-dependent.
   - Use a strong commercial model as an upper reference for empathy/human validation, but keep it separate from open-weight baselines.

### 5. EmpathyEditor Variants

1. **Full EmpathyEditor**
   - Agent 1 profile/ToM + Agent 2 fact extraction + Agent 3 fusion.

2. **No profile**
   - Remove Agent 1; fusion sees raw response and extracted facts only.

3. **No fact agent**
   - Remove Agent 2; fusion sees raw response and profile only.

4. **No ToM map**
   - Agent 1 sees only the current turn, not accumulated profile/history.

5. **Zero-shot Agent 1**
   - Replace fine-tuned Agent 1 with the base Qwen3.5-0.8B model.

6. **Fusion only**
   - Rewrite raw response without profile or extracted facts.

## Recommended Main Tables

### SoulChat Main Table

Rows:

1. Raw Qwen2.5-7B-Instruct.
2. Direct prompted Qwen2.5-7B-Instruct.
3. SoulChat-SFT Qwen2.5-7B-Instruct.
4. SoulChat (`scutcyr/SoulChat`) if runnable.
5. EmoLLM if runnable.
6. EmpathyEditor full.
7. EmpathyEditor ablations in a separate table.

Metrics:

1. BERTScore/ROUGE-L against SoulChat reference.
2. Fact coverage against extracted facts.
3. BERTScore to raw response.
4. Human empathy, fact preservation, helpfulness, fluency.
5. Pairwise win rate vs raw and vs SoulChat-SFT.

### TIDE Main Table

Rows:

1. Raw Qwen2.5-7B-Instruct.
2. Direct prompted Qwen2.5-7B-Instruct.
3. Direct prompted Qwen3.5-0.8B.
4. MINT-empathy-Qwen3-4B.
5. CounseLLM if runnable.
6. TIDE fine-tuned SLM if reproducible.
7. EmpathyEditor full.
8. EmpathyEditor ablations in a separate table.

Metrics:

1. BERTScore/ROUGE-L against TIDE reference.
2. Fact coverage against extracted facts.
3. Human empathy, fact preservation, helpfulness, fluency.
4. Pairwise win rate vs direct prompted Qwen2.5-7B and vs best external empathy model.

