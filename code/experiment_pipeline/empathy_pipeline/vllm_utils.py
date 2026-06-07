from __future__ import annotations

from typing import Any


class VLLMChatGenerator:
    def __init__(
        self,
        *,
        model: str,
        tokenizer_name: str | None = None,
        tensor_parallel_size: int = 1,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model, trust_remote_code=trust_remote_code)
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def generate(
        self,
        messages_list: list[list[dict[str, str]]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_p: float | None = None,
        max_tokens: int = 512,
    ) -> list[str]:
        from vllm import SamplingParams

        if not messages_list:
            return []
        texts = self.tokenizer.apply_chat_template(
            messages_list,
            tokenize=False,
            add_generation_prompt=True,
        )
        sampling_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if min_p is not None:
            sampling_kwargs["min_p"] = min_p
        outputs = self.llm.generate(texts, SamplingParams(**sampling_kwargs))
        return [output.outputs[0].text.strip() for output in outputs]


def clear_gpu_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def generate_chat_completions(
    *,
    model: str,
    messages_list: list[list[dict[str, str]]],
    tokenizer_name: str | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    min_p: float | None = None,
    max_tokens: int = 512,
    tensor_parallel_size: int = 1,
    trust_remote_code: bool = True,
    dtype: str = "auto",
    gpu_memory_utilization: float = 0.90,
) -> list[str]:
    generator = VLLMChatGenerator(
        model=model,
        tokenizer_name=tokenizer_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    try:
        return generator.generate(
            messages_list,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            max_tokens=max_tokens,
        )
    finally:
        del generator
        clear_gpu_memory()
