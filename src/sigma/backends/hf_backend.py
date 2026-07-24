"""A plain local Hugging Face causal LM used as an answer-generation backend.

This is deliberately independent of the SIGMA memory machinery (no adapters, no
`SharedLoRALinear`) -- it's for loading a *separate* local model purely as a comparison
point in `evaluate_sigma.py` (e.g. a bigger or different local model than the one the
memory is attached to). The SIGMA-adapted generation path stays in `evaluate_sigma.py`
itself, since it's tightly coupled to `memory/apply.py`'s adapter patching.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..reflection.dataset import build_prompt


class HFAnswerBackend:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        max_new_tokens: int = 16,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved_dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=resolved_dtype)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate(self, question: str) -> str:
        return self.generate_raw(build_prompt(question))

    def generate_raw(self, prompt: str) -> str:
        """Generate from ``prompt`` verbatim, skipping the question-only ``build_prompt``
        wrap -- used by ``baselines/`` scripts, whose prompts already embed retrieved
        context alongside the question.
        """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
