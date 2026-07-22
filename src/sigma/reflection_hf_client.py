"""Local, open-source-model client for the reflection pipeline, used by
``reflections.py --mode hf`` as an alternative to calling the OpenAI API.

Exposes the exact same ``client.chat.completions.create(model=, messages=, temperature=)
-> response.choices[0].message.content`` shape as the ``openai`` package's client, so
``reflection_llm.call_llm_json`` and every stage in ``reflection_pipeline.py`` work
completely unchanged regardless of which one they're handed -- pass an ``HFChatClient``
instance wherever the OpenAI client would otherwise go.

Uses the tokenizer's chat template (``apply_chat_template``), so this only works with an
instruction-tuned checkpoint (e.g. ``Qwen/Qwen2.5-7B-Instruct``, not the plain
``Qwen/Qwen2.5-7B`` base model) -- the reflection prompts are long, structured,
multi-step instructions asking for strict JSON output, which base models generally can't
follow.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _ChatCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, client: "HFChatClient") -> None:
        self._client = client

    def create(self, *, model: str, messages: list[dict[str, str]], temperature: float = 1.0) -> _ChatCompletion:
        # ``model`` is accepted (and ignored) purely to match the OpenAI client's call
        # signature -- this client is already bound to one loaded model at construction.
        content = self._client._generate(messages, temperature=temperature)
        return _ChatCompletion(content)


class _Chat:
    def __init__(self, client: "HFChatClient") -> None:
        self.completions = _Completions(client)


class HFChatClient:
    """Drop-in, local-model stand-in for ``openai.OpenAI``. Loads one model once and
    reuses it for every call ``reflection_pipeline.py`` makes during a run (sequential,
    same as the OpenAI backend -- no batching).
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        max_new_tokens: int = 4096,
        top_p: float = 0.9,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved_dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=resolved_dtype, trust_remote_code=True
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.to(self.device)
        self.model.eval()

        # Exposes the same "client.chat.completions.create(...)" shape as openai.OpenAI.
        self.chat = _Chat(self)

    def _generate(self, messages: list[dict[str, str]], *, temperature: float) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        do_sample = temperature > 0
        generate_kwargs: dict[str, object] = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = self.top_p

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generate_kwargs)

        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
