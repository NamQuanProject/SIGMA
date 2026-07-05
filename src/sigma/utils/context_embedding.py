"""Context embedding h(x; phi0 + B'^T A): the "double duty" embedding used both to
condition the coordinate generator and, in a future multi-task build, as the variable a
tree signature is fit over (SIGMA proposal, section 4.2.1 "Context embedding").

The caller is responsible for putting the desired adapter state (fundamentals-only, a
synthesized adapter, or none) onto the model before calling this -- it only handles the
forward pass and mean-pooling.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def compute_context_embedding(
    model, tokenizer, texts: list[str], *, max_length: int = 512, device=None
) -> torch.Tensor:
    """Mean-pool the last hidden state of ``model`` over each text's (non-padding) tokens."""

    device = device or next(model.parameters()).device
    encoded = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    ).to(device)

    outputs = model(**encoded, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1].float()  # (batch, seq, hidden)
    # Upcast to float32 regardless of the backbone's compute dtype (e.g. bf16) -- this is
    # what feeds CoordinateGenerator, whose params are float32, so keep it consistent
    # end to end rather than relying on every caller to cast.
    mask = encoded["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts
