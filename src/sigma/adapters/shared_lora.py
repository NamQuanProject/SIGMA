"""Shared-frozen-A LoRA: a single frozen down-projection shared across all bootstrapped
adapters, with only the per-adapter up-projection trained.

Paper convention (SIGMA proposal 4.2.1): "Use LoRA with a single shared, frozen
down-projection A across all M adapters; only the up-projection B_m is trained. Then
adapter m at a given layer is the update Δφ = B_m^T A." The proposal writes A and B_m both
in R^{r x d} for square layers. We use the standard generic LoRA shape convention instead,
so the same module works on the (generally non-square) attention projections of real
transformer blocks: A frozen (r, in_features), B_m trainable (out_features, r),
ΔW = B_m @ A. This preserves the property that matters -- one frozen shared down-projection,
many trainable up-projections -- without requiring square weight matrices.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Iterator

import torch
from torch import nn


class SharedLoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with ``base(x) + (x @ A^T) @ B^T``."""

    def __init__(self, base: nn.Linear, rank: int, shared_A: torch.Tensor) -> None:
        super().__init__()
        if tuple(shared_A.shape) != (rank, base.in_features):
            raise ValueError(
                f"shared_A shape {tuple(shared_A.shape)} != (rank={rank}, in_features={base.in_features})"
            )
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)

        self.rank = rank
        self.register_buffer("A", shared_A)
        # Zero-init B so the adapter starts as a no-op (delta W = 0).
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=shared_A.dtype))

    def reset_B(self) -> None:
        nn.init.zeros_(self.B)

    def delta_weight(self) -> torch.Tensor:
        return self.B @ self.A  # (out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype)
        return base_out + lora_out


def make_shared_A(rank: int, in_features: int, *, seed: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Deterministically initialize the frozen shared down-projection A.

    Replicates ``nn.init.kaiming_uniform_(a=sqrt(5))``'s bound by hand (rather than relying
    on that function's optional ``generator`` kwarg, which isn't available on all torch
    versions), so initialization is reproducible from ``seed`` alone.
    """

    generator = torch.Generator().manual_seed(seed)
    gain = math.sqrt(2.0 / (1 + 5))  # kaiming gain for a=sqrt(5)
    std = gain / math.sqrt(in_features)
    bound = math.sqrt(3.0) * std
    return torch.empty(rank, in_features, dtype=dtype).uniform_(-bound, bound, generator=generator)


def find_target_linears(model: nn.Module, target_modules: Iterable[str]) -> Iterator[tuple[str, nn.Linear]]:
    """Yield (qualified_name, module) pairs for linear layers matching target_modules regexes."""

    patterns = [re.compile(pattern) for pattern in target_modules]
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(pattern.search(name) for pattern in patterns):
            yield name, module


def attach_shared_lora(
    model: nn.Module,
    *,
    rank: int,
    target_modules: Iterable[str],
    seed: int,
) -> dict[str, SharedLoRALinear]:
    """Replace target linear layers in-place with SharedLoRALinear wrappers.

    Returns a mapping of qualified layer name -> SharedLoRALinear, so callers can
    reinitialize B between bootstrap adapters and later collect trained B matrices. All
    wrappers share the *same* frozen A tensor for a given layer, generated once here.
    """

    adapters: dict[str, SharedLoRALinear] = {}
    targets = list(find_target_linears(model, target_modules))
    if not targets:
        raise ValueError(f"No linear layers matched target_modules={list(target_modules)!r}")

    for name, linear in targets:
        shared_A = make_shared_A(rank, linear.in_features, seed=seed, dtype=linear.weight.dtype)
        wrapped = SharedLoRALinear(linear, rank, shared_A)
        _set_submodule(model, name, wrapped)
        adapters[name] = wrapped
    return adapters


def _set_submodule(model: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def reset_all_B(adapters: dict[str, SharedLoRALinear]) -> None:
    for adapter in adapters.values():
        adapter.reset_B()


def trainable_parameters(adapters: dict[str, SharedLoRALinear]) -> Iterator[nn.Parameter]:
    for adapter in adapters.values():
        yield adapter.B


def collect_B_matrices(adapters: dict[str, SharedLoRALinear]) -> dict[str, torch.Tensor]:
    """Detach and clone every adapter's B matrix, keyed by layer name."""

    return {name: adapter.B.detach().clone() for name, adapter in adapters.items()}
