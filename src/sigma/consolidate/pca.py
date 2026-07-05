"""PCA / Fisher-weighted PCA consolidation of bootstrapped adapter up-projections.

Implements eq. 16-20 of the SIGMA proposal: for each target layer and each rank index,
the up-projection columns {b_m} across the M bootstrapped adapters form a "cloud" in
R^{out_features}. Consolidation decomposes that cloud into a mean ("fundamentals") plus a
low-rank orthonormal steering basis, chosen via cumulative explained variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..reflection_dataset import IGNORE_INDEX, QAExample, build_prompt

if TYPE_CHECKING:
    from ..adapters.shared_lora import SharedLoRALinear


@dataclass
class LayerBasis:
    """Consolidated basis for one target layer, across all its rank columns.

    ``mean``: (rank, out_features) -- b̄ for every rank column ("fundamentals").
    ``basis``: (rank, out_features, L_max) -- {u_j} steering directions per rank column,
      zero-padded to the layer's max L (see ``basis_dims`` for the true per-column count).
    ``coordinates``: (num_adapters, rank, L_max) -- α_m per rank column, zero-padded likewise.
    ``basis_dims``: true L for each rank column (before padding).
    """

    mean: torch.Tensor
    basis: torch.Tensor
    coordinates: torch.Tensor
    basis_dims: list[int]


def _consolidate_single_cloud(
    cloud: torch.Tensor, *, explained_variance_threshold: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Consolidate one (M, d) cloud of vectors. Returns (mean[d], basis[d, L], coords[M, L])."""

    mean = cloud.mean(dim=0)
    centered = cloud - mean
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    energy = singular_values.pow(2)
    total_energy = energy.sum()

    if total_energy <= 0:
        # Degenerate cloud (e.g. all bootstrapped adapters landed identically): keep a
        # single all-zero steering direction so downstream shapes stay well-defined.
        basis = torch.zeros(cloud.shape[1], 1, dtype=cloud.dtype)
        coords = torch.zeros(cloud.shape[0], 1, dtype=cloud.dtype)
        return mean, basis, coords

    cumulative = torch.cumsum(energy, dim=0) / total_energy
    num_components = int(torch.searchsorted(cumulative, explained_variance_threshold).item()) + 1
    num_components = max(1, min(num_components, vh.shape[0]))

    basis = vh[:num_components].t()  # (d, L)
    coords = centered @ basis  # (M, L)
    return mean, basis, coords


def consolidate_cloud(
    b_matrices: dict[str, dict[str, torch.Tensor]],
    *,
    explained_variance_threshold: float = 0.9,
) -> dict[str, LayerBasis]:
    """PCA/SVD consolidation (Option 1) of bootstrapped adapters, per target layer.

    ``b_matrices`` maps adapter index (as str) -> {layer_name: B (out_features, rank)}.
    """

    return _consolidate(b_matrices, weights=None, explained_variance_threshold=explained_variance_threshold)


def fisher_weighted_consolidate(
    b_matrices: dict[str, dict[str, torch.Tensor]],
    fisher: dict[str, torch.Tensor],
    *,
    explained_variance_threshold: float = 0.9,
) -> dict[str, LayerBasis]:
    """Fisher-/curvature-weighted PCA (Option 2, eq. 19-20).

    ``fisher`` maps layer_name -> diagonal Fisher estimate, shape (out_features,), as
    produced by ``compute_diagonal_fisher``.
    """

    return _consolidate(b_matrices, weights=fisher, explained_variance_threshold=explained_variance_threshold)


def _consolidate(
    b_matrices: dict[str, dict[str, torch.Tensor]],
    *,
    weights: dict[str, torch.Tensor] | None,
    explained_variance_threshold: float,
) -> dict[str, LayerBasis]:
    adapter_ids = sorted(b_matrices.keys(), key=lambda x: int(x))
    if not adapter_ids:
        raise ValueError("No adapters provided for consolidation")

    layer_names = sorted(b_matrices[adapter_ids[0]].keys())
    result: dict[str, LayerBasis] = {}

    for layer_name in layer_names:
        stacked = torch.stack([b_matrices[aid][layer_name] for aid in adapter_ids], dim=0)
        # stacked: (M, out_features, rank)
        _, _, rank = stacked.shape

        sqrt_fisher = None
        if weights is not None:
            sqrt_fisher = weights[layer_name].clamp_min(1e-12).sqrt()  # (out_features,)

        means, bases, coords, dims = [], [], [], []
        for r in range(rank):
            cloud = stacked[:, :, r]  # (M, out_features)
            if sqrt_fisher is not None:
                rescaled = cloud * sqrt_fisher.unsqueeze(0)
                mean_r, basis_r, coords_r = _consolidate_single_cloud(
                    rescaled, explained_variance_threshold=explained_variance_threshold
                )
                # Map back to the original space (eq. 20, step 3): u_j = û_j / sqrt(f).
                # (Coordinates stay as computed in the rescaled space -- reconstruction
                # b̄ + Σ α_j u_j is exact in the original space with this pairing.)
                mean_r = mean_r / sqrt_fisher
                basis_r = basis_r / sqrt_fisher.unsqueeze(1)
            else:
                mean_r, basis_r, coords_r = _consolidate_single_cloud(
                    cloud, explained_variance_threshold=explained_variance_threshold
                )
            means.append(mean_r)
            bases.append(basis_r)
            coords.append(coords_r)
            dims.append(basis_r.shape[1])

        max_l = max(dims)
        padded_basis = torch.stack(
            [torch.nn.functional.pad(b, (0, max_l - b.shape[1])) for b in bases], dim=0
        )  # (rank, out_features, max_l)
        padded_coords = torch.stack(
            [torch.nn.functional.pad(c, (0, max_l - c.shape[1])) for c in coords], dim=1
        )  # (num_adapters, rank, max_l)
        mean_stack = torch.stack(means, dim=0)  # (rank, out_features)

        result[layer_name] = LayerBasis(mean=mean_stack, basis=padded_basis, coordinates=padded_coords, basis_dims=dims)

    return result


def compute_diagonal_fisher(
    model,
    adapters: dict[str, "SharedLoRALinear"],
    mean_b: dict[str, torch.Tensor],
    holdout_examples: list[QAExample],
    tokenizer,
    *,
    max_length: int = 512,
) -> dict[str, torch.Tensor]:
    """Estimate a diagonal Fisher over each layer's output-feature dimension (eq. 19).

    Sets every adapter's B to the consolidated mean ("fundamentals", no steering yet),
    then accumulates squared gradients of the per-example log-likelihood w.r.t. B over a
    held-out slice of Q_final, averaged over examples and over the rank dimension (rank
    columns share the same out_features basis, so they're pooled into one Fisher vector).
    """

    for name, adapter in adapters.items():
        adapter.B.data.copy_(mean_b[name].to(adapter.B.dtype))

    device = next(model.parameters()).device
    fisher_sums = {name: torch.zeros(adapter.B.shape[0], device=device) for name, adapter in adapters.items()}

    model.eval()
    for example in holdout_examples:
        for adapter in adapters.values():
            adapter.B.grad = None

        prompt = build_prompt(example.question)
        answer = " " + example.answer.strip()
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

        input_ids = torch.tensor([(prompt_ids + answer_ids)[:max_length]], device=device)
        labels = torch.tensor([([IGNORE_INDEX] * len(prompt_ids) + answer_ids)[:max_length]], device=device)

        outputs = model(input_ids=input_ids, labels=labels)
        outputs.loss.backward()

        for name, adapter in adapters.items():
            if adapter.B.grad is not None:
                fisher_sums[name] += adapter.B.grad.detach().pow(2).mean(dim=1)

    num_examples = max(len(holdout_examples), 1)
    return {name: value.cpu() / num_examples for name, value in fisher_sums.items()}
