"""A consolidated SIGMA memory entry: the "value" bootstrap-and-consolidate produces for
one task -- a per-layer basis plus a coordinate generator -- together with the shared
frozen A its B-matrices are expressed against.

This is what a leaf of SIGMA's cross-task memory tree (section 4.2.2) would store; for the
single-task HotpotQA build, ``single_entry.py`` provides the trivial one-leaf "tree" that
routes every query to this entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..consolidate.generator import CoordinateGenerator
from ..consolidate.pca import LayerBasis


@dataclass
class CoordinateLayout:
    """Bookkeeping to flatten/unflatten per-(layer, rank) alpha coordinates into one vector.

    The generator predicts a single flattened alpha vector; this class records the
    (layer, rank, L) structure needed to fold it back into per-layer coordinate tensors
    for reconstruction (eq. 23).
    """

    layer_names: list[str]
    rank_per_layer: dict[str, int]
    dims_per_layer: dict[str, list[int]]  # basis_dims (true L) per rank column

    @property
    def total_dim(self) -> int:
        return sum(sum(dims) for dims in self.dims_per_layer.values())

    @classmethod
    def from_layer_bases(cls, layer_bases: dict[str, LayerBasis]) -> "CoordinateLayout":
        layer_names = sorted(layer_bases.keys())
        rank_per_layer = {name: layer_bases[name].mean.shape[0] for name in layer_names}
        dims_per_layer = {name: list(layer_bases[name].basis_dims) for name in layer_names}
        return cls(layer_names=layer_names, rank_per_layer=rank_per_layer, dims_per_layer=dims_per_layer)

    def flatten(self, layer_bases: dict[str, LayerBasis], adapter_index: int) -> torch.Tensor:
        chunks = []
        for name in self.layer_names:
            basis = layer_bases[name]
            for rank_idx, dim in enumerate(basis.basis_dims):
                chunks.append(basis.coordinates[adapter_index, rank_idx, :dim])
        return torch.cat(chunks)

    def unflatten(self, alpha: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split a flattened alpha vector back into {layer_name: (rank, L_max) coords}."""

        offset = 0
        result: dict[str, torch.Tensor] = {}
        for name in self.layer_names:
            rank = self.rank_per_layer[name]
            dims = self.dims_per_layer[name]
            max_dim = max(dims) if dims else 0
            layer_coords = torch.zeros(rank, max_dim, dtype=alpha.dtype, device=alpha.device)
            for rank_idx, dim in enumerate(dims):
                layer_coords[rank_idx, :dim] = alpha[offset : offset + dim]
                offset += dim
            result[name] = layer_coords
        return result


@dataclass
class MemoryEntry:
    """One consolidated task memory: value = (layer_bases, generator)."""

    shared_A: dict[str, torch.Tensor]
    layer_bases: dict[str, LayerBasis]
    layout: CoordinateLayout
    generator: CoordinateGenerator

    def synthesize_adapter(self, context_embedding: torch.Tensor, *, num_samples: int = 1) -> dict[str, torch.Tensor]:
        """Reconstruct per-layer B' (eq. 23), averaging ``num_samples`` sampled alphas in
        alpha-space (eq. 24) -- equivalent to, but cheaper than, ensembling full adapters.
        """

        alpha_samples = self.generator.sample(context_embedding, num_samples=num_samples)
        alpha_mean = alpha_samples.mean(dim=0)
        if alpha_mean.dim() > 1:
            alpha_mean = alpha_mean.squeeze(0)
        coords_by_layer = self.layout.unflatten(alpha_mean)

        b_prime: dict[str, torch.Tensor] = {}
        for name, basis in self.layer_bases.items():
            coords = coords_by_layer[name]  # (rank, L_max)
            steering = torch.einsum("rol,rl->ro", basis.basis, coords)
            reconstructed = basis.mean + steering  # (rank, out_features)
            b_prime[name] = reconstructed.t()  # -> (out_features, rank), matches SharedLoRALinear.B
        return b_prime

    def save(self, path: Path) -> None:
        torch.save(
            {
                "shared_A": self.shared_A,
                "layer_bases": self.layer_bases,
                "layout": self.layout,
                "generator_state_dict": self.generator.state_dict(),
                "generator_config": {
                    "context_dim": self.generator.context_dim,
                    "alpha_dim": self.generator.alpha_dim,
                    "hidden_dim": self.generator.hidden_dim,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, map_location=None) -> "MemoryEntry":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        generator = CoordinateGenerator(**payload["generator_config"])
        generator.load_state_dict(payload["generator_state_dict"])
        generator.eval()
        return cls(
            shared_A=payload["shared_A"],
            layer_bases=payload["layer_bases"],
            layout=payload["layout"],
            generator=generator,
        )
