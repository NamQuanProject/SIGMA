"""Task signatures: the non-parametric "key" side of a SIGMA memory entry
(proposal section 4.2.2 -- "each memory entry stores ... a key: a Gaussian ... statistical
signature over a task-specific context-embedding space").

We fit a **diagonal** Gaussian rather than a full covariance. Each task's signature is
built from only `M` (the number of bootstrapped adapters, typically ~8-16) per-subset
context embeddings -- far fewer samples than the embedding dimension, so a full d x d
covariance would be massively ill-conditioned. A diagonal Gaussian, with the variance
estimate shrunk toward the average variance (a simple James-Stein-style estimator), is
well-conditioned even with few samples, and has a bonus: its "eigenvalues" (needed for
Gromov-Wasserstein distance in ``gw.py``) are just its diagonal entries -- no
eigendecomposition required.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TaskSignature:
    """A diagonal-Gaussian signature over one task's context-embedding space.

    ``mean``/``var`` have the dimensionality of that task's own context embedding (which
    differs from task to task, since each task has its own consolidated adapter). Two
    signatures from different tasks are therefore *not* directly comparable by mean/var
    -- only their sorted variance *spectra* are comparable, via Gromov-Wasserstein
    distance (``gw.py``), since sorted eigenvalues are invariant to the specific basis of
    each task's embedding space.
    """

    mean: torch.Tensor  # (d,)
    var: torch.Tensor  # (d,), diagonal covariance, already shrunk
    num_samples: int

    @property
    def spectrum(self) -> torch.Tensor:
        """Sorted-descending variance spectrum -- the "eigenvalues" GW distance compares."""
        return torch.sort(self.var, descending=True).values


def fit_signature(contexts: torch.Tensor, *, shrinkage: float = 0.1) -> TaskSignature:
    """Fit a diagonal Gaussian signature from a task's per-subset context embeddings.

    ``contexts``: (M, d) -- one row per bootstrapped adapter/subset (see
    ``run_consolidation.py``, where these are already computed for generator training).

    ``shrinkage`` (in [0, 1]) blends the per-dimension sample variance with the average
    variance across dimensions: ``var = (1 - shrinkage) * var_i + shrinkage * mean(var)``.
    At ``shrinkage=0`` this is the plain (biased) sample variance; higher values trade
    per-dimension precision for stability when `M` is small relative to `d`.
    """

    if contexts.dim() != 2:
        raise ValueError(f"contexts must be 2D (M, d), got shape {tuple(contexts.shape)}")
    num_samples = contexts.shape[0]

    mean = contexts.mean(dim=0)
    if num_samples > 1:
        var = contexts.var(dim=0, unbiased=False)
    else:
        # A single sample carries no variance information -- fall back to unit variance
        # everywhere so downstream Mahalanobis/GW math stays well-defined.
        var = torch.ones_like(mean)

    average_var = var.mean().clamp_min(1e-8)
    shrunk_var = (1 - shrinkage) * var + shrinkage * average_var
    shrunk_var = shrunk_var.clamp_min(1e-8)

    return TaskSignature(mean=mean, var=shrunk_var, num_samples=num_samples)


def mahalanobis(signature: TaskSignature, x: torch.Tensor) -> torch.Tensor:
    """Squared Mahalanobis distance of ``x`` to ``signature``, own-space (eq. 28).

    Diagonal covariance, so this is just a weighted sum of squared per-dimension
    deviations: ``sum_i (x_i - m_i)^2 / var_i``.
    """

    diff = x.reshape(-1) - signature.mean
    return (diff.pow(2) / signature.var).sum()
