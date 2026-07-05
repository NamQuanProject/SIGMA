"""Coordinate generator p_theta(alpha | c_x): a small heteroscedastic Gaussian MLP
(SIGMA proposal eq. 21-22) that maps a context embedding to a distribution over flattened
steering coordinates alpha, trained by Gaussian negative log-likelihood.

We use a single generator over one *flattened* alpha vector spanning every (layer, rank)
slot, rather than one generator per slot -- simpler for a v1 implementation, and the
reconstruction in ``memory/entry.py`` reshapes the flattened output back per eq. 23.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class CoordinateGenerator(nn.Module):
    """Small MLP g_theta(c_x) -> (mu, log_var) over a flattened alpha vector."""

    def __init__(self, context_dim: int, alpha_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.alpha_dim = alpha_dim
        self.hidden_dim = hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(hidden_dim, alpha_dim)
        self.log_var_head = nn.Linear(hidden_dim, alpha_dim)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(context)
        mu = self.mu_head(hidden)
        log_var = self.log_var_head(hidden).clamp(min=-10.0, max=10.0)
        return mu, log_var

    def sample(self, context: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Sample alpha ~ N(mu, Sigma). Returns (num_samples, *context.shape[:-1], alpha_dim)."""

        mu, log_var = self.forward(context)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn(num_samples, *mu.shape, device=mu.device, dtype=mu.dtype)
        return mu.unsqueeze(0) + eps * std.unsqueeze(0)


def gaussian_nll_loss(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood (eq. 22), diagonal covariance."""

    inv_var = torch.exp(-log_var)
    return 0.5 * (log_var + (target - mu).pow(2) * inv_var).sum(dim=-1).mean()


def train_generator(
    generator: CoordinateGenerator,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    *,
    num_epochs: int = 200,
    learning_rate: float = 1e-3,
    batch_size: int = 16,
) -> CoordinateGenerator:
    """Train the generator on the supervised set {(c_m, alpha_m)} built from bootstrap subsets."""

    optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    dataset = TensorDataset(contexts, targets)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

    generator.train()
    for _ in range(num_epochs):
        for context_batch, target_batch in loader:
            mu, log_var = generator(context_batch)
            loss = gaussian_nll_loss(mu, log_var, target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    generator.eval()
    return generator
