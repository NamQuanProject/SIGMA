"""Gromov-Wasserstein distance and barycenter over task signatures (proposal section
4.2.2, eq. 25-27).

The proposal is explicitly schematic here: eq. 26 is written
``GW2_2(Ni, Nj) ≍ (function of {λ(i)}, {η(i)}, di, dj)`` -- a proportionality, not a
closed-form formula. What *is* pinned down is the shape of the answer: for two Gaussians,
the distance depends only on the sorted covariance eigenvalues of each (padded to a
common length when the two tasks' embedding spaces have different dimensionality), and
is "sample-free" (computed purely from stored spectra, no need to re-sample either
task's data).

We implement the natural closed form matching those properties: **squared L2 distance
between the two sorted-descending, zero-padded variance spectra**. This is a documented
simplification of the exact Gromov-Wasserstein-for-Gaussians formula in the literature
this section cites (Salmona et al.), not a reproduction of it -- but it satisfies every
property the proposal calls out, and the closed-form barycenter that falls out of it
(the plain weighted average of padded spectra) is exactly what eq. 27 asks for: the node
minimizing the weighted sum of squared distances to its children.
"""

from __future__ import annotations

import torch

from .signature import TaskSignature


def _padded_spectrum(spectrum: torch.Tensor, length: int) -> torch.Tensor:
    if spectrum.shape[0] == length:
        return spectrum
    return torch.nn.functional.pad(spectrum, (0, length - spectrum.shape[0]))


def gw2_distance(sig_i: TaskSignature, sig_j: TaskSignature) -> torch.Tensor:
    """Squared Gromov-Wasserstein-style distance between two task signatures (eq. 26)."""

    spec_i, spec_j = sig_i.spectrum, sig_j.spectrum
    length = max(spec_i.shape[0], spec_j.shape[0])
    a = _padded_spectrum(spec_i, length)
    b = _padded_spectrum(spec_j, length)
    return (a - b).pow(2).sum()


def gw_barycenter(spectra: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    """Weighted average of sorted, zero-padded spectra (eq. 27's closed-form minimizer).

    Given the squared-L2 distance in ``gw2_distance``, the node minimizing
    ``sum_c w_c * gw2_distance(N, N_c)`` over a free spectrum ``N`` is, by ordinary
    weighted-least-squares, just the weighted mean of the (padded) children spectra --
    no iterative optimization needed.
    """

    if not spectra:
        raise ValueError("gw_barycenter requires at least one spectrum")
    length = max(spec.shape[0] for spec in spectra)
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value")

    barycenter = torch.zeros(length, dtype=spectra[0].dtype)
    for spectrum, weight in zip(spectra, weights):
        barycenter += (weight / total_weight) * _padded_spectrum(spectrum, length)
    return barycenter
