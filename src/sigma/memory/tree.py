"""SIGMA's cross-task memory tree (proposal section 4.2.2): task signatures organized
into a binary tree via Gromov-Wasserstein distance (``gw.py``), giving O(log n) routing
instead of a linear scan over every task, plus a mechanism for merging "confusable"
(very similar) tasks as the memory grows.

Two things the proposal leaves schematic, resolved concretely here (see also
``gw.py``'s docstring for the distance/barycenter choice):

- **Internal nodes store only a variance spectrum, no mean.** Different tasks' context
  embeddings live in genuinely different, incomparable coordinate frames (each task has
  its own consolidated adapter) -- that mismatch is exactly why GW distance is used
  instead of plain Wasserstein. So a barycenter across tasks can only meaningfully
  average their *spectra*, not their embedding-space locations. To make internal nodes
  usable for concrete routing anyway, each one caches a **representative leaf** (the
  descendant whose own signature is closest, by GW distance, to the node's barycenter).
- **Routing uses representative-Mahalanobis at internal nodes, exact eq. 28 at leaves.**
  At each internal node, the query's context embedding is computed under each child's
  representative task's own adapter and scored by Mahalanobis against that
  representative's own signature -- descend toward the lower score. This is the same
  Mahalanobis formula eq. 28 specifies, just applied hierarchically; by the time descent
  reaches an actual leaf, that leaf's own embedding/signature are used directly, matching
  eq. 28 exactly.

``consolidate_confusable`` (growth control) is a **mechanical merge** of two entries'
consolidated artifacts, not a proven error-bounding procedure -- the proposal specifies
*when* to merge (a distance threshold) and *why* (bounding retrieval error) but not a
concrete algorithm for *how*; this file's ``merge_entries`` is our chosen one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Union

import torch

from ..consolidate.generator import CoordinateGenerator, train_generator
from ..consolidate.pca import LayerBasis
from .entry import CoordinateLayout, MemoryEntry
from .gw import gw2_distance, gw_barycenter
from .signature import TaskSignature, fit_signature, mahalanobis


class Leaf:
    """A single task: one consolidated ``MemoryEntry`` plus a name for bookkeeping."""

    def __init__(self, name: str, entry: MemoryEntry) -> None:
        if entry.signature is None:
            raise ValueError(
                f"MemoryEntry {name!r} has no signature -- re-run run_consolidation.py "
                "(it fits and attaches one automatically) before building a memory tree"
            )
        self.name = name
        self.entry = entry

    @property
    def signature(self) -> TaskSignature:
        return self.entry.signature  # type: ignore[return-value]

    @property
    def weight(self) -> float:
        return float(self.entry.signature.num_samples)  # type: ignore[union-attr]

    def representative(self) -> "Leaf":
        return self

    def leaves(self) -> list["Leaf"]:
        return [self]

    def __repr__(self) -> str:
        return f"Leaf({self.name!r})"


class Internal:
    """An internal tree node: a GW-barycenter spectrum over its two children."""

    def __init__(self, left: "TreeNode", right: "TreeNode") -> None:
        self.left = left
        self.right = right
        self.weight = left.weight + right.weight
        spectrum = gw_barycenter(
            [left.signature.spectrum, right.signature.spectrum], [left.weight, right.weight]
        )
        # No mean: see module docstring -- an internal node has no coordinate frame of
        # its own, only a comparable spectrum.
        self.signature = TaskSignature(mean=torch.zeros(0), var=spectrum, num_samples=int(self.weight))
        candidates = [left.representative(), right.representative()]
        self._representative = min(
            candidates, key=lambda leaf: float(gw2_distance(leaf.signature, self.signature))
        )

    def representative(self) -> Leaf:
        return self._representative

    def leaves(self) -> list[Leaf]:
        return self.left.leaves() + self.right.leaves()

    def __repr__(self) -> str:
        return f"Internal(rep={self._representative.name!r})"


TreeNode = Union[Leaf, Internal]


class MemoryTree:
    """A binary tree over task signatures. Exposes the same
    ``route(context_fn) -> (MemoryEntry, Tensor)`` shape as ``SingleEntryMemory``,
    generalized to N >= 1 tasks.
    """

    def __init__(self, root: TreeNode) -> None:
        self.root = root

    @classmethod
    def build(cls, entries: dict[str, MemoryEntry]) -> "MemoryTree":
        """Bottom-up agglomerative clustering: repeatedly merge the two closest
        (sub)trees by GW2 distance between their signatures, until one root remains.
        """

        if not entries:
            raise ValueError("MemoryTree.build requires at least one entry")
        nodes: list[TreeNode] = [Leaf(name, entry) for name, entry in entries.items()]
        while len(nodes) > 1:
            best_pair, best_distance = None, None
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    distance = float(gw2_distance(nodes[i].signature, nodes[j].signature))
                    if best_distance is None or distance < best_distance:
                        best_distance, best_pair = distance, (i, j)
            i, j = best_pair  # type: ignore[misc]
            merged = Internal(nodes[i], nodes[j])
            nodes = [node for k, node in enumerate(nodes) if k not in (i, j)] + [merged]
        return cls(nodes[0])

    def insert(self, name: str, entry: MemoryEntry) -> None:
        """Add a new task leaf, descending toward the closer child at each step, then
        pairing the new leaf with whichever existing leaf it ends up next to. Ancestors
        along the path are rebuilt bottom-up so their spectra/representatives stay
        correct.
        """

        new_leaf = Leaf(name, entry)
        if isinstance(self.root, Leaf):
            self.root = Internal(self.root, new_leaf)
            return

        path: list[tuple[Internal, bool]] = []  # (ancestor, went_left)
        node: TreeNode = self.root
        replacement: TreeNode | None = None
        while isinstance(node, Internal):
            d_left = float(gw2_distance(new_leaf.signature, node.left.signature))
            d_right = float(gw2_distance(new_leaf.signature, node.right.signature))
            went_left = d_left <= d_right
            child = node.left if went_left else node.right
            path.append((node, went_left))
            if isinstance(child, Leaf):
                replacement = Internal(child, new_leaf)
                break
            node = child

        assert replacement is not None  # loop always terminates at a Leaf child
        for ancestor, went_left in reversed(path):
            replacement = Internal(replacement, ancestor.right) if went_left else Internal(ancestor.left, replacement)
        self.root = replacement

    def route(self, context_fn: Callable[[MemoryEntry], torch.Tensor]) -> tuple[MemoryEntry, torch.Tensor]:
        """Descend the tree for a query, returning the ``MemoryEntry`` to synthesize from
        plus its context embedding (reused from descent when possible, so the winning
        leaf's embedding usually doesn't need recomputing).

        ``context_fn(entry) -> Tensor`` must compute the query's context embedding under
        that specific entry's own (fundamentals-only) adapter -- the tree doesn't touch
        the model itself, so the caller (``evaluate_sigma.py``) supplies this. Same
        shape as ``SingleEntryMemory.route`` so callers don't need to branch on which
        one they have.
        """

        cache: dict[int, torch.Tensor] = {}

        def get_context(entry: MemoryEntry) -> torch.Tensor:
            key = id(entry)
            if key not in cache:
                cache[key] = context_fn(entry)
            return cache[key]

        def score(entry: MemoryEntry) -> torch.Tensor:
            return mahalanobis(entry.signature, get_context(entry))  # type: ignore[arg-type]

        node: TreeNode = self.root
        while isinstance(node, Internal):
            left_entry = node.left.representative().entry
            right_entry = node.right.representative().entry
            node = node.left if score(left_entry) <= score(right_entry) else node.right

        return node.entry, get_context(node.entry)

    def find_confusable_sibling(self, threshold: float) -> tuple[Internal, Leaf, Leaf] | None:
        """Find the first internal node whose two children are both leaves with GW2
        distance below ``threshold`` -- the growth-control trigger ("provably
        confusable" in the proposal's wording). Only sibling pairs are checked, since
        the tree structure already puts each task next to its closest match.
        """

        def walk(node: TreeNode) -> tuple[Internal, Leaf, Leaf] | None:
            if not isinstance(node, Internal):
                return None
            if isinstance(node.left, Leaf) and isinstance(node.right, Leaf):
                if float(gw2_distance(node.left.signature, node.right.signature)) < threshold:
                    return node, node.left, node.right
            return walk(node.left) or walk(node.right)

        return walk(self.root)

    def consolidate_confusable(
        self, threshold: float, *, generator_epochs: int = 200, generator_lr: float = 1e-3
    ) -> list[str]:
        """Repeatedly find and merge confusable sibling leaf pairs. Returns the names of
        the merged entries produced (e.g. ``"bridge+comparison"``).
        """

        merged_names = []
        while True:
            found = self.find_confusable_sibling(threshold)
            if found is None:
                break
            parent, left, right = found
            merged_entry = merge_entries(
                left.entry, right.entry, generator_epochs=generator_epochs, generator_lr=generator_lr
            )
            merged_leaf = Leaf(f"{left.name}+{right.name}", merged_entry)
            self.root = _replace_node(self.root, parent, merged_leaf)
            merged_names.append(merged_leaf.name)
        return merged_names

    def leaves(self) -> list[Leaf]:
        return self.root.leaves()

    def save(self, path: Path) -> None:
        torch.save(self.root, path)

    @classmethod
    def load(cls, path: Path, map_location=None) -> "MemoryTree":
        root = torch.load(path, map_location=map_location, weights_only=False)
        return cls(root)


def _replace_node(node: TreeNode, target: TreeNode, replacement: TreeNode) -> TreeNode:
    """Return a new tree with ``target`` swapped for ``replacement``, rebuilding every
    ancestor along the way (via ``Internal.__init__``) so spectra/representatives stay
    correct. Untouched subtrees are returned as-is.
    """

    if node is target:
        return replacement
    if isinstance(node, Internal):
        new_left = _replace_node(node.left, target, replacement)
        new_right = _replace_node(node.right, target, replacement)
        if new_left is node.left and new_right is node.right:
            return node
        return Internal(new_left, new_right)
    return node


def merge_entries(
    entry_a: MemoryEntry, entry_b: MemoryEntry, *, generator_epochs: int = 200, generator_lr: float = 1e-3
) -> MemoryEntry:
    """Merge two consolidated task entries into one (growth control, section 4.2.2).

    Requires both entries to share the same frozen ``A`` per layer -- i.e. to come from
    bootstrap runs with the same seed/rank/target_modules. This is a reasonable
    precondition rather than a real limitation: entries only get merged when they're
    "provably confusable" (GW distance below threshold) to begin with, which in practice
    means they were bootstrapped the same way.

    Merge procedure: basis directions from each task are *concatenated* (not averaged --
    averaging distinct steering directions would just cancel out the variation each one
    captures) so the merged entry can still express either task's steering, weighted
    means are combined by each task's original sample count, and one generator is
    retrained from scratch on the union of both tasks' (context, alpha) training pairs
    (requires ``training_contexts``/``training_targets`` to have been kept -- the
    default in ``run_consolidation.py``).
    """

    if entry_a.shared_A.keys() != entry_b.shared_A.keys():
        raise ValueError("Cannot merge entries targeting different sets of layers")
    for name, tensor_a in entry_a.shared_A.items():
        if tensor_a.shape != entry_b.shared_A[name].shape or not torch.allclose(tensor_a, entry_b.shared_A[name]):
            raise ValueError(
                f"Cannot merge entries with different shared A for layer {name!r} -- they "
                "must come from bootstrap runs sharing the same seed/rank/target_modules"
            )
    if entry_a.training_contexts is None or entry_b.training_contexts is None:
        raise ValueError(
            "Merging requires training_contexts/training_targets on both entries -- "
            "re-run run_consolidation.py (it keeps them by default) to populate them"
        )

    weight_a = float(entry_a.signature.num_samples)  # type: ignore[union-attr]
    weight_b = float(entry_b.signature.num_samples)  # type: ignore[union-attr]
    total_weight = weight_a + weight_b

    merged_layer_bases: dict[str, LayerBasis] = {}
    for name, basis_a in entry_a.layer_bases.items():
        basis_b = entry_b.layer_bases[name]
        rank, out_features = basis_a.mean.shape
        num_adapters_a, num_adapters_b = basis_a.coordinates.shape[0], basis_b.coordinates.shape[0]

        # Per rank-row, each task's *true* (pre-padding) directions must end up
        # contiguous from index 0 -- CoordinateLayout.flatten()/unflatten() assume that
        # layout (basis.coordinates[..., :dim] is "the valid part"). A naive
        # torch.cat of the already-padded tensors would leave task A's padding zeros
        # sitting *between* the two tasks' real directions, corrupting that contract --
        # so this is built row-by-row instead: task A's `da` real columns, then task B's
        # `db` real columns right after, then zero-padding out to the new max width.
        merged_dims = [da + db for da, db in zip(basis_a.basis_dims, basis_b.basis_dims)]
        max_l = max(merged_dims) if merged_dims else 0

        merged_mean = (weight_a * basis_a.mean + weight_b * basis_b.mean) / total_weight
        merged_basis = torch.zeros(rank, out_features, max_l, dtype=basis_a.basis.dtype)
        merged_coords_a = torch.zeros(num_adapters_a, rank, max_l, dtype=basis_a.coordinates.dtype)
        merged_coords_b = torch.zeros(num_adapters_b, rank, max_l, dtype=basis_b.coordinates.dtype)

        for r in range(rank):
            da, db = basis_a.basis_dims[r], basis_b.basis_dims[r]
            merged_basis[r, :, :da] = basis_a.basis[r, :, :da]
            merged_basis[r, :, da : da + db] = basis_b.basis[r, :, :db]
            merged_coords_a[:, r, :da] = basis_a.coordinates[:, r, :da]
            merged_coords_b[:, r, da : da + db] = basis_b.coordinates[:, r, :db]

        merged_coordinates = torch.cat([merged_coords_a, merged_coords_b], dim=0)
        merged_layer_bases[name] = LayerBasis(
            mean=merged_mean, basis=merged_basis, coordinates=merged_coordinates, basis_dims=merged_dims
        )

    merged_layout = CoordinateLayout.from_layer_bases(merged_layer_bases)

    contexts = torch.cat([entry_a.training_contexts, entry_b.training_contexts], dim=0)
    targets = torch.stack(
        [merged_layout.flatten(merged_layer_bases, adapter_index=i) for i in range(contexts.shape[0])]
    )

    generator = CoordinateGenerator(
        context_dim=contexts.shape[1], alpha_dim=targets.shape[1], hidden_dim=entry_a.generator.hidden_dim
    )
    train_generator(generator, contexts, targets, num_epochs=generator_epochs, learning_rate=generator_lr)

    merged_signature = fit_signature(contexts)

    return MemoryEntry(
        shared_A=entry_a.shared_A,
        layer_bases=merged_layer_bases,
        layout=merged_layout,
        generator=generator,
        signature=merged_signature,
        training_contexts=contexts,
        training_targets=targets,
    )
