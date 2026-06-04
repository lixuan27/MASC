"""Algorithm 1: Manifold-Aligned Semantic Clustering (MASC) construction.

Given the codebook embeddings ``Z = {v_1, ..., v_N}`` and a target number of
coarse clusters ``k``, this module builds the density-driven, bottom-up
hierarchical tree described in Section 3.2.2 / Algorithm 1 of the paper and
returns the flat mapping ``M : {1..N} -> {1..k}``.

The construction is *deterministic* (no random initialisation): the only input
is the codebook geometry.  At every step it merges the pair of currently active
clusters with the smallest inter-cluster distance (Eq. 4), and updates the
distance of the merged cluster to all others with the average-linkage
Lance--Williams recurrence — which is exactly equal to recomputing Eq. (3) on
the merged member set, but in ``O(N)`` rather than ``O(N^2)`` work per step::

    D[s*, u] <- (|C_s*| D[s*, u] + |C_t*| D[t*, u]) / (|C_s*| + |C_t*|)

Complexity is ``O(N^2 d + N^3)`` time / ``O(N^2)`` space (paper, App. B.1).
This reference implementation is intended to be read alongside Algorithm 1 and
is exercised on real codebooks via ``scripts/build_masc_tree.py``.  It is exact
for any ``N``; for very large codebooks the ``O(N^3)`` merge loop is the
practical bottleneck and is left to the user's preferred accelerated
agglomerative backend (the algorithm is unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .distance import pairwise_euclidean

__all__ = ["MASCTree", "build_masc_tree"]


@dataclass
class MASCTree:
    """Result of the MASC construction.

    Attributes
    ----------
    mapping : np.ndarray, shape ``(N,)``, dtype int64
        ``mapping[i]`` is the coarse cluster index in ``[0, k)`` of token ``i``.
    k : int
        Number of coarse clusters (branches of the cut tree).
    sizes : np.ndarray, shape ``(k,)``
        Number of fine tokens in each coarse cluster.
    merge_history : np.ndarray, shape ``(N - k, 3)``
        Rows ``(s, t, d)``: at that step active slot ``t`` was merged into ``s``
        at inter-cluster distance ``d``.  Retained for inspection / plotting the
        dendrogram; not required to apply the mapping.
    """

    mapping: np.ndarray
    k: int
    sizes: np.ndarray
    merge_history: np.ndarray


def _find(parent: np.ndarray, i: int) -> int:
    """Union-find root with path compression (tracks token -> active slot)."""
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:
        parent[i], i = root, parent[i]
    return root


def build_masc_tree(
    embeddings: np.ndarray,
    k: int,
    *,
    distance_matrix: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> MASCTree:
    """Run Algorithm 1 and return the coarse-vocabulary mapping.

    Parameters
    ----------
    embeddings : np.ndarray, shape ``(N, d)``
        Finalised codebook embeddings of the trained tokenizer.
    k : int
        Target number of coarse clusters; must satisfy ``1 <= k <= N``.
    distance_matrix : np.ndarray, optional
        Pre-computed ``N x N`` Euclidean distance matrix.  If ``None`` it is
        computed from ``embeddings`` via :func:`masc.distance.pairwise_euclidean`.
    verbose : bool
        Print progress every ~5% of merges.

    Returns
    -------
    MASCTree
    """
    x = np.asarray(embeddings, dtype=np.float64)
    n = x.shape[0]
    if not (1 <= k <= n):
        raise ValueError(f"k must be in [1, N]={n}, got {k}")

    # --- Initialisation -----------------------------------------------------
    # N active singleton clusters; D[s, t] = ||v_s - v_t||_2, diagonal = inf.
    d = pairwise_euclidean(x) if distance_matrix is None else np.array(
        distance_matrix, dtype=np.float64, copy=True
    )
    np.fill_diagonal(d, np.inf)

    active = np.ones(n, dtype=bool)
    size = np.ones(n, dtype=np.float64)          # |C_j|
    parent = np.arange(n, dtype=np.int64)        # union-find over original tokens
    merge_history = np.empty((n - k, 3), dtype=np.float64)

    # --- Bottom-up hierarchical construction --------------------------------
    n_merges = n - k
    for it in range(n_merges):
        # (s*, t*) = argmin_{s != t, both active} D[s, t]
        flat = int(np.argmin(d))
        s, t = divmod(flat, n)
        if s > t:                                # canonicalise: keep the lower slot
            s, t = t, s
        dist_st = d[s, t]

        # Lance--Williams average-linkage update of the surviving slot s.
        ws, wt = size[s], size[t]
        denom = ws + wt
        # vectorised over all (active) columns u; inactive columns stay inf.
        new_row = (ws * d[s, :] + wt * d[t, :]) / denom
        d[s, :] = new_row
        d[:, s] = new_row
        d[s, s] = np.inf

        # Deactivate the absorbed cluster t.
        d[t, :] = np.inf
        d[:, t] = np.inf
        active[t] = False
        size[s] = denom
        parent[t] = s                            # every token of C_t now in C_s

        merge_history[it] = (s, t, dist_st)
        if verbose and n_merges >= 20 and it % max(1, n_merges // 20) == 0:
            print(f"[MASC] merge {it + 1}/{n_merges}  d={dist_st:.4f}", flush=True)

    # --- Final mapping construction -----------------------------------------
    # The remaining k active slots are the coarse clusters; relabel them 0..k-1.
    roots = np.array([_find(parent, i) for i in range(n)], dtype=np.int64)
    active_slots = np.flatnonzero(active)
    assert active_slots.size == k, (active_slots.size, k)
    slot_to_label = {int(slot): lab for lab, slot in enumerate(active_slots)}
    mapping = np.array([slot_to_label[int(r)] for r in roots], dtype=np.int64)

    sizes = np.bincount(mapping, minlength=k).astype(np.int64)
    return MASCTree(mapping=mapping, k=k, sizes=sizes, merge_history=merge_history)
