"""Manifold-aligned distance for MASC.

This module implements the *centroid-free, instance-based average distance*
that MASC uses to measure dissimilarity between two clusters of codebook
tokens (Eq. 3 of the paper)::

    D(C_s, C_t) = 1 / (|C_s| |C_t|) * sum_{v_i in C_s} sum_{v_j in C_t} ||v_i - v_j||_2

Because the metric relies only on the locations of *actual* tokens (which are
guaranteed to lie on the semantic manifold) and never on an arithmetic
centroid, it is robust to the off-manifold / non-uniform-density issues that
make Euclidean-to-centroid metrics (e.g. k-means) ill-suited here.

The functions below are written for clarity and correctness; they are the
reference used by the unit tests in ``tests/test_distance.py``.  The
agglomerative construction in :mod:`masc.clustering` does *not* call
:func:`average_linkage_distance` in its inner loop — it maintains the same
quantity incrementally via the Lance--Williams update for numerical and
algorithmic efficiency (see the paper, Appendix B.1).
"""

from __future__ import annotations

import numpy as np

__all__ = ["pairwise_euclidean", "average_linkage_distance"]


def pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    """Return the dense ``N x N`` matrix of pairwise Euclidean distances.

    Parameters
    ----------
    x : np.ndarray, shape ``(N, d)``
        Codebook embeddings ``Z = {v_1, ..., v_N}``.

    Returns
    -------
    np.ndarray, shape ``(N, N)``
        ``D[s, t] = ||x[s] - x[t]||_2``.  The matrix is symmetric with a zero
        diagonal.
    """
    x = np.asarray(x, dtype=np.float64)
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b  (computed in float64 for stability)
    sq = np.sum(x * x, axis=1)
    g = x @ x.T
    d2 = sq[:, None] + sq[None, :] - 2.0 * g
    np.maximum(d2, 0.0, out=d2)  # clamp tiny negatives from round-off
    d = np.sqrt(d2)
    np.fill_diagonal(d, 0.0)
    return d


def average_linkage_distance(
    cluster_a: np.ndarray, cluster_b: np.ndarray, x: np.ndarray
) -> float:
    """Eq. (3): mean pairwise Euclidean distance between two token clusters.

    Parameters
    ----------
    cluster_a, cluster_b : array-like of int
        Indices into ``x`` of the tokens belonging to each cluster.
    x : np.ndarray, shape ``(N, d)``
        Codebook embeddings.

    Returns
    -------
    float
        ``D(C_a, C_b)`` as defined in Eq. (3).  This is the *exact* quantity
        the Lance--Williams recurrence in :mod:`masc.clustering` reproduces
        incrementally; the unit tests assert this equivalence.
    """
    a = np.asarray(cluster_a, dtype=np.int64)
    b = np.asarray(cluster_b, dtype=np.int64)
    va = np.asarray(x, dtype=np.float64)[a]
    vb = np.asarray(x, dtype=np.float64)[b]
    # pairwise distances between the two member sets
    sq_a = np.sum(va * va, axis=1)
    sq_b = np.sum(vb * vb, axis=1)
    d2 = sq_a[:, None] + sq_b[None, :] - 2.0 * (va @ vb.T)
    np.maximum(d2, 0.0, out=d2)
    return float(np.sqrt(d2).mean())
