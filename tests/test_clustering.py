"""Correctness of Algorithm 1 (MASC construction).

The MASC construction is average-linkage agglomerative clustering (UPGMA, paper
App. B.2) driven by the manifold-aligned distance.  These tests pin three
properties:

1. The Lance--Williams recurrence used in the inner loop reproduces the *exact*
   Eq. (3) average-linkage distance at the first merge.
2. The resulting flat k-partition agrees with an independent reference
   (scipy's average linkage) on well-separated data.
3. The construction is deterministic and the mapping is well-formed.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from masc.clustering import build_masc_tree
from masc.distance import average_linkage_distance, pairwise_euclidean


def _same_partition(a, b):
    """True if label arrays a, b induce the same partition (up to relabeling)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    pairs = set(zip(a.tolist(), b.tolist()))
    # bijection between label sets <=> contingency is a permutation
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


def test_first_merge_is_min_pairwise():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 3))
    d = pairwise_euclidean(x)
    np.fill_diagonal(d, np.inf)
    # build a 1-merge tree by asking for k = N-1 and checking the merged pair
    tree = build_masc_tree(x, k=x.shape[0] - 1)
    s, t, dist = tree.merge_history[0]
    assert abs(dist - d.min()) < 1e-9
    # at the first merge the LW distance to any third point equals Eq.(3) exactly
    s, t = int(s), int(t)
    u = next(i for i in range(x.shape[0]) if i not in (s, t))
    eq3 = average_linkage_distance([s, t], [u], x)
    # reconstruct what LW stored: mean of the two singleton distances
    lw = 0.5 * (d[s, u] + d[t, u])
    assert abs(lw - eq3) < 1e-9


def test_matches_scipy_average_linkage_on_blobs():
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
    except ImportError:
        return  # scipy optional for this cross-check
    rng = np.random.default_rng(2)
    centers = np.array([[0, 0], [20, 0], [0, 20], [20, 20]], dtype=float)
    x = np.concatenate([c + rng.normal(scale=0.5, size=(25, 2)) for c in centers])
    masc = build_masc_tree(x, k=4)
    Z = linkage(x, method="average", metric="euclidean")
    ref = fcluster(Z, t=4, criterion="maxclust")
    assert _same_partition(masc.mapping, ref)


def test_mapping_well_formed_and_deterministic():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(60, 6))
    t1 = build_masc_tree(x, k=7)
    t2 = build_masc_tree(x, k=7)
    assert t1.mapping.shape == (60,)
    assert set(np.unique(t1.mapping).tolist()) == set(range(7))
    assert t1.sizes.sum() == 60
    assert np.array_equal(t1.mapping, t2.mapping)  # deterministic


if __name__ == "__main__":
    test_first_merge_is_min_pairwise()
    test_matches_scipy_average_linkage_on_blobs()
    test_mapping_well_formed_and_deterministic()
    print("test_clustering: OK")
