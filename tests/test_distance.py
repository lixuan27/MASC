"""Correctness of the manifold-aligned distance (Eq. 3)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from masc.distance import average_linkage_distance, pairwise_euclidean


def test_pairwise_matches_bruteforce():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 5))
    d = pairwise_euclidean(x)
    for i in range(20):
        for j in range(20):
            assert abs(d[i, j] - np.linalg.norm(x[i] - x[j])) < 1e-9
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)


def test_average_linkage_equals_mean_pairwise():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 4))
    a = [0, 1, 2, 3, 4]
    b = [10, 11, 12]
    brute = np.mean([np.linalg.norm(x[i] - x[j]) for i in a for j in b])
    assert abs(average_linkage_distance(a, b, x) - brute) < 1e-9


if __name__ == "__main__":
    test_pairwise_matches_bruteforce()
    test_average_linkage_equals_mean_pairwise()
    print("test_distance: OK")
