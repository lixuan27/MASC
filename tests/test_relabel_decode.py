"""Target relabelling, mapping I/O, and random-sampling decode."""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from masc import (MASCMapping, load_mapping, random_sample_decode,
                  relabel_targets, save_mapping)
from masc.mapping import invert_mapping


def _toy_mapping():
    # 10 fine tokens -> 3 coarse clusters
    return np.array([0, 0, 1, 1, 1, 2, 2, 0, 2, 1], dtype=np.int64), 3


def test_relabel_shapes_and_values():
    mapping, _ = _toy_mapping()
    z = np.array([[0, 2, 5], [9, 7, 3]])
    out = relabel_targets(z, mapping)
    assert out.shape == z.shape
    assert out.tolist() == [[0, 1, 2], [1, 0, 1]]


def test_mapping_roundtrip():
    mapping, k = _toy_mapping()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.npz")
        save_mapping(p, mapping, k)
        mp = load_mapping(p)
    assert mp.k == k and mp.n == 10
    inv = invert_mapping(mapping, k)
    assert sorted(inv[0]) == [0, 1, 7]
    assert sorted(inv[1]) == [2, 3, 4, 9]
    assert sorted(inv[2]) == [5, 6, 8]


def test_random_decode_stays_in_cluster():
    mapping, k = _toy_mapping()
    mp = MASCMapping(mapping, k)
    rng = np.random.default_rng(0)
    coarse = rng.integers(0, k, size=(4, 16))
    fine = random_sample_decode(coarse, mp, rng=rng)
    assert fine.shape == coarse.shape
    # every decoded fine token must belong to the cluster it was decoded from
    assert np.array_equal(mp.coarse_of(fine), coarse)
    assert fine.min() >= 0  # no padding (-1) leaked through


if __name__ == "__main__":
    test_relabel_shapes_and_values()
    test_mapping_roundtrip()
    test_random_decode_stays_in_cluster()
    print("test_relabel_decode: OK")
