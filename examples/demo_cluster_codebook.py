#!/usr/bin/env python3
"""End-to-end MASC demo on a synthetic manifold-structured codebook.

Runs the full offline MASC preprocessing on a small synthetic codebook and shows
that the density-driven, average-linkage construction recovers the latent
semantic groups (the property visualised in Figure 3 of the paper).  Requires
only numpy; if matplotlib is available it also saves a scatter plot.

    python examples/demo_cluster_codebook.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from masc import build_masc_tree                                   # noqa: E402


def adjusted_purity(pred: np.ndarray, true: np.ndarray) -> float:
    """Fraction of points whose cluster's majority-true-label matches them."""
    correct = 0
    for c in np.unique(pred):
        members = true[pred == c]
        if members.size:
            correct += np.bincount(members).max()
    return correct / pred.size


def main() -> None:
    n_groups = 8
    per = 64
    n = n_groups * per
    rng = np.random.default_rng(0)

    # Build a manifold-structured codebook whose true group labels we know, so we
    # can score how well MASC recovers them (the property visualised in Fig. 3).
    pts, true = [], []
    for g in range(n_groups):
        center = rng.normal(scale=6.0, size=(8,))
        t = rng.uniform(0, np.pi, size=per)
        local = np.zeros((per, 8))
        local[:, 0] = 2.5 * np.cos(t)
        local[:, 1] = 2.5 * np.sin(t)
        local += 0.15 * rng.normal(size=(per, 8))
        pts.append(center[None, :] + local)
        true += [g] * per
    x = np.concatenate(pts, 0)
    true = np.asarray(true)

    tree = build_masc_tree(x, k=n_groups)
    pur = adjusted_purity(tree.mapping, true)
    print(f"[demo] N={n} tokens, true groups={n_groups}, MASC k={tree.k}")
    print(f"[demo] cluster sizes: {tree.sizes.tolist()}")
    print(f"[demo] cluster purity vs latent groups: {pur:.3f} "
          f"(1.0 = groups perfectly recovered)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].scatter(x[:, 0], x[:, 1], c=true, cmap="tab10", s=8)
        ax[0].set_title("Latent semantic groups")
        ax[1].scatter(x[:, 0], x[:, 1], c=tree.mapping, cmap="tab10", s=8)
        ax[1].set_title(f"MASC clusters (purity={pur:.2f})")
        out = os.path.join(os.path.dirname(__file__), "masc_demo.png")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"[demo] figure saved -> {out}")
    except Exception:
        print("[demo] matplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()
