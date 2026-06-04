#!/usr/bin/env python3
"""Build the MASC coarse-vocabulary mapping from a tokenizer codebook.

This is the one-time, offline MASC preprocessing step (Stage "MASC
Preprocessing" in Figure 2).  It is fully self-contained: it runs Algorithm 1
from :mod:`masc.clustering` on a codebook you provide, or on a synthetic
manifold-structured codebook (``--demo``) so the construction can be verified
without any external assets.

Examples
--------
Verify the algorithm end-to-end with no downloads::

    python scripts/build_masc_tree.py --demo --k 16 --out /tmp/masc_demo.npz

Build the real mapping from an exported codebook (a ``[N, d]`` array of the
finalised tokenizer embeddings; for the paper this is LlamaGen's VQ-VAE codebook
with N=16384, d=8/256 depending on the tokenizer variant)::

    python scripts/build_masc_tree.py --codebook codebook.npy --k 8192 \
        --out masc_mapping.npz

Exporting the codebook array itself depends on your tokenizer checkpoint; see
``docs/reproduction.md``.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from masc import build_masc_tree, save_mapping  # noqa: E402


def load_codebook(path: str) -> np.ndarray:
    """Load an ``[N, d]`` codebook from ``.npy`` / ``.npz`` / ``.pt``."""
    if path.endswith(".npy"):
        x = np.load(path)
    elif path.endswith(".npz"):
        data = np.load(path)
        key = "codebook" if "codebook" in data else list(data.keys())[0]
        x = data[key]
    elif path.endswith((".pt", ".pth")):
        import torch  # local import: only needed for torch checkpoints
        obj = torch.load(path, map_location="cpu")
        # A bare tensor, or a state_dict — pull the embedding weight if present.
        if hasattr(obj, "numpy"):
            x = obj.detach().cpu().numpy()
        else:
            cand = [v for k, v in obj.items() if "embedding" in k.lower()]
            if not cand:
                raise KeyError(
                    "Could not locate a codebook embedding in the checkpoint; "
                    "export it to a [N, d] .npy instead (see docs/reproduction.md)."
                )
            x = cand[0].detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported codebook format: {path}")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Codebook must be 2-D [N, d], got shape {x.shape}")
    return x


def synthetic_codebook(n: int = 512, d: int = 8, n_groups: int = 8,
                       seed: int = 0) -> np.ndarray:
    """A manifold-structured codebook for demonstration / testing.

    Points are drawn from ``n_groups`` low-dimensional curved patches embedded in
    ``R^d`` with non-uniform density — mimicking the semantic manifold structure
    that MASC is designed to recover.  Average-linkage agglomeration should
    recover the groups before merging across them.
    """
    rng = np.random.default_rng(seed)
    per = n // n_groups
    pts = []
    labels = []
    for g in range(n_groups):
        center = rng.normal(scale=6.0, size=(d,))
        t = rng.uniform(0, np.pi, size=per)
        # a curved 1-D arc in the first two dims + thin noise in the rest
        local = np.zeros((per, d))
        local[:, 0] = 2.5 * np.cos(t)
        local[:, 1] = 2.5 * np.sin(t)
        local += 0.15 * rng.normal(size=(per, d))
        pts.append(center[None, :] + local)
        labels.extend([g] * per)
    x = np.concatenate(pts, axis=0)
    perm = rng.permutation(x.shape[0])
    return x[perm].astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the MASC mapping (Algorithm 1).")
    ap.add_argument("--codebook", type=str, default=None,
                    help="Path to an [N, d] codebook (.npy/.npz/.pt).")
    ap.add_argument("--demo", action="store_true",
                    help="Use a synthetic manifold-structured codebook instead.")
    ap.add_argument("--demo-n", type=int, default=512)
    ap.add_argument("--demo-groups", type=int, default=8)
    ap.add_argument("--k", type=int, required=True, help="Target number of clusters.")
    ap.add_argument("--out", type=str, required=True, help="Output mapping (.npz).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.demo:
        x = synthetic_codebook(n=args.demo_n, n_groups=args.demo_groups)
        print(f"[MASC] synthetic codebook: N={x.shape[0]}, d={x.shape[1]}, "
              f"{args.demo_groups} latent groups")
    elif args.codebook:
        x = load_codebook(args.codebook)
        print(f"[MASC] loaded codebook: N={x.shape[0]}, d={x.shape[1]}")
    else:
        ap.error("provide --codebook PATH or --demo")

    tree = build_masc_tree(x, k=args.k, verbose=args.verbose)
    save_mapping(args.out, tree.mapping, tree.k)

    sizes = tree.sizes
    print(f"[MASC] built {tree.k} clusters from {x.shape[0]} tokens")
    print(f"[MASC] cluster size: min={sizes.min()} max={sizes.max()} "
          f"mean={sizes.mean():.1f} median={int(np.median(sizes))}")
    print(f"[MASC] mapping saved -> {args.out}")


if __name__ == "__main__":
    main()
