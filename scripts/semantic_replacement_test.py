#!/usr/bin/env python3
"""Semantic Replacement Test — intra-cluster coherence diagnostic (App. D.1).

This is the experiment behind Figure 4 and Table 11.  It directly probes whether
a clustering groups *semantically interchangeable* tokens: for each token in an
image's code map, replace it with another token sampled uniformly from the same
cluster, decode the perturbed codes back to an image, and measure how much the
reconstruction degrades (rFID / PSNR / SSIM).  A coherent clustering (MASC)
should degrade far less than a geometry-agnostic one (k-means).

It is self-contained *given* a tokenizer (to encode/decode images) and a mapping;
the tokenizer encode/decode is the only backbone-repo dependency.  The MASC
replacement logic itself is implemented here in full and reuses the exact
random-member sampling used at generation time
(:func:`masc.decode.random_sample_decode`).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def main() -> None:
    ap = argparse.ArgumentParser(description="Semantic Replacement Test (App. D.1).")
    ap.add_argument("--mapping", required=True, help="MASC (or k-means) mapping .npz")
    ap.add_argument("--images", required=True, help="Directory of evaluation images.")
    ap.add_argument("--limit", type=int, default=5000)
    args = ap.parse_args()

    import torch

    from masc import load_mapping, random_sample_decode

    mp = load_mapping(args.mapping)
    rng = np.random.default_rng(0)

    # The tokenizer (encoder + decoder) is the only external dependency here.
    try:
        from backbones import load_tokenizer
    except ImportError as e:
        raise SystemExit(
            "Need `backbones.load_tokenizer` (encode image -> z^q, decode z^q -> "
            "image), a thin adapter over the official tokenizer. "
            "See docs/reproduction.md."
        ) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer().to(device).eval()

    psnrs = []
    paths = [os.path.join(args.images, f) for f in sorted(os.listdir(args.images))][: args.limit]
    with torch.no_grad():
        for p in paths:
            img = load_image(p).to(device)                 # [1, 3, H, W] in [0,1]
            codes = tok.encode_indices(img)                # [1, L] fine z^q
            codes_np = codes.detach().cpu().numpy()
            # Replace every token by a random member of its own cluster.
            coarse = mp.coarse_of(codes_np)                # M(z^q)
            replaced = random_sample_decode(coarse, mp, rng=rng)
            recon = tok.decode_indices(torch.as_tensor(replaced, device=device))
            psnrs.append(psnr(img.cpu().numpy(), recon.cpu().numpy()))

    print(f"[srt] images={len(psnrs)}  PSNR(replaced vs original) "
          f"mean={np.mean(psnrs):.2f}")
    print("[srt] for rFID/SSIM, feed the replaced reconstructions to "
          "torch-fidelity / skimage.metrics against the originals "
          "(see Table 11 protocol).")


def load_image(path):
    from torchvision.io import read_image
    img = read_image(path).float() / 255.0
    return img.unsqueeze(0)


if __name__ == "__main__":
    main()
