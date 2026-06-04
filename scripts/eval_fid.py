#!/usr/bin/env python3
"""Evaluate a MASC-trained generator: FID / IS / Precision / Recall.

Implements the paper's evaluation protocol (Appendix A.4):

* 50,000 samples total — 50 images for each of the 1,000 ImageNet classes;
* FID computed against the standard pre-calculated Inception statistics of the
  ImageNet training set;
* FID / IS / Precision / Recall via ``torch-fidelity``.

The generation loop is backbone-specific: the AR model emits coarse cluster
indices ``z^b``, MASC decodes them to fine token indices ``z^q`` with
:func:`masc.decode.random_sample_decode` (the default strategy), and the
tokenizer decoder renders the image.  We wire the MASC decode here and defer the
backbone's sampling loop and the tokenizer decoder to your adapter / the official
repo (see ``docs/reproduction.md``).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    ap = argparse.ArgumentParser(description="MASC generation-quality evaluation.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--mapping", required=True, help="MASC mapping .npz")
    ap.add_argument("--ckpt", required=True, help="Trained generator checkpoint.")
    ap.add_argument("--out", required=True, help="Directory to write 50k samples.")
    ap.add_argument("--cfg-scale", type=float, default=2.5,
                    help="Classifier-free guidance scale (paper galleries use 2.5).")
    ap.add_argument("--samples-per-class", type=int, default=50)
    ap.add_argument("--num-classes", type=int, default=1000)
    args = ap.parse_args()

    import torch

    from masc import load_mapping, random_sample_decode

    mp = load_mapping(args.mapping)
    rng = np.random.default_rng(0)
    os.makedirs(args.out, exist_ok=True)

    # Backbone + tokenizer decoder come from your adapter / the official repo.
    try:
        from backbones import build_backbone, load_tokenizer_decoder
    except ImportError as e:
        raise SystemExit(
            "Need `backbones.build_backbone` and `backbones.load_tokenizer_decoder` "
            "(thin adapters over the official backbone/tokenizer). "
            "See docs/reproduction.md."
        ) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, adapter = build_backbone(load_config(args.config)["backbone"])
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model = model.to(device).eval()
    decoder = load_tokenizer_decoder().to(device).eval()

    n = 0
    with torch.no_grad():
        for cls in range(args.num_classes):
            for _ in range(args.samples_per_class):
                # 1) AR model samples a sequence of COARSE cluster indices z^b.
                coarse = model.sample_coarse(class_label=cls, cfg_scale=args.cfg_scale)
                # 2) MASC decode: coarse cluster -> fine token (random member).
                coarse_np = coarse.detach().cpu().numpy()
                fine = random_sample_decode(coarse_np, mp, rng=rng)
                fine_t = torch.as_tensor(fine, device=device)
                # 3) tokenizer decoder renders the image grid.
                img = decoder.decode_indices(fine_t)
                save_image(img, os.path.join(args.out, f"{n:06d}.png"))
                n += 1

    print(f"[eval] wrote {n} samples to {args.out}")
    print("[eval] now compute metrics, e.g.:")
    print(f"       fidelity --gpu 0 --fid --isc --prc --input1 {args.out} "
          "--input2 <imagenet_train_or_precomputed_stats>")


def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def save_image(img, path):
    # `img` is a [3, H, W] tensor in [0, 1]; use your backbone's renderer.
    from torchvision.utils import save_image as _save
    _save(img, path)


if __name__ == "__main__":
    main()
