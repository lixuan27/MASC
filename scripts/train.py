#!/usr/bin/env python3
"""Train an autoregressive image generator with the MASC coarse-vocabulary prior.

This entry point implements the *MASC-specific* parts of Stage 2 training
(Figure 2): it loads a pre-built MASC mapping, relabels the tokenizer codes to
coarse cluster indices on the fly, resizes the backbone's embedding/head to the
coarse vocabulary, and optimises the ``k``-way next-cluster cross-entropy (Eq. 5)
with the schedule reported in Appendix A.2 / Table 8.

What this script owns vs. what it expects from the backbone repo
----------------------------------------------------------------
MASC is a plug-and-play module.  To keep the change surface minimal and avoid
re-distributing third-party code, the *backbone* (its transformer, class
conditioning, CFG, KV-cache and AR sampling loop) is obtained from the official
repository of whichever generator you are reproducing and exposed through a thin
adapter implementing :class:`masc.integration.ARBackbone`.  Concretely you
provide a ``backbones`` package with::

    from backbones import build_backbone   # -> (model, ARBackbone adapter)

and a directory of pre-extracted tokenizer codes (the standard LlamaGen/VAR
practice of caching ``z^q`` per image).  See ``docs/reproduction.md`` for how to
produce both from the official LlamaGen / VAR / RandAR / IAR / CTF / GigaTok /
RAR releases.  Everything below this line is generic and backbone-independent.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def build_optimizer(model, cfg):
    import torch
    opt_cfg = cfg["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["peak_lr"],
        betas=tuple(opt_cfg.get("betas", (0.9, 0.95))),
        eps=float(opt_cfg.get("eps", 1e-8)),
        weight_decay=opt_cfg.get("weight_decay", 0.05),
    )


def cosine_lr(step, total_steps, peak_lr, final_lr, warmup_steps=0):
    """Cosine-annealing schedule used by all backbones (Table 8)."""
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return final_lr + 0.5 * (peak_lr - final_lr) * (1 + math.cos(math.pi * progress))


class CodeDataset:
    """Pre-extracted tokenizer codes ``z^q`` (+ class labels) on disk.

    Expects a directory of ``.npy`` shards, each ``[B, L]`` int token indices,
    plus a parallel ``labels.npy``.  Producing these from the tokenizer is a
    backbone-repo step (``docs/reproduction.md``); caching codes is standard
    practice and keeps the tokenizer out of the training loop.
    """

    def __init__(self, code_dir: str):
        self.files = sorted(
            os.path.join(code_dir, f) for f in os.listdir(code_dir)
            if f.endswith(".npy") and f != "labels.npy"
        )
        if not self.files:
            raise FileNotFoundError(
                f"No code shards in {code_dir}. Extract tokenizer codes from your "
                "backbone's tokenizer first (see docs/reproduction.md)."
            )
        self.labels = np.load(os.path.join(code_dir, "labels.npy"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return np.load(self.files[i]), self.labels[i]


def main() -> None:
    ap = argparse.ArgumentParser(description="Train AR generator + MASC prior.")
    ap.add_argument("--config", required=True, help="YAML config (see configs/).")
    ap.add_argument("--mapping", required=True, help="MASC mapping .npz from build_masc_tree.py")
    ap.add_argument("--codes", required=True, help="Directory of pre-extracted tokenizer codes.")
    ap.add_argument("--out", required=True, help="Checkpoint output directory.")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from masc import load_mapping
    from masc.integration import resize_token_embedding, resize_output_head, MASCObjective

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- MASC prior ---------------------------------------------------------
    mp = load_mapping(args.mapping)
    assert mp.k == cfg["masc"]["k"], (mp.k, cfg["masc"]["k"])
    objective = MASCObjective(mp.mapping, device=device)
    print(f"[train] MASC mapping: N={mp.n} -> k={mp.k}")

    # --- Backbone (from the official repo, via your adapter) ----------------
    # The adapter implements masc.integration.ARBackbone for your generator.
    try:
        from backbones import build_backbone  # provided by the user; see docs
    except ImportError as e:
        raise SystemExit(
            "Could not import `backbones.build_backbone`. MASC is plug-and-play: "
            "supply a thin adapter around the official backbone you are "
            "reproducing (LlamaGen/VAR/RandAR/IAR/CTF/GigaTok/RAR). "
            "See docs/reproduction.md."
        ) from e

    model, adapter = build_backbone(cfg["backbone"])
    model = model.to(device)
    resize_token_embedding(adapter, mp.k)   # k-row embedding table
    resize_output_head(adapter, mp.k)       # k-way logits

    # --- Data / optim -------------------------------------------------------
    ds = CodeDataset(args.codes)
    loader = DataLoader(ds, batch_size=None, shuffle=True,
                        num_workers=cfg.get("num_workers", 8), pin_memory=True,
                        persistent_workers=True)
    opt = build_optimizer(model, cfg)
    train_cfg = cfg["train"]
    total_steps = train_cfg["total_steps"]
    peak_lr = cfg["optimizer"]["peak_lr"]
    final_lr = cfg["optimizer"]["final_lr"]
    grad_clip = train_cfg.get("grad_clip", 1.0)
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.get("amp", True))

    os.makedirs(args.out, exist_ok=True)
    model.train()
    step = 0
    while step < total_steps:
        for codes, label in loader:
            codes = torch.as_tensor(codes, device=device)   # [B, L] fine z^q
            label = torch.as_tensor(label, device=device)
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total_steps, peak_lr, final_lr)
            with torch.cuda.amp.autocast(enabled=train_cfg.get("amp", True)):
                # The backbone consumes coarse inputs; targets are relabelled by
                # MASCObjective.  `model.forward_for_masc` is the one method your
                # adapter must route to the backbone's causal forward over the
                # coarse vocabulary (see docs/reproduction.md).
                coarse_in = objective.coarse_targets(codes)
                logits = model.forward_for_masc(coarse_in, class_labels=label)
                loss = objective.loss(logits, codes)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            if step % cfg.get("log_every", 50) == 0:
                print(f"[train] step {step}/{total_steps}  loss={loss.item():.4f}")
            if step > 0 and step % train_cfg.get("ckpt_every", 10000) == 0:
                torch.save({"model": model.state_dict(), "step": step},
                           os.path.join(args.out, f"ckpt_{step}.pt"))
            step += 1
            if step >= total_steps:
                break

    torch.save({"model": model.state_dict(), "step": step},
               os.path.join(args.out, "ckpt_final.pt"))
    print("[train] done.")


if __name__ == "__main__":
    main()
