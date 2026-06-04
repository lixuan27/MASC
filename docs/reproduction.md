# Reproducing the paper

MASC is a **plug-and-play prior**. Reproducing a number from the paper means:
(1) take a published AR generator and its tokenizer from the official
repository, (2) build the MASC coarse vocabulary from that tokenizer’s codebook,
(3) train the generator to predict coarse clusters, and (4) evaluate with the
standard protocol. Steps (2)–(4) use this repository; the backbone itself comes
from its authors. This page is the end-to-end checklist.

> All paper numbers were produced on 8× NVIDIA H100 (80 GB), PyTorch 2.1,
> CUDA 12.1, ImageNet-1K at 256×256, 300 epochs (Appendix A.1–A.2).

---

## Step 0 — Get a backbone and its tokenizer

Clone the official repository of the backbone you want to enhance and follow its
own setup. The paper evaluates:

| Backbone | Official source (as cited) | Tokenizer / codebook |
|---|---|---|
| LlamaGen-B/L/XL | Sun et al., 2024 | VQ-VAE, `N=16384` |
| VAR-d24 | Tian et al., 2024 | multi-scale VQ |
| RandAR-XL | Pang et al., 2025 | LlamaGen VQ |
| IAR-B/L/XL | Hu et al., 2025 | LlamaGen VQ |
| CTF-LlamaGen-L | Guo et al., 2025 | LlamaGen VQ |
| GigaTok-L | Xiong et al., 2025 | 16k-vocab tokenizer |
| RAR-L | Yu et al., 2024 | LlamaGen / GigaTok |

You need two things from the backbone repo: its **tokenizer** (to encode images
to token indices and to decode indices back to images) and its **transformer**
(the generator you will train).

## Step 1 — Export the tokenizer codebook

MASC operates on the *finalised* codebook embeddings `Z ∈ R^{N×d}`. Export them
to a plain array once:

```python
# pseudo-code; the attribute path depends on the tokenizer implementation
import numpy as np, torch
tok = load_official_tokenizer(ckpt)          # from the backbone repo
codebook = tok.quantize.embedding.weight     # [N, d], the learned codebook
np.save("codebook.npy", codebook.detach().cpu().numpy())
```

## Step 2 — Build the MASC mapping (this repo)

```bash
python scripts/build_masc_tree.py --codebook codebook.npy --k 8192 \
    --out masc_mapping.npz --verbose
```

This runs Algorithm 1 and writes the `token → cluster` mapping. `k=8192` is the
default (Table 1); `k=4096` gives the best FID on LlamaGen-XL (Table 2). The step
is deterministic and one-time. For `N=16384` the reference `O(N^3)` merge loop is
the practical bottleneck; the construction is exact and you may substitute any
average-linkage (UPGMA) agglomerative backend — the resulting partition is the
same (the unit tests pin this equivalence).

## Step 3 — Cache tokenizer codes (recommended)

As is standard for LlamaGen/VAR training, pre-extract the token indices for the
ImageNet training set so the tokenizer is out of the training loop:

```python
# for each image: codes = tok.encode_indices(img)  -> [L] int token ids
# save shards of [B, L] int arrays + a parallel labels.npy  (see scripts/train.py)
```

`scripts/train.py::CodeDataset` consumes exactly this layout.

## Step 4 — Wire MASC into the backbone (write a small adapter)

MASC needs three submodules of the backbone re-pointed at the coarse vocabulary:
the **token embedding** (now `k` rows), the **output head** (now `k` logits), and
the **targets** (relabelled with `M`). Implement the
`masc.integration.ARBackbone` protocol against your backbone’s source:

```python
from masc.integration import ARBackbone, resize_token_embedding, resize_output_head

class LlamaGenAdapter:                       # one per backbone family
    def __init__(self, model): self.m = model
    def get_token_embedding(self):  return self.m.tok_embeddings      # repo-specific
    def set_token_embedding(self, e): self.m.tok_embeddings = e
    def get_output_head(self):      return self.m.output              # repo-specific
    def set_output_head(self, h):   self.m.output = h

# expose a builder that scripts/train.py and scripts/eval_fid.py import:
def build_backbone(cfg):
    model = build_official_llamagen(cfg)     # from the official repo
    return model, LlamaGenAdapter(model)
```

Put `build_backbone` (and, for eval, `load_tokenizer_decoder` /
`load_tokenizer`) in a `backbones` package on your `PYTHONPATH`. The transformer
forward, causal masking, class-conditioning, CFG and the autoregressive sampling
loop are **unchanged** — MASC only swaps the in/out vocabulary and the targets.
`scripts/train.py` calls `model.forward_for_masc(coarse_in, class_labels=...)`
and `eval_fid.py` calls `model.sample_coarse(class_label=..., cfg_scale=...)`;
route these to your backbone’s existing forward / sampler over the coarse
vocabulary.

## Step 5 — Train

```bash
python scripts/train.py \
    --config configs/llamagen_xl_masc.yaml \
    --mapping masc_mapping.npz \
    --codes /path/to/cached_codes \
    --out runs/llamagen_xl_masc
```

The optimiser, LR schedule, batch size, gradient clipping and epoch count match
Appendix A.2 / Table 8. Set `train.total_steps = epochs × ceil(num_images /
global_batch_size)` for your dataset size.

## Step 6 — Evaluate (Appendix A.4 protocol)

Generate 50,000 samples (50 per class × 1,000 classes) and score with
`torch-fidelity` against the standard ImageNet training statistics:

```bash
python scripts/eval_fid.py \
    --config configs/llamagen_xl_masc.yaml \
    --mapping masc_mapping.npz \
    --ckpt runs/llamagen_xl_masc/ckpt_final.pt \
    --out samples/llamagen_xl_masc --cfg-scale 2.5

fidelity --gpu 0 --fid --isc --prc \
    --input1 samples/llamagen_xl_masc \
    --input2 <imagenet_train_or_precomputed_stats>
```

## Optional — diagnostics

* **Semantic Replacement Test** (Table 11): with only a tokenizer and a mapping,
  `scripts/semantic_replacement_test.py` reproduces the rFID/PSNR/SSIM
  degradation comparison that demonstrates MASC clusters are semantically
  interchangeable while k-means clusters are not.
* **Baselines**: build a k-means mapping (`scikit-learn`, `k=8192`,
  `init='random'`, 10 runs, lowest inertia — Appendix A.3) or a random mapping
  in the same `token → cluster` format and pass it to the same `train.py` /
  `eval_fid.py` to reproduce the `+ k-means` and `+ Random` rows of Table 1/3.

## Notes on compute

A single LlamaGen-XL + MASC run is ~300 epochs on 8× H100; the MASC
preprocessing itself is a one-time CPU/GPU step of well under an hour
(Appendix B.1). Plan accordingly — the headline tables span many backbone × k
configurations and represent substantial GPU time on the backbone side.
