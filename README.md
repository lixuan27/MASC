# MASC: Manifold-Aligned Semantic Clustering

Reference implementation for the ICML 2026 paper
**“A Flat Vocabulary or a Rich Hierarchy? Re-introducing Intrinsic Structure
Transforms the Autoregressive Image Generation.”**

📄 [Paper](docs/static/MASC_paper.pdf) &nbsp;|&nbsp; 🌐 [Project Page](https://lixuan27.github.io/MASC/) &nbsp;|&nbsp; 💻 [GitHub](https://github.com/lixuan27/MASC)

> **TL;DR.** Discrete autoregressive (AR) image generators predict over a vast,
> *flat* vocabulary of visual tokens, ignoring the fact that the codebook
> embeddings lie on a low-dimensional **semantic manifold**. MASC is a
> one-time, offline preprocessing module that builds a density-driven,
> manifold-aligned **hierarchical semantic tree** over the codebook and uses it
> to turn the hard `N`-way token prediction into a structured `k`-way
> coarse-cluster prediction. It is **plug-and-play**: it accelerates training by
> up to **71%** and improves quality (LlamaGen-XL FID **2.87 → 2.49**) across a
> wide range of AR backbones, with **no architectural changes** to the backbone.

---

## Install

```bash
git clone <this-repo> MASC && cd MASC
python -m pip install -r requirements.txt   # numpy/scipy required; torch for training
```

The MASC **preprocessing core** (clustering / mapping / relabel / decode) depends
only on `numpy`. PyTorch is needed only for the AR-backbone integration helpers
and the training/eval scripts.

## Quickstart — verify the algorithm in 5 seconds

No assets required. Build a MASC tree on a synthetic manifold-structured
codebook and confirm it recovers the latent semantic groups:

```bash
python examples/demo_cluster_codebook.py
#   cluster purity vs latent groups: 1.000  (groups perfectly recovered)
#   figure saved -> examples/masc_demo.png

python scripts/build_masc_tree.py --demo --k 16 --out /tmp/masc.npz --verbose
```

Run the test suite (the clustering is cross-checked against SciPy’s
average-linkage, confirming Algorithm 1 is exact):

```bash
python tests/test_distance.py
python tests/test_clustering.py
python tests/test_relabel_decode.py
# (or: pytest -q tests/)
```

---

## The three-line integration

MASC changes a training pipeline in exactly three places:

```python
from masc import build_masc_tree, load_mapping, relabel_targets, random_sample_decode

# (1) OFFLINE, ONCE: build the coarse vocabulary from the tokenizer codebook.
tree = build_masc_tree(codebook_embeddings, k=8192)          # Algorithm 1
save_mapping("masc_mapping.npz", tree.mapping, tree.k)

# (2) TRAIN: predict the next coarse cluster instead of the next fine token.
coarse_targets = relabel_targets(fine_token_ids, tree.mapping)   # Eq. 5

# (3) SAMPLE: decode a predicted cluster back to a token (default strategy).
fine_tokens = random_sample_decode(coarse_cluster_indices, mapping)
```

Everything else — the transformer, class conditioning, CFG, KV-cache and the
AR sampling loop — is the **unmodified backbone**.

---

## Citation

```bibtex
@inproceedings{he2026masc,
  title     = {A Flat Vocabulary or a Rich Hierarchy? Re-introducing Intrinsic
               Structure Transforms the Autoregressive Image Generation},
  author    = {He, Lixuan and Zheng, Shikang},
  booktitle = {Proceedings of the 43rd International Conference on Machine
               Learning (ICML)},
  year      = {2026}
}
```
