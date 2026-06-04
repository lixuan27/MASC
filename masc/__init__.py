"""MASC — Manifold-Aligned Semantic Clustering.

Reference implementation accompanying the paper
"A Flat Vocabulary or a Rich Hierarchy? Re-introducing Intrinsic Structure
Transforms the Autoregressive Image Generation" (ICML 2026).

The package is organised so that the *offline MASC preprocessing* — the part
that turns a tokenizer codebook into a coarse-vocabulary mapping — is fully
self-contained and depends only on numpy:

    from masc import build_masc_tree, save_mapping, load_mapping
    tree = build_masc_tree(codebook_embeddings, k=8192)
    save_mapping("masc_mapping.npz", tree.mapping, tree.k)

Relabelling targets and random-sampling decode are likewise numpy-only:

    from masc import relabel_targets, random_sample_decode, MASCMapping

The AR-backbone surgery (masc.integration) and the optional hierarchical
refiner (masc.refine) require PyTorch and are imported lazily.
"""

from .distance import average_linkage_distance, pairwise_euclidean
from .clustering import MASCTree, build_masc_tree
from .mapping import MASCMapping, invert_mapping, load_mapping, save_mapping
from .relabel import relabel_targets
from .decode import random_sample_decode

__version__ = "1.0.0"

__all__ = [
    "average_linkage_distance",
    "pairwise_euclidean",
    "MASCTree",
    "build_masc_tree",
    "MASCMapping",
    "invert_mapping",
    "load_mapping",
    "save_mapping",
    "relabel_targets",
    "random_sample_decode",
    "__version__",
]
