"""Hierarchical decoding (extended strategy) — network interface only.

Section 3.3 describes an optional secondary refinement network ``G_refine`` that,
given the main model's hidden state and the predicted coarse cluster ``z^b``,
predicts the *fine* token by computing a distribution restricted to the tokens
within that cluster.  Per Table 10 / App. C.2 it is a single-layer transformer
decoder (4 heads, embed dim 512, ~25M params) and buys only a marginal FID gain
(2.92 -> 2.87 on LlamaGen-L) over Random Sampling.

All numbers reported as the paper's main results use the parameter-free Random
Sampling decoder (``masc.decode.random_sample_decode``).  The refinement network
is therefore *not* needed to reproduce them, and only its module interface is
provided here.  The intra-cluster restriction and the training recipe for
``G_refine`` belong to the extended pipeline.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise ImportError("masc.refine requires PyTorch.") from exc

__all__ = ["HierarchicalRefiner"]


class HierarchicalRefiner(nn.Module):
    """Interface for the optional fine-token refinement network ``G_refine``.

    The forward pass restricts the prediction to the members of the predicted
    cluster (a masked softmax over the fine vocabulary).  The masking tensor that
    encodes ``{i : M(i) = z^b}`` and the training objective are part of the
    extended pipeline and are not included in this reference release.
    """

    def __init__(self, hidden_dim: int, n_fine_tokens: int, n_clusters: int,
                 embed_dim: int = 512, n_heads: int = 4):
        super().__init__()
        self.cluster_embed = nn.Embedding(n_clusters, embed_dim)
        self.in_proj = nn.Linear(hidden_dim, embed_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=1)
        self.fine_head = nn.Linear(embed_dim, n_fine_tokens)

    def forward(self, hidden_state: "torch.Tensor", coarse_index: "torch.Tensor"):
        raise NotImplementedError(
            "HierarchicalRefiner is provided as an interface for the extended "
            "decoding strategy. The main results use Random Sampling "
            "(masc.decode.random_sample_decode); see App. C.2."
        )
