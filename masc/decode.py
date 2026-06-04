"""Decoding a predicted coarse cluster index back to a fine token index.

Section 3.3, "Integrating the MASC Prior".  The autoregressive model emits a
sequence of coarse cluster indices ``z^b``; before the tokenizer decoder can
render an image these must be turned back into fine token indices ``z^q``.

Two strategies are described in the paper:

* **Random Sampling (default).**  For each predicted coarse index, sample a fine
  token *uniformly at random* from that cluster's members
  ``{v_i | M(i) = z^b_t}``.  Because MASC clusters are semantically coherent
  (App. D), any member is a good representative, so this parameter-free decoder
  already yields high-quality results (Table 10).  Implemented in full below.

* **Hierarchical Decoding (extended).**  A small refinement network predicts the
  fine token within the cluster, trading +25M params for a marginal fidelity
  gain.  Only the network *interface* is provided here (``masc/refine.py``); its
  training recipe is part of the extended pipeline and is not required to
  reproduce the main results, which all use Random Sampling.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .mapping import MASCMapping

__all__ = ["random_sample_decode"]


def random_sample_decode(
    coarse_indices: np.ndarray,
    mapping: MASCMapping,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Default decoder: uniform random member sampling within each cluster.

    Parameters
    ----------
    coarse_indices : np.ndarray of int, any shape ``(...)``
        Predicted coarse cluster indices ``z^b`` in ``[0, k)``.
    mapping : MASCMapping
        The MASC mapping handle (provides the padded member table).
    rng : np.random.Generator, optional
        Source of randomness; if ``None`` a default generator is used.

    Returns
    -------
    np.ndarray
        Fine token indices ``z^q`` with the same shape as ``coarse_indices``,
        ready to be embedded by the tokenizer decoder.
    """
    if rng is None:
        rng = np.random.default_rng()
    c = np.asarray(coarse_indices, dtype=np.int64)
    counts = mapping.member_counts[c]                      # [...]
    # uniform offset in [0, count) per position, then gather from the padded table
    offsets = (rng.random(c.shape) * counts).astype(np.int64)
    offsets = np.minimum(offsets, counts - 1)
    flat_c = c.reshape(-1)
    flat_o = offsets.reshape(-1)
    tokens = mapping.member_table[flat_c, flat_o]
    return tokens.reshape(c.shape)
