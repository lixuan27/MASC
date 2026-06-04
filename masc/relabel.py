"""Training-target relabelling: fine token indices -> coarse cluster indices.

Section 3.3 of the paper.  During training the ground-truth sequence of fine
token indices ``z^q`` is converted into a sequence of coarse cluster indices
``z^b = M(z^q)`` (Eq. 5), and the autoregressive model is trained to predict the
next *cluster* index with a standard cross-entropy over ``k`` classes instead of
``N``.

This relabelling is deliberately framework-agnostic: it accepts either a numpy
array or any object exposing numpy-style fancy indexing (e.g. it works on a
torch tensor moved to CPU, or you can index the mapping on-device — see
``masc/integration.py``).  It is intentionally the *only* change MASC makes to
the data side of the training loop; the much larger task of wiring the coarse
targets and the resized output head into a specific backbone is handled in
``masc/integration.py`` and the official backbone repositories.
"""

from __future__ import annotations

import numpy as np

__all__ = ["relabel_targets"]


def relabel_targets(token_indices, mapping) -> np.ndarray:
    """Map a batch of fine token indices to coarse cluster indices.

    Parameters
    ----------
    token_indices : array-like of int, any shape ``(...)``
        Fine token indices ``z^q`` produced by the tokenizer's nearest-neighbour
        quantisation (Eq. 1).
    mapping : array-like of int, shape ``(N,)``
        The MASC mapping ``M``.

    Returns
    -------
    np.ndarray
        Coarse cluster indices ``z^b = M(z^q)`` with the same shape as the input.
    """
    mapping = np.asarray(mapping, dtype=np.int64)
    idx = np.asarray(token_indices, dtype=np.int64)
    return mapping[idx]
