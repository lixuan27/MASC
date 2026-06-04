"""Persisting and applying the MASC mapping ``M : token -> coarse cluster``.

The mapping is the single artefact produced by the one-time, offline MASC
preprocessing step.  It is a small integer array of length ``N`` and is all the
downstream autoregressive training/inference needs from MASC.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

__all__ = ["save_mapping", "load_mapping", "invert_mapping", "MASCMapping"]


def save_mapping(path: str, mapping: np.ndarray, k: int) -> None:
    """Save the token->cluster mapping to ``path`` (``.npz``)."""
    mapping = np.asarray(mapping, dtype=np.int64)
    if mapping.ndim != 1:
        raise ValueError("mapping must be 1-D of length N")
    np.savez(path, mapping=mapping, k=np.int64(k), n=np.int64(mapping.shape[0]))


def load_mapping(path: str) -> "MASCMapping":
    """Load a mapping previously written by :func:`save_mapping`."""
    data = np.load(path)
    return MASCMapping(mapping=data["mapping"].astype(np.int64), k=int(data["k"]))


def invert_mapping(mapping: np.ndarray, k: int) -> Dict[int, List[int]]:
    """Return ``{cluster_index: [token ids in that cluster]}``.

    Used by the random-sampling decoder to map a predicted coarse index back to
    the set of fine tokens it represents.
    """
    mapping = np.asarray(mapping, dtype=np.int64)
    members: Dict[int, List[int]] = {c: [] for c in range(k)}
    for tok, clu in enumerate(mapping.tolist()):
        members[int(clu)].append(int(tok))
    return members


class MASCMapping:
    """Convenience handle bundling the mapping with its derived structures."""

    def __init__(self, mapping: np.ndarray, k: int):
        self.mapping = np.asarray(mapping, dtype=np.int64)
        self.k = int(k)
        self.n = int(self.mapping.shape[0])
        self._members = invert_mapping(self.mapping, self.k)
        # Padded [k, max_cluster_size] table of token ids for fast batched
        # sampling; -1 marks padding.
        self._max_members = max(len(v) for v in self._members.values())
        table = np.full((self.k, self._max_members), -1, dtype=np.int64)
        counts = np.zeros(self.k, dtype=np.int64)
        for c, toks in self._members.items():
            table[c, : len(toks)] = toks
            counts[c] = len(toks)
        self.member_table = table       # [k, max_members]
        self.member_counts = counts     # [k]

    def coarse_of(self, token_ids: np.ndarray) -> np.ndarray:
        """Vectorised ``M(token)`` lookup."""
        return self.mapping[np.asarray(token_ids, dtype=np.int64)]

    def members(self, cluster_index: int) -> List[int]:
        return self._members[int(cluster_index)]
