# MASC in detail

This note maps each paper equation to its implementation.

## The high-dimensional prediction challenge (§3.1)

A tokenizer quantises an image into a grid of indices `z^q ∈ {1..N}^{h×w}` via
nearest-neighbour lookup in a codebook `Z = {v_1..v_N} ⊂ R^d` (Eq. 1). The AR
model then performs an `N`-way classification at every step (Eq. 2). With
`N=16384` this flat task discards the geometry of `Z` — the embeddings are not
scattered randomly but lie on a low-dimensional semantic manifold.

## Manifold-aligned distance (§3.2.1, Eq. 3)

MASC measures cluster dissimilarity with the **centroid-free, instance-based
average distance**

```
D(C_s, C_t) = 1/(|C_s||C_t|) · Σ_{v_i∈C_s} Σ_{v_j∈C_t} ‖v_i − v_j‖₂
```

→ `masc/distance.py::average_linkage_distance`. It is manifold-aligned because it
only ever uses the locations of real on-manifold tokens, never an arithmetic
centroid (which may fall off the manifold — the failure mode of k-means).

## Density-driven construction (§3.2.2, Algorithm 1)

Bottom-up agglomeration: start with `N` singletons; repeatedly merge the closest
pair under `D` (Eq. 4) until `k` clusters remain. The inner loop maintains `D`
for the merged cluster with the average-linkage **Lance–Williams** recurrence

```
D[s*, u] ← (|C_s*|·D[s*,u] + |C_t*|·D[t*,u]) / (|C_s*| + |C_t*|)
```

which equals recomputing Eq. 3 on the merged member set, in `O(N)` per step.
→ `masc/clustering.py::build_masc_tree`. The merge order is implicitly
**density-driven**: dense, semantically tight regions are merged first, so the
hierarchy faithfully reflects the non-uniform token density. App. B.2 motivates
average-linkage over single-/complete-linkage (robust to chaining and outliers).

Complexity: `O(N²d + N³)` time, `O(N²)` space (App. B.1); a one-time offline cost
of well under a GPU-hour for `N=16384`.

## Integrating the prior (§3.3)

* **Vocabulary reduction.** Cut the tree at `k` branches → mapping
  `M : {1..N} → {1..k}` (`masc/mapping.py`).
* **Training target (Eq. 5).** `z^b = M(z^q)`; the model predicts the next
  *cluster* with `k`-way cross-entropy (`masc/relabel.py`,
  `masc/integration.py::MASCObjective`). This lowers the normalised prediction
  entropy and the gradient variance (Tables 6, 12), which is what yields the
  training acceleration.
* **Decoding.**
  * *Random sampling (default):* a predicted cluster is decoded to a token by
    uniform sampling among its members (`masc/decode.py`). Parameter-free and,
    thanks to intra-cluster coherence, already high-quality (Table 10).
  * *Hierarchical (extended):* a ~25M refinement net picks the fine token; a
    marginal gain at extra cost (`masc/refine.py` interface; App. C.2).

## Why it works (empirical anchors)

* **Intra-cluster coherence** — the Semantic Replacement Test shows MASC
  clusters are semantically interchangeable (low rFID, high SSIM) while k-means
  clusters are not (App. D.1 / Table 11; `scripts/semantic_replacement_test.py`).
* **Lower-entropy objective** — MASC reduces normalised prediction entropy
  across all scales (Table 12), directly evidencing the simplified task.
* **Convergence enabler** — for large-vocabulary permutation models (RAR) that
  diverge on the flat task, MASC restores stable training (Table 5).
