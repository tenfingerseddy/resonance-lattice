"""Hard-negative mining for InfoNCE training.

Spec (mrl_fabric_remote_train.py:372-404 — original-positive-results path):
  For each synth query: cosine vs all passages → argpartition(-sims, k=64) →
  drop positive → np.shuffle (seed=0) → take 7. Random sample from a hard pool,
  not deterministic top-7-hardest.

The earlier v2.0 implementation took k=8 and the 7 cosine-ranked nearest non-
positive neighbours. That's "7 hardest-possible" mining, which biases training
against likely false-negatives (high-cosine paraphrases of the positive that
*are* relevant — just not the same passage_idx). InfoNCE then pushes the
projection AWAY from valid relevant content. Spec's k=64 + shuffle gives 7
*sampled* hard negatives — still hard, but spread enough to avoid systemic
false-negative contamination.

Phase 4 deliverable; Phase B spec-compliance fixed 2026-04-26.
"""

from __future__ import annotations

import numpy as np

NEGATIVES_PER_QUERY = 7
# Spec §4d: k=64 candidates → drop positive → shuffle → take 7. Wide hard
# pool reduces false-negative bias vs the previous tight k=8 selection.
DEFAULT_TOP_K = 64
SEED = 0

# Per-chunk Q×N similarity tile cap. ~960 MB matches the v2.0 worked
# example in the module docstring (40K passages × 6K queries × 4B). Tiling
# along the query axis keeps peak RSS bounded for larger advertised
# corpora (e.g. 100K passages × 6K queries → 2.4 GB unchunked) without
# changing per-row results — argpartition runs row-by-row inside.
_CHUNK_BYTES = 960 * 1024 * 1024


def mine(
    query_embeddings: np.ndarray,
    passage_embeddings: np.ndarray,
    query_passage_idx: list[int] | np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    seed: int = SEED,
) -> np.ndarray:
    """Return `(N_queries, NEGATIVES_PER_QUERY)` int32 array of hard negatives.

    Both `query_embeddings` and `passage_embeddings` are L2-normalised
    `float32`; their dot product is cosine similarity. A single batched
    matmul produces all per-query top-k candidates in one pass — much
    faster than per-query `dense.search` calls and indistinguishable in
    output.

    Spec sampling (§4d): top-k (default 64) candidates → drop positive →
    np.shuffle(seed=0) → take first NEGATIVES_PER_QUERY. The shuffle is
    seeded so re-runs of the same `(query_embeddings, query_passage_idx)`
    produce identical negative sets — required for the harness suite's
    bit-exact roundtrip check.

    Edge case: if the positive isn't in the top-k (the query encoder
    produced a sufficiently off-distribution vector), all k candidates
    are eligible and we shuffle + take 7 from those.
    """
    if query_embeddings.shape[0] == 0:
        return np.zeros((0, NEGATIVES_PER_QUERY), dtype=np.int32)
    if top_k < NEGATIVES_PER_QUERY + 1:
        raise ValueError(
            f"top_k={top_k} too small; need ≥{NEGATIVES_PER_QUERY + 1} so a "
            f"positive in the top-k still leaves {NEGATIVES_PER_QUERY} negatives"
        )

    # Chunk along the query axis so peak RSS stays bounded by `_CHUNK_BYTES`
    # regardless of corpus size. The matmul is BLAS-accelerated within each
    # tile and `argpartition` runs row-by-row, so the chunked path is
    # numerically identical to one big matmul (no cross-row reduction).
    n_queries = query_embeddings.shape[0]
    n_passages = passage_embeddings.shape[0]
    bytes_per_row = max(1, n_passages * np.dtype(np.float32).itemsize)
    rows_per_chunk = max(1, _CHUNK_BYTES // bytes_per_row)
    k = min(top_k, n_passages)

    # int32 indices are sufficient for any conceivable v2.0 corpus
    # (n_passages << 2^31). Halves the persistent allocation vs int64.
    partitioned = np.empty((n_queries, k), dtype=np.int32)
    for start in range(0, n_queries, rows_per_chunk):
        end = min(start + rows_per_chunk, n_queries)
        sims_chunk = query_embeddings[start:end] @ passage_embeddings.T
        # Top-k unsorted candidates per row — sufficient since we shuffle next.
        partitioned[start:end] = np.argpartition(
            -sims_chunk, kth=k - 1, axis=1
        )[:, :k].astype(np.int32, copy=False)

    rng = np.random.default_rng(seed)
    out = np.empty((n_queries, NEGATIVES_PER_QUERY), dtype=np.int32)
    pos_array = np.asarray(query_passage_idx)
    for i, pos_idx in enumerate(pos_array):
        candidates = partitioned[i]
        non_positive = candidates[candidates != pos_idx]
        if non_positive.shape[0] == 0:
            raise ValueError(
                f"no non-positive candidates for query {i} "
                f"(passage_idx={int(pos_idx)}, corpus N={n_passages}). "
                f"Mining requires at least 2 distinct passages."
            )
        rng.shuffle(non_positive)
        negatives = non_positive[:NEGATIVES_PER_QUERY]
        if negatives.shape[0] < NEGATIVES_PER_QUERY:
            # Cycle-pad from the available pool — duplicates produce weaker
            # but non-degenerate contrastive signal.
            reps = (NEGATIVES_PER_QUERY // non_positive.shape[0]) + 1
            negatives = np.tile(non_positive, reps)[:NEGATIVES_PER_QUERY]
        out[i] = negatives.astype(np.int32, copy=False)
    return out
