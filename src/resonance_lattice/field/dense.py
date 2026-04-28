"""Dense cosine retrieval — base or optimised band.

Single retrieval strategy applied uniformly. The retrieval pipeline branches
on band presence at knowledge-model load time, not on a flag. Cross-knowledge-
model ops always use the base band (see Phase 3 `cli/compare.py`).

Phase 1 deliverable. Base plan §3.1, §3.3, §3.4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import _runtime_common as common

if TYPE_CHECKING:
    from collections.abc import Sequence


def search(
    query_embedding: np.ndarray,
    band_embeddings: np.ndarray,
    registry: "Sequence | None" = None,
    projection_matrix: np.ndarray | None = None,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Top-k cosine retrieval against a single band.

    query_embedding: (D,) L2-normalised float32 from `Encoder.encode([query])[0]`.
    band_embeddings: (N, d_band) L2-normalised float32. d_band is 768 for base,
        512 for the MRL optimised (after W projection).
    registry: sequence of objects exposing `.source_file: str` and
        `.char_offset: int` for query-time dedup. Passing None skips dedup.
    projection_matrix: (d_band, D) MRL W matrix. Pass when `band_embeddings`
        is the optimised band; pass None for the base band.
    top_k: number of results returned after dedup.

    Returns: list of (passage_idx, score) sorted by score descending,
    deduplicated by (source_file, char_offset) when `registry` is provided.
    Cosine == dot product when both vectors are L2-normalised.
    """
    if top_k <= 0:
        return []

    q = query_embedding
    if projection_matrix is not None:
        q = q @ projection_matrix.T
        common.l2_normalize(q)

    scores = band_embeddings @ q
    n = len(scores)

    if registry is None:
        # No dedup; one partition + sort produces exactly top_k hits.
        budget = min(top_k, n)
        idx_sorted = topk_indices(scores, budget)
        return [(int(i), float(s)) for i, s in zip(idx_sorted, scores[idx_sorted])]

    # With dedup, a fixed candidate_k can underflow on duplicate-heavy
    # registries — keep doubling the budget until we have enough distinct
    # hits or we've scanned the whole band.
    budget = min(top_k * common.CANDIDATE_MULTIPLIER, n)
    while True:
        idx_sorted = topk_indices(scores, budget)
        hits = [(int(i), float(s)) for i, s in zip(idx_sorted, scores[idx_sorted])]
        hits = dedup_by_source(hits, registry)
        if len(hits) >= top_k or budget >= n:
            return hits[:top_k]
        budget = min(budget * 2, n)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k scores in descending order. argpartition
    is O(N); we sort only the partitioned slice. Public so RQL ops
    (`navigate.neighbors` etc.) reuse the same idiom rather than reimplementing
    argpartition + argsort everywhere."""
    n = len(scores)
    if k >= n:
        return np.argsort(-scores)
    partitioned = np.argpartition(-scores, k - 1)[:k]
    return partitioned[np.argsort(-scores[partitioned])]


# Soft cap on the (query × target × 4 byte) similarity matrix produced by
# `max_cosines_against`. 512 MB at the default lets a 512-row query tile
# scan up to ~256K targets before chunking. Beyond that, target rows are
# streamed in chunks so peak RSS stays bounded.
COSINE_CHUNK_BYTES = 512 * 1024 * 1024


def sampled_mean_max_cosine(
    src: np.ndarray, dst: np.ndarray, *, sample_size: int, seed: int = 0,
) -> float:
    """Mean max-cosine of a deterministic `sample_size` sample from `src`
    against all of `dst`. Used by `cli/compare._mutual_coverage` and
    `rql/compare.compare` — single home for "sample-then-mean-max."

    Sampling is deterministic via `seed=` (np.random.default_rng); identical
    inputs produce identical numbers across runs. Empty src or dst → 0.0
    so callers don't need to special-case before calling.
    """
    n_src = src.shape[0]
    n_dst = dst.shape[0]
    if n_src == 0 or n_dst == 0:
        return 0.0
    if n_src <= sample_size:
        sample = src
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_src, sample_size, replace=False)
        sample = src[idx]
    return float(max_cosines_against(sample, dst).mean())


def max_cosines_against(
    query_band: np.ndarray,
    target_band: np.ndarray,
    *,
    chunk_bytes: int = COSINE_CHUNK_BYTES,
) -> np.ndarray:
    """For each row in `query_band`, return its max cosine vs `target_band`.

    Output: `(N_query,)` float32. Both inputs assumed L2-normalised — cosine
    == dot product. Used by `cli/compare._mutual_coverage` (with sampling)
    and `rql/compare.unique` / `rql/compare.intersect_max` (without).

    Memory-bounded: `query @ target.T` would materialise a
    `(N_query, N_target)` matrix; for huge `target` this gets chunked along
    axis 0 with a running per-row max accumulated incrementally. Peak RSS
    stays under `chunk_bytes` regardless of corpus size.

    Empty target → returns a `(N_query,)` array of `-inf` so callers know
    "no neighbour at any threshold" rather than getting a silent zero.
    """
    n_query = query_band.shape[0]
    n_target = target_band.shape[0]
    if n_query == 0:
        return np.zeros(0, dtype=np.float32)
    if n_target == 0:
        return np.full(n_query, -np.inf, dtype=np.float32)
    bytes_per_target_row = n_query * 4
    chunk_rows = max(1, chunk_bytes // bytes_per_target_row)
    if chunk_rows >= n_target:
        sims = query_band @ target_band.T
        return sims.max(axis=1).astype(np.float32, copy=False)
    running_max = np.full(n_query, -np.inf, dtype=np.float32)
    for start in range(0, n_target, chunk_rows):
        chunk = target_band[start:start + chunk_rows]
        chunk_max = (query_band @ chunk.T).max(axis=1)
        np.maximum(running_max, chunk_max, out=running_max)
    return running_max


def dedup_by_source(
    hits: list[tuple[int, float]],
    registry: "Sequence",
) -> list[tuple[int, float]]:
    """Drop hits sharing `(source_file, char_offset)`. First-seen wins.

    Two passages can be near-duplicates when one is a substring of another or
    when overlapping windows produce nearly-identical embeddings; this prunes
    them so the top-k slice carries `top_k` distinct sources.
    """
    seen: set[tuple[str, int]] = set()
    out: list[tuple[int, float]] = []
    for idx, score in hits:
        coord = registry[idx]
        key = (coord.source_file, coord.char_offset)
        if key in seen:
            continue
        seen.add(key)
        out.append((idx, score))
    return out
