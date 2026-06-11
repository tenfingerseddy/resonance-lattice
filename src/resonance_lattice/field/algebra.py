"""Field geometry helpers — the band-level primitives production actually uses.

Surface: `centroid` (cli/compare, cli/summary, rql/compare), `greedy_cluster`
(rql/inspect, rql/compose dedup, benchmarks), `complete_linkage_cluster`
(curator/signals intent clustering).

The v0.11 algebra surface (~271 ops) collapsed here, and the 2026-06 review
removed the last five elementwise ops (`merge`/`intersect`/`diff`/`subtract`/
`empty`) — they had no production caller (RQL's merge/intersect use different
machinery: physical archive merge + `dense.max_cosines_against`). The
sign-aware `intersect` design is preserved in git history if a future RQL
surface wants it.
"""

from __future__ import annotations

import numpy as np

from ._runtime_common import l2_normalize


def centroid(band: np.ndarray) -> np.ndarray:
    """Mean of an `(N, D)` band, L2-renormalised.

    The mean of unit vectors is not unit; renormalising lets callers compare
    two centroids by raw dot product as cosine, and lets a single centroid
    serve as a synthetic query for "what is this corpus about?" retrieval.

    Empty bands (`N == 0`) return a zero vector — caller short-circuits any
    cosine to 0.0 rather than letting NaN leak into JSON output.
    """
    if band.shape[0] == 0:
        return np.zeros(band.shape[1], dtype=np.float32)
    out = np.ascontiguousarray(band.mean(axis=0), dtype=np.float32)
    l2_normalize(out)
    return out


def greedy_cluster(
    embeddings: np.ndarray, threshold: float,
) -> list[list[int]]:
    """Single-linkage clustering by cosine ≥ threshold via connected
    components. O(N²) — the (N, N) cosine matrix dominates.

    Two rows are in the same cluster iff there's a path of pairwise-
    above-threshold edges between them. Transitive chains are honoured:
    if A↔B at 0.96 and B↔C at 0.96 but A↔C at 0.91 (below threshold),
    {A, B, C} still cluster together because B bridges A and C.

    The returned lists hold row indices in ascending order; cluster order
    follows the lowest member id. Singletons are kept — callers that
    want pairs-only filter on `len(cluster) >= 2`.

    Assumes rows are L2-normalised (cosine == dot product). Used by
    `rql.inspect.near_duplicates` (within-corpus dedup at 0.95) and
    `rql.compose` (semantic dedupe in set unions). Single home for the
    algorithm so the threshold-tuning history doesn't fork.

    Implementation: union-find over the strict upper-triangle threshold
    graph. O(N²) for the cosine pass; the union-find amortises near-O(1)
    per edge so the total cost is dominated by the matmul + the np.where
    over the threshold mask. The full-precision (N, N) cosine matrix
    means the practical ceiling for a single call is ~50K rows (~10 GB
    float32); callers at higher scale must pre-shard.
    """
    n = embeddings.shape[0]
    if n == 0:
        return []
    sims = embeddings @ embeddings.T
    # Union-find: each row starts as its own cluster, then we union along
    # every above-threshold edge. Path compression keeps `find` near-O(1).
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Lower index becomes root so cluster ids stay deterministic
            # (matches "lowest member id is the seed" property callers rely on).
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # Strict upper triangle: i < j only — avoids double-processing each pair.
    rows, cols = np.where(np.triu(sims >= threshold, k=1))
    for i, j in zip(rows.tolist(), cols.tolist()):
        union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    # Sort each cluster by index, then sort clusters by their lowest member.
    return [sorted(members) for _, members in sorted(groups.items())]


def complete_linkage_cluster(
    embeddings: np.ndarray, threshold: float,
) -> list[list[int]]:
    """Complete-linkage clustering by cosine ≥ threshold — the chaining-RESISTANT
    counterpart to `greedy_cluster`.

    Single-linkage (`greedy_cluster`) honours transitive chains: A↔B and B↔C merge
    {A, B, C} even when A↔C is below threshold. That is right for near-duplicate
    dedup (0.95) but WRONG for intent clustering (0.70), where same-frame queries on
    DIFFERENT topics chain into one mega-cluster (measured: single @ 0.70 → 1 cluster
    / pairwise-F1 0.13; complete @ 0.70 → the true clusters / F1 1.0 — see
    `benchmarks/bench_intent_clustering.py`). Complete-linkage merges two clusters
    only if EVERY cross pair clears the threshold (max intra-cluster cosine distance
    ≤ 1 − threshold), so a cluster is a genuine clique of mutually-similar items —
    no chaining.

    Same return contract as `greedy_cluster`: row-index lists, members ascending,
    clusters ordered by lowest member; singletons kept. Assumes L2-normalised rows.
    O(N²) distances + O(N² log N) linkage — fine for the buffered query counts intent
    clustering runs on (capped well under `greedy_cluster`'s ceiling)."""
    n = embeddings.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    # Cosine distance from the dot product (rows are L2-normalised by contract), NOT
    # scipy's pdist(metric="cosine") — pdist divides by the norm and a sanitised
    # ZERO row (a junk/non-finite embedding the caller zeroed) yields a non-finite
    # distance that crashes linkage. From the dot, a zero row scores cosine 0 →
    # distance 1.0 (finite), so it simply never merges — the intended "junk is its
    # own singleton" behaviour.
    sims = embeddings @ embeddings.T
    dist = np.clip(1.0 - sims, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method="complete")
    labels = fcluster(z, t=1.0 - threshold, criterion="distance")
    groups: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels.tolist()):
        groups.setdefault(lab, []).append(idx)
    # Members ascending, clusters ordered by lowest member (greedy_cluster's contract).
    return sorted((sorted(members) for members in groups.values()), key=lambda m: m[0])
