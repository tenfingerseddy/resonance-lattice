"""Field algebra — minimal operators used by RQL composition ops.

Surface kept tight: `merge`, `intersect`, `diff`, `subtract`, `empty`. No PSD
projection, no temporal derivatives, no symplectic ops, no reaction-diffusion
heads. The v0.11 surface (~271 ops) collapses here; the rest were deleted
because they're either NumPy wrappers or relics of the field-as-content
thesis.

All operators are **elementwise** and shape-agnostic — they work on a single
`(D,)` concept vector or on an `(N, D)` band tensor as long as the operands
share a shape. Operators are linear and intentionally do not L2-renormalise
their output: callers that need a unit vector apply
`_runtime_common.l2_normalize` themselves. This keeps `merge` strictly
associative and commutative.

Invariants (validated by `tests/harness/property.py`):
- `merge(a, merge(b, c)) == merge(merge(a, b), c)`        (associativity)
- `merge(a, empty(...)) == a`                              (identity)
- `merge(a, b) == merge(b, a)`                             (commutativity)
- `intersect(a, b) == intersect(b, a)`                     (commutativity)
- `subtract(a, a) == empty(...)`

Phase 1 deliverable. RQL audit (Phase 6) may add 1-2 more.
"""

from __future__ import annotations

import numpy as np

from ._runtime_common import l2_normalize


def empty(shape: int | tuple[int, ...], dtype=np.float32) -> np.ndarray:
    """Identity element for `merge`. Zero vector / zero band."""
    return np.zeros(shape, dtype=dtype)


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


def merge(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Additive combination — `a + b`.

    Strict-associative and strict-commutative additive union over concept
    vectors. Callers L2-renormalise the output if a unit-vector is needed
    downstream (skipped here so the algebra invariants are exact).
    """
    return a + b


def intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sign-aware bottleneck overlap.

    Same-sign components → smaller magnitude wins, preserving the sign:
    `intersect([0.5], [0.3]) == [0.3]`, `intersect([-0.7], [-0.2]) == [-0.2]`.
    Disagreeing signs → 0 (no shared signal). Plain `np.minimum` would
    inflate negative coordinates (`min(-0.2, -0.7) == -0.7`), which is the
    wrong reading for the L2-normalised mixed-sign cosine vectors this module
    targets. Commutative and associative.
    """
    same_sign = np.sign(a) == np.sign(b)
    magnitude = np.minimum(np.abs(a), np.abs(b))
    out = np.where(same_sign, magnitude * np.sign(a), 0)
    return out.astype(a.dtype, copy=False)


def diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Asymmetric residual — `max(a - b, 0)`.

    "What's in `a` that isn't already in `b`" — components where `a` exceeds
    `b` survive at their excess magnitude; equal-or-lower components zero
    out. Asymmetric on purpose: `diff(a, b) != diff(b, a)` in general.
    """
    return np.maximum(a - b, 0)


def subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed residual — `a - b`.

    The strict additive inverse of `merge`. `subtract(a, a) == empty(...)`,
    `merge(a, subtract(b, a)) == b`. Distinct from `diff`: `subtract`
    preserves sign, so negative components are valid output.
    """
    return a - b


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
    `memory.consolidation` (episodic→semantic promotion at threshold 0.92)
    and `rql.inspect.near_duplicates` (within-corpus dedup at 0.95). Single
    home for the algorithm so the threshold-tuning history doesn't fork.

    Implementation: union-find over the strict upper-triangle threshold
    graph. O(N²) for the cosine pass; the union-find amortises near-O(1)
    per edge so the total cost is dominated by the matmul + the np.where
    over the threshold mask. The full-precision (N, N) cosine matrix
    means the practical ceiling for a single call is ~50K rows (~10 GB
    float32); callers at higher scale must pre-shard. Episodic-tier
    consolidation (≤2000) and most knowledge-model dedup workflows fit
    comfortably.
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
