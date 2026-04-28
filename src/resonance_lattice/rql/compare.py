"""Cross-knowledge-model comparison ops.

`compare`   — overall similarity metrics (centroid + asymmetric coverage).
`unique`    — passages in A with no near-match in B.
`intersect` — passage pairs from A and B with cosine ≥ threshold.

ALL ops here use the base band (cross-model rule per CLAUDE.md). Optimised
bands are corpus-specific projections; their dimensions and orientations
are not interoperable across knowledge models. `select_band(prefer="base")`
enforces this — every op raises a clear error if either KM is missing the
base band, rather than silently producing nonsense numbers.

Phase 6 deliverable.
"""

from __future__ import annotations

import numpy as np

from ..field.algebra import centroid
from ..field.dense import (
    COSINE_CHUNK_BYTES,
    max_cosines_against,
    sampled_mean_max_cosine,
)
from ..store.archive import ArchiveContents
from .types import Citation, CompareResult, PassagePair


def _require_base_pair(
    contents_a: ArchiveContents, contents_b: ArchiveContents,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve both knowledge models' base bands or raise with a clear message.

    Cross-model ops have one entry condition: both KMs must carry a base
    band of matching dim. Optimised bands are explicitly excluded — they
    aren't comparable across KMs.

    This is the *dim-only* guard — cosine ordering is still meaningful
    across distinct backbone revisions even though magnitudes differ.
    Use `_require_compatible_pair` for ops that need exact-revision match
    (corpus_diff, merge — where a mismatched cosine threshold silently
    invalidates the result classification).
    """
    handle_a = contents_a.select_band(prefer="base")
    handle_b = contents_b.select_band(prefer="base")
    if handle_a.band.shape[1] != handle_b.band.shape[1]:
        raise ValueError(
            f"base-band dim mismatch ({handle_a.band.shape[1]} vs "
            f"{handle_b.band.shape[1]}) — incompatible knowledge models"
        )
    return handle_a.band, handle_b.band


def _require_compatible_pair(
    contents_a: ArchiveContents, contents_b: ArchiveContents,
) -> tuple[np.ndarray, np.ndarray]:
    """Strict variant of `_require_base_pair`: dim AND backbone revision must
    match. Raises on either mismatch.

    Used by ops where threshold-based classification is meaningless if the
    two corpora's embeddings live in different distributions (corpus_diff,
    merge). For thematic-comparison ops where ordering still holds across
    revisions (compare, intersect, unique), use `_require_base_pair` and
    let the caller surface the revision delta as a UserWarning or report
    field instead.
    """
    band_a, band_b = _require_base_pair(contents_a, contents_b)
    rev_a = contents_a.metadata.backbone.revision
    rev_b = contents_b.metadata.backbone.revision
    if rev_a != rev_b:
        raise ValueError(
            f"backbone revisions differ ({rev_a!r} vs {rev_b!r}); "
            f"this op requires identical revisions because mixing distinct "
            f"embedding distributions makes the cosine threshold meaningless"
        )
    return band_a, band_b


def compare(
    contents_a: ArchiveContents,
    contents_b: ArchiveContents,
    *,
    sample_size: int = 512,
    seed: int = 0,
) -> CompareResult:
    """Cross-KM thematic alignment + asymmetric coverage on the base band.

    `centroid_cosine` is the cosine between the two L2-normalised mean
    vectors — a single thematic-alignment number. Coverage is the mean
    max-cosine of a `sample_size` sample from one corpus against the
    other (asymmetric).

    `overlap_score` is the symmetric average of the two coverages. Use
    `compare(...).overlap_score` instead of computing a separate `overlap`
    op — `overlap` was folded into this dataclass during Phase 6 design.

    Sampling is deterministic via `seed=` — identical inputs produce
    identical numbers across runs.
    """
    band_a, band_b = _require_base_pair(contents_a, contents_b)
    n_a, n_b = band_a.shape[0], band_b.shape[0]

    centroid_cosine = float(centroid(band_a) @ centroid(band_b))

    # Spawn two independent seed streams from `seed` so the B-into-A sample
    # is genuinely independent of the A-into-B sample. SeedSequence.spawn
    # is the idiomatic numpy way to derive sub-seeds — `seed + 1` would
    # work but is less honest about the sub-stream contract.
    seq_a, seq_b = np.random.SeedSequence(seed).spawn(2)
    cov_a_b = sampled_mean_max_cosine(
        band_a, band_b, sample_size=sample_size,
        seed=int(seq_a.generate_state(1)[0]),
    )
    cov_b_a = sampled_mean_max_cosine(
        band_b, band_a, sample_size=sample_size,
        seed=int(seq_b.generate_state(1)[0]),
    )

    rev_a = contents_a.metadata.backbone.revision
    rev_b = contents_b.metadata.backbone.revision
    return CompareResult(
        a_passage_count=n_a,
        b_passage_count=n_b,
        a_backbone_revision=rev_a,
        b_backbone_revision=rev_b,
        revision_match=rev_a == rev_b,
        centroid_cosine=centroid_cosine,
        coverage_a_in_b=cov_a_b,
        coverage_b_in_a=cov_b_a,
        overlap_score=(cov_a_b + cov_b_a) / 2.0,
        sample_size=min(sample_size, n_a, n_b),
    )


def unique(
    contents_a: ArchiveContents,
    contents_b: ArchiveContents,
    *,
    threshold: float = 0.7,
) -> list[Citation]:
    """Citations of passages in A whose top-1 match in B has cosine < threshold.

    Returns A's "distinctive content" — what A has that B doesn't cover.
    O(N_A × N_B) compute via `max_cosines_against` (chunked memory). Always
    base band per cross-model rule.

    `threshold=0.7` is a reasonable "approximate semantic match" floor
    for gte-mb-base — paraphrases typically clear 0.7, lexically-different
    same-topic passages typically don't. Tighten to 0.85+ for
    "near-paraphrase only."
    """
    band_a, band_b = _require_base_pair(contents_a, contents_b)
    if band_a.shape[0] == 0:
        return []
    if band_b.shape[0] == 0:
        # B is empty — every A passage is "unique" by definition.
        return [Citation.from_coord(c) for c in contents_a.registry]
    max_cos = max_cosines_against(band_a, band_b)
    keep = np.flatnonzero(max_cos < threshold)
    return [Citation.from_coord(contents_a.registry[int(i)]) for i in keep]


def intersect(
    contents_a: ArchiveContents,
    contents_b: ArchiveContents,
    *,
    threshold: float = 0.7,
    max_pairs: int = 10_000,
) -> list[PassagePair]:
    """Passage pairs `(A, B)` with base-band cosine ≥ `threshold`.

    Returns up to `max_pairs` pairs sorted by descending cosine. The
    `max_pairs` cap exists because for two corpora of N=10K passages each
    at threshold=0.7, the result set can be millions of pairs — caller
    almost always wants the strongest matches, not the full O(N²) cross
    product.

    `threshold=0.7` matches `unique`'s default. `max_pairs=10_000` is a
    practical UI limit; raise it for batch / scripting workloads.

    Memory: chunks `band_b` along axis 0 so `max_pairs` are accumulated
    incrementally rather than materialising the full `(N_A, N_B)` matrix.
    """
    band_a, band_b = _require_base_pair(contents_a, contents_b)
    n_a, n_b = band_a.shape[0], band_b.shape[0]
    if n_a == 0 or n_b == 0:
        return []

    # We need the (i, j, sim) triples above threshold, not just the per-row
    # max — so we can't reuse `max_cosines_against`. We walk band_b in
    # chunks to keep peak memory bounded by ~`COSINE_CHUNK_BYTES` worth of
    # the (N_a, chunk_b) sims tile.
    bytes_per_target_row = n_a * 4
    chunk_rows = max(1, COSINE_CHUNK_BYTES // bytes_per_target_row)

    triples: list[tuple[float, int, int]] = []
    for start in range(0, n_b, chunk_rows):
        chunk = band_b[start:start + chunk_rows]
        sims = band_a @ chunk.T
        # np.where on (N_a, chunk_size) → row/col indices in this tile.
        rows, cols = np.where(sims >= threshold)
        for r, c in zip(rows, cols):
            triples.append((float(sims[r, c]), int(r), int(start + c)))
        # Truncate the running list once it's overflowing by an order of
        # magnitude. Two thresholds: 10× the cap triggers truncation
        # (absorbs many later-chunk reorderings before paying the sort
        # cost); 2× survivors keeps a safety margin so the next chunk's
        # contributions can re-order the cut without losing any of the
        # final top-k.
        if len(triples) > 10 * max_pairs:
            triples.sort(key=lambda t: -t[0])
            triples = triples[: max_pairs * 2]

    triples.sort(key=lambda t: -t[0])
    triples = triples[:max_pairs]

    return [
        PassagePair(
            citation_a=Citation.from_coord(contents_a.registry[i]),
            citation_b=Citation.from_coord(contents_b.registry[j]),
            cosine=cos,
        )
        for cos, i, j in triples
    ]
