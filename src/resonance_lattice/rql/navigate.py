"""Navigation + evidence ops.

`neighbors`   — top-K nearest passages to a given passage_idx (excludes self).
`evidence`    — flagship search op: hits + calibrated confidence + citations.
`drift`       — within-KM drift status (passages with stale source).
`corpus_diff` — cross-KM passage-level diff (added / removed / unchanged).

`audit` is reserved for the experimental commit (Phase 6 #5); omitted
here so each commit lands a coherent surface.

Phase 6 deliverable.
"""

from __future__ import annotations

import numpy as np

from ..field import dense
from ..field.dense import max_cosines_against, topk_indices
from ..store.archive import ArchiveContents
from ..store.base import Store
from ..store.verified import verify_hits
from .compare import _require_base_pair, _require_compatible_pair
from .types import (
    Citation,
    CitationHit,
    ConfidenceMetrics,
    CorpusDiff,
    DriftRecord,
    DriftReport,
    EvidenceReport,
    NeighborHit,
)


def neighbors(
    contents: ArchiveContents,
    passage_idx: int,
    *,
    top_k: int = 10,
    prefer: str | None = None,
) -> list[NeighborHit]:
    """Top-K nearest neighbours of `passage_idx` by cosine, excludes self.

    Within-corpus op — defaults to `prefer=None` so optimised-when-present
    is used (in-corpus retrieval gets the trained-on-this-corpus lift).
    Cross-knowledge-model navigation is not supported here; that's
    `compose` + `search` territory.

    Returns `top_k` hits sorted by descending cosine. If the corpus has
    fewer than `top_k + 1` passages, returns however many are available
    (after self-exclusion).

    Raises `IndexError` if `passage_idx` is out of range.
    """
    handle = contents.select_band(prefer)
    band = handle.band
    n = band.shape[0]
    if not (0 <= passage_idx < n):
        raise IndexError(
            f"passage_idx {passage_idx} out of range for {n}-row band"
        )

    query_vec = band[passage_idx]
    sims = band @ query_vec
    sims[passage_idx] = -1.0

    k = min(top_k, n - 1)
    if k <= 0:
        return []
    ordered = topk_indices(sims, k)
    return [
        NeighborHit(
            citation=Citation.from_coord(contents.registry[int(idx)]),
            cosine=float(sims[idx]),
        )
        for idx in ordered
    ]


def evidence(
    contents: ArchiveContents,
    store: Store,
    query_embedding: np.ndarray,
    *,
    top_k: int = 10,
    prefer: str | None = None,
) -> EvidenceReport:
    """Search + per-result citation + corpus-level calibrated confidence.

    The flagship RQL op. Runs cosine retrieval (`field.dense.search`) on
    the selected band, materialises text via the store, and computes:

    - `top1_top2_gap`: separation of best from second-best score
    - `source_diversity`: unique source files in top-K / K
    - `drift_fraction`: fraction of top-K hits with non-verified drift
    - `band_used`: which band actually served retrieval

    Drifted hits are NOT filtered — they're surfaced via `drift_fraction`
    so the caller can decide whether to trust the evidence base. Pair
    with `--verified-only` post-filter (or `filter_verified()`) when
    drift must be excluded.

    `query_embedding` is expected L2-normalised in the same dim as the
    band (for the optimised band: 768d; the projection matrix is applied
    inside `dense.search`).

    `prefer=None` picks optimised when present (in-corpus retrieval
    benefit); pass `prefer="base"` to force base — useful when comparing
    evidence against another knowledge model's base-band output.
    """
    handle = contents.select_band(prefer)
    raw_hits = dense.search(
        query_embedding,
        handle.band,
        contents.registry,
        handle.projection,
        top_k=top_k,
    )
    if not raw_hits:
        return EvidenceReport(
            hits=[],
            confidence=ConfidenceMetrics.from_verified([], handle.name),
        )
    verified = verify_hits(raw_hits, store, contents.registry)

    citation_hits = [
        CitationHit(
            citation=Citation.from_coord(contents.registry[v.passage_idx]),
            score=v.score,
            text=v.text,
        )
        for v in verified
    ]
    return EvidenceReport(
        hits=citation_hits,
        confidence=ConfidenceMetrics.from_verified(verified, handle.name),
    )


def drift(contents: ArchiveContents, store: Store) -> DriftReport:
    """Within-KM drift report: passages whose source has drifted or gone missing.

    Walks every registry row, calls `Store.verify` once each. The Store
    base class caches per-source-file text, so `verify` over N passages
    collapses to N-distinct-files reads regardless of registry size.

    Returns `(drifted, missing, verified_count)` — drifted means source
    exists but content_hash mismatches; missing means the source file is
    gone. Verified passages are not enumerated (they're the common case);
    only their count is returned.
    """
    drifted_records: list[DriftRecord] = []
    missing_records: list[DriftRecord] = []
    verified_count = 0
    for coord in contents.registry:
        status = store.verify(
            coord.source_file,
            coord.char_offset,
            coord.char_length,
            coord.content_hash,
        )
        if status == "verified":
            verified_count += 1
            continue
        record = DriftRecord(
            citation=Citation.from_coord(coord),
            drift_status=status,
        )
        if status == "drifted":
            drifted_records.append(record)
        else:
            missing_records.append(record)
    return DriftReport(
        drifted=drifted_records,
        missing=missing_records,
        verified_count=verified_count,
    )


def corpus_diff(
    contents_old: ArchiveContents,
    contents_new: ArchiveContents,
    *,
    threshold: float = 0.95,
) -> CorpusDiff:
    """Cross-KM passage-level diff between two corpus snapshots.

    A passage is "removed" if it's in old with NO near-match (cosine
    ≥ `threshold`) in new, "added" if it's in new with no near-match in
    old, "unchanged" if it has a near-match in the other side. The
    threshold is intentionally tight (0.95) — Material content edits
    typically drop cosine well below this.

    Cross-KM op: uses the base band per cross-model rule, raises on dim
    OR backbone-revision mismatch via `_require_compatible_pair`. The
    cosine threshold is meaningless across distinct embedding distributions
    so the strict guard is mandatory — passages from different revisions
    would silently classify as added/removed regardless of content.

    For "what's modified between snapshots" semantics, run `corpus_diff`
    + `intersect(old, new, threshold=0.95)` — the intersect pairs are
    the unchanged set, by definition.
    """
    band_old, band_new = _require_compatible_pair(contents_old, contents_new)
    n_old = band_old.shape[0]
    n_new = band_new.shape[0]
    if n_old == 0 and n_new == 0:
        return CorpusDiff(added=[], removed=[], unchanged_count=0)
    if n_old == 0:
        added = [Citation.from_coord(c) for c in contents_new.registry]
        return CorpusDiff(added=added, removed=[], unchanged_count=0)
    if n_new == 0:
        removed = [Citation.from_coord(c) for c in contents_old.registry]
        return CorpusDiff(added=[], removed=removed, unchanged_count=0)

    max_to_new = max_cosines_against(band_old, band_new)
    removed_idxs = np.flatnonzero(max_to_new < threshold)
    removed = [
        Citation.from_coord(contents_old.registry[int(i)]) for i in removed_idxs
    ]
    max_to_old = max_cosines_against(band_new, band_old)
    added_idxs = np.flatnonzero(max_to_old < threshold)
    added = [
        Citation.from_coord(contents_new.registry[int(i)]) for i in added_idxs
    ]
    unchanged_count = int(n_old - len(removed))
    return CorpusDiff(added=added, removed=removed, unchanged_count=unchanged_count)
