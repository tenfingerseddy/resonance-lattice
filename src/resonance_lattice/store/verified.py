"""Verified-retrieval surface (WS3 #292).

Every search hit carries: source_file, char_offset, char_length, content_hash,
drift_status. Users can pass --verified-only to filter out drifted chunks.

`DriftStatus` and `compute_hash` are owned by `store.base` (single source of
truth — `Store.verify` consumes both); re-exported here so callers can keep
importing from `store.verified` without knowing the internal split.

Drift status:
- "verified" — content_hash matches authoritative source.
- "drifted"  — source exists but hash mismatches.
- "missing"  — source file no longer exists.

InsightHit is the sibling type for the lensed-knowledge insight layer. The
two stay structurally distinct on purpose (trust-contract foundation 5:
source and insight must be visibly different at every output surface).
Code that consumes hits dispatches on type / `layer` field.

Phase 2 deliverable. Ports the v0.11 surface verbatim. Day 1 of the
lensed-knowledge build adds InsightHit + composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .base import DriftStatus, RemoteShaMismatch, Store, compute_hash
from .insight import InsightCitation, InsightKind, InsightPassage, VerdictState
from .registry import PassageCoord

__all__ = [
    "DriftStatus",
    "InsightHit",
    "VerifiedHit",
    "compute_hash",
    "filter_verified",
    "verify_hits",
    "verify_insight_hits",
]


@dataclass(frozen=True)
class VerifiedHit:
    passage_idx: int
    source_file: str
    char_offset: int
    char_length: int
    content_hash: str
    drift_status: DriftStatus
    score: float
    text: str
    # Layer discriminator — kept on both hit types so a mixed list of
    # SourceHit-shaped VerifiedHits and InsightHits can be filtered by
    # `[h for h in hits if h.layer == "source"]` without isinstance checks.
    layer: Literal["source"] = "source"


# Insight verdict_state → display-layer drift_status mapping. Source-only
# rendering paths can treat InsightHit.drift_status uniformly with VerifiedHit's
# even though the underlying semantics differ.
_INSIGHT_STATE_TO_DRIFT: dict[VerdictState, DriftStatus] = {
    "accepted": "verified",
    "stale": "drifted",
    "candidate": "drifted",
    "rejected": "missing",
    "rejected_corrected": "missing",
    "retired": "missing",
}


@dataclass(frozen=True)
class InsightHit:
    """One insight-layer retrieval hit. Sibling of VerifiedHit.

    `insight_idx` joins the insight band to the row. `insight_id` is the
    stable content-derived id (see `store.insight.compute_insight_id`) and
    serves the same role as `content_hash` does for source hits.

    `verdict_state` is the canonical state; `drift_status` is derived
    from it via `_INSIGHT_STATE_TO_DRIFT` so callers can apply
    `filter_verified` uniformly across source and insight hits.
    """
    insight_idx: int
    insight_id: str
    kind: InsightKind
    content: str
    citations: tuple[InsightCitation, ...]
    source_passage_hashes: tuple[str, ...]
    verdict_state: VerdictState
    confidence: float
    generated_at: str
    intent_context: str | None
    score: float
    layer: Literal["insight"] = "insight"

    @property
    def drift_status(self) -> DriftStatus:
        return _INSIGHT_STATE_TO_DRIFT[self.verdict_state]


def filter_verified(hits: "list[VerifiedHit] | list[InsightHit] | list") -> list:
    """Drop drifted/missing hits across either source or insight layers.

    Mixed lists are supported — VerifiedHit and InsightHit both carry
    `drift_status` (InsightHit's is derived from verdict_state) so the
    filter is uniform.
    """
    return [h for h in hits if h.drift_status == "verified"]


def verify_hits(
    hits: list[tuple[int, float]],
    store: Store,
    registry: list[PassageCoord],
) -> list[VerifiedHit]:
    """Resolve `(passage_idx, score)` hits into `VerifiedHit`s with drift status.

    Looks up each hit's coordinate via `registry[passage_idx]`, calls
    `store.verify` to detect drift against the build-time content_hash, and
    `store.fetch` to retrieve the current authoritative text — skipped on
    `"missing"` and `"drifted"` status (text becomes empty), since fetch
    would raise `FileNotFoundError` for a missing source and
    `RemoteShaMismatch` for a remote-mode SHA-pin mismatch. The base
    class's per-instance text cache makes verify+fetch amortise to a single
    full-file read per source file regardless of hit count.

    `RemoteShaMismatch` from `store.fetch` is *also* caught explicitly
    (and demoted to drifted+empty) for the case where `verify` returned
    "verified" against the recorded `content_hash` but the SHA-pin guard
    fired during the actual byte fetch — that combination is rare but
    representable, e.g. a remote-mode entry whose recorded passage hash
    matches the cached bytes but whose cached bytes' file-level SHA no
    longer matches the manifest pin (cache corruption). Without this
    catch, `rlat search`, `rlat skill-context`, and `rql.evidence` would
    crash on remote drift instead of surfacing it as a drift status.

    Output is in input order; sorting and `--verified-only` filtering are
    the caller's job (see `filter_verified`).

    Raises `IndexError` if any `passage_idx` is out of range — that means
    the caller passed hits from a different knowledge model's registry,
    which is a programming error, not a runtime drift.
    """
    out: list[VerifiedHit] = []
    for passage_idx, score in hits:
        coord = registry[passage_idx]
        drift_status = store.verify(
            coord.source_file,
            coord.char_offset,
            coord.char_length,
            coord.content_hash,
        )
        if drift_status == "missing" or drift_status == "drifted":
            # Skip fetch on missing (would FileNotFoundError) and on drifted
            # (text doesn't match recorded content_hash; returning the live
            # bytes would be a footgun for citation paths). Empty text keeps
            # the row in output so callers see the drift status rather than
            # silently dropped hits.
            text = ""
        else:
            try:
                text = store.fetch(
                    coord.source_file,
                    coord.char_offset,
                    coord.char_length,
                )
            except RemoteShaMismatch:
                # Remote SHA-pin mismatch fired during fetch (e.g. cache
                # corruption surfaced after `verify` returned verified
                # against the per-passage hash). Demote to drifted+empty
                # rather than crashing the retrieval path.
                drift_status = "drifted"
                text = ""
        out.append(VerifiedHit(
            passage_idx=passage_idx,
            source_file=coord.source_file,
            char_offset=coord.char_offset,
            char_length=coord.char_length,
            content_hash=coord.content_hash,
            drift_status=drift_status,
            score=float(score),
            text=text,
        ))
    return out


def verify_insight_hits(
    hits: list[tuple[int, float]],
    insights: list[InsightPassage],
    *,
    include_stale: bool = False,
) -> list[InsightHit]:
    """Resolve `(insight_idx, score)` hits into `InsightHit`s.

    `verdict_state` drives the derived `drift_status` for uniform rendering:
    accepted → verified, stale/candidate → drifted, rejected* / retired →
    missing. `include_stale=False` (the default) filters stale rows out
    silently — the same way `--verified-only` filters drifted source hits.

    Retired and rejected rows are always excluded regardless of
    `include_stale` because they're not "fresh-but-stale" — they're final
    states with no path back into retrieval.

    Raises `IndexError` if any `insight_idx` is out of range — programming
    error, same as `verify_hits`.
    """
    from .insight import FINAL_STATES, PENDING_STATES

    out: list[InsightHit] = []
    for insight_idx, score in hits:
        row = insights[insight_idx]
        if row.verdict_state in FINAL_STATES:
            continue
        if row.verdict_state in PENDING_STATES and not include_stale:
            continue
        out.append(InsightHit(
            insight_idx=insight_idx,
            insight_id=row.insight_id,
            kind=row.kind,
            content=row.content,
            citations=row.citations,
            source_passage_hashes=row.source_passage_hashes,
            verdict_state=row.verdict_state,
            confidence=row.confidence,
            generated_at=row.generated_at,
            intent_context=row.intent_context,
            score=float(score),
        ))
    return out
