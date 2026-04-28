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

Phase 2 deliverable. Ports the v0.11 surface verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import DriftStatus, RemoteShaMismatch, Store, compute_hash
from .registry import PassageCoord

__all__ = [
    "DriftStatus",
    "VerifiedHit",
    "compute_hash",
    "filter_verified",
    "verify_hits",
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


def filter_verified(hits: list[VerifiedHit]) -> list[VerifiedHit]:
    """Drop drifted/missing hits."""
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
