"""Recall — §0.6 two-stage cosine retrieval over the per-user band.

Pipeline (in order, all gates active by default):

    1. Filter: drop is_bad rows; drop rows below `cosine_floor`.
    2. Workspace gate: keep rows whose polarity contains the caller's
       `cwd_hash` workspace tag OR `cross-workspace`.
    3. Confidence gate: keep only if top1 ≥ floor AND
       (top1 - top2) ≥ gap. Empty result if either fails.
    4. Recurrence gate: keep rows with recurrence_count ≥ M.
    5. Sort by cosine descending; return top_k.

Spec: `.claude/plans/fabric-agent-flat-memory.md` §0.6 + §0.4.

Two surfaces:

- `rank(query, *, rows, band, encoder, ...)` — pure algorithm; no I/O.
  The future MVP daemon caches `(rows, band)` and calls `rank()` per
  request, so the §12.4 30/80ms gate measures only the cosine + gate
  + sort cost.
- `recall(query, *, store, encoder=None, ...)` — full pipeline. Calls
  `Memory.read_all` then `rank`; what the CLI one-shot path uses.

The CLI `rlat memory recall`, the future daemon socket, and the future
UserPromptSubmit hook all delegate to one of these — they never
bypass the gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..field._runtime_common import l2_normalize
from ..field.encoder import Encoder
from ._common import workspace_hash
from .store import IntentKind, Memory, Row

# Defaults from §0.4 / §0.6. Callers can override per-invocation; the
# §0.6 contract says all gates run by default.
DEFAULT_COSINE_FLOOR = 0.7
DEFAULT_TOP1_TOP2_GAP = 0.05
DEFAULT_MIN_RECURRENCE = 3
DEFAULT_TOP_K = 5

# Cold-start auto-relaxation. When the row count is below the threshold,
# the gates relax so sparse-memory workloads can surface anything at all
# — the longitudinal benchmark (run #2) showed the v1 defaults are a
# blackout on the first ~30 sessions of a fresh workspace because no
# cluster ever forms above (0.85 cosine, recurrence ≥5) and recall's
# (0.7, ≥0.05, ≥3) cuts every event row. Diagnostic during the run #3
# bench prep showed top hits also cluster tightly (top1=0.548, top2=0.533
# → gap=0.015) so the top1_top2_gap also needs relaxing or recall still
# returns nothing. Callers opt in via the daemon's
# RecallRequest.auto_tune_cold_start flag.
COLD_START_ROW_THRESHOLD = 200
COLD_START_COSINE_FLOOR = 0.5
COLD_START_TOP1_TOP2_GAP = 0.0
COLD_START_MIN_RECURRENCE = 1


def cold_start_gates(n_rows: int) -> tuple[float, float, int] | None:
    """Return `(cosine_floor, top1_top2_gap, min_recurrence)` relaxed
    gates when memory is sparse, else None. Callers compose with
    default values:

        relaxed = cold_start_gates(len(rows))
        if relaxed is not None:
            cosine_floor, top1_top2_gap, min_recurrence = relaxed
    """
    if n_rows < COLD_START_ROW_THRESHOLD:
        return (
            COLD_START_COSINE_FLOOR,
            COLD_START_TOP1_TOP2_GAP,
            COLD_START_MIN_RECURRENCE,
        )
    return None


@dataclass(frozen=True)
class RecallHit:
    """One row above all four §0.6 gates, with its query cosine attached.

    `cosine` is the raw value before any filter; callers that need to
    surface confidence (e.g., the `--explain` CLI flag in MVP) read it
    directly. Sort order is descending by `cosine`.
    """

    row: Row
    cosine: float


# Reasons a recall attempt returned empty. The hook persists this string
# verbatim to `recall_diagnostic.jsonl` so future bench runs can attribute
# misses precisely — "16/20 sessions had no recall" turns into per-call
# `dropped_at` counts instead of a guess.
DROPPED_NO_ROWS = "no_rows"
DROPPED_BELOW_COSINE_FLOOR = "below_cosine_floor"
DROPPED_WRONG_WORKSPACE = "wrong_workspace"
DROPPED_BELOW_CONFIDENCE_GAP = "below_confidence_gap"
DROPPED_BELOW_RECURRENCE = "below_recurrence"
DROPPED_OK = "ok"


@dataclass(frozen=True)
class RankDiagnostic:
    """Per-recall metrics + dropped-at reason for the query-shape diagnostic.

    Computed in lockstep with `rank()`'s pipeline. Daemon ships it back
    in `RecallReply.diagnostic`; the hook writes one entry per call to
    `recall_diagnostic.jsonl` so future bench misses are attributable
    instead of mysterious.

    `top1_raw_cosine` is the highest cosine in the snapshot before any
    gate — useful for "did the query find anything semantically close at
    all?". `top1_cosine` / `top2_cosine` are post-workspace + post-floor
    values the confidence gate actually saw; -inf when fewer than that
    many rows survived to that stage.
    """

    n_rows: int
    top1_raw_cosine: float
    n_above_cosine_floor: int
    n_after_workspace: int
    top1_cosine: float
    top2_cosine: float
    n_after_recurrence: int
    n_hits: int
    dropped_at: str
    effective_cosine_floor: float
    effective_top1_top2_gap: float
    effective_min_recurrence: int


def _row_matches_cwd(row: Row, cwd_hash: str) -> bool:
    """§0.6 step 2 — row in scope iff it has the cwd's workspace tag or
    the cross-workspace bypass.
    """
    target = f"workspace:{cwd_hash}"
    for tag in row.polarity:
        if tag == target or tag == "cross-workspace":
            return True
    return False


def _encode_query(query: str, encoder: Encoder) -> np.ndarray:
    """Encode the query under the same `text + " | intent: " + intent`
    convention rows use, with intent="" since queries don't carry one.
    The L2-normalised result lets us compute cosine as a dot product.
    """
    embedding = encoder.encode([query])[0]
    l2_normalize(embedding)
    return embedding


def rank_with_diagnostic(
    query: str,
    *,
    rows: list[Row],
    band: np.ndarray,
    encoder: Encoder,
    cwd_hash: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    cosine_floor: float = DEFAULT_COSINE_FLOOR,
    top1_top2_gap: float = DEFAULT_TOP1_TOP2_GAP,
    min_recurrence: int = DEFAULT_MIN_RECURRENCE,
    intent_kind: IntentKind | None = None,
) -> tuple[list[RecallHit], RankDiagnostic]:
    """Run the §0.6 retrieval pipeline AND emit per-gate diagnostic counts.

    Same algorithm as `rank()`; additionally returns a `RankDiagnostic`
    naming the gate at which the result emptied (or "ok" when hits
    survive). The daemon ships this in `RecallReply.diagnostic` so the
    UserPromptSubmit hook can log "why no hit" per call.
    """
    neg_inf = float("-inf")
    eff_floor = float(cosine_floor)
    eff_gap = float(top1_top2_gap)
    eff_recurrence = int(min_recurrence)

    def _diag(
        *,
        top1_raw: float = neg_inf,
        n_above_floor: int = 0,
        n_after_workspace: int = 0,
        top1: float = neg_inf,
        top2: float = neg_inf,
        n_after_recurrence: int = 0,
        n_hits: int = 0,
        dropped_at: str,
    ) -> RankDiagnostic:
        return RankDiagnostic(
            n_rows=len(rows),
            top1_raw_cosine=top1_raw,
            n_above_cosine_floor=n_above_floor,
            n_after_workspace=n_after_workspace,
            top1_cosine=top1,
            top2_cosine=top2,
            n_after_recurrence=n_after_recurrence,
            n_hits=n_hits,
            dropped_at=dropped_at,
            effective_cosine_floor=eff_floor,
            effective_top1_top2_gap=eff_gap,
            effective_min_recurrence=eff_recurrence,
        )

    if not rows:
        return [], _diag(dropped_at=DROPPED_NO_ROWS)
    if cwd_hash is None:
        cwd_hash = workspace_hash(str(Path.cwd()))

    query_emb = _encode_query(query, encoder)

    # Step 1: dot-product cosines (band is L2-normalised at write time).
    cosines = band @ query_emb
    top1_raw = float(cosines.max()) if len(cosines) else neg_inf

    # Walk once, accumulating per-stage counts so the diagnostic can
    # report exactly which gate emptied the result. Single pass keeps
    # the hot-path cost equal to the original rank().
    n_above_floor = 0
    eligible: list[tuple[Row, float]] = []
    for row, cos in zip(rows, cosines):
        if cos < eff_floor:
            continue
        if row.is_bad:
            continue
        n_above_floor += 1
        if not _row_matches_cwd(row, cwd_hash):
            continue
        eligible.append((row, float(cos)))

    if n_above_floor == 0:
        return [], _diag(top1_raw=top1_raw, dropped_at=DROPPED_BELOW_COSINE_FLOOR)
    if not eligible:
        return [], _diag(
            top1_raw=top1_raw, n_above_floor=n_above_floor,
            dropped_at=DROPPED_WRONG_WORKSPACE,
        )

    eligible.sort(key=lambda pair: pair[1], reverse=True)
    top1_cos = eligible[0][1]
    top2_cos = eligible[1][1] if len(eligible) >= 2 else neg_inf

    # Step 3: confidence gate. Single-row passes the gap check (no
    # second contender to make the result ambiguous).
    if top1_cos < eff_floor:
        return [], _diag(
            top1_raw=top1_raw, n_above_floor=n_above_floor,
            n_after_workspace=len(eligible), top1=top1_cos, top2=top2_cos,
            dropped_at=DROPPED_BELOW_COSINE_FLOOR,
        )
    if len(eligible) >= 2 and (top1_cos - top2_cos) < eff_gap:
        return [], _diag(
            top1_raw=top1_raw, n_above_floor=n_above_floor,
            n_after_workspace=len(eligible), top1=top1_cos, top2=top2_cos,
            dropped_at=DROPPED_BELOW_CONFIDENCE_GAP,
        )

    above_recurrence = [
        (row, cos) for row, cos in eligible if row.recurrence_count >= eff_recurrence
    ]
    if not above_recurrence:
        return [], _diag(
            top1_raw=top1_raw, n_above_floor=n_above_floor,
            n_after_workspace=len(eligible), top1=top1_cos, top2=top2_cos,
            dropped_at=DROPPED_BELOW_RECURRENCE,
        )

    hits = [RecallHit(row=row, cosine=cos) for row, cos in above_recurrence]
    if intent_kind is not None and intent_kind != "none":
        from .rerank import rerank as _rerank
        hits = _rerank(hits, intent_kind=intent_kind)
    hits = hits[:top_k]
    return hits, _diag(
        top1_raw=top1_raw, n_above_floor=n_above_floor,
        n_after_workspace=len(eligible), top1=top1_cos, top2=top2_cos,
        n_after_recurrence=len(above_recurrence), n_hits=len(hits),
        dropped_at=DROPPED_OK,
    )


def rank(
    query: str,
    *,
    rows: list[Row],
    band: np.ndarray,
    encoder: Encoder,
    cwd_hash: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    cosine_floor: float = DEFAULT_COSINE_FLOOR,
    top1_top2_gap: float = DEFAULT_TOP1_TOP2_GAP,
    min_recurrence: int = DEFAULT_MIN_RECURRENCE,
    intent_kind: IntentKind | None = None,
) -> list[RecallHit]:
    """Run the §0.6 retrieval pipeline against an already-loaded snapshot.

    Pure algorithm, no I/O. Thin wrapper over `rank_with_diagnostic`
    that discards the diagnostic — preserves the v2.1 wire shape every
    existing harness suite pins.

    `cwd_hash` defaults to the caller's `Path.cwd()` so the surface is
    usable without explicit cwd plumbing. Pass an explicit value when
    serving a different working directory.

    `intent_kind` opts the post-gate hits into the manifesto re-rank
    (`memory.rerank`). When None (default), surviving hits come back in
    cosine-descending order. When set, the four binary gates still
    decide eligibility, then the strength × valence × level × confidence
    re-rank decides order.
    """
    hits, _ = rank_with_diagnostic(
        query, rows=rows, band=band, encoder=encoder, cwd_hash=cwd_hash,
        top_k=top_k, cosine_floor=cosine_floor, top1_top2_gap=top1_top2_gap,
        min_recurrence=min_recurrence, intent_kind=intent_kind,
    )
    return hits


def recall(
    query: str,
    *,
    store: Memory,
    cwd_hash: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    encoder: Encoder | None = None,
    cosine_floor: float = DEFAULT_COSINE_FLOOR,
    top1_top2_gap: float = DEFAULT_TOP1_TOP2_GAP,
    min_recurrence: int = DEFAULT_MIN_RECURRENCE,
    intent_kind: IntentKind | None = None,
) -> list[RecallHit]:
    """Run the §0.6 retrieval pipeline against the per-user band.

    Convenience wrapper that loads the snapshot from `store` then runs
    `rank()`. One-shot CLI / fallback path; the daemon shape calls
    `store.read_all()` once and invokes `rank()` per request.

    `intent_kind` is forwarded to `rank` — opt-in manifesto re-rank.
    """
    rows, band = store.read_all()
    if encoder is None:
        encoder = store._ensure_encoder()  # type: ignore[attr-defined]
    return rank(
        query,
        rows=rows,
        band=band,
        encoder=encoder,
        cwd_hash=cwd_hash,
        top_k=top_k,
        cosine_floor=cosine_floor,
        top1_top2_gap=top1_top2_gap,
        min_recurrence=min_recurrence,
        intent_kind=intent_kind,
    )
