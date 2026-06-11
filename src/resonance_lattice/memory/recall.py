"""Recall — §0.6 two-stage cosine retrieval over the per-user band.

Pipeline (in order, all gates active by default):

    1. Filter: drop is_bad claims; drop claims below `cosine_floor`.
    2. Workspace gate: keep claims whose polarity contains the caller's
       `cwd_hash` workspace tag OR `cross-workspace`.
    3. Confidence gate: keep only if top1 ≥ floor AND
       (top1 - top2) ≥ gap. Empty result if either fails.
    4. Recurrence gate: keep claims with recurrence_count ≥ M
       (the event-noise filter).
    5. Sort by cosine descending; return top_k.

Spec: `.claude/plans/fabric-agent-flat-memory.md` §0.6 + §0.4.

Two surfaces:

- `rank(query, *, claims, band, encoder, ...)` — pure algorithm; no I/O.
  The future MVP daemon caches `(claims, band)` and calls `rank()` per
  request, so the §12.4 30/80ms gate measures only the cosine + gate
  + sort cost.
- `recall(query, *, store, encoder=None, ...)` — full pipeline. Calls
  `ExperienceClaimStore.read_all_with_band` then `rank`; what the CLI
  one-shot path uses.

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
from ..state.claim import Claim
from ._common import workspace_hash
from .claim_store import ExperienceClaimStore
from .store import IntentKind

# Defaults from §0.4 / §0.6. Callers can override per-invocation; the
# §0.6 contract says all gates run by default.
DEFAULT_COSINE_FLOOR = 0.7
DEFAULT_TOP1_TOP2_GAP = 0.05
DEFAULT_MIN_RECURRENCE = 3
DEFAULT_TOP_K = 5

# Cold-start auto-relaxation. When the claim count is below the threshold,
# the gates relax so sparse-memory workloads can surface anything at all
# — the longitudinal benchmark (run #2) showed the v1 defaults are a
# blackout on the first ~30 sessions of a fresh workspace because no
# cluster ever forms above (0.85 cosine, recurrence ≥5) and recall's
# (0.7, ≥0.05, ≥3) cuts every event claim. Diagnostic during the run #3
# bench prep showed top hits also cluster tightly (top1=0.548, top2=0.533
# → gap=0.015) so the top1_top2_gap also needs relaxing or recall still
# returns nothing. Callers opt in via the daemon's
# RecallRequest.auto_tune_cold_start flag.
COLD_START_ROW_THRESHOLD = 200
COLD_START_COSINE_FLOOR = 0.5
COLD_START_TOP1_TOP2_GAP = 0.0
COLD_START_MIN_RECURRENCE = 2


def cold_start_gates(n_claims: int) -> tuple[float, float, int] | None:
    """Return `(cosine_floor, top1_top2_gap, min_recurrence)` relaxed
    gates when memory is sparse, else None. Callers compose with
    default values:

        relaxed = cold_start_gates(len(claims))
        if relaxed is not None:
            cosine_floor, top1_top2_gap, min_recurrence = relaxed
    """
    if n_claims < COLD_START_ROW_THRESHOLD:
        return (
            COLD_START_COSINE_FLOOR,
            COLD_START_TOP1_TOP2_GAP,
            COLD_START_MIN_RECURRENCE,
        )
    return None


@dataclass(frozen=True)
class RecallHit:
    """One claim above all four §0.6 gates, with its query cosine attached.

    `cosine` is the raw value before any filter; callers that need to
    surface confidence (e.g., the `--explain` CLI flag in MVP) read it
    directly. Sort order is descending by `cosine`.
    """

    claim: Claim
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
    many claims survived to that stage.
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


def _claim_matches_cwd(claim: Claim, cwd_hash: str) -> bool:
    """§0.6 step 2 — claim in scope iff its polarity has the cwd's
    workspace tag or the cross-workspace bypass.
    """
    target = f"workspace:{cwd_hash}"
    for tag in claim.facts.polarity:
        if tag == target or tag == "cross-workspace":
            return True
    return False


def _encode_query(query: str, encoder: Encoder) -> np.ndarray:
    """Encode the query under the same `text + " | intent: " + intent`
    convention claims use, with intent="" since queries don't carry one.
    The L2-normalised result lets us compute cosine as a dot product.
    """
    embedding = encoder.encode([query])[0]
    l2_normalize(embedding)
    return embedding


def rank_with_diagnostic(
    query: str,
    *,
    claims: list[Claim],
    band: np.ndarray,
    encoder: Encoder,
    cwd_hash: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    cosine_floor: float = DEFAULT_COSINE_FLOOR,
    top1_top2_gap: float = DEFAULT_TOP1_TOP2_GAP,
    min_recurrence: int = DEFAULT_MIN_RECURRENCE,
    intent_kind: IntentKind | None = None,
    query_emb: np.ndarray | None = None,
) -> tuple[list[RecallHit], RankDiagnostic]:
    """Run the §0.6 retrieval pipeline AND emit per-gate diagnostic counts.

    Same algorithm as `rank()`; additionally returns a `RankDiagnostic`
    naming the gate at which the result emptied (or "ok" when hits
    survive). The daemon ships this in `RecallReply.diagnostic` so the
    UserPromptSubmit hook can log "why no hit" per call.

    `query_emb` is an optional pre-computed L2-normalised query embedding. The
    daemon passes the one it embeds for corpus-band recall too, so the query is
    encoded once per request rather than twice; omitted, it is encoded here as
    before (the encoder must encode + normalise it identically — see
    `_encode_query`).
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
            n_rows=len(claims),
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

    if not claims:
        return [], _diag(dropped_at=DROPPED_NO_ROWS)
    if cwd_hash is None:
        cwd_hash = workspace_hash(str(Path.cwd()))

    if query_emb is None:
        query_emb = _encode_query(query, encoder)

    # Step 1: dot-product cosines (band is L2-normalised at write time).
    cosines = band @ query_emb
    top1_raw = float(cosines.max()) if len(cosines) else neg_inf

    # Walk once, accumulating per-stage counts so the diagnostic can
    # report exactly which gate emptied the result. Single pass keeps
    # the hot-path cost equal to the original rank().
    n_above_floor = 0
    eligible: list[tuple[Claim, float]] = []
    for claim, cos in zip(claims, cosines):
        if cos < eff_floor:
            continue
        if claim.facts.is_bad:
            continue
        n_above_floor += 1
        if not _claim_matches_cwd(claim, cwd_hash):
            continue
        eligible.append((claim, float(cos)))

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

    # Step 3: confidence gate. Single-claim passes the gap check (no
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

    # Step 4: recurrence gate — the event-noise filter.
    above_recurrence = [
        (claim, cos)
        for claim, cos in eligible
        if claim.facts.recurrence_count >= eff_recurrence
    ]
    if not above_recurrence:
        return [], _diag(
            top1_raw=top1_raw, n_above_floor=n_above_floor,
            n_after_workspace=len(eligible), top1=top1_cos, top2=top2_cos,
            dropped_at=DROPPED_BELOW_RECURRENCE,
        )

    hits = [RecallHit(claim=claim, cosine=cos) for claim, cos in above_recurrence]
    # Cold-start: with 6-7 close-cosine hits and intent_kind="none" (the
    # classifier defaults for most bench prompts), cosine-only ordering
    # leaves the rank to noise. Run the manifesto rerank even when
    # intent_kind is "none"/None — the neutral valence multiplier
    # collapses the score to `cosine × strength × confidence_floor`,
    # which still factors in recency, log-recurrence, criticality, and
    # confidence_floor.
    rerank_intent = intent_kind if (intent_kind is not None) else "none"
    is_intent_driven_rerank = (
        intent_kind is not None and intent_kind != "none"
    )
    is_cold_start_rerank = len(claims) < COLD_START_ROW_THRESHOLD
    if is_intent_driven_rerank or is_cold_start_rerank:
        from .rerank import rerank as _rerank
        hits = _rerank(hits, intent_kind=rerank_intent)
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
    claims: list[Claim],
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
    that discards the diagnostic — preserves the wire shape every
    existing harness suite pins.

    `cwd_hash` defaults to the caller's `Path.cwd()` so the surface is
    usable without explicit cwd plumbing. Pass an explicit value when
    serving a different working directory.

    `intent_kind` opts the post-gate hits into the manifesto re-rank
    (`memory.rerank`). When None (default), surviving hits come back in
    cosine-descending order. When set, the four binary gates still
    decide eligibility, then the strength × valence × confidence
    re-rank decides order.
    """
    hits, _ = rank_with_diagnostic(
        query, claims=claims, band=band, encoder=encoder, cwd_hash=cwd_hash,
        top_k=top_k, cosine_floor=cosine_floor, top1_top2_gap=top1_top2_gap,
        min_recurrence=min_recurrence, intent_kind=intent_kind,
    )
    return hits


def recall(
    query: str,
    *,
    store: ExperienceClaimStore,
    cwd_hash: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    encoder: Encoder | None = None,
    cosine_floor: float = DEFAULT_COSINE_FLOOR,
    top1_top2_gap: float = DEFAULT_TOP1_TOP2_GAP,
    min_recurrence: int = DEFAULT_MIN_RECURRENCE,
    intent_kind: IntentKind | None = None,
    auto_tune_cold_start: bool = False,
) -> list[RecallHit]:
    """Run the §0.6 retrieval pipeline against the per-user band.

    Convenience wrapper that loads the snapshot from `store` then runs
    `rank()`. One-shot CLI / fallback path; the daemon shape calls
    `store.read_all_with_band()` once and invokes `rank()` per request.

    `intent_kind` is forwarded to `rank` — opt-in manifesto re-rank.

    `auto_tune_cold_start=True` relaxes the three gates to their
    cold-start values when the store is below `COLD_START_ROW_THRESHOLD`,
    exactly as the daemon does for the UserPromptSubmit hook — callers
    that want recall to match production behaviour on a sparse store
    opt in. Explicit gate overrides still win (relax only fills the
    defaults).
    """
    claims, band = store.read_all_with_band()
    if encoder is None:
        encoder = store._ensure_encoder()  # type: ignore[attr-defined]
    if auto_tune_cold_start:
        relaxed = cold_start_gates(len(claims))
        if relaxed is not None:
            cold_floor, cold_gap, cold_recurrence = relaxed
            if cosine_floor == DEFAULT_COSINE_FLOOR:
                cosine_floor = cold_floor
            if top1_top2_gap == DEFAULT_TOP1_TOP2_GAP:
                top1_top2_gap = cold_gap
            if min_recurrence == DEFAULT_MIN_RECURRENCE:
                min_recurrence = cold_recurrence
    return rank(
        query,
        claims=claims,
        band=band,
        encoder=encoder,
        cwd_hash=cwd_hash,
        top_k=top_k,
        cosine_floor=cosine_floor,
        top1_top2_gap=top1_top2_gap,
        min_recurrence=min_recurrence,
        intent_kind=intent_kind,
    )
