"""Verdict lifecycle + drift cascade for insight passages.

The state machine specified in `.claude/plans/lensed-knowledge-architecture.md`
§4.4, implemented as pure functions over `InsightPassage`. Every
transition produces a NEW frozen row — callers replace the list element
explicitly. No hidden mutation.

States and the events that drive them:

    candidate ──/accept────→ accepted        (test passes + signal ≥ threshold)
       │
       └────/reject──────→ rejected
       │
       └────/correct─────→ rejected_corrected  (replacement queued)
       │
       └──fail test─────→ rejected            (compression test fails)

    accepted ──source-drift──→ stale          (cited content_hash changed)

    stale    ──re-verify pass──→ accepted
             ──re-verify fail──→ retired

    rejected* / retired  are final.

The compression-test gate (Day 4) governs the candidate→accepted path;
this module owns the verdict-signal path and the drift path.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Iterable, Mapping

from .insight import (
    FINAL_STATES,
    PENDING_STATES,
    InsightPassage,
    VerdictPolarity,
    VerdictSignal,
    VerdictSource,
    VerdictState,
    append_verdict,
)

# States from which a transition to rejected is reachable via user reject.
_REJECTABLE_STATES: frozenset[VerdictState] = frozenset({"candidate", "accepted", "stale"})

# ---------------------------------------------------------------------------
# Confidence calculation
# ---------------------------------------------------------------------------

# Source authority weights (matches three-tier attribution philosophy from
# the agent-harness architecture — user > mechanical > llm).
_AUTHORITY: dict[VerdictSource, float] = {
    "user": 1.0,
    "mechanical": 0.6,
    "llm": 0.3,
}

_POLARITY: dict[VerdictPolarity, float] = {
    "accept": 1.0,
    "neutral": 0.0,
    "reject": -1.0,
}

# Compression-test prerequisites (Day 4 wires the actual test; this
# threshold gates "would the candidate even be considered?").
PROMOTE_CONFIDENCE_THRESHOLD = 0.5
MIN_DISTINCT_CITATIONS = 2      # anti-paraphrase guard (architecture §7.2)

# --- Beta-accumulation confidence (docs/internal/GROUNDING_MODEL.md) ---
#
# Each insight carries two tallies — corroboration and falsification —
# behind the derived `InsightPassage.confidence` (a Beta mean: bounded
# 0..1, naturally slow). The faithfulness score seeds the prior at
# promotion (`insight.seed_confidence`); outcome reducers add weight via
# `accumulate_outcome`. Nothing else moves confidence.

# Source drift is one falsification outcome (see propagate_drift); a
# passing re-verification is its corroborating inverse, same magnitude.
_SOURCE_DRIFT_WEIGHT = 1.0


def accumulate_outcome(
    insight: InsightPassage,
    *,
    corroboration: float = 0.0,
    falsification: float = 0.0,
) -> InsightPassage:
    """Add outcome weight to the tallies; `confidence` derives from them.

    The single mutation point for the Beta model — every outcome reducer
    (diffuse, agent-reported, temporal) lands here. Pure: returns a new row.
    """
    return replace(
        insight,
        corroboration=insight.corroboration + corroboration,
        falsification=insight.falsification + falsification,
    )


def compute_verdict_score(signals: tuple[VerdictSignal, ...]) -> float:
    """Weighted average of verdict signals.

    Each signal contributes `authority(source) * polarity` and the result is
    normalised by the total authority weight (so 5 LLM accepts don't drown
    out 1 user reject). Empty signal list → 0.0 (neutral).
    """
    if not signals:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    for s in signals:
        w = _AUTHORITY[s.source]
        numerator += w * _POLARITY[s.polarity]
        denominator += w
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """ISO 8601 UTC timestamp; injectable for deterministic tests via
    `datetime.now` replacement, but the production path goes through here
    so every state transition is wall-clock-attributable."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_verdict(
    insight: InsightPassage,
    *,
    source: VerdictSource,
    polarity: VerdictPolarity,
    lens_id: str | None = None,
    timestamp: str | None = None,
) -> InsightPassage:
    """Append a verdict signal to the insight's history.

    Does NOT trigger state transitions — those are decided by the
    consolidation pass (`consolidate_state`) which considers the full
    accumulated verdict history alongside the compression-test outcome.
    Splitting "record signal" from "decide state" keeps the per-turn path
    fast (just append) and the per-session path coherent (full re-eval).
    """
    sig = VerdictSignal(
        source=source,
        polarity=polarity,
        timestamp=timestamp or _utc_now(),
        lens_id=lens_id,
    )
    return append_verdict(insight, sig)


def consolidate_state(
    insight: InsightPassage,
    *,
    compression_test_pass: bool | None = None,
    correction_replacement: InsightPassage | None = None,
) -> InsightPassage:
    """Decide the next verdict_state from accumulated signals + test outcome.

    Called at consolidation cadence per insight. Inputs:

    - `compression_test_pass`: outcome of the compression test (Day 4).
      `None` means the test wasn't run this cycle — leave state untouched
      unless verdict signals alone force a transition (e.g. explicit reject).
    - `correction_replacement`: if a `/correct` was issued, the new
      synthesis candidate that replaces this row. The current row flips
      to `rejected_corrected`; the caller is responsible for adding the
      replacement as a new candidate.

    Transitions are conservative — only fire when the evidence is
    unambiguous. Ambiguous cases (e.g. mixed verdict signals on a
    candidate that hasn't seen the compression test yet) stay in
    candidate so the next cycle gets another chance.

    Pure — returns a new row.
    """
    state = insight.verdict_state
    verdict = compute_verdict_score(insight.verdict_signals)

    if state in FINAL_STATES:
        return insight

    # /correct path — replacement supplied means we're retiring this row.
    if correction_replacement is not None:
        return replace(insight, verdict_state="rejected_corrected")

    # User verdict authority — borrowed from memory.forget's "user_declared"
    # protection. The MOST RECENT user signal is authoritative: a user
    # reject sends the row to rejected; a user accept on an accepted row
    # blocks any later downgrade from compression-test failure. This
    # mirrors the memory layer's user-declared protection over automated
    # signal noise (memory.forget §protections).
    user_signals = [s for s in insight.verdict_signals if s.source == "user"]
    if user_signals:
        latest_user = user_signals[-1]   # append-only log → last is latest
        if latest_user.polarity == "reject" and state in _REJECTABLE_STATES:
            return replace(insight, verdict_state="rejected")
        if (latest_user.polarity == "accept" and state == "accepted"
                and compression_test_pass is False):
            # User accepted; the next compression cycle failed but the
            # user's authority blocks downgrade — the row stays accepted.
            return insight

    if state == "candidate":
        if compression_test_pass is False:
            return replace(insight, verdict_state="rejected")
        if compression_test_pass is True:
            # Test passed — check verdict signal strength + citation
            # diversity. The architecture §7.1 specifies the compression
            # test ALSO enforces citation diversity, so this check is a
            # defence-in-depth.
            distinct = {c.passage_id for c in insight.citations}
            if (verdict > 0 and len(distinct) >= MIN_DISTINCT_CITATIONS
                    and insight.confidence >= PROMOTE_CONFIDENCE_THRESHOLD):
                return replace(insight, verdict_state="accepted")
            # Stay in candidate — compression test passed but signal isn't
            # there yet. Next session might bring more verdicts.
            return insight
        return insight   # no compression-test outcome this cycle

    if state == "stale":
        # `stale` is set by the drift cascade; the re-verification pass
        # (a separate consolidator step) calls back here with
        # compression_test_pass to commit to accepted or retired.
        if compression_test_pass is True:
            return replace(insight, verdict_state="accepted")
        if compression_test_pass is False:
            return replace(insight, verdict_state="retired")
        return insight

    return insight


# ---------------------------------------------------------------------------
# Drift cascade
# ---------------------------------------------------------------------------

def detect_drift(
    insights: list[InsightPassage],
    fresh_source_hashes: Mapping[str, str],
) -> list[int]:
    """Indices of insights whose cited source hashes no longer match.

    `fresh_source_hashes`: `{passage_id: current_content_hash}`.

    Drift is detected by position-aligned comparison of
    `insight.citations[i].passage_id` against
    `insight.source_passage_hashes[i]`. The promotion pipeline guarantees
    this alignment (each citation is paired with the source's hash-at-
    promotion-time); writers must preserve it. A source removed entirely
    also cascades as drift.

    Final-state insights (rejected, rejected_corrected, retired) are
    skipped — they don't surface in retrieval and have no reason to be
    re-evaluated.

    Returns indices into `insights`, in input order.
    """
    drifted: list[int] = []
    for idx, ins in enumerate(insights):
        if not ins.stale_if_sources_drift:
            continue
        if ins.verdict_state in FINAL_STATES:
            continue
        stored = ins.source_passage_hashes
        for cit_idx, c in enumerate(ins.citations):
            current = fresh_source_hashes.get(c.passage_id)
            if current is None:
                drifted.append(idx)
                break
            if cit_idx < len(stored) and stored[cit_idx] != current:
                drifted.append(idx)
                break
    return drifted


def propagate_drift(
    insights: list[InsightPassage],
    fresh_source_hashes: Mapping[str, str],
) -> tuple[list[InsightPassage], list[int]]:
    """Return updated insight list with drift-detected rows flipped to stale.

    Returns `(new_insights, drifted_indices)`. The list ordering and
    `insight_idx` positions are preserved (the band-row join must
    remain valid). Already-stale rows are left untouched — re-verification
    is the only path out of stale.
    """
    drifted = detect_drift(insights, fresh_source_hashes)
    if not drifted:
        return insights, []
    drift_set = set(drifted)
    updated: list[InsightPassage] = []
    for idx, ins in enumerate(insights):
        if idx in drift_set and ins.verdict_state == "accepted":
            # Drift is one falsification outcome — the stale row carries
            # visibly lower confidence until re-verification corroborates
            # or retires it.
            stale = replace(ins, verdict_state="stale")
            updated.append(
                accumulate_outcome(stale, falsification=_SOURCE_DRIFT_WEIGHT)
            )
        else:
            updated.append(ins)
    return updated, drifted


# ---------------------------------------------------------------------------
# Convenience: bulk re-evaluation
# ---------------------------------------------------------------------------

def apply_drift_cascade_to_archive(km_path, contents=None) -> tuple[int, int]:
    """Run the drift cascade against the current source layer; rewrite the
    insight layer in place if anything flipped to stale.

    `contents` is an optional pre-loaded `ArchiveContents` — pass it when
    you've just written the archive and want to avoid re-reading the
    (potentially large) source band. The function falls back to a fresh
    `archive.read` when contents is None.

    Returns `(n_drifted, n_total_insights)`. A no-drift result rewrites
    nothing. Returns `(0, 0)` when the archive has no insight layer.
    """
    from pathlib import Path
    from . import archive

    p = Path(km_path)
    if contents is None:
        contents = archive.read(p)
    if not contents.insights:
        return 0, 0

    fresh_hashes = {c.passage_id: c.content_hash for c in contents.registry}

    updated, drifted_idx = propagate_drift(contents.insights, fresh_hashes)
    if not drifted_idx:
        return 0, len(contents.insights)

    insight_band = contents.bands[archive.INSIGHT_BAND_NAME]
    archive.write_insight_layer_in_place(p, updated, insight_band)
    return len(drifted_idx), len(contents.insights)


def consolidate_all(
    insights: list[InsightPassage],
    *,
    compression_test_results: Mapping[str, bool] | None = None,
    corrections: Mapping[str, InsightPassage] | None = None,
) -> list[InsightPassage]:
    """Apply consolidate_state to every insight in one pass.

    `compression_test_results`: `{insight_id: passed?}` — pass `None` to
    skip the test gate for that row.
    `corrections`: `{insight_id: replacement_candidate}`.

    Used by the session-end consolidator (Day 4 wires this into the
    harness consolidation pipeline).
    """
    compression_test_results = compression_test_results or {}
    corrections = corrections or {}
    updated: list[InsightPassage] = []
    for ins in insights:
        updated.append(consolidate_state(
            ins,
            compression_test_pass=compression_test_results.get(ins.insight_id),
            correction_replacement=corrections.get(ins.insight_id),
        ))
    return updated
