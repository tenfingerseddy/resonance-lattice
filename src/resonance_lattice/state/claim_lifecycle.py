"""Claim lifecycle primitives — corpus + experience state transitions,
Beta math, experience reseed.

`consolidate_corpus` drives a corpus `Claim` between `candidate`,
`active`, `stale`, and `retired` on compression-test outcomes + verdict
signals. `consolidate_experience` is its experience analog — an asserted
(uncited) claim has no compression test, so its earning gate is recurrence
+ outcome trust. The *gates* that judge "did this candidate earn promotion"
(compression test for fresh corpus candidates, LLM reverify for stale
claims, user verdicts) stay where they are and call the source-matched
consolidator to drive the transition through one named seam.

`accumulate_outcome` is the single Beta-tally mutation point, shared by
both sources (corpus ledger reducers and experience outcome corroboration
both land here). `record_verdict` appends a verdict signal without moving
state (corpus only — verdict signals are a `CorpusFacts` field).
`retune_to_rung` is the experience-side reseed seam used by the forget
pass — same Beta arithmetic, no state move.

Every function is pure over `Claim` — frozen records in, new frozen
records out via `evolve`. No hidden mutation; no store I/O. Each
consolidator rejects the other source's claims, so a mis-routed claim
fails loudly rather than reading a field its `facts` lacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from ..memory._common import utcnow_iso
from .claim import (
    FINAL_STATES,
    Claim,
    evolve,
)

# Verdict-signal primitives live in `store.insight` (leaf record types —
# the spine imports them the same way `state.claim` already does at
# function-scope for its derived properties).
from ..store.insight import VerdictPolarity, VerdictSignal, VerdictSource

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Compression-test prerequisites — `consolidate_corpus` only flips a
# candidate claim to `active` when the verdict signals + trust clear these.
PROMOTE_CONFIDENCE_THRESHOLD = 0.5
MIN_DISTINCT_CITATIONS = 2

# Experience earning gate — `consolidate_experience` flips a candidate to
# `active` only once it has recurred this many times AND its trust has risen
# strictly above the neutral seed. Recurrence is a cross-session *repetition*
# signal (the agent re-derived the same claim), NOT independent-source
# grounding the way `MIN_DISTINCT_CITATIONS` is — the capture path bumps it on
# identical re-captured text, so it can be self-corroborating. It is therefore
# the weaker of the two gates; the strict trust gate carries the earned
# evidence.
MIN_RECURRENCE = 2

# A claim whose trust drops below this floor retires permanently. Above
# the faithfulness gate's ~0.6 admit threshold, the seed prior lands
# comfortably above the floor; sustained falsification (paced by the
# Beta model's slowness) is what reaches it.
RETIRE_FLOOR = 0.3

# A source-drift event counts as one falsification (a passing
# re-verification is its corroborating inverse, same magnitude).
_SOURCE_DRIFT_WEIGHT = 1.0

# States from which a user reject can still send a claim to `retired`.
_REJECTABLE_STATES: frozenset[str] = frozenset(
    {"candidate", "active", "stale"}
)

# Source authority weights (architecture: user > mechanical > LLM).
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


# ---------------------------------------------------------------------------
# Gate signals — the typed payload a gate hands the spine
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateSignals:
    """Signals a gate hands `consolidate_corpus` to drive a state
    transition. All fields optional; the spine reads what the claim's
    state warrants. Today's two signals — the compression-test outcome
    and a correction replacement.
    """

    compression_test_pass: bool | None = None
    correction_replacement: Claim | None = None
    reason: str = ""


# ---------------------------------------------------------------------------
# Beta-tally arithmetic — the single mutation point
# ---------------------------------------------------------------------------

def accumulate_outcome(
    claim: Claim,
    *,
    corroboration: float = 0.0,
    falsification: float = 0.0,
) -> Claim:
    """Add outcome weight to the Beta tallies; `trust` derives from them.

    The single mutation point for DIRECT (non-consolidate) Beta evidence:
    reverification corroboration (`store.reverification`) and drift
    falsification (`propagate_drift`). The session/intent attribution apply does
    NOT come through here — it re-derives via `rederive_outcome`.

    For a SEEDED corpus claim, the born seed is the *non-ledger baseline* the
    consolidate apply re-derives the tally from (`tally = seed + ledger`). So
    direct evidence must grow the seed too, or the next re-derivation would
    silently discard it. Experience claims and corpus claims minted before the
    seed existed (sentinel < 0) have no baseline and accumulate the tally only.
    """
    new_corr = claim.corroboration + corroboration
    new_fals = claim.falsification + falsification
    if claim.source == "corpus" and claim.facts.seed_corroboration >= 0.0:
        return evolve(
            claim,
            corroboration=new_corr,
            falsification=new_fals,
            seed_corroboration=claim.facts.seed_corroboration + corroboration,
            seed_falsification=claim.facts.seed_falsification + falsification,
        )
    return evolve(claim, corroboration=new_corr, falsification=new_fals)


def rederive_outcome(
    claim: Claim,
    *,
    seed_corroboration: float,
    seed_falsification: float,
    corroboration: float = 0.0,
    falsification: float = 0.0,
) -> Claim:
    """SET the Beta tallies to `born seed + cumulative outcome weight` — the
    idempotent counterpart to `accumulate_outcome`.

    The corpus attribution apply re-derives trust from the IMMUTABLE born seed
    plus the full outcome ledger's weight each pass, so re-running
    consolidation is a no-op rather than re-folding the ledger onto the prior
    result (§B BLOCKER). Mirrors the experience confidence pass, which
    re-derives from a neutral baseline. `evolve` leaves `facts` (and thus the
    seed) untouched, so the seed stays constant across passes.
    """
    return evolve(
        claim,
        corroboration=seed_corroboration + corroboration,
        falsification=seed_falsification + falsification,
    )


def compute_verdict_score(signals: Iterable[VerdictSignal]) -> float:
    """Weighted average of verdict signals — `authority(source) * polarity`,
    normalised by the total authority weight (so 5 LLM accepts don't
    drown out 1 user reject). Empty signal list → 0.0 (neutral)."""
    sigs = list(signals)
    if not sigs:
        return 0.0
    num = 0.0
    den = 0.0
    for s in sigs:
        w = _AUTHORITY[s.source]
        num += w * _POLARITY[s.polarity]
        den += w
    return num / den if den else 0.0


def record_verdict(
    claim: Claim,
    *,
    source: VerdictSource,
    polarity: VerdictPolarity,
    lens_id: str | None = None,
    timestamp: str | None = None,
) -> Claim:
    """Append a verdict signal to the claim's history.

    Does NOT trigger a state transition — `consolidate_corpus` decides
    that on the full accumulated history. Splitting "record signal" from
    "decide state" keeps the per-turn path fast (append only) and the
    per-session path coherent (full re-eval).
    """
    sig = VerdictSignal(
        source=source,
        polarity=polarity,
        timestamp=timestamp or utcnow_iso(),
        lens_id=lens_id,
    )
    return evolve(
        claim, verdict_signals=claim.facts.verdict_signals + (sig,)
    )


# ---------------------------------------------------------------------------
# Consolidate — the state transition
# ---------------------------------------------------------------------------

def consolidate_corpus(
    claim: Claim,
    *,
    signals: GateSignals = GateSignals(),
) -> Claim:
    """Decide the next `ClaimState` for a corpus claim from outcome
    evidence + gate signals.

    Verdict-signal history + compression-test outcome drive the §4.4
    state-machine table. Transitions are conservative — only fire when
    the evidence is unambiguous; ambiguous cases stay in the current
    state so the next cycle gets another chance.
    """
    if claim.source != "corpus":
        # Mirror `consolidate_experience`'s door guard so a mis-routed claim
        # fails loudly at the seam (the module contract: each consolidator
        # rejects the other source) rather than AttributeError-ing mid-body
        # on the CorpusFacts-only `verdict_signals`/`citations` reads below.
        raise TypeError(
            f"consolidate_corpus expects a corpus claim; "
            f"got source={claim.source!r}"
        )
    state = claim.state
    if state in FINAL_STATES:
        return claim

    if signals.correction_replacement is not None:
        return evolve(claim, state="retired")

    verdict = compute_verdict_score(claim.facts.verdict_signals)

    # User verdict authority — the most-recent user signal wins.
    user_signals = [
        s for s in claim.facts.verdict_signals if s.source == "user"
    ]
    if user_signals:
        latest = user_signals[-1]
        if latest.polarity == "reject" and state in _REJECTABLE_STATES:
            return evolve(claim, state="retired")
        if (latest.polarity == "accept" and state == "active"
                and signals.compression_test_pass is False):
            # User accepted; a subsequent test fail does not downgrade.
            return claim

    if claim.trust < RETIRE_FLOOR:
        return evolve(claim, state="retired")

    if state == "candidate":
        if signals.compression_test_pass is False:
            return evolve(claim, state="retired")
        if signals.compression_test_pass is True:
            # Promote on a passing compression test, provided the verdict
            # history is not net-negative, citation diversity holds, and
            # trust clears the floor. `verdict >= 0` (not `> 0`): the
            # autonomous corpus pipeline promotes a fresh claim with no
            # verdict signals at all (verdict 0.0) — a strict `> 0` would
            # strand every autonomously-promoted claim in `candidate`.
            # A genuinely net-rejected candidate (verdict < 0) is held.
            distinct = {c.passage_id for c in claim.facts.citations}
            if (verdict >= 0
                    and len(distinct) >= MIN_DISTINCT_CITATIONS
                    and claim.trust >= PROMOTE_CONFIDENCE_THRESHOLD):
                return evolve(claim, state="active")
            return claim
        return claim

    if state == "stale":
        if signals.compression_test_pass is True:
            return evolve(claim, state="active")
        if signals.compression_test_pass is False:
            return evolve(claim, state="retired")
        return claim

    return claim


def consolidate_experience(claim: Claim) -> Claim:
    """Decide the next `ClaimState` for an experience claim — the
    experience analog of `consolidate_corpus`.

    An experience claim is asserted, not source-cited, so it has no
    compression test and no citation diversity; its earning gate is
    **recurrence + outcome trust** (architecture §2/§5). Born `candidate`,
    it earns `active` once it has recurred (`recurrence_count >=
    MIN_RECURRENCE`) AND its trust has risen **strictly above** the neutral
    seed (`> PROMOTE_CONFIDENCE_THRESHOLD`) — genuine net-positive outcome
    evidence, not the un-earned 0.5 prior. The trust gate is strict where
    `consolidate_corpus`'s is inclusive: a corpus claim at trust 0.5 still
    carries independent grounding (≥2 distinct citations + a coverage lift),
    whereas an experience claim's only structural signal is recurrence — a
    repetition (the capture path bumps it on identical re-captured text),
    possibly self-corroborating — so its trust gate must reflect real
    positive outcomes, never the neutral seed alone. (This also makes a
    zero-net attribution a no-op rather than a free promotion.) Sustained
    falsification below `RETIRE_FLOOR` retires it. `active` stays active
    above the floor — a claim that earned retrieval keeps it while it stays
    above the floor, mirroring `consolidate_corpus`, which has no
    active→candidate demotion; `retired` is absorbing.

    Trust already carries the outcome evidence — `accumulate_outcome` (the
    shared Beta mutation point) lands the corroboration/falsification before
    this commits the transition, exactly as the corpus path does. Pure over
    `Claim`; no store I/O.
    """
    if claim.source != "experience":
        raise TypeError(
            f"consolidate_experience expects an experience claim; "
            f"got source={claim.source!r}"
        )
    state = claim.state
    if state in FINAL_STATES:
        return claim
    if claim.trust < RETIRE_FLOOR:
        return evolve(claim, state="retired")
    if state == "candidate":
        if (claim.facts.recurrence_count >= MIN_RECURRENCE
                and claim.trust > PROMOTE_CONFIDENCE_THRESHOLD):
            return evolve(claim, state="active")
        return claim
    return claim


# ---------------------------------------------------------------------------
# Drift cascade — corpus-side
# ---------------------------------------------------------------------------

def detect_drift(
    insights: list[Claim],
    fresh_source_hashes: Mapping[str, str],
) -> list[int]:
    """Indices of corpus claims whose cited source hashes no longer match.

    Drift is detected by position-aligned comparison of
    `claim.facts.citations[i].passage_id` against
    `claim.facts.source_passage_hashes[i]`. The promotion pipeline
    guarantees this alignment; writers must preserve it. A source removed
    entirely also cascades as drift.

    Final-state claims (`retired`) are skipped — they don't surface in
    retrieval. Returns indices into `insights`, in input order.
    """
    drifted: list[int] = []
    for idx, ins in enumerate(insights):
        # Only corpus claims cite sources and can drift; an experience claim
        # has no citations / `stale_if_sources_drift`. Skip it before the
        # `CorpusFacts`-only reads below (the unified band carries both).
        if ins.source != "corpus":
            continue
        if not ins.facts.stale_if_sources_drift:
            continue
        if ins.state in FINAL_STATES:
            continue
        stored = ins.facts.source_passage_hashes
        for cit_idx, c in enumerate(ins.facts.citations):
            # External citations (verified web/source fills) are NOT corpus passages,
            # so corpus-registry hash drift does not apply — a missing lookup means
            # "not in this corpus", not "the source vanished". Skip them; their
            # durability is governed by the claim lifecycle (outcomes), not corpus
            # hashes. Otherwise every external fill is falsely flagged drifted on the
            # next refresh and evicted.
            if c.is_external:
                continue
            current = fresh_source_hashes.get(c.passage_id)
            if current is None:
                drifted.append(idx)
                break
            if cit_idx < len(stored) and stored[cit_idx] != current:
                drifted.append(idx)
                break
    return drifted


def propagate_drift(
    insights: list[Claim],
    fresh_source_hashes: Mapping[str, str],
) -> tuple[list[Claim], list[int]]:
    """Flip drift-detected `active` claims to `stale`, with one
    falsification outcome each. Returns `(new_insights, drifted_indices)`
    in input order — the band-row join must remain valid."""
    drifted = detect_drift(insights, fresh_source_hashes)
    if not drifted:
        return insights, []
    drift_set = set(drifted)
    updated: list[Claim] = []
    for idx, ins in enumerate(insights):
        if idx in drift_set and ins.state == "active":
            stale = evolve(ins, state="stale")
            updated.append(
                accumulate_outcome(stale, falsification=_SOURCE_DRIFT_WEIGHT)
            )
        else:
            updated.append(ins)
    return updated, drifted


# ---------------------------------------------------------------------------
# Re-tune — experience-side seam
# ---------------------------------------------------------------------------

def retune_to_rung(claim: Claim, target_rung: str) -> Claim:
    """Re-seed the Beta tallies to the target rung's prior; everything
    else stays. The forget-pass condition-4 downgrade path lands here."""
    from ..memory.store import seed_tallies_for_rung

    corroboration, falsification = seed_tallies_for_rung(target_rung)
    return evolve(
        claim,
        corroboration=corroboration,
        falsification=falsification,
    )
