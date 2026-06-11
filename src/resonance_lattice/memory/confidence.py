"""Confidence raising — the five architecture §"Calibration" mechanisms.

Distillation dilutes confidence on every promotion. Without a recovery path,
learnings and principles would be permanently downranked despite being the
most valuable claims. The architecture specifies five mechanisms; all five
are implemented here:

  Mechanism 1 — Outcome corroboration
    Each satisfied outcome adds corroboration weight, each failure adds
    falsification weight, to the claim's Beta tallies; the 4-rung confidence
    is a band over the Beta mean. In practice 2 clean successes land a claim
    at `medium`, 3 at `high`, 5 at `verified`.

  Mechanism 2 — Corpus verification
    Manual scan, on-demand via `rlat memory verify <km.rlat>`:
    high-criticality claims at `low` or `verified` confidence are
    checked against the current corpus via retrieval + an LLM judge.
    Confirmed → verified; contradicted → low (the corpus-drift response
    when a once-verified claim is no longer supported); silent →
    unchanged. `corpus_verification_pass`. Not scheduled — scheduling
    is gated on a Phase E ablation showing the scan lifts a measurable
    metric.

  Mechanism 3 — Implicit corroboration
    A claim surfaced in recall whose session then satisfied its intent, with
    no explicit outcome attributed to the claim, earns a fractional bump —
    3 such events add one unit of corroboration weight.
    `implicit_corroboration_events`.

  Mechanism 4 — User corroboration
    `rlat memory corroborate <claim_id>` → immediate one-step raise.
    `corroborate_claim`.

  Mechanism 5 — Cross-domain accumulation
    Breadth of evidence is distinct from depth: each intent_kind beyond
    the first a claim won in adds extra corroboration weight, so a claim
    proven across domains reaches the verified band sooner. A weight, not
    a gate — confidence stays a pure derived view of trust.

One trust math: confidence is a Beta mean — `corroboration / (corroboration
+ falsification)` — the same primitive the corpus insight layer uses, and
`Claim.confidence` is a derived read-only band over it, never a stored
field. Each pass re-derives every claim's tallies from the outcome ledger:
a fixed neutral prior plus the ledger evidence, scoped to outcomes resolved
after the claim's `trust_as_of` so a repaired claim is judged on fresh
evidence. Stateless — running the pass twice is a no-op, and there is no
checkpoint file to keep consistent. A claim the ledger carries no evidence
for is left untouched, so confidence set by the user (mechanism 4) or the
corpus scan (mechanism 2) survives the pass.

Symmetric: a failure adds falsification weight exactly as a success adds
corroboration. Forget condition 3 still handles the *drop-claim* extreme;
this module handles the confidence-drift gradient.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Literal

if TYPE_CHECKING:
    from .._pricing import CostMeter

from ..state.claim import Claim, evolve
from ..state.claim_outcome import ClaimOutcomeLog
from ..store.insight import (
    CONFIDENCE_HIGH_BAND as _HIGH_BAND,
    CONFIDENCE_MEDIUM_BAND as _MEDIUM_BAND,
    CONFIDENCE_VERIFIED_BAND as _VERIFIED_BAND,
    beta_mean,
    confidence_band,
    seed_confidence,
)
from ._common import parse_llm_json
from .claim_store import ExperienceClaimStore
from ._llm import LLMClient
from .store import (
    CONFIDENCE_VALUES,
    Confidence,
    seed_tallies_for_rung,
)

# The confidence ladder, weakest → strongest. `CONFIDENCE_VALUES` is an
# unordered set for enum validation; this is the ordered axis the
# one-step raise/drop mechanisms walk.
CONFIDENCE_LADDER: tuple[Confidence, ...] = (
    "low", "medium", "high", "verified",
)


def raise_one_step(current: Confidence) -> Confidence:
    """Return the next confidence level up, capped at `verified`."""
    idx = CONFIDENCE_LADDER.index(current)
    return CONFIDENCE_LADDER[min(idx + 1, len(CONFIDENCE_LADDER) - 1)]


# Neutral prior — a memory claim has no faithfulness gate, so its Beta prior
# is `seed_confidence(None)`: the neutral Beta(2, 2). Ledger evidence
# accumulates on top of it.
_NEUTRAL_CORR, _NEUTRAL_FALS = seed_confidence(None)

# The net outcome score at which a claim counts as failed. The confidence
# pass itself works in Beta bands now; this constant remains the shared
# definition of the drop floor that `memory/redistil.py` triggers on.
_DROP_FLOOR_NET = -2

# Mechanism 5 — cross-domain accumulation. Breadth of evidence is a
# distinct signal from depth: a claim corroborated across several
# intent_kinds generalises better than one proven only in one domain.
# Each intent_kind beyond the first the claim won in adds this much extra
# corroboration weight — breadth *accelerates* trust toward the verified
# band rather than gating it (confidence stays a pure derived view of
# trust). Tuned below the restraint-test ceiling: two cross-domain wins
# must still read `medium`, never `high`.
_CROSS_DOMAIN_BONUS = 0.25


@dataclass(frozen=True)
class ConfidenceChange:
    """One claim's confidence transition with the evidence that drove it.

    `corroboration` / `falsification` are the claim's re-derived Beta tallies
    after this pass — the neutral prior plus the ledger evidence (mechanism
    1), the mechanism-3 implicit count, and the mechanism-5 cross-domain
    bonus. `to_confidence` is the band over their Beta mean.
    """

    claim_id: str
    from_confidence: Confidence
    to_confidence: Confidence
    corroboration: float
    falsification: float
    distinct_intent_kinds: int
    implicit_events: int = 0


def _bucket_evidence_by_claim(
    outcomes: Iterable,
    cutoffs: dict[str, str] | None = None,
) -> dict[str, tuple[int, int, set[str]]]:
    """Single pass over the ledger → `{claim_id: (corroboration_weight,
    falsification_weight, intent_kinds_with_wins)}`.

    Buckets outcomes by attributed claim_id once, so per-claim lookups are
    O(1). A satisfied roll-up adds one unit of corroboration weight, a
    not_satisfied one unit of falsification weight; `incidental`-tier
    attributions are excluded per architecture §"How attribution flows
    downstream".

    `cutoffs` maps a claim_id to its `trust_as_of` timestamp: an outcome
    resolved at or before a claim's cutoff does not count for that claim, so
    a repaired claim is judged on fresh evidence (Phase C2). An empty or
    missing cutoff counts every outcome.
    """
    cutoffs = cutoffs or {}
    by_claim: dict[str, tuple[int, int, set[str]]] = {}
    for record in outcomes:
        for att in record.attribution:
            if att.tier == "incidental":
                continue
            cutoff = cutoffs.get(att.claim_id)
            if cutoff and record.resolved_at <= cutoff:
                continue
            corr, fals, kinds = by_claim.get(att.claim_id, (0, 0, set()))
            if record.roll_up_verdict == "satisfied":
                corr += 1
                # `none` is the unclassified sentinel, not a real domain —
                # it never counts toward the mechanism-5 cross-domain bonus.
                kind = record.details.intent_kind
                if kind and kind != "none":
                    kinds = kinds | {kind}
            elif record.roll_up_verdict == "not_satisfied":
                fals += 1
            by_claim[att.claim_id] = (corr, fals, kinds)
    return by_claim


# Mechanism 3 — implicit corroboration. 3 implicit events fold in as one
# unit of corroboration weight, so M3 feeds the same Beta tallies M1 does.
# An implicit event is one distinct satisfied intent the claim was recalled
# for but never explicitly attributed to.
_IMPLICIT_EVENTS_PER_CORROBORATION = 3


def implicit_corroboration_events(
    claim_id: str, *, recalls: Iterable, outcomes: Iterable, cutoff: str = "",
) -> int:
    """Count mechanism-3 implicit-corroboration events for `claim_id`.

    An implicit event is a distinct intent that (a) was satisfied,
    (b) had `claim_id` surfaced in at least one recall stamped with that
    intent_id, and (c) did NOT explicitly attribute `claim_id` in its
    outcome (those are mechanism 1's job — M3 is the no-explicit-
    outcome complement, so the two never double-count).

    Counted per distinct intent, not per recall: a chatty session that
    re-surfaces the claim across many recalls for one satisfied intent is
    one event, not many.

    `cutoff` is the claim's `trust_as_of`: an outcome resolved at or before
    it is excluded, so a repaired claim's implicit corroboration is also
    scoped to fresh evidence (Phase C2). Empty cutoff counts every outcome.
    """
    recalls = list(recalls)
    intents_with_claim_surfaced: set[str] = set()
    for entry in recalls:
        if entry.intent_id is None:
            continue
        if any(m.claim_id == claim_id for m in entry.row_metadata):
            intents_with_claim_surfaced.add(entry.intent_id)
    if not intents_with_claim_surfaced:
        return 0

    implicit_intent_ids: set[str] = set()
    for record in outcomes:
        intent_id = getattr(record, "intent_id", None)
        if intent_id is None or intent_id not in intents_with_claim_surfaced:
            continue
        if record.roll_up_verdict != "satisfied":
            continue
        if cutoff and getattr(record, "resolved_at", "") <= cutoff:
            continue
        explicitly_attributed = any(
            att.claim_id == claim_id and att.tier != "incidental"
            for att in record.attribution
        )
        if explicitly_attributed:
            continue  # mechanism 1 owns this intent for this claim
        implicit_intent_ids.add(intent_id)
    return len(implicit_intent_ids)


def _beta_band(beta: float) -> Confidence:
    """Label a Beta mean with a 4-rung confidence.

    A thin alias for `store.insight.confidence_band` — the one band cut,
    shared with `Claim.confidence`. `verified` is purely the top trust
    band; there is no cross-domain gate."""
    return confidence_band(beta)  # type: ignore[return-value]


def _derive_confidence(
    claim: Claim, corr_weight: int, fals_weight: int, intent_kinds: set[str],
) -> tuple[float, float, Confidence]:
    """Re-derive a claim's `(corroboration, falsification, rung)` from its
    cumulative outcome weight — the neutral prior plus the ledger evidence.

    Mechanism 5: each distinct intent_kind beyond the first the claim won
    in adds `_CROSS_DOMAIN_BONUS` corroboration, so breadth of evidence
    accelerates trust. The rung is purely the Beta-mean band — `verified`
    is the top band, never a separate gate."""
    cross_domain = _CROSS_DOMAIN_BONUS * max(0, len(intent_kinds) - 1)
    corroboration = _NEUTRAL_CORR + corr_weight + cross_domain
    falsification = _NEUTRAL_FALS + fals_weight
    rung = _beta_band(beta_mean(corroboration, falsification))
    return corroboration, falsification, rung


def target_confidence(
    claim: Claim, outcomes: Iterable,
) -> Confidence | None:
    """Map cumulative explicit-outcome evidence to a confidence rung.

    Returns None when the ledger carries no counted (primary/secondary)
    evidence for the claim, or when the derived rung matches the current
    one — the caller skips those claims. Mechanism 3 (implicit
    corroboration) is folded in only by `raise_confidence_pass`, which
    has the recall cache. Evidence is scoped to the claim's `trust_as_of`.
    """
    evidence = _bucket_evidence_by_claim(
        outcomes, {claim.claim_id: claim.trust_as_of})
    if claim.claim_id not in evidence:
        return None
    corr_w, fals_w, intent_kinds = evidence[claim.claim_id]
    _, _, rung = _derive_confidence(claim, corr_w, fals_w, intent_kinds)
    return rung if rung != claim.confidence else None


def raise_confidence_pass(
    memory: ExperienceClaimStore,
    *,
    state_root: Path | None = None,
    outcomes: Iterable | None = None,
    recalls: Iterable | None = None,
    dry_run: bool = False,
) -> list[ConfidenceChange]:
    """Re-derive every claim's Beta tallies and confidence from the ledger.

    `outcomes` overrides the on-disk ledger (used by tests). `recalls`
    overrides the on-disk recall cache. When neither `outcomes` nor
    `state_root` is supplied, the pass returns immediately — there's no
    evidence to fold in.

    Folds mechanisms 1 + 3 into one Beta re-derivation: each satisfied
    outcome adds corroboration weight and each failure falsification
    weight (M1); 3 implicit-corroboration events add one corroboration
    unit (M3). The derived rung is purely the Beta-mean band. M2 (corpus
    verification) runs separately as `corpus_verification_pass` — it
    needs the corpus.

    Stateless: each pass re-derives every claim's tallies from the full
    ledger, so running it twice is a no-op. A claim the ledger has no
    counted evidence for is skipped — confidence the user (M4) or the
    corpus scan (M2) set is left untouched.

    `dry_run=True` skips the per-claim write; the returned changes list
    still describes what *would* have been written.
    """
    if outcomes is None:
        if state_root is None:
            return []
        outcomes = list(
            ClaimOutcomeLog(state_root).iter_records(kind="intent")
        )
    else:
        outcomes = list(outcomes)
    if recalls is None:
        if state_root is not None:
            from ..state.recall_cache import RecallCache
            recalls = RecallCache(state_root).read_recent(limit=None)
        else:
            recalls = []
    else:
        recalls = list(recalls)
    claims = memory.read_all()
    # One pass over the ledger to bucket evidence per claim_id; per-claim
    # lookups below are O(1). Evidence is scoped to each claim's
    # `trust_as_of` so a repaired claim is judged on fresh outcomes.
    cutoffs = {c.claim_id: c.trust_as_of for c in claims}
    evidence = _bucket_evidence_by_claim(outcomes, cutoffs)
    changes: list[ConfidenceChange] = []
    pending: list[Claim] = []
    for claim in claims:
        entry = evidence.get(claim.claim_id)
        implicit = implicit_corroboration_events(
            claim.claim_id, recalls=recalls, outcomes=outcomes,
            cutoff=claim.trust_as_of,
        )
        implicit_weight = implicit // _IMPLICIT_EVENTS_PER_CORROBORATION
        # A claim the ledger carries no counted evidence for is left
        # untouched — confidence the user or the corpus scan set survives.
        if entry is None and implicit_weight == 0:
            continue
        corr_w, fals_w, intent_kinds = entry or (0, 0, set())
        corroboration, falsification, target = _derive_confidence(
            claim, corr_w + implicit_weight, fals_w, intent_kinds,
        )
        rung_changed = target != claim.confidence
        tallies_changed = (
            corroboration != claim.corroboration
            or falsification != claim.falsification
        )
        if not rung_changed and not tallies_changed:
            continue
        if rung_changed:
            changes.append(ConfidenceChange(
                claim_id=claim.claim_id,
                from_confidence=claim.confidence,
                to_confidence=target,
                corroboration=corroboration,
                falsification=falsification,
                distinct_intent_kinds=len(intent_kinds),
                implicit_events=implicit,
            ))
        if not dry_run:
            # Confidence is derived — update only the Beta tallies; the
            # rung follows from `Claim.confidence`. Batched into one
            # `write_many` so a whole-store pass is O(N), not O(N²).
            pending.append(evolve(
                claim,
                corroboration=corroboration,
                falsification=falsification,
            ))
    if pending:
        memory.write_many(pending)
    return changes


def corroborate_claim(
    memory: ExperienceClaimStore, claim_id: str, *, dry_run: bool = False,
) -> ConfidenceChange | None:
    """Mechanism 4 — user corroboration. Immediate one-step raise.

    The user has explicitly confirmed `claim_id` is trustworthy (via
    `rlat memory corroborate`). Raise its confidence one rung and reseed
    its Beta tallies to match — a user vouch is a rung authority, not a
    ledger outcome, so it sets the tallies for the target rung directly
    and the derived rung follows. Returns the `ConfidenceChange`, or None
    when the claim is missing or already at `verified` (one-step raise is
    a no-op). `distinct_intent_kinds` is 0 — this mechanism is not
    ledger-derived.

    `dry_run=True` reports the change without writing.
    """
    claim = memory.read(claim_id)
    if claim is None:
        return None
    target = raise_one_step(claim.confidence)  # type: ignore[arg-type]
    if target == claim.confidence:
        return None
    corroboration, falsification = seed_tallies_for_rung(target)
    if not dry_run:
        memory.write(evolve(
            claim, corroboration=corroboration, falsification=falsification,
        ))
    return ConfidenceChange(
        claim_id=claim_id,
        from_confidence=claim.confidence,
        to_confidence=target,
        corroboration=corroboration,
        falsification=falsification,
        distinct_intent_kinds=0,
    )


# ---------------------------------------------------------------------------
# Mechanism 2 — corpus verification
# ---------------------------------------------------------------------------

# Retriever seam: `(query_text, top_k) -> passage texts`. Injected so the
# pass stays hermetic in tests and decoupled from the knowledge-model
# store — the CLI wires the real `rlat search` retrieval; the harness
# injects a synthetic corpus.
CorpusRetriever = Callable[[str, int], list[str]]

# Final per-claim verdict. The LLM is prompted for the imperative verbs
# (`confirm` / `contradict` / `unverifiable`); `_corpus_judge` maps them
# to this past-tense result vocabulary.
CorpusVerdict = Literal["confirmed", "contradicted", "unverifiable"]
_LLM_VERDICT_TO_RESULT: dict[str, CorpusVerdict] = {
    "confirm": "confirmed",
    "contradict": "contradicted",
    "unverifiable": "unverifiable",
}

# M2 scans high-criticality claims at the two ends of the ladder: `low`
# (the downranked-but-important case the scan exists to rescue) and
# `verified` (re-checked so corpus drift can pull a no-longer-supported
# claim back down). Mid-ladder claims are left to outcome corroboration.
_CORPUS_VERIFY_CRITICALITY: tuple[str, ...] = ("high", "severe")
_CORPUS_VERIFY_CONFIDENCE: tuple[Confidence, ...] = ("low", "verified")
_CORPUS_VERIFY_TOP_K = 5
_CORPUS_VERIFY_MAX_TOKENS = 200

_CORPUS_VERIFY_SYSTEM = """\
You judge whether a project's reference corpus supports, contradicts, or
is silent on a stored memory claim.

You are given one CLAIM and several CORPUS PASSAGES — the closest matches
to that claim retrieved from the corpus.

OUTPUT FORMAT — read carefully:
  Output ONLY a JSON object. No prose, no markdown, no code fences. The
  first character is `{`, the last is `}`.

  {"verdict": "confirm|contradict|unverifiable", "reason": "<short reason>"}

  confirm       — a passage states or directly entails the claim.
  contradict    — a passage states something incompatible with the claim.
  unverifiable  — the passages neither support nor contradict the claim:
                  off-topic, or too thin to judge.

Judge ONLY against the passages shown. Do not use outside knowledge."""


@dataclass(frozen=True)
class CorpusVerification:
    """One claim's mechanism-2 outcome.

    `to_confidence` is the claim's confidence after the scan: `verified`
    on a confirmed claim, `low` on a contradicted one, unchanged when the
    corpus is silent.
    """

    claim_id: str
    verdict: CorpusVerdict
    to_confidence: Confidence
    reason: str


def _corpus_judge(
    claim: Claim, passages: list[str], llm: LLMClient, max_tokens: int,
    meter: "CostMeter | None" = None,
) -> tuple[CorpusVerdict, str]:
    """One LLM round-trip → `(verdict, reason)`.

    Any LLM/parse failure degrades to `unverifiable` so a flaky judge
    can never wrongly raise or drop a claim. When `meter` is supplied,
    the observed token usage is recorded so a per-session cap can
    enforce upstream.
    """
    body = (
        "CLAIM:\n" + claim.content.strip() + "\n\nCORPUS PASSAGES:\n"
        + "\n---\n".join(p.strip() for p in passages)
    )
    try:
        response = llm(
            _CORPUS_VERIFY_SYSTEM,
            [{"role": "user", "content": body
              + "\n\nJudge the claim against the corpus."}],
            max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 — judge failure must not raise
        return "unverifiable", f"llm error: {type(exc).__name__}: {exc}"
    if meter is not None:
        meter.add(response.input_tokens, response.output_tokens)
    try:
        payload = parse_llm_json(response.text)
    except json.JSONDecodeError as exc:
        return "unverifiable", f"non-JSON response: {exc}"
    if not isinstance(payload, dict):
        return "unverifiable", "malformed response"
    raw = payload.get("verdict")
    reason = str(payload.get("reason", ""))[:200]
    result = _LLM_VERDICT_TO_RESULT.get(raw) if isinstance(raw, str) else None
    if result is None:
        return "unverifiable", f"bad verdict: {raw!r}"
    return result, reason


def corpus_verification_pass(
    memory: ExperienceClaimStore,
    *,
    corpus: CorpusRetriever,
    llm: LLMClient,
    top_k: int = _CORPUS_VERIFY_TOP_K,
    max_tokens: int = _CORPUS_VERIFY_MAX_TOKENS,
    dry_run: bool = False,
    cost_cap_usd: float | None = None,
) -> list[CorpusVerification]:
    """Mechanism 2 — the on-demand corpus-verification scan.

    Selects claims with `criticality in {high, severe}` AND `confidence in
    {low, verified}`, retrieves the closest corpus passages for each
    claim's content, and asks `llm` to judge whether the corpus confirms,
    contradicts, or is silent on the claim:

      confirm      → confidence set to `verified`.
      contradict   → confidence dropped to `low`. On a `verified` claim
                     this is the corpus-drift response — the source has
                     moved and no longer supports the claim.
      unverifiable → confidence unchanged; absence of corpus support is
                     not refutation, so a `verified` claim stays verified.

    Scanning `verified` claims (not just `low` ones) is what closes the
    drift loop: a claim the corpus once confirmed is re-judged every scan,
    so a contradicting edit pulls it back down without needing a
    separate drift signal.

    `cost_cap_usd` caps cumulative LLM spend in USD across all calls this
    invocation makes. The pass stops before the next call once observed
    spend crosses the cap. Remaining qualifying claims appear in the
    outcome list as `unverifiable` with the cap reason; their confidence
    is unchanged so they stay re-scannable next pass. Matches the
    contract in `store/reverification.py::reverify_stale_insights`.

    Returns one `CorpusVerification` per scanned claim (the empty list
    when nothing qualifies). `dry_run=True` runs the judge but skips the
    confidence write.
    """
    from .._pricing import CostMeter

    meter = CostMeter(cap_usd=cost_cap_usd)
    claims = memory.read_all()
    results: list[CorpusVerification] = []
    pending: list[Claim] = []
    for claim in claims:
        if (claim.facts.criticality not in _CORPUS_VERIFY_CRITICALITY
                or claim.confidence not in _CORPUS_VERIFY_CONFIDENCE):
            continue
        if meter.has_exceeded_cap():
            results.append(CorpusVerification(
                claim.claim_id, "unverifiable", claim.confidence,
                f"cost cap crossed "
                f"(${meter.cost_so_far():.4f} of ${meter.cap_usd:.4f})",
            ))
            continue
        passages = list(corpus(claim.content, top_k))
        if not passages:
            results.append(CorpusVerification(
                claim.claim_id, "unverifiable", claim.confidence,
                "no corpus passages retrieved",
            ))
            continue
        verdict, reason = _corpus_judge(
            claim, passages, llm, max_tokens, meter=meter,
        )
        if verdict == "confirmed":
            to_confidence: Confidence = "verified"
        elif verdict == "contradicted":
            to_confidence = "low"
        else:
            to_confidence = claim.confidence
        if to_confidence != claim.confidence and not dry_run:
            # The corpus scan is a rung authority — it sets the target
            # rung's Beta tallies and the derived confidence follows
            # (see `corroborate_claim`).
            seed_corr, seed_fals = seed_tallies_for_rung(to_confidence)
            pending.append(evolve(
                claim, corroboration=seed_corr, falsification=seed_fals,
            ))
        results.append(CorpusVerification(
            claim.claim_id, verdict, to_confidence, reason,
        ))
    if pending:
        memory.write_many(pending)
    return results


# Sanity-checks at import time so a future Confidence enum drift or a
# band-tuning typo can't slip through silently.
assert all(level in CONFIDENCE_VALUES for level in CONFIDENCE_LADDER), (
    "CONFIDENCE_LADDER references unknown Confidence values"
)
assert 0.0 < _MEDIUM_BAND < _HIGH_BAND < _VERIFIED_BAND <= 1.0, (
    "Beta confidence bands must be strictly increasing within (0, 1]"
)
# The rung→Beta seed table lives in store.py; this check fails the import
# if a seed no longer bands back to its own rung — the two can't drift.
assert all(
    _beta_band(beta_mean(*seed_tallies_for_rung(rung))) == rung
    for rung in CONFIDENCE_LADDER
), "seed_tallies_for_rung must band back to its own rung"
