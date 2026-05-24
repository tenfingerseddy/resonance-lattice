"""Confidence raising — the five architecture §"Calibration" mechanisms.

Distillation dilutes confidence on every promotion. Without a recovery path,
learnings and principles would be permanently downranked despite being the
most valuable rows. The architecture specifies five mechanisms; all five
are implemented here:

  Mechanism 1 — Outcome corroboration
    Threshold-based: 2 successes raise low → medium; 3 raise medium → high;
    5 (across ≥2 intent_kinds) raise high → verified.

  Mechanism 2 — Corpus verification
    Scheduled scan: high-criticality rows at `low` or `verified`
    confidence are checked against the current corpus via retrieval +
    an LLM judge. Confirmed → verified; contradicted → low (the
    corpus-drift response when a once-verified row is no longer
    supported); silent → unchanged. `corpus_verification_pass`.

  Mechanism 3 — Implicit corroboration
    A row surfaced in recall whose session then satisfied its intent, with
    no explicit outcome attributed to the row, earns a fractional bump —
    3 such events raise one step. `implicit_corroboration_events`.

  Mechanism 4 — User corroboration
    `rlat memory corroborate <row_id>` → immediate one-step raise.
    `corroborate_row`.

  Mechanism 5 — Cross-domain accumulation (principle-only)
    Principle attributed to a successful outcome in a NEW intent_kind →
    one-step raise after the cross-domain count threshold.

Stateless re-derivation: each pass scans the cumulative outcome ledger and
maps `(net_score, distinct_intent_kinds_with_wins)` to the target
confidence level. That avoids the per-pass checkpoint dance — one less
state file to keep consistent.

Symmetric: failures attributed to a row count as -1 to net_score
(architecture §"Calibration mechanisms — failed outcomes drop confidence
by one step"). Forget condition 3 still handles the *drop-row* extreme;
this module handles the confidence-drift gradient.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

from ..state.ledger import OutcomeLedger
from ._common import parse_llm_json
from .distil import LLMClient
from .store import CONFIDENCE_VALUES, Confidence, Memory, Row

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

# Threshold cuts. Values come from architecture §"Mechanism 1 — Outcome
# corroboration" — engineering-spec parameters; tunable without rewriting
# the manifesto.
_NET_TO_CONFIDENCE: list[tuple[int, Confidence]] = [
    # net_score >= 5 → verified (cross-domain check applied separately)
    (5, "verified"),
    (3, "high"),
    (2, "medium"),
]
_DROP_FLOOR_NET = -2
_VERIFIED_REQUIRES_INTENT_KINDS = 2
_PRINCIPLE_LEVEL = "principle"


@dataclass(frozen=True)
class ConfidenceChange:
    """One row's confidence transition with the evidence that drove it.

    `net_score` is the explicit-outcome net (mechanism 1).
    `implicit_events` is the mechanism-3 count folded in alongside it.
    """

    row_id: str
    from_confidence: Confidence
    to_confidence: Confidence
    net_score: int
    distinct_intent_kinds: int
    implicit_events: int = 0


def _bucket_evidence_by_row(
    outcomes: Iterable,
) -> dict[str, tuple[int, set[str]]]:
    """Single pass over the ledger → `{row_id: (net_score, intent_kinds_with_wins)}`.

    Replaces the prior O(M×N) inner-loop scan (one full ledger walk per
    row) with O(M+N): bucket outcomes by attributed row_id once, look
    up per row. `incidental` tier excluded per architecture §"How
    attribution flows downstream".
    """
    by_row: dict[str, tuple[int, set[str]]] = {}
    for record in outcomes:
        for att in record.attribution:
            if att.tier == "incidental":
                continue
            net, kinds = by_row.get(att.row_id, (0, set()))
            if record.roll_up_verdict == "satisfied":
                net += 1
                if record.intent_kind:
                    kinds = kinds | {record.intent_kind}
            elif record.roll_up_verdict == "not_satisfied":
                net -= 1
            by_row[att.row_id] = (net, kinds)
    return by_row


def _row_evidence(
    row_id: str, outcomes: Iterable,
) -> tuple[int, set[str]]:
    """Count primary+secondary attributions for `row_id`.

    Single-row convenience wrapper used by tests + `target_confidence`'s
    public surface; the pass-level path uses `_bucket_evidence_by_row`
    once and indexes it.
    """
    return _bucket_evidence_by_row(outcomes).get(row_id, (0, set()))


# Mechanism 3 — implicit corroboration. 3 implicit events fold into the
# net_score as +1, so M3 raises confidence through the same threshold
# table M1 uses. An implicit event is one distinct satisfied intent the
# row was recalled for but never explicitly attributed to.
_IMPLICIT_EVENTS_PER_NET_POINT = 3


def implicit_corroboration_events(
    row_id: str, *, recalls: Iterable, outcomes: Iterable,
) -> int:
    """Count mechanism-3 implicit-corroboration events for `row_id`.

    An implicit event is a distinct intent that (a) was satisfied,
    (b) had `row_id` surfaced in at least one recall stamped with that
    intent_id, and (c) did NOT explicitly attribute `row_id` in its
    outcome (those are mechanism 1's job — M3 is the no-explicit-
    outcome complement, so the two never double-count).

    Counted per distinct intent, not per recall: a chatty session that
    re-surfaces the row across many recalls for one satisfied intent is
    one event, not many.
    """
    recalls = list(recalls)
    intents_with_row_surfaced: set[str] = set()
    for entry in recalls:
        if entry.intent_id is None:
            continue
        if any(m.row_id == row_id for m in entry.row_metadata):
            intents_with_row_surfaced.add(entry.intent_id)
    if not intents_with_row_surfaced:
        return 0

    implicit_intent_ids: set[str] = set()
    for record in outcomes:
        intent_id = getattr(record, "intent_id", None)
        if intent_id is None or intent_id not in intents_with_row_surfaced:
            continue
        if record.roll_up_verdict != "satisfied":
            continue
        explicitly_attributed = any(
            att.row_id == row_id and att.tier != "incidental"
            for att in record.attribution
        )
        if explicitly_attributed:
            continue  # mechanism 1 owns this intent for this row
        implicit_intent_ids.add(intent_id)
    return len(implicit_intent_ids)


def _target_from_evidence(
    row: Row, net: int, intent_kinds: set[str],
) -> Confidence | None:
    """Map (net_score, intent_kinds) → target confidence for `row`."""
    target: Confidence | None = None
    for threshold, level in _NET_TO_CONFIDENCE:
        if net >= threshold:
            if level == "verified":
                # Cross-domain requirement (mechanism 5). Principles can
                # reach verified via cross-domain accumulation; non-
                # principle rows cap at high until corpus verification
                # (mechanism 2, deferred) lands.
                if (row.level == _PRINCIPLE_LEVEL
                        and len(intent_kinds) >= _VERIFIED_REQUIRES_INTENT_KINDS):
                    target = "verified"
                else:
                    target = "high"
            else:
                target = level
            break
    if target is None and net <= _DROP_FLOOR_NET:
        target = "low"
    if target is None or target == row.confidence:
        return None
    return target


def target_confidence(
    row: Row, outcomes: Iterable,
) -> Confidence | None:
    """Map cumulative evidence to a confidence level for `row`.

    Returns None when the evidence is too thin to suggest a change OR
    when the suggested level matches the current one. Caller skips
    rows where this returns None.
    """
    net, intent_kinds = _row_evidence(row.row_id, outcomes)
    return _target_from_evidence(row, net, intent_kinds)


def raise_confidence_pass(
    memory: Memory,
    *,
    state_root: Path | None = None,
    outcomes: Iterable | None = None,
    recalls: Iterable | None = None,
    dry_run: bool = False,
) -> list[ConfidenceChange]:
    """Re-derive every row's confidence from the cumulative ledger.

    `outcomes` overrides the on-disk ledger (used by tests). `recalls`
    overrides the on-disk recall cache. When neither `outcomes` nor
    `state_root` is supplied, the pass returns immediately — there's no
    evidence to fold in.

    Folds mechanisms 1 + 3 + 5: explicit-outcome net (M1), implicit-
    corroboration events (M3, every 3 = +1 to the effective net), and
    the cross-domain verified gate (M5). M2 (corpus verification) runs
    separately as `corpus_verification_pass` — it needs the corpus.

    `dry_run=True` skips the per-row update; the returned changes list
    still describes what *would* have been written.

    Architecture's "step at a time" framing is preserved by the threshold
    bands: a row with 2 wins lands at medium (not verified) regardless of
    its starting confidence; a row with 5 wins + cross-domain lands at
    verified. Walking through every level isn't necessary because we re-
    derive from cumulative state every pass.
    """
    if outcomes is None:
        if state_root is None:
            return []
        outcomes = list(OutcomeLedger(state_root).iter_records())
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
    rows, _ = memory.read_all()
    # One pass over the ledger to bucket evidence per row_id; per-row
    # lookups below are O(1). Avoids the per-row full-ledger scan that
    # the simpler `target_confidence` path does.
    evidence = _bucket_evidence_by_row(outcomes)
    changes: list[ConfidenceChange] = []
    for row in rows:
        net, intent_kinds = evidence.get(row.row_id, (0, set()))
        implicit = implicit_corroboration_events(
            row.row_id, recalls=recalls, outcomes=outcomes,
        )
        # M3 folds in as fractional net: 3 implicit events = +1 point.
        effective_net = net + implicit // _IMPLICIT_EVENTS_PER_NET_POINT
        target = _target_from_evidence(row, effective_net, intent_kinds)
        if target is None:
            continue
        changes.append(ConfidenceChange(
            row_id=row.row_id,
            from_confidence=row.confidence,
            to_confidence=target,
            net_score=net,
            distinct_intent_kinds=len(intent_kinds),
            implicit_events=implicit,
        ))
        if not dry_run:
            memory.update_row(row.row_id, confidence=target)
    return changes


def corroborate_row(
    memory: Memory, row_id: str, *, dry_run: bool = False,
) -> ConfidenceChange | None:
    """Mechanism 4 — user corroboration. Immediate one-step raise.

    The user has explicitly confirmed `row_id` is trustworthy (via
    `rlat memory corroborate`). Raise its confidence one level. Returns
    the `ConfidenceChange`, or None when the row is missing or already
    at `verified` (one-step raise is a no-op). `net_score` /
    `distinct_intent_kinds` are 0 — this mechanism is user-driven, not
    ledger-derived.

    `dry_run=True` reports the change without writing.
    """
    rows, _ = memory.read_all()
    row = next((r for r in rows if r.row_id == row_id), None)
    if row is None:
        return None
    target = raise_one_step(row.confidence)
    if target == row.confidence:
        return None
    if not dry_run:
        memory.update_row(row_id, confidence=target)
    return ConfidenceChange(
        row_id=row_id,
        from_confidence=row.confidence,
        to_confidence=target,
        net_score=0,
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

# Final per-row verdict. The LLM is prompted for the imperative verbs
# (`confirm` / `contradict` / `unverifiable`); `_corpus_judge` maps them
# to this past-tense result vocabulary.
CorpusVerdict = Literal["confirmed", "contradicted", "unverifiable"]
_LLM_VERDICT_TO_RESULT: dict[str, CorpusVerdict] = {
    "confirm": "confirmed",
    "contradict": "contradicted",
    "unverifiable": "unverifiable",
}

# M2 scans high-criticality rows at the two ends of the ladder: `low`
# (the downranked-but-important case the scan exists to rescue) and
# `verified` (re-checked so corpus drift can pull a no-longer-supported
# row back down). Mid-ladder rows are left to outcome corroboration.
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
    """One row's mechanism-2 outcome.

    `to_confidence` is the row's confidence after the scan: `verified`
    on a confirmed row, `low` on a contradicted one, unchanged when the
    corpus is silent.
    """

    row_id: str
    verdict: CorpusVerdict
    to_confidence: Confidence
    reason: str


def _corpus_judge(
    row: Row, passages: list[str], llm: LLMClient, max_tokens: int,
) -> tuple[CorpusVerdict, str]:
    """One LLM round-trip → `(verdict, reason)`.

    Any LLM/parse failure degrades to `unverifiable` so a flaky judge
    can never wrongly raise or drop a row.
    """
    body = (
        "CLAIM:\n" + row.text.strip() + "\n\nCORPUS PASSAGES:\n"
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
    memory: Memory,
    *,
    corpus: CorpusRetriever,
    llm: LLMClient,
    top_k: int = _CORPUS_VERIFY_TOP_K,
    max_tokens: int = _CORPUS_VERIFY_MAX_TOKENS,
    dry_run: bool = False,
) -> list[CorpusVerification]:
    """Mechanism 2 — the scheduled corpus-verification scan.

    Selects rows with `criticality in {high, severe}` AND `confidence in
    {low, verified}`, retrieves the closest corpus passages for each
    row's text, and asks `llm` to judge whether the corpus confirms,
    contradicts, or is silent on the claim:

      confirm      → confidence set to `verified`.
      contradict   → confidence dropped to `low`. On a `verified` row
                     this is the corpus-drift response — the source has
                     moved and no longer supports the row.
      unverifiable → confidence unchanged; absence of corpus support is
                     not refutation, so a `verified` row stays verified.

    Scanning `verified` rows (not just `low` ones) is what closes the
    drift loop: a row the corpus once confirmed is re-judged every scan,
    so a contradicting edit pulls it back down without needing a
    separate drift signal.

    Returns one `CorpusVerification` per scanned row (the empty list
    when nothing qualifies). `dry_run=True` runs the judge but skips the
    confidence write.
    """
    rows, _ = memory.read_all()
    results: list[CorpusVerification] = []
    for row in rows:
        if (row.criticality not in _CORPUS_VERIFY_CRITICALITY
                or row.confidence not in _CORPUS_VERIFY_CONFIDENCE):
            continue
        passages = list(corpus(row.text, top_k))
        if not passages:
            results.append(CorpusVerification(
                row.row_id, "unverifiable", row.confidence,
                "no corpus passages retrieved",
            ))
            continue
        verdict, reason = _corpus_judge(row, passages, llm, max_tokens)
        if verdict == "confirmed":
            to_confidence: Confidence = "verified"
        elif verdict == "contradicted":
            to_confidence = "low"
        else:
            to_confidence = row.confidence
        if to_confidence != row.confidence and not dry_run:
            memory.update_row(row.row_id, confidence=to_confidence)
        results.append(CorpusVerification(
            row.row_id, verdict, to_confidence, reason,
        ))
    return results


# Sanity-check at import time so a future Confidence enum drift can't
# silently break the threshold map.
assert all(level in CONFIDENCE_VALUES for _, level in _NET_TO_CONFIDENCE), (
    "_NET_TO_CONFIDENCE references unknown Confidence values"
)
assert all(level in CONFIDENCE_VALUES for level in CONFIDENCE_LADDER), (
    "CONFIDENCE_LADDER references unknown Confidence values"
)
