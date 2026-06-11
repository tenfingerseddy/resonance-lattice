"""Measure evaluators — the join's first semantic layer (H1/S4 deliverable 1).

A `success_criteria` entry carries a `measure` string naming *how* the
criterion is verified — one of `mechanical:<spec>`, `user_confirms`, or
`llm_judges:<rubric>` (architecture §"Success criteria", `state/intent.py`
`Criterion`). Until now those strings were **uninterpreted**: nothing read
them at outcome time. This module gives each one a runnable evaluator that
turns the observed signals into a `CriterionCheck` verdict.

The dispatch is by `measure` *kind*; each kind designates its authoritative
signal source, and a `user` signal always outranks the others (the standing
authority rule — user > mechanical > llm). Rather than re-implement that
rule, every evaluator funnels the relevant signals through the existing
`claim_outcome.combine_signals`, so a user override of a mechanical or llm
verdict is handled by one tested code path.

  mechanical:<spec> — verified by the deterministic `mechanical` signals the
                      PostToolUse hook already distilled (exit code, file
                      existence, …). The `<spec>` is the human-readable
                      description of what was checked; v1 trusts the hook's
                      pre-distilled per-signal verdict and does not re-run the
                      spec. A `user` signal can override.
  user_confirms     — verified by the `user` accept/reject signal. No prompt
                      is ever raised (Principle 2 — harvest, don't ask);
                      absent a user signal the criterion is `unknown`.
  llm_judges:<rubric> — verified by an injected LLM judge over the `<rubric>`
                      and the resolution-time evidence. The judge is a seam
                      (`LLMJudge`) so the evaluator stays hermetic in tests;
                      the CLI wires the real client. An llm verdict is low
                      authority by construction, so the poison guard (S4
                      deliverable 4) attenuates its trust impact for free.

The evaluator is pure: signals in, a `CriterionCheck` out, no I/O. The
caller (the signal→criterion synthesiser, deliverable 2) owns reading the
pending-signal log and converting `PendingSignal` → `Signal`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from .claim_outcome import (
    CriterionCheck,
    Signal,
    Verdict,
    VerdictConfidence,
    combine_signals,
)

if TYPE_CHECKING:  # `Criterion` is a TypedDict — dict at runtime, no import.
    from .intent import Criterion
    from .signals import PendingSignal

# The three measure kinds. A `measure` string is `<kind>` or `<kind>:<spec>`.
MEASURE_KINDS: frozenset[str] = frozenset(
    {"mechanical", "user_confirms", "llm_judges"}
)

# An injected LLM judge: `(rubric, evidence) -> (verdict, confidence)`. The
# evaluator never constructs a client itself — the CLI passes a real one, the
# harness a synthetic one. Returns `("unknown", "low")` when it cannot decide.
# The returned confidence is audit-only (recorded in the signal value); the
# authority rule pins an llm verdict to low `verdict_confidence` regardless, so
# a confident-but-wrong judge cannot inflate a criterion's trust weight.
LLMJudge = Callable[[str, str], "tuple[Verdict, VerdictConfidence]"]


def parse_measure(measure: str) -> tuple[str, str]:
    """Split a `measure` string into `(kind, spec)`.

    `user_confirms` → `("user_confirms", "")`; `mechanical:exit_code==0` →
    `("mechanical", "exit_code==0")`; `llm_judges:answer is correct` →
    `("llm_judges", "answer is correct")`. An unrecognised kind returns
    `("", measure)` so the evaluator can yield `unknown` rather than guess.
    """
    kind, _, spec = measure.partition(":")
    if kind not in MEASURE_KINDS:
        return "", measure
    return kind, spec


def evaluate_criterion(
    criterion: "Criterion | dict",
    signals: list[Signal],
    *,
    judge: LLMJudge | None = None,
    evidence: str = "",
) -> CriterionCheck:
    """Evaluate one success criterion against the observed `signals`.

    `signals` are the `Signal`s seen for this intent (the caller filters by
    intent and converts `PendingSignal` → `Signal`). Each measure has a
    **native** source that supplies its verdict; a `user` signal then layers
    on via the standing authority rule (`combine_signals`, user > mechanical
    > llm):

      user_confirms      — native source is the user; verdict from the user
                           signal(s), `verdict_source="signal"`. No user
                           signal → `unknown`.
      mechanical:<spec>  — native source is the `mechanical` signals. With
                           native evidence the user can conflict it, but a
                           failed test is never whitewashed by an accept
                           (combine_signals fails safe to not_satisfied +
                           conflict on disagreement).
      llm_judges:<rubric>— native verdict is the injected `judge` over the
                           rubric + `evidence`. The judge's *returned*
                           confidence is recorded only in `signals_seen`; the
                           authority rule pins an llm verdict to low
                           `verdict_confidence`, so it never lifts the rung.

    When a non-user measure has **no native evidence** but the user explicitly
    resolved the intent, the user's verdict is honoured (not dropped to
    `unknown`) and marked `verdict_source="user_override"` — distinguishable
    from a measured pass, since the declared check never ran. With neither
    native evidence nor a user signal → `unknown` (unmet-but-undecided, which
    the roll-up treats as not-promotable without falsifying). An
    uninterpretable measure is likewise `unknown`.
    """
    measure = criterion["measure"]
    kind, spec = parse_measure(measure)

    text = criterion["text"]
    if kind == "":
        # Uninterpretable measure — cannot be verified, so it cannot pass.
        return CriterionCheck(
            criterion_text=text, measure=measure, verdict="unknown",
            signals_seen=[], verdict_confidence="low",
        )

    user_signals = [s for s in signals if s.source == "user"]

    if kind == "user_confirms":
        # The user *is* the native verifier — a user verdict is a measurement,
        # `verdict_source="signal"`, not an override.
        verdict, vc, conflict, source = combine_signals(user_signals)
        relevant = user_signals
    else:
        if kind == "mechanical":
            native = [s for s in signals if s.source == "mechanical"]
        else:  # llm_judges — only spend a judge call the user won't override.
            native = []
            if judge is not None and not user_signals:
                jv, jc = judge(spec, evidence)
                native = [Signal(
                    source="llm",
                    value={"verdict": jv, "confidence": jc},
                    timestamp=_latest_timestamp(signals),
                )]
        relevant = native + user_signals
        if native:
            # Measured: native verdict, and the user can conflict it (a failed
            # test is not whitewashed by a user accept — combine_signals fails
            # safe to not_satisfied + conflict on disagreement).
            verdict, vc, conflict, source = combine_signals(relevant)
        elif user_signals:
            # The declared check never ran, but the user explicitly resolved
            # the intent — honour that verdict rather than dropping it, marked
            # `user_override` so it is distinguishable from a measured pass
            # (the S4 d4 poison guard weights override vs measured).
            verdict, vc, conflict, _ = combine_signals(user_signals)
            source = "user_override"
        else:
            verdict, vc, conflict, source = combine_signals([])  # unknown

    return CriterionCheck(
        criterion_text=text,
        measure=measure,
        verdict=verdict,
        signals_seen=relevant,
        verdict_confidence=vc,
        conflict_flag=conflict,
        verdict_source=source,
    )


def evaluate_criteria(
    criteria: "list[Criterion] | list[dict]",
    signals: list[Signal],
    *,
    judge: LLMJudge | None = None,
    evidence: str = "",
) -> list[CriterionCheck]:
    """Evaluate every criterion against the same signal set — the per-criterion
    map the synthesiser drives. v1 passes the intent's whole signal set to
    each criterion; the per-criterion source filter in `evaluate_criterion`
    does the coarse routing. (An intent with two same-kind criteria therefore
    shares that kind's signal pool — finer spec→signal binding is a later
    refinement, documented not silently assumed.)"""
    return [
        evaluate_criterion(c, signals, judge=judge, evidence=evidence)
        for c in criteria
    ]


def pending_to_signal(pending: "PendingSignal") -> Signal:
    """Convert a pending-signal record to a `Signal`. The pending log's
    `captured_at` becomes the signal's `timestamp`. Typed under
    `TYPE_CHECKING` (no runtime import of `state.signals`) — attribute access
    still accepts any duck-typed record a test supplies."""
    return Signal(
        source=pending.source,
        value=pending.value,
        timestamp=pending.captured_at,
    )


def synthesize_criterion_checks(
    criteria: "list[Criterion] | list[dict]",
    pending_signals: list,
    *,
    fallback: "Criterion | dict | None" = None,
    judge: LLMJudge | None = None,
    evidence: str = "",
) -> list[CriterionCheck]:
    """The signal→criterion synthesiser (S4 deliverable 2): read an intent's
    declared `success_criteria` and the pending signals seen during its
    lifetime, evaluate each criterion, and return per-criterion
    `CriterionCheck`s.

    When the intent declares **no** criteria (the pre-S5 common case — most
    intents are added with `success_criteria=[]`), `fallback` — a single
    `user_confirms` criterion synthesised from the user's accept/reject — is
    evaluated instead, preserving today's user-confirms behaviour until S5
    auto-harvests real criteria. With neither criteria nor fallback the result
    is empty (the roll-up is then `unknown`).

    `pending_signals` are raw pending records (converted here); `judge` /
    `evidence` are forwarded to any `llm_judges` criterion.
    """
    signals = [pending_to_signal(s) for s in pending_signals]
    if criteria:
        effective: list = list(criteria)
    elif fallback is not None:
        effective = [fallback]
    else:
        effective = []
    return evaluate_criteria(effective, signals, judge=judge, evidence=evidence)


def _latest_timestamp(signals: list[Signal]) -> str:
    """The latest signal timestamp, or `now` when there are none — the
    synthesised llm verdict is stamped at the most recent evidence it judged.
    Cosmetic only: `combine_signals` resolves by source, never by time."""
    if signals:
        return max(s.timestamp for s in signals)
    from .claim_outcome import now_iso
    return now_iso()
