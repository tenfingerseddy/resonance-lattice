"""memory_v22_confidence — confidence raising contracts.

Pins architecture §"Calibration mechanisms — how rows earn back trust",
all five mechanisms. Contracts:

  (a) 2 wins raise low → medium.

  (b) 3 wins raise medium → high.

  (c) Non-principle row caps at high — verified requires principle level
      AND cross-domain (mechanism 5).

  (d) Principle with 5 wins across ≥2 intent_kinds reaches verified.

  (e) Principle with 5 wins in only 1 intent_kind caps at high.

  (f) Symmetric drop — 2 net losses pull confidence down to low.

  (g) Incidental tier doesn't count — primary/secondary only.

  (h) End-to-end — `raise_confidence_pass` mutates the store.

  (i) Mechanism 4 — `corroborate_row` one-step raise, caps at verified.

  (j) Mechanism 3 — implicit corroboration folds in as fractional net.

  (k) Mechanism 2 — corpus confirms → row raised to verified.

  (l) Mechanism 2 — corpus contradicts → row stays low, flagged.

  (m) Mechanism 2 — empty corpus retrieval → unverifiable, no LLM call.

  (n) Mechanism 2 — only high/severe-criticality rows at low or
      verified confidence are scanned.

  (o) Mechanism 2 — a verified row the corpus now contradicts drops to
      low (the corpus-drift response).

  (p) Mechanism 2 — a verified row the corpus is silent on stays
      verified.

Hermetic — synthetic outcomes + temp memory store; fake corpus + LLM.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from resonance_lattice.memory.confidence import (
    raise_confidence_pass,
    target_confidence,
)
from resonance_lattice.memory.store import Memory, Row
from resonance_lattice.state import (
    Attribution,
    CriterionCheck,
    OutcomeRecord,
)


def _row(
    *,
    row_id: str = "01HZ_R1",
    level: str = "pattern",
    confidence: str = "low",
) -> Row:
    return Row(
        row_id=row_id,
        text="row",
        polarity=["factual", "workspace:abc123"],
        recurrence_count=2,
        created_at="2026-05-01T00:00:00Z",
        last_corroborated_at="2026-05-01T00:00:00Z",
        transcript_hash="manual",
        is_bad=False,
        level=level,
        confidence=confidence,
        origin="distilled",
    )


def _outcome(
    *,
    row_id: str,
    verdict: str,
    intent_kind: str = "implement",
    tier: str = "primary",
) -> OutcomeRecord:
    return OutcomeRecord(
        intent_id=f"01HZ_INTENT_{verdict}",
        intent_level="task",
        criterion_checks=[CriterionCheck(
            criterion_text="x", measure="user_confirms", verdict=verdict,
        )],
        roll_up_verdict=verdict,
        attribution=[Attribution(row_id=row_id, tier=tier)],
        resolved_at="2026-05-07T00:00:00Z",
        intent_kind=intent_kind,
    )


def _check_2_wins_low_to_medium() -> int:
    row = _row(level="pattern", confidence="low")
    outcomes = [_outcome(row_id="01HZ_R1", verdict="satisfied") for _ in range(2)]
    target = target_confidence(row, outcomes)
    if target != "medium":
        print(f"[memory_v22_confidence] FAIL (a): target={target!r}",
              file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (a) 2 wins → medium OK", file=sys.stderr)
    return 0


def _check_3_wins_medium_to_high() -> int:
    row = _row(level="pattern", confidence="medium")
    outcomes = [_outcome(row_id="01HZ_R1", verdict="satisfied") for _ in range(3)]
    target = target_confidence(row, outcomes)
    if target != "high":
        print(f"[memory_v22_confidence] FAIL (b): target={target!r}",
              file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (b) 3 wins → high OK", file=sys.stderr)
    return 0


def _check_non_principle_caps_at_high() -> int:
    row = _row(level="pattern", confidence="high")
    outcomes = [
        _outcome(row_id="01HZ_R1", verdict="satisfied", intent_kind=k)
        for k in ["debug", "design", "implement", "review", "explain"]
    ]
    target = target_confidence(row, outcomes)
    # Pattern-level row should NOT be promoted to verified, cross-domain
    # or otherwise — only principles can hit verified.
    if target == "verified":
        print(f"[memory_v22_confidence] FAIL (c): pattern hit verified",
              file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (c) non-principle caps at high OK",
          file=sys.stderr)
    return 0


def _check_principle_cross_domain_to_verified() -> int:
    row = _row(level="principle", confidence="high")
    outcomes = (
        [_outcome(row_id="01HZ_R1", verdict="satisfied", intent_kind="debug")
         for _ in range(3)]
        + [_outcome(row_id="01HZ_R1", verdict="satisfied", intent_kind="design")
           for _ in range(2)]
    )
    target = target_confidence(row, outcomes)
    if target != "verified":
        print(f"[memory_v22_confidence] FAIL (d): target={target!r}",
              file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (d) principle cross-domain → verified OK",
          file=sys.stderr)
    return 0


def _check_principle_single_domain_caps_at_high() -> int:
    row = _row(level="principle", confidence="high")
    outcomes = [
        _outcome(row_id="01HZ_R1", verdict="satisfied", intent_kind="debug")
        for _ in range(5)
    ]
    target = target_confidence(row, outcomes)
    if target == "verified":
        print(f"[memory_v22_confidence] FAIL (e): single-domain principle "
              f"hit verified", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (e) single-domain principle caps OK",
          file=sys.stderr)
    return 0


def _check_symmetric_drop_to_low() -> int:
    row = _row(level="pattern", confidence="medium")
    # 2 net losses (3 losses, 1 win) → low
    outcomes = (
        [_outcome(row_id="01HZ_R1", verdict="not_satisfied") for _ in range(3)]
        + [_outcome(row_id="01HZ_R1", verdict="satisfied")]
    )
    target = target_confidence(row, outcomes)
    if target != "low":
        print(f"[memory_v22_confidence] FAIL (f): target={target!r}",
              file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (f) symmetric drop to low OK",
          file=sys.stderr)
    return 0


def _check_incidental_tier_excluded() -> int:
    row = _row(level="pattern", confidence="low")
    outcomes = [
        _outcome(row_id="01HZ_R1", verdict="satisfied", tier="incidental")
        for _ in range(10)
    ]
    target = target_confidence(row, outcomes)
    # Incidental shouldn't credit anything — row stays at low (target=None).
    if target is not None:
        print(f"[memory_v22_confidence] FAIL (g): incidental credited; "
              f"target={target!r}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (g) incidental tier excluded OK",
          file=sys.stderr)
    return 0


def _check_end_to_end_pass() -> int:
    """raise_confidence_pass actually mutates the store."""
    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        row_id = memory.add_row(
            text="distilled pattern",
            polarity=["factual", "workspace:abc123"],
            transcript_hash="distilled:x",
            embedding=np.zeros(768, dtype=np.float32),
            level="pattern",
            confidence="low",
            origin="distilled",
        )
        outcomes = [
            _outcome(row_id=row_id, verdict="satisfied") for _ in range(2)
        ]
        changes = raise_confidence_pass(memory, outcomes=outcomes)
        rows, _ = memory.read_all()
    if len(changes) != 1:
        print(f"[memory_v22_confidence] FAIL (h): changes={len(changes)}",
              file=sys.stderr)
        return 1
    if rows[0].confidence != "medium":
        print(f"[memory_v22_confidence] FAIL (h): confidence="
              f"{rows[0].confidence!r}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (h) end-to-end pass OK", file=sys.stderr)
    return 0


def _check_m4_user_corroboration() -> int:
    """(i) Mechanism 4 — `corroborate_row` raises one step per call and
    caps at verified; a missing row returns None."""
    from resonance_lattice.memory.confidence import corroborate_row

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        rid = memory.add_row(
            text="a row", polarity=["factual"], transcript_hash="manual",
            embedding=np.zeros(768, dtype=np.float32), confidence="low",
        )
        ladder = []
        for _ in range(4):
            c = corroborate_row(memory, rid)
            ladder.append(c.to_confidence if c is not None else None)
        if ladder != ["medium", "high", "verified", None]:
            print(f"[memory_v22_confidence] FAIL (i): ladder={ladder!r}",
                  file=sys.stderr)
            return 1
        if corroborate_row(memory, "no-such-row") is not None:
            print("[memory_v22_confidence] FAIL (i): missing row should "
                  "return None", file=sys.stderr)
            return 1
    print("[memory_v22_confidence] (i) M4 user corroboration OK",
          file=sys.stderr)
    return 0


def _check_m3_implicit_corroboration() -> int:
    """(j) Mechanism 3 — implicit corroboration. 6 distinct satisfied
    intents where the row was recalled but never explicitly attributed
    fold in as +2 net (3 events = +1), raising low → medium. An
    explicitly-attributed intent does NOT also count as implicit.
    """
    from resonance_lattice.memory.confidence import (
        implicit_corroboration_events,
        raise_confidence_pass,
    )
    from resonance_lattice.state import RecallEntry, RecallHitMetadata

    def _sat(intent_id: str, *, attribution: list) -> OutcomeRecord:
        return OutcomeRecord(
            intent_id=intent_id, intent_level="task",
            criterion_checks=[CriterionCheck(
                criterion_text="x", measure="user_confirms",
                verdict="satisfied")],
            roll_up_verdict="satisfied", attribution=attribution,
            resolved_at="2026-05-07T00:00:00Z", intent_kind="design",
        )

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        rid = memory.add_row(
            text="distilled pattern", polarity=["factual", "workspace:abc"],
            transcript_hash="distilled:x",
            embedding=np.zeros(768, dtype=np.float32),
            level="pattern", confidence="low", origin="distilled",
        )
        recalls = [
            RecallEntry(
                turn_id=f"t{i}", timestamp="2026-05-07T00:00:00Z",
                prompt_hash="p", intent_kind="design",
                intent_id=f"intent-{i}",
                row_metadata=[
                    RecallHitMetadata(row_id=rid, rank=0, cosine=0.9)],
            )
            for i in range(6)
        ]
        outcomes = [_sat(f"intent-{i}", attribution=[]) for i in range(6)]

        n = implicit_corroboration_events(
            rid, recalls=recalls, outcomes=outcomes)
        if n != 6:
            print(f"[memory_v22_confidence] FAIL (j): implicit count={n} "
                  f"(want 6)", file=sys.stderr)
            return 1
        raise_confidence_pass(memory, outcomes=outcomes, recalls=recalls)
        rows, _ = memory.read_all()
        if rows[0].confidence != "medium":
            print(f"[memory_v22_confidence] FAIL (j): confidence="
                  f"{rows[0].confidence!r} (want medium)", file=sys.stderr)
            return 1

        # An explicitly-attributed intent is mechanism 1's, not implicit.
        explicit = [_sat(
            "intent-0",
            attribution=[Attribution(row_id=rid, tier="primary")])]
        if implicit_corroboration_events(
            rid, recalls=recalls[:1], outcomes=explicit) != 0:
            print("[memory_v22_confidence] FAIL (j): explicit attribution "
                  "double-counted as implicit", file=sys.stderr)
            return 1
    print("[memory_v22_confidence] (j) M3 implicit corroboration OK",
          file=sys.stderr)
    return 0


def _fake_corpus_llm(verdict: str):
    """Canned LLM client returning a fixed corpus-verification verdict —
    the `(system, messages, max_tokens) -> LLMResponse` seam shape."""
    from resonance_lattice.memory.distil import LLMResponse

    def _call(system, messages, max_tokens):
        return LLMResponse(
            text=json.dumps({"verdict": verdict, "reason": f"test-{verdict}"}),
            input_tokens=0, output_tokens=0,
        )

    return _call


def _check_m2_confirm() -> int:
    """(k) Mechanism 2 — a high-criticality low-confidence row the corpus
    confirms is raised to verified."""
    from resonance_lattice.memory.confidence import corpus_verification_pass

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        memory.add_row(
            text="always pin the encoder revision",
            polarity=["factual", "workspace:abc"],
            transcript_hash="manual",
            embedding=np.zeros(768, dtype=np.float32),
            criticality="high", confidence="low",
        )
        results = corpus_verification_pass(
            memory,
            corpus=lambda q, k: ["The encoder revision must be pinned."],
            llm=_fake_corpus_llm("confirm"),
        )
        rows, _ = memory.read_all()
    if len(results) != 1 or results[0].verdict != "confirmed":
        print(f"[memory_v22_confidence] FAIL (k): results={results!r}",
              file=sys.stderr)
        return 1
    if rows[0].confidence != "verified":
        print(f"[memory_v22_confidence] FAIL (k): confidence="
              f"{rows[0].confidence!r}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (k) M2 corpus confirm → verified OK",
          file=sys.stderr)
    return 0


def _check_m2_contradict() -> int:
    """(l) Mechanism 2 — a contradicted row stays low, flagged for review;
    M2 itself never drops."""
    from resonance_lattice.memory.confidence import corpus_verification_pass

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        memory.add_row(
            text="the default top-k is 20", polarity=["factual"],
            transcript_hash="manual",
            embedding=np.zeros(768, dtype=np.float32),
            criticality="severe", confidence="low",
        )
        results = corpus_verification_pass(
            memory,
            corpus=lambda q, k: ["The default top-k is 10."],
            llm=_fake_corpus_llm("contradict"),
        )
        rows, _ = memory.read_all()
    if len(results) != 1 or results[0].verdict != "contradicted":
        print(f"[memory_v22_confidence] FAIL (l): results={results!r}",
              file=sys.stderr)
        return 1
    if rows[0].confidence != "low":
        print(f"[memory_v22_confidence] FAIL (l): confidence moved to "
              f"{rows[0].confidence!r}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (l) M2 contradict stays low OK",
          file=sys.stderr)
    return 0


def _check_m2_unverifiable_empty_corpus() -> int:
    """(m) Mechanism 2 — an empty corpus retrieval yields `unverifiable`
    without an LLM call; the row stays low."""
    from resonance_lattice.memory.confidence import corpus_verification_pass

    llm_calls = []

    def _tracking_llm(system, messages, max_tokens):
        llm_calls.append(1)
        raise AssertionError("LLM must not be called with no passages")

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        memory.add_row(
            text="some high-criticality claim", polarity=["factual"],
            transcript_hash="manual",
            embedding=np.zeros(768, dtype=np.float32),
            criticality="high", confidence="low",
        )
        results = corpus_verification_pass(
            memory, corpus=lambda q, k: [], llm=_tracking_llm,
        )
        rows, _ = memory.read_all()
    if len(results) != 1 or results[0].verdict != "unverifiable":
        print(f"[memory_v22_confidence] FAIL (m): results={results!r}",
              file=sys.stderr)
        return 1
    if llm_calls or rows[0].confidence != "low":
        print(f"[memory_v22_confidence] FAIL (m): llm_calls={llm_calls!r} "
              f"confidence={rows[0].confidence!r}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (m) M2 empty corpus → unverifiable OK",
          file=sys.stderr)
    return 0


def _check_m2_selection_gate() -> int:
    """(n) Mechanism 2 — only high/severe-criticality rows at low or
    verified confidence are scanned; a medium-confidence or normal-
    criticality row is skipped entirely (no result, no LLM call)."""
    from resonance_lattice.memory.confidence import corpus_verification_pass

    def _never_llm(system, messages, max_tokens):
        raise AssertionError("no row should reach the judge")

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        # high-criticality but already medium — out of scope.
        memory.add_row(
            text="medium row", polarity=["factual"], transcript_hash="manual",
            embedding=np.zeros(768, dtype=np.float32),
            criticality="high", confidence="medium",
        )
        # low-confidence but only normal-criticality — out of scope.
        memory.add_row(
            text="normal row", polarity=["factual"], transcript_hash="manual",
            embedding=np.zeros(768, dtype=np.float32),
            criticality="normal", confidence="low",
        )
        results = corpus_verification_pass(
            memory, corpus=lambda q, k: ["x"], llm=_never_llm,
        )
    if results:
        print(f"[memory_v22_confidence] FAIL (n): scanned out-of-scope "
              f"rows: {results!r}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (n) M2 selection gate OK", file=sys.stderr)
    return 0


def _check_m2_verified_contradicted_drops() -> int:
    """(o) Mechanism 2 re-checks verified rows — a verified high-
    criticality row the corpus now contradicts drops to low. This is the
    corpus-drift response that closes the loop."""
    from resonance_lattice.memory.confidence import corpus_verification_pass

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        memory.add_row(
            text="the encoder is gte-small", polarity=["factual"],
            transcript_hash="manual",
            embedding=np.zeros(768, dtype=np.float32),
            criticality="high", confidence="verified",
        )
        results = corpus_verification_pass(
            memory,
            corpus=lambda q, k: ["The encoder is gte-modernbert-base."],
            llm=_fake_corpus_llm("contradict"),
        )
        rows, _ = memory.read_all()
    if len(results) != 1 or results[0].verdict != "contradicted":
        print(f"[memory_v22_confidence] FAIL (o): results={results!r}",
              file=sys.stderr)
        return 1
    if rows[0].confidence != "low":
        print(f"[memory_v22_confidence] FAIL (o): verified row not dropped — "
              f"confidence={rows[0].confidence!r}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (o) M2 verified+contradict → low OK",
          file=sys.stderr)
    return 0


def _check_m2_verified_unverifiable_stays() -> int:
    """(p) A verified row the corpus is silent on stays verified —
    absence of corpus support is not refutation."""
    from resonance_lattice.memory.confidence import corpus_verification_pass

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        memory.add_row(
            text="prefer small, reviewable commits", polarity=["factual"],
            transcript_hash="manual",
            embedding=np.zeros(768, dtype=np.float32),
            criticality="high", confidence="verified",
        )
        results = corpus_verification_pass(
            memory,
            corpus=lambda q, k: ["An unrelated passage about caching."],
            llm=_fake_corpus_llm("unverifiable"),
        )
        rows, _ = memory.read_all()
    if len(results) != 1 or results[0].verdict != "unverifiable":
        print(f"[memory_v22_confidence] FAIL (p): results={results!r}",
              file=sys.stderr)
        return 1
    if rows[0].confidence != "verified":
        print(f"[memory_v22_confidence] FAIL (p): verified row moved to "
              f"{rows[0].confidence!r}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (p) M2 verified+silent stays verified OK",
          file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_2_wins_low_to_medium,
        _check_3_wins_medium_to_high,
        _check_non_principle_caps_at_high,
        _check_principle_cross_domain_to_verified,
        _check_principle_single_domain_caps_at_high,
        _check_symmetric_drop_to_low,
        _check_incidental_tier_excluded,
        _check_end_to_end_pass,
        _check_m4_user_corroboration,
        _check_m3_implicit_corroboration,
        _check_m2_confirm,
        _check_m2_contradict,
        _check_m2_unverifiable_empty_corpus,
        _check_m2_selection_gate,
        _check_m2_verified_contradicted_drops,
        _check_m2_verified_unverifiable_stays,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v22_confidence] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
