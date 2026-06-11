"""memory_v22_confidence — confidence raising contracts.

Pins architecture §"Calibration mechanisms — how claims earn back trust".
Confidence is a DERIVED 4-rung band over a claim's Beta trust — never a
stored field. The confidence passes write only Beta tallies; the rung
follows. The cross-domain `verified` gate was dropped — `verified` is now
purely the top trust band (trust ≥ 0.76), reachable by any claim with
enough corroboration regardless of level or intent-kind spread.

Re-derived expectations (neutral prior Beta(2,2); N satisfied → (2+N, 2);
band cuts medium ≥ 0.40, high ≥ 0.70, verified ≥ 0.76):
  2 wins → (4,2) = 0.667 → medium
  3 wins → (5,2) = 0.714 → high
  5 wins → (7,2) = 0.778 → verified

Contracts:

  (a) 2 wins raise low → medium.

  (b) 3 wins raise medium → high.

  (c) 5 wins on a pattern reach verified — no level gate (gate dropped).

  (d) Principle with 5 wins across ≥2 intent_kinds reaches verified.

  (e) Principle with 5 wins in only 1 intent_kind also reaches verified —
      single-domain no longer caps at high (gate dropped).

  (f) Symmetric drop — 2 net losses pull confidence down to low.

  (g) Incidental tier doesn't count — primary/secondary only.

  (h) End-to-end — `raise_confidence_pass` mutates the store.

  (i) Mechanism 4 — `corroborate_claim` one-step raise, caps at verified.

  (j) Mechanism 3 — implicit corroboration folds in as fractional net.

  (k) Mechanism 2 — corpus confirms → claim raised to verified.

  (l) Mechanism 2 — corpus contradicts → claim stays low, flagged.

  (m) Mechanism 2 — empty corpus retrieval → unverifiable, no LLM call.

  (n) Mechanism 2 — only high/severe-criticality claims at low or
      verified confidence are scanned.

  (o) Mechanism 2 — a verified claim the corpus now contradicts drops to
      low (the corpus-drift response).

  (p) Mechanism 2 — a verified claim the corpus is silent on stays
      verified.

  (q) B2 — `raise_confidence_pass` re-derives the Beta tallies, is
      idempotent (a second identical pass is a no-op), and leaves a claim
      the ledger has no evidence for untouched.

  (r) B3 — M2 (corpus verification) and M4 (user corroboration) reseed
      the Beta tallies to match the rung they set.

  (s) C2 — Beta accumulation counts only outcomes resolved after a
      claim's `trust_as_of`; pre-cutoff outcomes are excluded.

Hermetic — synthetic outcomes + temp claim store; fake corpus + LLM.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from resonance_lattice.memory.claim_store import ExperienceClaimStore
from resonance_lattice.memory.confidence import (
    raise_confidence_pass,
    target_confidence,
)
from resonance_lattice.memory.store import seed_tallies_for_rung
from resonance_lattice.state import (
    Attribution,
    ClaimOutcomeRecord,
    CriterionCheck,
    IntentOutcomeDetails,
)
from resonance_lattice.state.claim import Claim, ExperienceFacts


def _claim(
    *,
    claim_id: str = "01HZ_R1",
    kind: str = "event",
    confidence: str = "low",
) -> Claim:
    corr, fals = seed_tallies_for_rung(confidence)
    return Claim(
        claim_id=claim_id,
        source="experience",
        kind=kind,
        content="row",
        created_at="2026-05-01T00:00:00Z",
        corroboration=corr,
        falsification=fals,
        trust_as_of="",
        state="active",
        parent_ids=(),
        facts=ExperienceFacts(
            polarity=("factual", "workspace:abc123"),
            recurrence_count=2,
            criticality="normal",
            created_under_intent_kind="none",
            transcript_hash="manual",
            origin="distilled",
            last_corroborated_at="2026-05-01T00:00:00Z",
            is_bad=False,
        ),
    )


def _store(root: Path) -> ExperienceClaimStore:
    return ExperienceClaimStore(root=root, encoder=None)


def _add(
    memory: ExperienceClaimStore,
    *,
    claim_id: str,
    content: str = "row",
    polarity: tuple[str, ...] = ("factual", "workspace:abc123"),
    kind: str = "event",
    confidence: str = "low",
    criticality: str = "normal",
    transcript_hash: str = "manual",
    origin: str = "manual",
) -> str:
    """Build + write a claim seeded at `confidence`; return the claim_id."""
    corr, fals = seed_tallies_for_rung(confidence)
    memory.write(Claim(
        claim_id=claim_id,
        source="experience",
        kind=kind,
        content=content,
        created_at="2026-05-01T00:00:00Z",
        corroboration=corr,
        falsification=fals,
        trust_as_of="",
        state="active",
        parent_ids=(),
        facts=ExperienceFacts(
            polarity=polarity,
            recurrence_count=2,
            criticality=criticality,
            created_under_intent_kind="none",
            transcript_hash=transcript_hash,
            origin=origin,
            last_corroborated_at="2026-05-01T00:00:00Z",
            is_bad=False,
        ),
    ), embedding=np.zeros(768, dtype=np.float32))
    return claim_id


def _outcome(
    *,
    row_id: str,
    verdict: str,
    intent_kind: str = "implement",
    tier: str = "primary",
) -> ClaimOutcomeRecord:
    return ClaimOutcomeRecord(
        intent_id=f"01HZ_INTENT_{verdict}",
        details=IntentOutcomeDetails(
            intent_level="task",
            criterion_checks=[CriterionCheck(
                criterion_text="x", measure="user_confirms", verdict=verdict,
            )],
            intent_kind=intent_kind,
        ),
        roll_up_verdict=verdict,
        attribution=[Attribution(claim_id=row_id, tier=tier)],
        resolved_at="2026-05-07T00:00:00Z",
    )


def _check_2_wins_low_to_medium() -> int:
    claim = _claim(kind="event", confidence="low")
    outcomes = [_outcome(row_id="01HZ_R1", verdict="satisfied")
                for _ in range(2)]
    target = target_confidence(claim, outcomes)
    if target != "medium":
        print(f"[memory_v22_confidence] FAIL (a): target={target!r}",
              file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (a) 2 wins → medium OK", file=sys.stderr)
    return 0


def _check_3_wins_medium_to_high() -> int:
    claim = _claim(kind="event", confidence="medium")
    outcomes = [_outcome(row_id="01HZ_R1", verdict="satisfied")
                for _ in range(3)]
    target = target_confidence(claim, outcomes)
    if target != "high":
        print(f"[memory_v22_confidence] FAIL (b): target={target!r}",
              file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (b) 3 wins → high OK", file=sys.stderr)
    return 0


def _check_pattern_5_wins_reach_verified() -> int:
    """The cross-domain verified gate was dropped: a pattern with 5 wins
    reaches `verified` purely on its trust band (5 wins → (7,2) = 0.778).
    """
    claim = _claim(kind="event", confidence="high")
    outcomes = [
        _outcome(row_id="01HZ_R1", verdict="satisfied", intent_kind=k)
        for k in ["debug", "design", "implement", "review", "explain"]
    ]
    target = target_confidence(claim, outcomes)
    if target != "verified":
        print(f"[memory_v22_confidence] FAIL (c): pattern with 5 wins did "
              f"not reach verified; target={target!r}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (c) pattern 5 wins → verified (no level "
          "gate) OK", file=sys.stderr)
    return 0


def _check_principle_cross_domain_to_verified() -> int:
    claim = _claim(kind="event", confidence="high")
    outcomes = (
        [_outcome(row_id="01HZ_R1", verdict="satisfied", intent_kind="debug")
         for _ in range(3)]
        + [_outcome(row_id="01HZ_R1", verdict="satisfied", intent_kind="design")
           for _ in range(2)]
    )
    target = target_confidence(claim, outcomes)
    if target != "verified":
        print(f"[memory_v22_confidence] FAIL (d): target={target!r}",
              file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (d) principle cross-domain → verified OK",
          file=sys.stderr)
    return 0


def _check_principle_single_domain_reaches_verified() -> int:
    """The verified gate was dropped — a principle with 5 single-domain
    wins also reaches `verified` (5 wins → 0.778, the top band)."""
    claim = _claim(kind="event", confidence="high")
    outcomes = [
        _outcome(row_id="01HZ_R1", verdict="satisfied", intent_kind="debug")
        for _ in range(5)
    ]
    target = target_confidence(claim, outcomes)
    if target != "verified":
        print(f"[memory_v22_confidence] FAIL (e): single-domain principle "
              f"with 5 wins did not reach verified; target={target!r}",
              file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (e) single-domain principle 5 wins → "
          "verified (no domain gate) OK", file=sys.stderr)
    return 0


def _check_symmetric_drop_to_low() -> int:
    claim = _claim(kind="event", confidence="medium")
    # 2 net losses (3 losses, 1 win) → low.
    outcomes = (
        [_outcome(row_id="01HZ_R1", verdict="not_satisfied") for _ in range(3)]
        + [_outcome(row_id="01HZ_R1", verdict="satisfied")]
    )
    target = target_confidence(claim, outcomes)
    if target != "low":
        print(f"[memory_v22_confidence] FAIL (f): target={target!r}",
              file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (f) symmetric drop to low OK",
          file=sys.stderr)
    return 0


def _check_incidental_tier_excluded() -> int:
    claim = _claim(kind="event", confidence="low")
    outcomes = [
        _outcome(row_id="01HZ_R1", verdict="satisfied", tier="incidental")
        for _ in range(10)
    ]
    target = target_confidence(claim, outcomes)
    # Incidental shouldn't credit anything — claim stays at low (target=None).
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
        memory = _store(Path(td) / "u")
        claim_id = _add(
            memory, claim_id="01HZE2EPASS00000000000001",
            content="distilled pattern", transcript_hash="distilled:x",
            kind="event", confidence="low", origin="distilled",
        )
        outcomes = [
            _outcome(row_id=claim_id, verdict="satisfied") for _ in range(2)
        ]
        changes = raise_confidence_pass(memory, outcomes=outcomes)
        claims = memory.read_all()
    if len(changes) != 1:
        print(f"[memory_v22_confidence] FAIL (h): changes={len(changes)}",
              file=sys.stderr)
        return 1
    if claims[0].confidence != "medium":
        print(f"[memory_v22_confidence] FAIL (h): confidence="
              f"{claims[0].confidence!r}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (h) end-to-end pass OK", file=sys.stderr)
    return 0


def _check_m4_user_corroboration() -> int:
    """(i) Mechanism 4 — `corroborate_claim` raises one step per call and
    caps at verified; a missing claim returns None."""
    from resonance_lattice.memory.confidence import corroborate_claim

    with tempfile.TemporaryDirectory() as td:
        memory = _store(Path(td) / "u")
        rid = _add(memory, claim_id="01HZM4ROW0000000000000001",
                   content="a row", polarity=("factual",),
                   transcript_hash="manual", confidence="low")
        ladder = []
        for _ in range(4):
            c = corroborate_claim(memory, rid)
            ladder.append(c.to_confidence if c is not None else None)
        if ladder != ["medium", "high", "verified", None]:
            print(f"[memory_v22_confidence] FAIL (i): ladder={ladder!r}",
                  file=sys.stderr)
            return 1
        if corroborate_claim(memory, "no-such-row") is not None:
            print("[memory_v22_confidence] FAIL (i): missing claim should "
                  "return None", file=sys.stderr)
            return 1
    print("[memory_v22_confidence] (i) M4 user corroboration OK",
          file=sys.stderr)
    return 0


def _check_m3_implicit_corroboration() -> int:
    """(j) Mechanism 3 — implicit corroboration. 6 distinct satisfied
    intents where the claim was recalled but never explicitly attributed
    fold in as +2 net (3 events = +1), raising low → medium. An
    explicitly-attributed intent does NOT also count as implicit.
    """
    from resonance_lattice.memory.confidence import (
        implicit_corroboration_events,
        raise_confidence_pass,
    )
    from resonance_lattice.state import RecallEntry, RecallHitMetadata

    def _sat(intent_id: str, *, attribution: list) -> ClaimOutcomeRecord:
        return ClaimOutcomeRecord(
            intent_id=intent_id,
            details=IntentOutcomeDetails(
                intent_level="task",
                criterion_checks=[CriterionCheck(
                    criterion_text="x", measure="user_confirms",
                    verdict="satisfied")],
                intent_kind="design",
            ),
            roll_up_verdict="satisfied", attribution=attribution,
            resolved_at="2026-05-07T00:00:00Z",
        )

    with tempfile.TemporaryDirectory() as td:
        memory = _store(Path(td) / "u")
        rid = _add(memory, claim_id="01HZM3ROW0000000000000001",
                   content="distilled pattern",
                   polarity=("factual", "workspace:abc"),
                   transcript_hash="distilled:x", kind="event",
                   confidence="low", origin="distilled")
        recalls = [
            RecallEntry(
                turn_id=f"t{i}", timestamp="2026-05-07T00:00:00Z",
                prompt_hash="p", intent_kind="design",
                intent_id=f"intent-{i}",
                row_metadata=[
                    RecallHitMetadata(claim_id=rid, rank=0, cosine=0.9)],
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
        claims = memory.read_all()
        if claims[0].confidence != "medium":
            print(f"[memory_v22_confidence] FAIL (j): confidence="
                  f"{claims[0].confidence!r} (want medium)", file=sys.stderr)
            return 1

        # An explicitly-attributed intent is mechanism 1's, not implicit.
        explicit = [_sat(
            "intent-0",
            attribution=[Attribution(claim_id=rid, tier="primary")])]
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
    from resonance_lattice.memory._llm import LLMResponse

    def _call(system, messages, max_tokens):
        return LLMResponse(
            text=json.dumps({"verdict": verdict, "reason": f"test-{verdict}"}),
            input_tokens=0, output_tokens=0,
        )

    return _call


def _check_m2_confirm() -> int:
    """(k) Mechanism 2 — a high-criticality low-confidence claim the corpus
    confirms is raised to verified."""
    from resonance_lattice.memory.confidence import corpus_verification_pass

    with tempfile.TemporaryDirectory() as td:
        memory = _store(Path(td) / "u")
        _add(memory, claim_id="01HZM2CONFIRM0000000000001",
             content="always pin the encoder revision",
             polarity=("factual", "workspace:abc"),
             transcript_hash="manual", criticality="high", confidence="low")
        results = corpus_verification_pass(
            memory,
            corpus=lambda q, k: ["The encoder revision must be pinned."],
            llm=_fake_corpus_llm("confirm"),
        )
        claims = memory.read_all()
    if len(results) != 1 or results[0].verdict != "confirmed":
        print(f"[memory_v22_confidence] FAIL (k): results={results!r}",
              file=sys.stderr)
        return 1
    if claims[0].confidence != "verified":
        print(f"[memory_v22_confidence] FAIL (k): confidence="
              f"{claims[0].confidence!r}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (k) M2 corpus confirm → verified OK",
          file=sys.stderr)
    return 0


def _check_m2_contradict() -> int:
    """(l) Mechanism 2 — a contradicted claim stays low, flagged for review;
    M2 itself never drops."""
    from resonance_lattice.memory.confidence import corpus_verification_pass

    with tempfile.TemporaryDirectory() as td:
        memory = _store(Path(td) / "u")
        _add(memory, claim_id="01HZM2CONTRADICT000000001",
             content="the default top-k is 20", polarity=("factual",),
             transcript_hash="manual", criticality="severe", confidence="low")
        results = corpus_verification_pass(
            memory,
            corpus=lambda q, k: ["The default top-k is 10."],
            llm=_fake_corpus_llm("contradict"),
        )
        claims = memory.read_all()
    if len(results) != 1 or results[0].verdict != "contradicted":
        print(f"[memory_v22_confidence] FAIL (l): results={results!r}",
              file=sys.stderr)
        return 1
    if claims[0].confidence != "low":
        print(f"[memory_v22_confidence] FAIL (l): confidence moved to "
              f"{claims[0].confidence!r}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (l) M2 contradict stays low OK",
          file=sys.stderr)
    return 0


def _check_m2_unverifiable_empty_corpus() -> int:
    """(m) Mechanism 2 — an empty corpus retrieval yields `unverifiable`
    without an LLM call; the claim stays low."""
    from resonance_lattice.memory.confidence import corpus_verification_pass

    llm_calls = []

    def _tracking_llm(system, messages, max_tokens):
        llm_calls.append(1)
        raise AssertionError("LLM must not be called with no passages")

    with tempfile.TemporaryDirectory() as td:
        memory = _store(Path(td) / "u")
        _add(memory, claim_id="01HZM2EMPTY000000000000001",
             content="some high-criticality claim", polarity=("factual",),
             transcript_hash="manual", criticality="high", confidence="low")
        results = corpus_verification_pass(
            memory, corpus=lambda q, k: [], llm=_tracking_llm,
        )
        claims = memory.read_all()
    if len(results) != 1 or results[0].verdict != "unverifiable":
        print(f"[memory_v22_confidence] FAIL (m): results={results!r}",
              file=sys.stderr)
        return 1
    if llm_calls or claims[0].confidence != "low":
        print(f"[memory_v22_confidence] FAIL (m): llm_calls={llm_calls!r} "
              f"confidence={claims[0].confidence!r}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (m) M2 empty corpus → unverifiable OK",
          file=sys.stderr)
    return 0


def _check_m2_selection_gate() -> int:
    """(n) Mechanism 2 — only high/severe-criticality claims at low or
    verified confidence are scanned; a medium-confidence or normal-
    criticality claim is skipped entirely (no result, no LLM call)."""
    from resonance_lattice.memory.confidence import corpus_verification_pass

    def _never_llm(system, messages, max_tokens):
        raise AssertionError("no claim should reach the judge")

    with tempfile.TemporaryDirectory() as td:
        memory = _store(Path(td) / "u")
        # high-criticality but already medium — out of scope.
        _add(memory, claim_id="01HZM2GATE0000000000000001",
             content="medium row", polarity=("factual",),
             transcript_hash="manual", criticality="high", confidence="medium")
        # low-confidence but only normal-criticality — out of scope.
        _add(memory, claim_id="01HZM2GATE0000000000000002",
             content="normal row", polarity=("factual",),
             transcript_hash="manual", criticality="normal", confidence="low")
        results = corpus_verification_pass(
            memory, corpus=lambda q, k: ["x"], llm=_never_llm,
        )
    if results:
        print(f"[memory_v22_confidence] FAIL (n): scanned out-of-scope "
              f"claims: {results!r}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (n) M2 selection gate OK", file=sys.stderr)
    return 0


def _check_m2_verified_contradicted_drops() -> int:
    """(o) Mechanism 2 re-checks verified claims — a verified high-
    criticality claim the corpus now contradicts drops to low. This is the
    corpus-drift response that closes the loop."""
    from resonance_lattice.memory.confidence import corpus_verification_pass

    with tempfile.TemporaryDirectory() as td:
        memory = _store(Path(td) / "u")
        _add(memory, claim_id="01HZM2DRIFT000000000000001",
             content="the encoder is gte-small", polarity=("factual",),
             transcript_hash="manual", criticality="high",
             confidence="verified")
        results = corpus_verification_pass(
            memory,
            corpus=lambda q, k: ["The encoder is gte-modernbert-base."],
            llm=_fake_corpus_llm("contradict"),
        )
        claims = memory.read_all()
    if len(results) != 1 or results[0].verdict != "contradicted":
        print(f"[memory_v22_confidence] FAIL (o): results={results!r}",
              file=sys.stderr)
        return 1
    if claims[0].confidence != "low":
        print(f"[memory_v22_confidence] FAIL (o): verified claim not dropped — "
              f"confidence={claims[0].confidence!r}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (o) M2 verified+contradict → low OK",
          file=sys.stderr)
    return 0


def _check_m2_verified_unverifiable_stays() -> int:
    """(p) A verified claim the corpus is silent on stays verified —
    absence of corpus support is not refutation."""
    from resonance_lattice.memory.confidence import corpus_verification_pass

    with tempfile.TemporaryDirectory() as td:
        memory = _store(Path(td) / "u")
        _add(memory, claim_id="01HZM2SILENT00000000000001",
             content="prefer small, reviewable commits", polarity=("factual",),
             transcript_hash="manual", criticality="high",
             confidence="verified")
        results = corpus_verification_pass(
            memory,
            corpus=lambda q, k: ["An unrelated passage about caching."],
            llm=_fake_corpus_llm("unverifiable"),
        )
        claims = memory.read_all()
    if len(results) != 1 or results[0].verdict != "unverifiable":
        print(f"[memory_v22_confidence] FAIL (p): results={results!r}",
              file=sys.stderr)
        return 1
    if claims[0].confidence != "verified":
        print(f"[memory_v22_confidence] FAIL (p): verified claim moved to "
              f"{claims[0].confidence!r}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (p) M2 verified+silent stays verified OK",
          file=sys.stderr)
    return 0


def _check_beta_rederivation() -> int:
    """(q) B2 — the pass re-derives the Beta tallies, is idempotent, and
    leaves a claim the ledger has no evidence for untouched (the property
    that protects user- and corpus-set confidence)."""
    with tempfile.TemporaryDirectory() as td:
        memory = _store(Path(td) / "u")
        evidenced = _add(
            memory, claim_id="01HZBETAEVIDENCED00000001",
            content="evidenced pattern",
            polarity=("factual", "workspace:abc"),
            transcript_hash="distilled:x", kind="event",
            confidence="low", origin="distilled",
        )
        # A claim the ledger never references — its confidence must survive.
        untouched = _add(
            memory, claim_id="01HZBETAUNTOUCHED00000001",
            content="user-vouched row",
            polarity=("factual", "workspace:abc"),
            transcript_hash="manual", confidence="high",
        )
        outcomes = [
            _outcome(row_id=evidenced, verdict="satisfied") for _ in range(2)
        ]
        changes = raise_confidence_pass(memory, outcomes=outcomes)
        again = raise_confidence_pass(memory, outcomes=outcomes)
        by_id = {c.claim_id: c for c in memory.read_all()}

    if len(changes) != 1 or changes[0].claim_id != evidenced:
        print(f"[memory_v22_confidence] FAIL (q): first pass changes="
              f"{changes!r}", file=sys.stderr)
        return 1
    if again:
        print(f"[memory_v22_confidence] FAIL (q): second pass not a no-op — "
              f"{again!r}", file=sys.stderr)
        return 1
    ev = by_id[evidenced]
    # 2 satisfied on the neutral Beta(2, 2) prior → corroboration 4,
    # falsification 2.
    if (ev.corroboration, ev.falsification) != (4.0, 2.0):
        print(f"[memory_v22_confidence] FAIL (q): tallies not re-derived — "
              f"{ev.corroboration}/{ev.falsification}", file=sys.stderr)
        return 1
    if by_id[untouched].confidence != "high":
        print(f"[memory_v22_confidence] FAIL (q): no-evidence claim moved to "
              f"{by_id[untouched].confidence!r}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (q) Beta re-derivation + idempotency OK",
          file=sys.stderr)
    return 0


def _check_m2_m4_reseed_tallies() -> int:
    """(r) B3 — M2 (corpus) and M4 (user corroboration) reseed the Beta
    tallies to match the rung they set, so `trust` stays consistent with
    `confidence` for claims the ledger has no evidence for."""
    from resonance_lattice.memory.confidence import (
        corpus_verification_pass,
        corroborate_claim,
    )

    with tempfile.TemporaryDirectory() as td:
        memory = _store(Path(td) / "u")
        m4 = _add(memory, claim_id="01HZB3M4ROW000000000000001",
                  content="user-vouched row", polarity=("factual",),
                  transcript_hash="manual", confidence="low")
        corroborate_claim(memory, m4)
        m2 = _add(memory, claim_id="01HZB3M2ROW000000000000001",
                  content="corpus-checked claim", polarity=("factual",),
                  transcript_hash="manual", criticality="high",
                  confidence="low")
        corpus_verification_pass(
            memory, corpus=lambda q, k: ["a supporting passage"],
            llm=_fake_corpus_llm("confirm"),
        )
        by_id = {c.claim_id: c for c in memory.read_all()}

    m4_row = by_id[m4]
    if (m4_row.confidence != "medium"
            or (m4_row.corroboration, m4_row.falsification)
            != seed_tallies_for_rung("medium")):
        print(f"[memory_v22_confidence] FAIL (r): M4 claim conf="
              f"{m4_row.confidence!r} tallies={m4_row.corroboration}/"
              f"{m4_row.falsification}", file=sys.stderr)
        return 1
    m2_row = by_id[m2]
    if (m2_row.confidence != "verified"
            or (m2_row.corroboration, m2_row.falsification)
            != seed_tallies_for_rung("verified")):
        print(f"[memory_v22_confidence] FAIL (r): M2 claim conf="
              f"{m2_row.confidence!r} tallies={m2_row.corroboration}/"
              f"{m2_row.falsification}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (r) M2/M4 reseed tallies OK",
          file=sys.stderr)
    return 0


def _check_trust_as_of_scoping() -> int:
    """(s) C2 — `raise_confidence_pass` counts only outcomes resolved
    after a claim's `trust_as_of`. A claim whose cutoff post-dates all its
    outcomes earns nothing; an unscoped claim with the same outcomes is
    raised."""
    from resonance_lattice.state.claim import evolve

    with tempfile.TemporaryDirectory() as td:
        memory = _store(Path(td) / "u")
        # `_outcome` stamps resolved_at 2026-05-07 — the scoped claim's
        # trust_as_of post-dates that, so its outcomes are all pre-cutoff.
        scoped = _add(memory, claim_id="01HZSCOPED0000000000000001",
                      content="repaired claim", polarity=("factual",),
                      transcript_hash="manual", kind="event",
                      confidence="low")
        memory.write(evolve(memory.read(scoped),
                            trust_as_of="2026-05-10T00:00:00Z"))
        unscoped = _add(memory, claim_id="01HZUNSCOPED00000000000001",
                        content="ordinary claim", polarity=("factual",),
                        transcript_hash="manual", kind="event",
                        confidence="low")
        outcomes = (
            [_outcome(row_id=scoped, verdict="satisfied") for _ in range(3)]
            + [_outcome(row_id=unscoped, verdict="satisfied")
               for _ in range(3)]
        )
        changes = raise_confidence_pass(memory, outcomes=outcomes)
        by_id = {c.claim_id: c for c in memory.read_all()}

    if by_id[scoped].confidence != "low":
        print(f"[memory_v22_confidence] FAIL (s): scoped claim moved to "
              f"{by_id[scoped].confidence!r} — pre-cutoff outcomes counted",
              file=sys.stderr)
        return 1
    if by_id[unscoped].confidence != "high":
        print(f"[memory_v22_confidence] FAIL (s): unscoped claim at "
              f"{by_id[unscoped].confidence!r} (want high)", file=sys.stderr)
        return 1
    if {c.claim_id for c in changes} != {unscoped}:
        print(f"[memory_v22_confidence] FAIL (s): changes="
              f"{[c.claim_id for c in changes]}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (s) trust_as_of scoping OK",
          file=sys.stderr)
    return 0


def _check_m2_cost_cap() -> int:
    """(t) Mechanism 2 — `cost_cap_usd` halts the loop. After the first
    LLM call's metered spend crosses the cap, subsequent qualifying
    claims record as `unverifiable` with the cap reason and the LLM is
    not called again. Confidence is preserved so the rows stay
    re-scannable next pass. Mirrors `reverification` g8."""
    from resonance_lattice.memory._llm import LLMResponse
    from resonance_lattice.memory.confidence import corpus_verification_pass

    # Build an LLM that reports >$0.20-worth of token usage on every call
    # — see `_pricing.cost_usd` for the per-token rates. ~100K input +
    # ~1K output Sonnet tokens lands around $0.315.
    calls: list[int] = []

    def _expensive_llm(system, messages, max_tokens):
        calls.append(1)
        return LLMResponse(
            text=json.dumps({"verdict": "confirm", "reason": "ok"}),
            input_tokens=100_000, output_tokens=1_000,
        )

    with tempfile.TemporaryDirectory() as td:
        memory = _store(Path(td) / "u")
        for i in range(3):
            _add(memory, claim_id=f"01HZM2COSTCAP00000000000{i:02d}",
                 content=f"some severe claim number {i}",
                 polarity=("factual",), transcript_hash="manual",
                 criticality="severe", confidence="low")
        results = corpus_verification_pass(
            memory,
            corpus=lambda q, k: ["passage"],
            llm=_expensive_llm,
            cost_cap_usd=0.20,
        )
        claims = memory.read_all()

    verdicts = [r.verdict for r in results]
    confirmed = [r for r in results if r.verdict == "confirmed"]
    unverifiable = [r for r in results if r.verdict == "unverifiable"]
    cap_reasoned = [
        r for r in unverifiable if "cost cap" in r.reason
    ]
    ok = (
        len(results) == 3
        and len(confirmed) == 1
        and len(unverifiable) == 2
        and len(cap_reasoned) == 2
        and len(calls) == 1
        and all(c.confidence in {"low", "verified"} for c in claims)
    )
    if not ok:
        print(f"[memory_v22_confidence] FAIL (t): verdicts={verdicts!r} "
              f"calls={len(calls)} cap_reasoned={len(cap_reasoned)}",
              file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (t) M2 cost_cap_usd halts the loop OK",
          file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_2_wins_low_to_medium,
        _check_3_wins_medium_to_high,
        _check_pattern_5_wins_reach_verified,
        _check_principle_cross_domain_to_verified,
        _check_principle_single_domain_reaches_verified,
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
        _check_beta_rederivation,
        _check_m2_m4_reseed_tallies,
        _check_trust_as_of_scoping,
        _check_m2_cost_cap,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v22_confidence] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
