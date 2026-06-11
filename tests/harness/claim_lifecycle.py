"""claim_lifecycle — corpus state transitions + Beta math.

Pins `state.claim_lifecycle`. Six contracts:

  (a) `consolidate_corpus` drives a corpus claim through the §4.4
      state-machine table — candidate/stale promote or retire on the
      compression-test outcome; an active claim with no test signal is
      untouched; `retired` is absorbing.
  (c) `accumulate_outcome` adds weight to a corpus claim's Beta tallies;
      `trust` derives.
  (d) `record_verdict` appends to the signal history and does NOT change
      the claim's state.
  (e) `compute_verdict_score` weights signals by authority and
      normalises; empty → 0.0.
  (g) `retune_to_rung` reseeds the Beta tallies on an experience claim;
      nothing else moves.
  (h) `consolidate_corpus` promotes a candidate with NO verdict signals
      to `active` on a passing compression test — the autonomous corpus
      pipeline — but holds a candidate whose verdict history is
      net-negative.

Hermetic — pure-function tests, no temp dir.
"""

from __future__ import annotations

import sys

_P = "claim_lifecycle"


def _make_corpus(state: str = "candidate", trust_corr: float = 3.0,
                 trust_fals: float = 1.0, *, citations: int = 2):
    from resonance_lattice.state.claim import Claim, CorpusFacts
    from resonance_lattice.store.insight import InsightCitation

    cits = tuple(
        InsightCitation(passage_id=f"p{i}", char_span=None, confidence=0.9)
        for i in range(citations)
    )
    return Claim(
        claim_id="01HZCORPUS00000000000000A1",
        source="corpus",
        kind="synthesis",
        content="cited synthesis",
        created_at="2026-05-21T10:00:00Z",
        corroboration=trust_corr,
        falsification=trust_fals,
        trust_as_of="2026-05-21T10:00:00Z",
        state=state,
        parent_ids=(),
        facts=CorpusFacts(
            citations=cits,
            content_fingerprint="01HZCORPUS00000000000000A1",
            source_model_hash="m",
            source_passage_hashes=tuple(f"h{i}" for i in range(citations)),
            verdict_signals=(),
            query=None,
            intent_context=None,
            stale_if_sources_drift=True,
            encoder_version="",
        ),
    )


def _make_experience(state: str = "active"):
    from resonance_lattice.state.claim import Claim, ExperienceFacts

    return Claim(
        claim_id="01HZEXP000000000000000001",
        source="experience",
        kind="pattern",
        content="prefer the standard library",
        created_at="2026-05-21T10:00:00Z",
        corroboration=3.0,
        falsification=1.0,
        trust_as_of="2026-05-21T10:00:00Z",
        state=state,
        parent_ids=("01HZEVENT00000000000000Z1",),
        facts=ExperienceFacts(
            polarity=("prefer", "workspace:abc"),
            recurrence_count=4,
            criticality="normal",
            created_under_intent_kind="implement",
            transcript_hash="distilled:x",
            origin="distilled",
            last_corroborated_at="2026-05-21T10:00:00Z",
        ),
    )


def _check_corpus_consolidate() -> int:
    from resonance_lattice.state.claim_lifecycle import (
        GateSignals, consolidate_corpus,
    )

    # `_make_corpus` seeds trust 0.75 + 2 distinct citations + no verdict
    # signals — a healthy claim that clears the spine's promote gate.
    # (label, state, compression_test_pass, expected next state)
    cases = [
        ("candidate + test pass → active", "candidate", True, "active"),
        ("candidate + test fail → retired", "candidate", False, "retired"),
        ("active + no test → unchanged", "active", None, "active"),
        ("stale + reverify pass → active", "stale", True, "active"),
        ("stale + reverify fail → retired", "stale", False, "retired"),
        ("retired is absorbing", "retired", True, "retired"),
    ]
    for label, state, test, want in cases:
        out = consolidate_corpus(
            _make_corpus(state),
            signals=GateSignals(compression_test_pass=test),
        )
        if out.state != want:
            print(f"[{_P}] FAIL (a) {label}: got {out.state!r}, "
                  f"want {want!r}", file=sys.stderr)
            return 1
    print(f"[{_P}] (a) corpus state machine per the §4.4 table OK",
          file=sys.stderr)
    return 0


def _check_accumulate_outcome() -> int:
    from resonance_lattice.state.claim_lifecycle import accumulate_outcome

    claim = _make_corpus(trust_corr=2.0, trust_fals=2.0)
    plus = accumulate_outcome(claim, corroboration=1.0)
    minus = accumulate_outcome(claim, falsification=2.0)
    ok = (
        plus.corroboration == 3.0 and plus.falsification == 2.0
        and minus.corroboration == 2.0 and minus.falsification == 4.0
        and abs(plus.trust - 3.0 / 5.0) < 1e-9
        and abs(minus.trust - 2.0 / 6.0) < 1e-9
    )
    if not ok:
        print(f"[{_P}] FAIL (c): accumulate drifted — plus={plus.trust:.3f} "
              f"minus={minus.trust:.3f}", file=sys.stderr)
        return 1
    print(f"[{_P}] (c) accumulate_outcome adds weight, trust derives OK",
          file=sys.stderr)
    return 0


def _check_record_verdict() -> int:
    from resonance_lattice.state.claim_lifecycle import record_verdict

    claim = _make_corpus("active")
    out = record_verdict(claim, source="user", polarity="accept")
    ok = (
        len(out.facts.verdict_signals) == 1
        and out.facts.verdict_signals[0].source == "user"
        and out.facts.verdict_signals[0].polarity == "accept"
        and out.state == claim.state            # no transition
    )
    if not ok:
        print(f"[{_P}] FAIL (d): record_verdict mis-shaped or transitioned",
              file=sys.stderr)
        return 1
    print(f"[{_P}] (d) record_verdict appends + does not transition OK",
          file=sys.stderr)
    return 0


def _check_compute_verdict_score() -> int:
    from resonance_lattice.state.claim_lifecycle import compute_verdict_score
    from resonance_lattice.store.insight import VerdictSignal

    def sig(src, pol):
        return VerdictSignal(
            source=src, polarity=pol,
            timestamp="2026-05-21T10:00:00Z", lens_id=None,
        )
    # user accept (auth 1.0) + llm reject (auth 0.3) →
    # (1.0*1 + 0.3*-1) / (1.0 + 0.3) = 0.7 / 1.3 ≈ 0.538
    score = compute_verdict_score([sig("user", "accept"), sig("llm", "reject")])
    if abs(score - 0.7 / 1.3) > 1e-9:
        print(f"[{_P}] FAIL (e): wrong weighted score {score:.4f}",
              file=sys.stderr)
        return 1
    if compute_verdict_score([]) != 0.0:
        print(f"[{_P}] FAIL (e): empty signals not 0.0", file=sys.stderr)
        return 1
    print(f"[{_P}] (e) compute_verdict_score authority-weighted OK",
          file=sys.stderr)
    return 0


def _check_retune_to_rung() -> int:
    from resonance_lattice.state.claim_lifecycle import retune_to_rung
    from resonance_lattice.memory.store import seed_tallies_for_rung

    claim = _make_experience("active")
    out = retune_to_rung(claim, "low")
    expected_corr, expected_fals = seed_tallies_for_rung("low")
    ok = (
        out.corroboration == expected_corr
        and out.falsification == expected_fals
        and out.content == claim.content                  # untouched
        and out.facts.polarity == claim.facts.polarity    # untouched
        and out.state == claim.state                      # untouched
    )
    if not ok:
        print(f"[{_P}] FAIL (g): retune drifted — {out!r}", file=sys.stderr)
        return 1
    print(f"[{_P}] (g) retune_to_rung reseeds tallies only OK",
          file=sys.stderr)
    return 0


def _check_autonomous_promotion() -> int:
    from resonance_lattice.state.claim_lifecycle import (
        GateSignals, consolidate_corpus, record_verdict,
    )

    # A freshly-promoted corpus claim has no verdict signals; a passing
    # compression test alone must take it candidate→active.
    fresh = _make_corpus("candidate")  # trust 0.75, 2 citations, no signals
    promoted = consolidate_corpus(
        fresh, signals=GateSignals(compression_test_pass=True),
    )
    if promoted.state != "active":
        print(f"[{_P}] FAIL (h): no-signal candidate did not promote — "
              f"state={promoted.state!r}", file=sys.stderr)
        return 1
    # A candidate whose verdict history is net-negative (a non-user
    # reject) is held in `candidate` even on a passing test.
    rejected = record_verdict(
        _make_corpus("candidate"), source="llm", polarity="reject",
    )
    held = consolidate_corpus(
        rejected, signals=GateSignals(compression_test_pass=True),
    )
    if held.state != "candidate":
        print(f"[{_P}] FAIL (h): net-negative candidate promoted — "
              f"state={held.state!r}", file=sys.stderr)
        return 1
    print(f"[{_P}] (h) autonomous promotion + net-negative hold OK",
          file=sys.stderr)
    return 0


def _exp(state: str, *, recurrence: int, corr: float, fals: float):
    """An experience claim with explicit state / recurrence / Beta tallies."""
    from resonance_lattice.state.claim import Claim, ExperienceFacts

    return Claim(
        claim_id="01HZEXP000000000000000002",
        source="experience",
        kind="event",
        content="asserted fact",
        created_at="2026-05-21T10:00:00Z",
        corroboration=corr,
        falsification=fals,
        trust_as_of="2026-05-21T10:00:00Z",
        state=state,
        parent_ids=(),
        facts=ExperienceFacts(
            polarity=("factual",),
            recurrence_count=recurrence,
            criticality="normal",
            created_under_intent_kind="none",
            transcript_hash="manual",
            origin="manual",
            last_corroborated_at="2026-05-21T10:00:00Z",
        ),
    )


def _check_experience_consolidate() -> int:
    from resonance_lattice.state.claim_lifecycle import consolidate_experience

    # Trust by tallies: (3,1)→0.75 (>0.5), (2,2)→0.50 (the neutral seed,
    # NOT > 0.5), (2,3)→0.40 (≥floor, <0.5), (1,3)→0.25 (<RETIRE_FLOOR 0.3).
    # (label, state, recurrence, corr, fals, expected next state)
    cases = [
        ("candidate + recurred + net-positive trust → active",
         "candidate", 2, 3.0, 1.0, "active"),
        ("candidate + single capture → held (recurrence floor)",
         "candidate", 1, 3.0, 1.0, "candidate"),
        ("candidate + recurred + trust AT neutral seed → held (strict gate)",
         "candidate", 4, 2.0, 2.0, "candidate"),
        ("candidate + recurred + trust below promote → held",
         "candidate", 4, 2.0, 3.0, "candidate"),
        ("candidate + trust below retire floor → retired",
         "candidate", 4, 1.0, 3.0, "retired"),
        ("active above floor → stays active (no demotion)",
         "active", 1, 2.0, 3.0, "active"),
        ("active below retire floor → retired",
         "active", 4, 1.0, 3.0, "retired"),
        ("retired is absorbing", "retired", 4, 3.0, 1.0, "retired"),
    ]
    for label, state, rec, corr, fals, want in cases:
        out = consolidate_experience(_exp(state, recurrence=rec,
                                          corr=corr, fals=fals))
        if out.state != want:
            print(f"[{_P}] FAIL (i) {label}: got {out.state!r}, "
                  f"want {want!r} (trust={out.trust:.3f})", file=sys.stderr)
            return 1

    # A corpus claim routed here is a programming error — fail loud, never
    # read an ExperienceFacts field off a CorpusFacts claim.
    try:
        consolidate_experience(_make_corpus("candidate"))
    except TypeError:
        pass
    else:
        print(f"[{_P}] FAIL (i): consolidate_experience accepted a corpus claim",
              file=sys.stderr)
        return 1
    print(f"[{_P}] (i) experience earning gate (recurrence + trust) OK",
          file=sys.stderr)
    return 0


def _check_drift_skips_experience() -> int:
    from resonance_lattice.state.claim_lifecycle import (
        detect_drift, propagate_drift,
    )

    # Mixed unified band: a corpus claim citing a drifted source + an
    # experience claim (no citations). detect_drift must flag only the
    # corpus row and never touch the experience claim's missing fields.
    corpus = _make_corpus("active")              # cites p0/p1, hashes h0/h1
    experience = _make_experience("active")
    band = [corpus, experience]
    fresh = {"p0": "h0_new", "p1": "h1"}          # p0 drifted
    drifted = detect_drift(band, fresh)
    if drifted != [0]:
        print(f"[{_P}] FAIL (j): detect_drift={drifted}, expected [0] "
              f"(experience row must be skipped, not crash)", file=sys.stderr)
        return 1
    updated, idx = propagate_drift(band, fresh)
    ok = (
        idx == [0]
        and updated[0].state == "stale"
        and updated[1].state == "active"          # experience untouched
        and updated[1] is experience              # identity-preserved
    )
    if not ok:
        print(f"[{_P}] FAIL (j): propagate_drift mishandled the mixed band — "
              f"idx={idx}, states={[u.state for u in updated]}",
              file=sys.stderr)
        return 1
    print(f"[{_P}] (j) drift cascade skips experience claims OK",
          file=sys.stderr)
    return 0


def _check_seeded_corpus_rederivation() -> int:
    """(k) §B BLOCKER fix: for a SEEDED corpus claim the born seed is the
    non-ledger baseline. `accumulate_outcome` (reverification / drift evidence)
    grows BOTH the tally and the seed, so a consolidate `rederive_outcome`
    (`tally = seed + ledger weight`) PRESERVES that non-ledger evidence and is
    idempotent on re-run. Experience claims carry no seed and accumulate the
    tally only."""
    from resonance_lattice.state.claim_lifecycle import (
        accumulate_outcome,
        rederive_outcome,
    )

    from ._testutil import make_corpus_claim

    c = make_corpus_claim("seeded synthesis", ["p0"], state="active")
    born = c.facts.seed_corroboration
    if born < 0.0:
        print(f"[{_P}] FAIL (k): make_corpus_claim left the seed unset",
              file=sys.stderr)
        return 1

    # Direct non-ledger evidence grows the tally AND the seed baseline.
    rev = accumulate_outcome(c, corroboration=1.0)
    if (rev.corroboration != born + 1.0
            or rev.facts.seed_corroboration != born + 1.0):
        print(f"[{_P}] FAIL (k): non-ledger evidence did not grow the seed — "
              f"tally={rev.corroboration} seed={rev.facts.seed_corroboration}",
              file=sys.stderr)
        return 1

    # Consolidate re-derivation: tally = seed + ledger weight. The +1.0
    # non-ledger corroboration survives (it lives in the seed); +0.5 is the
    # ledger weight. rederive_outcome must NOT touch the seed.
    led = rederive_outcome(
        rev, seed_corroboration=rev.facts.seed_corroboration,
        seed_falsification=rev.facts.seed_falsification, corroboration=0.5)
    if (led.corroboration != born + 1.5
            or led.facts.seed_corroboration != born + 1.0):
        print(f"[{_P}] FAIL (k): re-derivation dropped non-ledger evidence or "
              f"moved the seed — tally={led.corroboration} "
              f"seed={led.facts.seed_corroboration} (want {born + 1.5}/"
              f"{born + 1.0})", file=sys.stderr)
        return 1

    # Idempotent: same seed + same ledger weight → unchanged tally.
    led2 = rederive_outcome(
        led, seed_corroboration=led.facts.seed_corroboration,
        seed_falsification=led.facts.seed_falsification, corroboration=0.5)
    if led2.corroboration != led.corroboration:
        print(f"[{_P}] FAIL (k): re-derivation not idempotent — "
              f"{led.corroboration} → {led2.corroboration}", file=sys.stderr)
        return 1

    # Experience claim: accumulate adds the tally only, no seed, no crash.
    e = _make_experience()
    e2 = accumulate_outcome(e, corroboration=1.0)
    if e2.corroboration != e.corroboration + 1.0:
        print(f"[{_P}] FAIL (k): experience accumulate wrong "
              f"({e2.corroboration})", file=sys.stderr)
        return 1
    print(f"[{_P}] (k) seeded-corpus non-ledger evidence survives re-derivation "
          f"+ idempotent OK", file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_corpus_consolidate,
        _check_accumulate_outcome,
        _check_record_verdict,
        _check_compute_verdict_score,
        _check_retune_to_rung,
        _check_autonomous_promotion,
        _check_experience_consolidate,
        _check_drift_skips_experience,
        _check_seeded_corpus_rederivation,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print(f"[{_P}] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
