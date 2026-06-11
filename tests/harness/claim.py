"""claim — the unified Claim record.

Pins the `Claim` record, the typed source-discriminated facts
(`ExperienceFacts` / `CorpusFacts`), and the `evolve` rewriter:

  (a) An experience Claim constructs with ExperienceFacts; `trust`
      derives the Beta mean; `is_bad` defaults False.
  (b) A corpus Claim constructs with CorpusFacts; defaults hold.
  (c) Claim and the facts records are frozen — mutation raises.
  (e) The discrimination invariant — a Claim whose `facts` type does not
      match its `source` is rejected at construction.
  (f) An unknown `source` value is rejected — not silently treated as
      corpus by the discrimination fallthrough.
  (g) `evolve` routes keyword changes to the core record or the typed
      `facts` sub-record, leaves the frozen original untouched, and
      rejects an unknown field; `confidence` derives from `trust`.

Hermetic — pure construction, no I/O.
"""

from __future__ import annotations

import dataclasses
import sys

from ._testutil import check_guarantee

_P = "claim"


def _experience_claim():
    from resonance_lattice.state.claim import Claim, ExperienceFacts

    return Claim(
        claim_id="01HZCLAIM000000000000000001",
        source="experience",
        kind="pattern",
        content="prefer the standard library",
        created_at="2026-05-18T00:00:00Z",
        corroboration=3.0,
        falsification=1.0,
        trust_as_of="",
        state="active",
        parent_ids=("01HZEVENT0000000000000000A",),
        facts=ExperienceFacts(
            polarity=("prefer", "workspace:abc"),
            recurrence_count=4,
            criticality="normal",
            created_under_intent_kind="implement",
            transcript_hash="distilled:arrow1:x",
            origin="distilled",
            last_corroborated_at="2026-05-18T00:00:00Z",
        ),
    )


def _corpus_claim():
    from resonance_lattice.state.claim import Claim, CorpusFacts

    return Claim(
        claim_id="01HZCLAIM000000000000000002",
        source="corpus",
        kind="synthesis",
        content="Tokens persist in Redis.",
        created_at="2026-05-18T00:00:00Z",
        corroboration=2.0,
        falsification=2.0,
        trust_as_of="",
        state="candidate",
        parent_ids=(),
        facts=CorpusFacts(
            citations=(),
            content_fingerprint="abcdef0123456789",
            source_model_hash="model-x",
            source_passage_hashes=("p0", "p1"),
        ),
    )


def _check_experience() -> int:
    from resonance_lattice.state.claim import ExperienceFacts

    c = _experience_claim()
    ok = (
        c.source == "experience"
        and isinstance(c.facts, ExperienceFacts)
        and c.facts.is_bad is False                 # default
        and abs(c.trust - 0.75) < 1e-9              # beta_mean(3, 1)
    )
    return 0 if check_guarantee(ok, "(a) experience Claim + trust", _P) else 1


def _check_corpus() -> int:
    from resonance_lattice.state.claim import CorpusFacts

    c = _corpus_claim()
    ok = (
        c.source == "corpus"
        and isinstance(c.facts, CorpusFacts)
        and c.facts.stale_if_sources_drift is True  # default
        and c.facts.verdict_signals == ()           # default
        and c.trust == 0.5
    )
    return 0 if check_guarantee(ok, "(b) corpus Claim + defaults", _P) else 1


def _check_frozen() -> int:
    c = _experience_claim()
    claim_frozen = facts_frozen = False
    try:
        c.content = "mutated"                       # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        claim_frozen = True
    try:
        c.facts.recurrence_count = 99               # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        facts_frozen = True
    return 0 if check_guarantee(
        claim_frozen and facts_frozen, "(c) Claim + facts frozen", _P) else 1


def _check_facts_discrimination() -> int:
    from resonance_lattice.state.claim import Claim, CorpusFacts

    raised = False
    try:
        Claim(
            claim_id="01HZCLAIM000000000000000003",
            source="experience",                   # mismatched with facts
            kind="pattern",
            content="c",
            created_at="2026-05-18T00:00:00Z",
            corroboration=1.0,
            falsification=1.0,
            trust_as_of="",
            state="active",
            parent_ids=(),
            facts=CorpusFacts(
                citations=(),
                content_fingerprint="f",
                source_model_hash="m",
                source_passage_hashes=(),
            ),
        )
    except TypeError:
        raised = True
    return 0 if check_guarantee(
        raised, "(e) source/facts mismatch rejected", _P) else 1


def _check_unknown_source() -> int:
    from resonance_lattice.state.claim import Claim, CorpusFacts

    raised = False
    try:
        Claim(
            claim_id="01HZCLAIM000000000000000004",
            source="experence",                   # typo — unknown source
            kind="synthesis",
            content="c",
            created_at="2026-05-18T00:00:00Z",
            corroboration=1.0,
            falsification=1.0,
            trust_as_of="",
            state="candidate",
            parent_ids=(),
            facts=CorpusFacts(
                citations=(),
                content_fingerprint="f",
                source_model_hash="m",
                source_passage_hashes=(),
            ),
        )
    except ValueError:
        raised = True
    return 0 if check_guarantee(
        raised, "(f) unknown source rejected", _P) else 1


def _check_evolve() -> int:
    from resonance_lattice.state.claim import evolve

    c = _experience_claim()
    c2 = evolve(c, content="edited", is_bad=True, corroboration=9.0)
    unknown_rejected = facts_rejected = immutable_rejected = False
    try:
        evolve(c, nonsense=1)
    except ValueError:
        unknown_rejected = True
    try:
        evolve(c, facts=None)
    except ValueError:
        facts_rejected = True
    try:
        evolve(c, claim_id="01HZCLAIM00000000000000OTHER")
    except ValueError:
        immutable_rejected = True
    ok = (
        c2.content == "edited"                      # core field routed
        and c2.facts.is_bad is True                 # facts field routed
        and c2.corroboration == 9.0
        and c.content != "edited"                   # frozen original untouched
        and c.facts.is_bad is False
        and c2.confidence in ("low", "medium", "high", "verified")
        and unknown_rejected
        and facts_rejected
        and immutable_rejected                      # claim_id is immutable
    )
    return 0 if check_guarantee(ok, "(g) evolve + confidence", _P) else 1


def run() -> int:
    failures = 0
    for check in (
        _check_experience,
        _check_corpus,
        _check_frozen,
        _check_facts_discrimination,
        _check_unknown_source,
        _check_evolve,
    ):
        failures += check()
    if failures:
        print(f"[{_P}] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print(f"[{_P}] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
