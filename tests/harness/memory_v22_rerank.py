"""memory_v22_rerank — manifesto recall re-rank contracts.

Pins architecture §"Recall — Scoring formula" + §"Layer manifesto scoring
factors as a re-rank over the existing gates". Contracts:

  (a) Default behaviour unchanged — `intent_kind` omitted (or `none`)
      keeps cosine ordering. Every existing memory_v21_recall contract
      pins this.

  (b) Debugging intent + an `avoid` claim outranks a `prefer` claim of
      equal cosine — `valence_match` favours warnings when about to act
      under a debug profile (avoid coefficient 1.3 in the debugging row).

  (d) Confidence floor punishes `low` more than `verified` at equal
      cosine — `low` is 0.6 vs `verified` 1.0.

  (e) Strength factor — recurrence > 1 outranks recurrence == 1 at equal
      everything else (log(1+rec) is monotonically increasing).

  (f) `severe avoid` claims survive recency decay — even with a stale
      `last_corroborated_at`, the severe floor keeps the recency factor
      ≥ 0.6 so the claim stays surfaceable.

  (g) `effective_score` composition — `cosine × strength × valence_match
      × confidence_floor` for an experience claim.

Hermetic — fixed cosines so we exercise rerank math directly, without
running the encoder.
"""

from __future__ import annotations

import datetime as _dt
import sys

from resonance_lattice.memory.recall import RecallHit
from resonance_lattice.memory.rerank import (
    effective_score,
    recency_factor,
    rerank,
)
from resonance_lattice.memory.store import seed_tallies_for_rung
from resonance_lattice.state.claim import Claim, ExperienceFacts


def _claim(
    *,
    text: str = "row",
    polarity: list[str] | None = None,
    recurrence_count: int = 5,
    criticality: str = "normal",
    confidence: str = "medium",
    last_corroborated_at: str = "2026-05-07T00:00:00Z",
) -> Claim:
    corr, fals = seed_tallies_for_rung(confidence)
    return Claim(
        claim_id=text,
        source="experience",
        kind="event",
        content=text,
        created_at=last_corroborated_at,
        corroboration=corr,
        falsification=fals,
        trust_as_of="",
        state="active",
        parent_ids=(),
        facts=ExperienceFacts(
            polarity=tuple(
                polarity if polarity is not None
                else ["factual", "workspace:abc123"]
            ),
            recurrence_count=recurrence_count,
            criticality=criticality,
            created_under_intent_kind="none",
            transcript_hash="manual",
            origin="manual",
            last_corroborated_at=last_corroborated_at,
            is_bad=False,
        ),
    )


def _check_default_unchanged() -> int:
    a = _claim(text="A")
    b = _claim(text="B")
    hits = [RecallHit(claim=a, cosine=0.9), RecallHit(claim=b, cosine=0.95)]
    out = rerank(hits, intent_kind="none")
    if [h.claim.claim_id for h in out] != ["B", "A"]:
        print(f"[memory_v22_rerank] FAIL (a): "
              f"{[h.claim.claim_id for h in out]}", file=sys.stderr)
        return 1
    print("[memory_v22_rerank] (a) neutral profile preserves cosine order OK",
          file=sys.stderr)
    return 0


def _check_avoid_promoted_under_debug() -> int:
    avoid = _claim(text="avoid", polarity=["avoid", "workspace:abc123"])
    prefer = _claim(text="prefer", polarity=["prefer", "workspace:abc123"])
    hits = [
        RecallHit(claim=prefer, cosine=0.9), RecallHit(claim=avoid, cosine=0.9),
    ]
    out = rerank(hits, intent_kind="debug")
    if out[0].claim.claim_id != "avoid":
        print(f"[memory_v22_rerank] FAIL (b): "
              f"order={[h.claim.claim_id for h in out]}", file=sys.stderr)
        return 1
    print("[memory_v22_rerank] (b) debug intent promotes avoid OK",
          file=sys.stderr)
    return 0


def _check_confidence_floor() -> int:
    verified = _claim(text="verified", confidence="verified")
    low = _claim(text="low", confidence="low")
    hits = [
        RecallHit(claim=low, cosine=0.9), RecallHit(claim=verified, cosine=0.9),
    ]
    out = rerank(hits, intent_kind="implement")
    if out[0].claim.claim_id != "verified":
        print(f"[memory_v22_rerank] FAIL (d): "
              f"order={[h.claim.claim_id for h in out]}", file=sys.stderr)
        return 1
    print("[memory_v22_rerank] (d) verified outranks low at equal cosine OK",
          file=sys.stderr)
    return 0


def _check_strength_recurrence() -> int:
    one = _claim(text="rec1", recurrence_count=1)
    many = _claim(text="rec20", recurrence_count=20)
    hits = [
        RecallHit(claim=one, cosine=0.9), RecallHit(claim=many, cosine=0.9),
    ]
    out = rerank(hits, intent_kind="implement")
    if out[0].claim.claim_id != "rec20":
        print(f"[memory_v22_rerank] FAIL (e): "
              f"order={[h.claim.claim_id for h in out]}", file=sys.stderr)
        return 1
    print("[memory_v22_rerank] (e) higher recurrence outranks lower OK",
          file=sys.stderr)
    return 0


def _check_severe_floor() -> int:
    stale = _dt.datetime(2021, 1, 1, tzinfo=_dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    factor = recency_factor(stale, criticality="severe")
    if factor < 0.6 - 1e-9:
        print(f"[memory_v22_rerank] FAIL (f): severe factor={factor!r}",
              file=sys.stderr)
        return 1
    weak = recency_factor(stale, criticality="low")
    if weak > 0.6:
        print(f"[memory_v22_rerank] FAIL (f): low factor={weak!r} "
              f"(should be tiny)", file=sys.stderr)
        return 1
    print(f"[memory_v22_rerank] (f) severe floor holds (factor={factor:.3f}) OK",
          file=sys.stderr)
    return 0


def _check_effective_score_components() -> int:
    """Sanity-check that effective_score = cosine × strength × valence
    × confidence_floor for a known experience-claim input."""
    claim = _claim(
        text="ck",
        polarity=["factual", "workspace:abc123"],
        recurrence_count=5,
        criticality="high",
        confidence="high",
    )
    cosine = 0.85
    score = effective_score(claim, cosine, intent_kind="design")
    from resonance_lattice.memory.rerank import (
        confidence_floor,
        strength,
        valence_match,
    )
    expected = (
        cosine
        * strength(claim)
        * valence_match("design", claim)
        * confidence_floor(claim)
    )
    if abs(score - expected) > 1e-9:
        print(f"[memory_v22_rerank] FAIL (g): score={score!r} "
              f"expected={expected!r}", file=sys.stderr)
        return 1
    print(f"[memory_v22_rerank] (g) effective_score composition OK "
          f"(={score:.4f})", file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_default_unchanged,
        _check_avoid_promoted_under_debug,
        _check_confidence_floor,
        _check_strength_recurrence,
        _check_severe_floor,
        _check_effective_score_components,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v22_rerank] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
