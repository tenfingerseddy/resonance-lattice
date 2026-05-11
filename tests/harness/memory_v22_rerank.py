"""memory_v22_rerank — manifesto recall re-rank contracts.

Pins architecture §"Recall — Scoring formula" + §"Layer manifesto scoring
factors as a re-rank over the existing gates". Six contracts:

  (a) Default behaviour unchanged — `intent_kind` omitted (or `none`)
      keeps cosine ordering. Every existing memory_v21_recall contract
      pins this.

  (b) Debugging intent + an `avoid` row outranks a `prefer` row of equal
      cosine — `valence_match` favours warnings when about to act under
      a debug profile (avoid coefficient 1.3 in the debugging row).

  (c) Designing intent biases toward `principle` over `event` at equal
      cosine — `level_match` for the designing profile (principle 1.3
      vs event 0.7).

  (d) Confidence floor punishes `low` more than `verified` at equal
      cosine + level — `low/event` is 0.6 vs `verified/event` 1.0.

  (e) Strength factor — recurrence > 1 outranks recurrence == 1 at equal
      everything else (log(1+rec) is monotonically increasing).

  (f) `severe avoid` rows survive recency decay — even with a stale
      `last_corroborated_at`, the severe floor keeps the recency factor
      ≥ 0.6 so the row stays surfaceable.

Hermetic — fixed cosines + ZeroEncoder so we exercise rerank math directly,
without running the encoder.
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
from resonance_lattice.memory.store import Row


def _row(
    *,
    text: str = "row",
    polarity: list[str] | None = None,
    recurrence_count: int = 5,
    level: str = "event",
    criticality: str = "normal",
    confidence: str = "medium",
    last_corroborated_at: str = "2026-05-07T00:00:00Z",
) -> Row:
    return Row(
        row_id=text,
        text=text,
        polarity=polarity if polarity is not None else ["factual", "workspace:abc123"],
        recurrence_count=recurrence_count,
        created_at=last_corroborated_at,
        last_corroborated_at=last_corroborated_at,
        transcript_hash="manual",
        is_bad=False,
        level=level,
        criticality=criticality,
        confidence=confidence,
        origin="manual",
    )


def _check_default_unchanged() -> int:
    a = _row(text="A")
    b = _row(text="B")
    hits = [RecallHit(row=a, cosine=0.9), RecallHit(row=b, cosine=0.95)]
    # No intent_kind → identity re-rank: rerank() with neutral profile
    # returns scoring-driven order, but the same order callers got
    # from cosine-descending must hold when no intent is provided.
    # `rank()` short-circuits the rerank call entirely on intent_kind=None;
    # exercising that path lives in memory_v21_recall (which still passes).
    out = rerank(hits, intent_kind="none")
    if [h.row.row_id for h in out] != ["B", "A"]:
        # rerank with intent_kind='none' is neutral — order should follow
        # strength; both rows have identical strength; cosine breaks tie.
        # B has higher cosine, so it should rank first.
        print(f"[memory_v22_rerank] FAIL (a): {[h.row.row_id for h in out]}",
              file=sys.stderr)
        return 1
    print("[memory_v22_rerank] (a) neutral profile preserves cosine order OK",
          file=sys.stderr)
    return 0


def _check_avoid_promoted_under_debug() -> int:
    avoid = _row(
        text="avoid",
        polarity=["avoid", "workspace:abc123"],
    )
    prefer = _row(
        text="prefer",
        polarity=["prefer", "workspace:abc123"],
    )
    # Equal cosine. Debugging profile gives avoid 1.3, prefer 0.9.
    hits = [RecallHit(row=prefer, cosine=0.9), RecallHit(row=avoid, cosine=0.9)]
    out = rerank(hits, intent_kind="debug")
    if out[0].row.row_id != "avoid":
        print(f"[memory_v22_rerank] FAIL (b): order={[h.row.row_id for h in out]}",
              file=sys.stderr)
        return 1
    print("[memory_v22_rerank] (b) debug intent promotes avoid OK",
          file=sys.stderr)
    return 0


def _check_principle_promoted_under_design() -> int:
    event = _row(text="event-row", level="event")
    principle = _row(text="principle-row", level="principle")
    hits = [RecallHit(row=event, cosine=0.9), RecallHit(row=principle, cosine=0.9)]
    out = rerank(hits, intent_kind="design")
    if out[0].row.row_id != "principle-row":
        print(f"[memory_v22_rerank] FAIL (c): order={[h.row.row_id for h in out]}",
              file=sys.stderr)
        return 1
    print("[memory_v22_rerank] (c) design intent promotes principle OK",
          file=sys.stderr)
    return 0


def _check_confidence_floor() -> int:
    verified = _row(text="verified", confidence="verified")
    low = _row(text="low", confidence="low")
    hits = [RecallHit(row=low, cosine=0.9), RecallHit(row=verified, cosine=0.9)]
    out = rerank(hits, intent_kind="implement")
    if out[0].row.row_id != "verified":
        print(f"[memory_v22_rerank] FAIL (d): order={[h.row.row_id for h in out]}",
              file=sys.stderr)
        return 1
    print("[memory_v22_rerank] (d) verified outranks low at equal cosine OK",
          file=sys.stderr)
    return 0


def _check_strength_recurrence() -> int:
    one = _row(text="rec1", recurrence_count=1)
    many = _row(text="rec20", recurrence_count=20)
    hits = [RecallHit(row=one, cosine=0.9), RecallHit(row=many, cosine=0.9)]
    out = rerank(hits, intent_kind="implement")
    if out[0].row.row_id != "rec20":
        print(f"[memory_v22_rerank] FAIL (e): order={[h.row.row_id for h in out]}",
              file=sys.stderr)
        return 1
    print("[memory_v22_rerank] (e) higher recurrence outranks lower OK",
          file=sys.stderr)
    return 0


def _check_severe_floor() -> int:
    # 5 years stale, severe criticality — recency factor must stay ≥ 0.6
    # via the severe floor (architecture §"Field interactions worth knowing").
    stale = _dt.datetime(2021, 1, 1, tzinfo=_dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    factor = recency_factor(stale, criticality="severe")
    if factor < 0.6 - 1e-9:
        print(f"[memory_v22_rerank] FAIL (f): severe factor={factor!r}",
              file=sys.stderr)
        return 1
    # And the un-floored low criticality should be much smaller.
    weak = recency_factor(stale, criticality="low")
    if weak > 0.6:
        print(f"[memory_v22_rerank] FAIL (f): low factor={weak!r} (should be tiny)",
              file=sys.stderr)
        return 1
    print(f"[memory_v22_rerank] (f) severe floor holds (factor={factor:.3f}) OK",
          file=sys.stderr)
    return 0


def _check_effective_score_components() -> int:
    """Sanity-check that effective_score = cosine × strength × valence
    × level × confidence_floor for a known input."""
    row = _row(
        text="ck",
        polarity=["factual", "workspace:abc123"],
        recurrence_count=5,
        level="learning",
        criticality="high",
        confidence="high",
    )
    cosine = 0.85
    score = effective_score(row, cosine, intent_kind="design")
    # Manual recomputation under same parameters.
    from resonance_lattice.memory.rerank import (
        confidence_floor,
        level_match,
        strength,
        valence_match,
    )
    expected = (
        cosine
        * strength(row)
        * valence_match("design", row)
        * level_match("design", row)
        * confidence_floor(row)
    )
    if abs(score - expected) > 1e-9:
        print(f"[memory_v22_rerank] FAIL (g): score={score!r} expected={expected!r}",
              file=sys.stderr)
        return 1
    print(f"[memory_v22_rerank] (g) effective_score composition OK "
          f"(={score:.4f})", file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_default_unchanged,
        _check_avoid_promoted_under_debug,
        _check_principle_promoted_under_design,
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
