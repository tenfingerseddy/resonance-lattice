"""memory_v22_forget — Forget operation contracts.

Pins architecture §"Forget" — five drop conditions and five protections.
Eight contracts:

  (a) Decay below floor — old + low-recurrence + normal-criticality drops.

  (b) Redundant after promotion — event with confident pattern parent and
      no independent strength signal drops; bypasses active_provenance.

  (c) Falsified by outcomes — low-confidence row with ≥3 failed attributions
      and ≤1 success drops; primary/secondary tiers count, incidental doesn't.

  (d) Trivial from start — ALL five sub-conditions must hold.

  (e) Active provenance protects — referenced parents stay (except the
      condition-2 carve-out).

  (f) Severe avoid protects — once-burned-twice-shy floor.

  (g) User-declared protects — origin: manual stays.

  (h) Recently active protects — corroborated within window stays.

Hermetic — synthetic rows + temp dir; no encoder, no LLM.
"""

from __future__ import annotations

import datetime as _dt
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

from resonance_lattice.memory.forget import (
    DEFAULT_RECENT_ACTIVITY_DAYS,
    forget_pass,
)
from resonance_lattice.memory.store import Row
from resonance_lattice.state import (
    Attribution,
    CriterionCheck,
    OutcomeRecord,
)


def _row(
    *,
    row_id: str = "01HZ0000000000000000000001",
    text: str = "row",
    polarity: list[str] | None = None,
    recurrence_count: int = 1,
    created_at: str = "2025-01-01T00:00:00Z",
    last_corroborated_at: str | None = None,
    is_bad: bool = False,
    level: str = "event",
    criticality: str = "normal",
    confidence: str = "medium",
    parent_ids: list[str] | None = None,
    origin: str = "manual",
) -> Row:
    return Row(
        row_id=row_id,
        text=text,
        polarity=polarity or ["factual", "workspace:abc123"],
        recurrence_count=recurrence_count,
        created_at=created_at,
        last_corroborated_at=last_corroborated_at or created_at,
        transcript_hash=("manual" if origin == "manual" else "distilled:x"),
        is_bad=is_bad,
        level=level,
        criticality=criticality,
        confidence=confidence,
        parent_ids=parent_ids or [],
        origin=origin,
    )


_NOW = _dt.datetime(2026, 5, 8, tzinfo=_dt.timezone.utc)


def _stale(days: int) -> str:
    return (_NOW - _dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _verdict_for(row_id: str, verdicts) -> object:
    return next(v for v in verdicts if v.row_id == row_id)


def _check_decay() -> int:
    # Distilled origin so user_declared protection doesn't fire. Old +
    # recurrence 2 + normal criticality skips trivial (which requires
    # recurrence 1) and lands on the decay branch.
    old = _row(
        row_id="01HZ_OLD", recurrence_count=2, origin="distilled",
        created_at=_stale(365), last_corroborated_at=_stale(365),
    )
    fresh = _row(
        row_id="01HZ_FRESH", recurrence_count=10, origin="distilled",
        created_at=_stale(1), last_corroborated_at=_stale(1),
    )
    verdicts = forget_pass([old, fresh], now=_NOW)
    if not _verdict_for("01HZ_OLD", verdicts).drop:
        print(f"[memory_v22_forget] FAIL (a): old not dropped: "
              f"{_verdict_for('01HZ_OLD', verdicts)!r}", file=sys.stderr)
        return 1
    if _verdict_for("01HZ_FRESH", verdicts).drop:
        print(f"[memory_v22_forget] FAIL (a): fresh dropped: "
              f"{_verdict_for('01HZ_FRESH', verdicts)!r}", file=sys.stderr)
        return 1
    print("[memory_v22_forget] (a) decay below floor OK", file=sys.stderr)
    return 0


def _check_redundant_after_promotion() -> int:
    # Event with a confident pattern parent + no independent signal drops.
    event = _row(
        row_id="01HZ_EVENT", level="event", recurrence_count=2,
        created_at=_stale(2), last_corroborated_at=_stale(60),
    )
    confident_pattern = _row(
        row_id="01HZ_PATTERN", level="pattern", confidence="high",
        parent_ids=["01HZ_EVENT"], origin="distilled",
        created_at=_stale(1), last_corroborated_at=_stale(1),
    )
    verdicts = forget_pass([event, confident_pattern], now=_NOW)
    ev_verdict = _verdict_for("01HZ_EVENT", verdicts)
    if not ev_verdict.drop or ev_verdict.condition != "redundant":
        print(f"[memory_v22_forget] FAIL (b): {ev_verdict!r}", file=sys.stderr)
        return 1
    if _verdict_for("01HZ_PATTERN", verdicts).drop:
        print(f"[memory_v22_forget] FAIL (b): pattern dropped", file=sys.stderr)
        return 1

    # Independent strength signal blocks redundancy drop.
    strong_event = replace(event, recurrence_count=20)
    verdicts2 = forget_pass([strong_event, confident_pattern], now=_NOW)
    if _verdict_for("01HZ_EVENT", verdicts2).drop:
        print(f"[memory_v22_forget] FAIL (b): strong event dropped despite "
              f"recurrence", file=sys.stderr)
        return 1
    print("[memory_v22_forget] (b) redundant-after-promotion OK", file=sys.stderr)
    return 0


def _check_falsified() -> int:
    losing_row = _row(
        row_id="01HZ_LOSER", confidence="low", recurrence_count=5,
        origin="distilled",
        created_at=_stale(60), last_corroborated_at=_stale(60),
    )

    def _make_outcome(verdict: str, tier: str) -> OutcomeRecord:
        return OutcomeRecord(
            intent_id="t",
            intent_level="task",
            criterion_checks=[CriterionCheck(
                criterion_text="x", measure="user_confirms", verdict=verdict,
            )],
            roll_up_verdict=verdict,
            attribution=[Attribution(row_id="01HZ_LOSER", tier=tier)],
            resolved_at="2026-05-01T00:00:00Z",
        )
    outcomes = [
        _make_outcome("not_satisfied", "primary"),
        _make_outcome("not_satisfied", "primary"),
        _make_outcome("not_satisfied", "secondary"),
        # An incidental success doesn't count toward the 1-success threshold.
        _make_outcome("satisfied", "incidental"),
    ]
    verdicts = forget_pass([losing_row], outcomes=outcomes, now=_NOW)
    if not _verdict_for("01HZ_LOSER", verdicts).drop:
        print(f"[memory_v22_forget] FAIL (c): falsification didn't fire: "
              f"{_verdict_for('01HZ_LOSER', verdicts)!r}", file=sys.stderr)
        return 1
    if _verdict_for("01HZ_LOSER", verdicts).condition != "falsified":
        print(f"[memory_v22_forget] FAIL (c): wrong condition: "
              f"{_verdict_for('01HZ_LOSER', verdicts).condition!r}",
              file=sys.stderr)
        return 1
    print("[memory_v22_forget] (c) falsified by outcomes OK", file=sys.stderr)
    return 0


def _check_trivial() -> int:
    # All five sub-conditions hold → drop with `trivial`. Distilled origin
    # to bypass user_declared protection.
    trivial = _row(
        row_id="01HZ_TRIVIAL", recurrence_count=1, criticality="low",
        origin="distilled",
        created_at=_stale(21), last_corroborated_at=_stale(21),
    )
    verdicts = forget_pass([trivial], now=_NOW)
    v = _verdict_for("01HZ_TRIVIAL", verdicts)
    if not v.drop or v.condition != "trivial":
        print(f"[memory_v22_forget] FAIL (d): {v!r}", file=sys.stderr)
        return 1
    # Break one sub-condition (recurrence) — drop should not fire as trivial.
    less_trivial = replace(trivial, recurrence_count=2)
    verdicts2 = forget_pass([less_trivial], now=_NOW)
    v2 = _verdict_for("01HZ_TRIVIAL", verdicts2)
    if v2.condition == "trivial":
        print(f"[memory_v22_forget] FAIL (d): trivial fired with recurrence=2",
              file=sys.stderr)
        return 1
    print("[memory_v22_forget] (d) trivial-from-start OK", file=sys.stderr)
    return 0


def _check_active_provenance() -> int:
    parent = _row(row_id="01HZ_P1", origin="manual",
                  created_at=_stale(30), last_corroborated_at=_stale(30))
    child = _row(row_id="01HZ_C1", origin="distilled",
                 parent_ids=["01HZ_P1"],
                 created_at=_stale(30), last_corroborated_at=_stale(30))
    verdicts = forget_pass([parent, child], now=_NOW)
    pv = _verdict_for("01HZ_P1", verdicts)
    if pv.drop:
        print(f"[memory_v22_forget] FAIL (e): provenance parent dropped: "
              f"{pv!r}", file=sys.stderr)
        return 1
    print("[memory_v22_forget] (e) active provenance protects OK",
          file=sys.stderr)
    return 0


def _check_severe_avoid() -> int:
    sticky = _row(
        row_id="01HZ_STICKY",
        polarity=["avoid", "workspace:abc123"],
        criticality="severe",
        confidence="low", recurrence_count=1,
        created_at=_stale(365), last_corroborated_at=_stale(365),
    )
    verdicts = forget_pass([sticky], now=_NOW)
    if _verdict_for("01HZ_STICKY", verdicts).drop:
        print(f"[memory_v22_forget] FAIL (f): severe avoid dropped",
              file=sys.stderr)
        return 1
    print("[memory_v22_forget] (f) severe avoid protects OK", file=sys.stderr)
    return 0


def _check_user_declared() -> int:
    declared = _row(
        row_id="01HZ_USER", origin="manual", recurrence_count=1,
        created_at=_stale(120), last_corroborated_at=_stale(120),
    )
    verdicts = forget_pass([declared], now=_NOW)
    if _verdict_for("01HZ_USER", verdicts).drop:
        print(f"[memory_v22_forget] FAIL (g): user-declared dropped",
              file=sys.stderr)
        return 1
    print("[memory_v22_forget] (g) user-declared protects OK", file=sys.stderr)
    return 0


def _check_recently_active() -> int:
    fresh = _row(
        row_id="01HZ_FRESH", recurrence_count=1, criticality="low",
        origin="distilled",
        created_at=_stale(60),
        last_corroborated_at=_stale(DEFAULT_RECENT_ACTIVITY_DAYS - 1),
    )
    verdicts = forget_pass([fresh], now=_NOW)
    if _verdict_for("01HZ_FRESH", verdicts).drop:
        print(f"[memory_v22_forget] FAIL (h): recently-active dropped",
              file=sys.stderr)
        return 1
    print("[memory_v22_forget] (h) recently-active protects OK",
          file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_decay,
        _check_redundant_after_promotion,
        _check_falsified,
        _check_trivial,
        _check_active_provenance,
        _check_severe_avoid,
        _check_user_declared,
        _check_recently_active,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v22_forget] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
