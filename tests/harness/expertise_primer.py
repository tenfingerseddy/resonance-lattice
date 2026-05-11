"""expertise_primer — synthesis of memory + intent into a session-start primer.

Pins the v0 expertise primer's render contracts:

  (a) Empty inputs → both sections present with `_(none)_` body.

  (b) Active intents are surfaced; satisfied / abandoned / superseded
      intents are filtered out.

  (c) Intent ordering: status priority (active > blocked > proposed),
      then level priority (direction > goal > task > step), then
      created_at ascending so long-standing intents stay visible.

  (d) Memory rows: intent-shaped levels (direction/goal/task/step) are
      filtered out — they live in the live store and surface in the
      Active intents section. `is_bad` rows are dropped.

  (e) Memory ordering: confidence × level × recurrence — verified
      principles before high learnings before medium patterns.

  (f) Caps respected: `max_intents` and `max_memory_rows` truncate.

  (g) Long row text is ellipsised at the configured char cap.

Hermetic — no encoder, no LLM, no I/O.
"""

from __future__ import annotations

import sys

import numpy as np


def _row(
    row_id: str, *, level: str = "event", text: str = "row text",
    confidence: str = "medium", recurrence_count: int = 1,
    is_bad: bool = False, polarity: list[str] | None = None,
) -> object:
    from resonance_lattice.memory.store import Row

    return Row(
        row_id=row_id,
        text=text,
        polarity=polarity or ["factual", "workspace:abc123"],
        recurrence_count=recurrence_count,
        created_at="2026-05-01T00:00:00Z",
        last_corroborated_at="2026-05-01T00:00:00Z",
        transcript_hash="manual",
        is_bad=is_bad,
        level=level,
        criticality="normal",
        confidence=confidence,
        parent_ids=[],
        origin="manual",
    )


def _intent(
    intent_id: str, *, level: str = "task", text: str = "do the thing",
    status: str = "active", created_at: str = "2026-05-09T00:00:00Z",
) -> object:
    from resonance_lattice.state.intent import LiveIntent

    return LiveIntent(
        intent_id=intent_id,
        level=level,
        text=text,
        parent_ids=[],
        blocks=[],
        stance="do",
        achievability="medium",
        status=status,
        success_criteria=[],
        constraints=[],
        created_under_intent_kind="implement",
        created_at=created_at,
        updated_at=created_at,
    )


def _check_empty_inputs() -> int:
    from resonance_lattice.expertise import render_expertise_primer

    body = render_expertise_primer(
        intents=[], memory_rows=[], now="2026-05-09T00:00:00Z",
    )
    if "## Active intents (0)" not in body or "## Project memory (top 0)" not in body:
        print(f"[expertise_primer] FAIL (a): empty headings missing\n{body}",
              file=sys.stderr)
        return 1
    if body.count("_(none)_") != 2:
        print(f"[expertise_primer] FAIL (a): expected two _(none)_ markers\n{body}",
              file=sys.stderr)
        return 1
    print("[expertise_primer] (a) empty inputs render OK", file=sys.stderr)
    return 0


def _check_intent_filtering_and_order() -> int:
    """(b) terminal statuses excluded; (c) status > level > created_at order."""
    from resonance_lattice.expertise import render_expertise_primer

    intents = [
        _intent("01HZSAT", text="satisfied-intent", status="satisfied"),
        _intent("01HZABN", text="abandoned-intent", status="abandoned"),
        _intent("01HZSUP", text="superseded-intent", status="superseded"),
        _intent("01HZBLK", text="blocked-intent", status="blocked",
                created_at="2026-05-09T01:00:00Z"),
        _intent("01HZACT", text="active-intent", status="active",
                created_at="2026-05-09T02:00:00Z"),
        _intent("01HZPRO", text="proposed-intent", status="proposed",
                created_at="2026-05-09T03:00:00Z"),
        _intent("01HZGOAL", text="active-goal", status="active",
                level="goal", created_at="2026-05-09T04:00:00Z"),
    ]
    body = render_expertise_primer(intents=intents, memory_rows=[])
    for excluded in ("satisfied-intent", "abandoned-intent",
                     "superseded-intent"):
        if excluded in body:
            print(f"[expertise_primer] FAIL (b): {excluded!r} not filtered\n{body}",
                  file=sys.stderr)
            return 1

    # (c) Order: active-goal (active+goal) → active-intent (active+task) →
    # blocked-intent (blocked) → proposed-intent (proposed)
    expected_order = [
        "active-goal",
        "active-intent",
        "blocked-intent",
        "proposed-intent",
    ]
    indices = [body.index(s) for s in expected_order]
    if indices != sorted(indices):
        print(f"[expertise_primer] FAIL (c): wrong intent order\n"
              f"expected={expected_order} indices={indices}\n{body}",
              file=sys.stderr)
        return 1
    print("[expertise_primer] (b) terminal statuses filtered + (c) "
          "status > level > created_at order OK", file=sys.stderr)
    return 0


def _check_memory_filtering_and_order() -> int:
    """(d) intent-shaped levels + is_bad filtered; (e) confidence × level
    × recurrence ordering."""
    from resonance_lattice.expertise import render_expertise_primer

    rows = [
        _row("01HZTASK", level="task", text="intent-shaped task"),
        _row("01HZGOAL", level="goal", text="intent-shaped goal"),
        _row("01HZBAD", text="bad-row", is_bad=True),
        _row("01HZP1", level="principle", text="verified-principle",
             confidence="verified", recurrence_count=1),
        _row("01HZL1", level="learning", text="high-learning",
             confidence="high", recurrence_count=1),
        _row("01HZE1", level="event", text="medium-event-recur5",
             confidence="medium", recurrence_count=5),
        _row("01HZE2", level="event", text="medium-event-recur1",
             confidence="medium", recurrence_count=1),
    ]
    body = render_expertise_primer(intents=[], memory_rows=rows)
    for excluded in ("intent-shaped task", "intent-shaped goal", "bad-row"):
        if excluded in body:
            print(f"[expertise_primer] FAIL (d): {excluded!r} not filtered\n{body}",
                  file=sys.stderr)
            return 1
    # (e) verified > high > medium-recur5 > medium-recur1
    expected_order = [
        "verified-principle",
        "high-learning",
        "medium-event-recur5",
        "medium-event-recur1",
    ]
    indices = [body.index(s) for s in expected_order]
    if indices != sorted(indices):
        print(f"[expertise_primer] FAIL (e): wrong memory order\n"
              f"indices={indices}\n{body}", file=sys.stderr)
        return 1
    print("[expertise_primer] (d) intent-shaped + bad rows filtered + "
          "(e) confidence × level × recurrence order OK", file=sys.stderr)
    return 0


def _check_caps_respected() -> int:
    """(f) max_intents and max_memory_rows truncate cleanly."""
    from resonance_lattice.expertise import render_expertise_primer

    intents = [
        _intent(f"01HZIN{i}", text=f"intent-{i}",
                created_at=f"2026-05-09T{i:02d}:00:00Z")
        for i in range(15)
    ]
    rows = [_row(f"01HZRO{i}", text=f"row-{i}") for i in range(20)]
    body = render_expertise_primer(
        intents=intents, memory_rows=rows,
        max_intents=3, max_memory_rows=4,
    )
    if "## Active intents (3)" not in body:
        print(f"[expertise_primer] FAIL (f): intent cap not honoured\n{body}",
              file=sys.stderr)
        return 1
    if "## Project memory (top 4)" not in body:
        print(f"[expertise_primer] FAIL (f): memory cap not honoured\n{body}",
              file=sys.stderr)
        return 1
    print("[expertise_primer] (f) caps respected OK", file=sys.stderr)
    return 0


def _check_long_text_ellipsis() -> int:
    """(g) row text exceeding the char cap is ellipsised mid-line."""
    from resonance_lattice.expertise import render_expertise_primer

    long_text = "a" * 500
    rows = [_row("01HZLONG", text=long_text)]
    body = render_expertise_primer(
        intents=[], memory_rows=rows, memory_row_chars=50,
    )
    if long_text in body:
        print(f"[expertise_primer] FAIL (g): long text not truncated",
              file=sys.stderr)
        return 1
    if "…" not in body:
        print(f"[expertise_primer] FAIL (g): ellipsis missing\n{body}",
              file=sys.stderr)
        return 1
    print("[expertise_primer] (g) long text ellipsis OK", file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_empty_inputs,
        _check_intent_filtering_and_order,
        _check_memory_filtering_and_order,
        _check_caps_respected,
        _check_long_text_ellipsis,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[expertise_primer] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
