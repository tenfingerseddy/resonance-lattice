"""memory_v22_what_next — what-next recommendation contracts.

Pins architecture §"What-next — the user-facing recommendation operation".
Six contracts:

  (a) Empty graph → empty result + safe stub line.

  (b) Single active task surfaces as the recommendation.

  (c) Status priority — in_progress (recently updated) outranks ready
      (active but stale) outranks blocked.

  (d) Level depth — at equal status, deeper level (step > task) wins.

  (e) Achievability — at equal status + level, high outranks medium.

  (f) Cheap-path stub when llm=None renders the top candidate's text.

Hermetic — pure scoring, no LLM, no network.
"""

from __future__ import annotations

import datetime as _dt
import sys

from resonance_lattice.memory.what_next import (
    pick_candidates,
    synthesise_recommendation,
)
from resonance_lattice.state.intent import Intent

_NOW = _dt.datetime(2026, 5, 8, 12, tzinfo=_dt.timezone.utc)


def _intent(
    *,
    intent_id: str = "01HZ_I1",
    text: str = "do thing",
    level: str = "task",
    status: str = "active",
    achievability: str = "medium",
    updated_hours_ago: float = 0.5,
    blocks: list[str] | None = None,
) -> Intent:
    updated = (_NOW - _dt.timedelta(hours=updated_hours_ago)).strftime(
        "%Y-%m-%dT%H:%M:%SZ",
    )
    return Intent(
        intent_id=intent_id,
        level=level,
        text=text,
        parent_ids=[],
        blocks=list(blocks) if blocks else [],
        stance="do",
        achievability=achievability,
        status=status,
        success_criteria=[],
        constraints=[],
        created_under_intent_kind="implement",
        created_at=updated,
        updated_at=updated,
    )


def _check_empty_graph() -> int:
    candidates = pick_candidates([], now=_NOW)
    if candidates:
        print(f"[memory_v22_what_next] FAIL (a): non-empty: {candidates!r}",
              file=sys.stderr)
        return 1
    line = synthesise_recommendation([], llm=None)
    if not line or "/want" not in line:
        print(f"[memory_v22_what_next] FAIL (a): stub line bad: {line!r}",
              file=sys.stderr)
        return 1
    print("[memory_v22_what_next] (a) empty graph + safe stub OK",
          file=sys.stderr)
    return 0


def _check_single_active_task() -> int:
    intent = _intent(intent_id="01HZ_T1", text="ship the harness")
    candidates = pick_candidates([intent], now=_NOW)
    if len(candidates) != 1 or candidates[0].text != "ship the harness":
        print(f"[memory_v22_what_next] FAIL (b): {candidates!r}",
              file=sys.stderr)
        return 1
    print("[memory_v22_what_next] (b) single active task surfaces OK",
          file=sys.stderr)
    return 0


def _check_status_priority() -> int:
    in_progress = _intent(
        intent_id="01HZ_PROG", text="in progress",
        updated_hours_ago=1,  # within 24h window → in_progress
    )
    ready = _intent(
        intent_id="01HZ_RDY", text="ready", updated_hours_ago=72,
    )
    blocked = _intent(
        intent_id="01HZ_BLK", text="blocked", status="blocked",
        updated_hours_ago=1,
    )
    candidates = pick_candidates([blocked, ready, in_progress], now=_NOW)
    if [c.text for c in candidates] != ["in progress", "ready", "blocked"]:
        print(f"[memory_v22_what_next] FAIL (c): "
              f"{[c.text for c in candidates]!r}", file=sys.stderr)
        return 1
    print("[memory_v22_what_next] (c) status priority OK", file=sys.stderr)
    return 0


def _check_level_depth_at_equal_status() -> int:
    step = _intent(
        intent_id="01HZ_S", text="step now", level="step",
        updated_hours_ago=1,
    )
    task = _intent(
        intent_id="01HZ_T", text="task now", level="task",
        updated_hours_ago=1,
    )
    candidates = pick_candidates([task, step], now=_NOW)
    if candidates[0].text != "step now":
        print(f"[memory_v22_what_next] FAIL (d): "
              f"{[c.text for c in candidates]!r}", file=sys.stderr)
        return 1
    print("[memory_v22_what_next] (d) deeper level wins at equal status OK",
          file=sys.stderr)
    return 0


def _check_achievability_breaks_tie() -> int:
    high = _intent(
        intent_id="01HZ_H", text="high reach", achievability="high",
        updated_hours_ago=1,
    )
    medium = _intent(
        intent_id="01HZ_M", text="medium reach", achievability="medium",
        updated_hours_ago=1,
    )
    candidates = pick_candidates([medium, high], now=_NOW)
    if candidates[0].text != "high reach":
        print(f"[memory_v22_what_next] FAIL (e): "
              f"{[c.text for c in candidates]!r}", file=sys.stderr)
        return 1
    print("[memory_v22_what_next] (e) achievability breaks tie OK",
          file=sys.stderr)
    return 0


def _check_stub_renders_top_candidate() -> int:
    intent = _intent(intent_id="01HZ_T1", text="run the harness")
    candidates = pick_candidates([intent], now=_NOW)
    line = synthesise_recommendation(candidates, llm=None)
    if "run the harness" not in line:
        print(f"[memory_v22_what_next] FAIL (f): line missing top text: "
              f"{line!r}", file=sys.stderr)
        return 1
    if "Want me to start" not in line:
        print(f"[memory_v22_what_next] FAIL (f): line missing offer: "
              f"{line!r}", file=sys.stderr)
        return 1
    print("[memory_v22_what_next] (f) stub renders top candidate OK",
          file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_empty_graph,
        _check_single_active_task,
        _check_status_priority,
        _check_level_depth_at_equal_status,
        _check_achievability_breaks_tie,
        _check_stub_renders_top_candidate,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v22_what_next] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
