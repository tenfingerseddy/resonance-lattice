"""Train operator surface — row-level mutations from `/rlat-train` flags.

The full §8 GRPO loop (4 runner subagents → grader → distiller, ~10-15
min per task) ships as the `/rlat-train` slash command in Day 9-10 —
spawning parallel `Task` subagents requires the Claude Code primitive
that a CLI subprocess can't drive. This module covers the *operator*
surface only: the cheap synchronous mutations a user invokes from the
CLI to manage individual rows after a training run or distil pass
mis-classified one.

Maps to §0.7's three operator flags:

  rlat memory train --bad-vote <row_id> [--why "..."]
  rlat memory train --good-vote <row_id>
  rlat memory train --corroborate <row_id>

`bad-vote` is the §18.1 mitigation — the hot fix when distil writes a
wrong-polarity lesson and recall starts injecting it. `corroborate`
is the manual escape hatch for the §0.6 recurrence ≥ 3 gate when the
operator has independent evidence the lesson generalises.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ._common import utcnow_iso
from .store import Memory, Row


@dataclass(frozen=True)
class TrainResult:
    """One operator action against a single row."""

    row_id: str
    action: Literal["bad-vote", "good-vote", "corroborate"]
    field_changed: Literal["is_bad", "recurrence_count"]
    before: object
    after: object
    why: str | None = None


_AUDIT_FILENAME = "train_audit.log"


def _audit_log_path(store: Memory) -> Path:
    return store.root / _AUDIT_FILENAME


def _append_audit(store: Memory, row_id: str, action: str, *, why: str | None) -> None:
    """One line per operator action. Append-only, never reads back. Mirrors
    the §6.4 redaction-log shape so operators have one mental model.
    """
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts}  action={action}  row_id={row_id}"
    if why:
        line += f"  why={why!r}"
    path = _audit_log_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _find_row(store: Memory, row_id: str) -> Row:
    rows, _ = store.read_all()
    for row in rows:
        if row.row_id == row_id:
            return row
    raise KeyError(f"row_id {row_id!r} not in memory")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bad_vote(*, store: Memory, row_id: str, why: str | None = None) -> TrainResult:
    """Mark `row_id` as `is_bad: true`. §0.6 retrieval drops it at gate 1.

    Idempotent — bad-voting an already-bad row is a no-op (returns a
    TrainResult with `before == after == True` and no audit-log entry).
    """
    existing = _find_row(store, row_id)
    if existing.is_bad:
        return TrainResult(
            row_id=row_id, action="bad-vote",
            field_changed="is_bad", before=True, after=True, why=why,
        )
    store.update_row(row_id, is_bad=True)
    _append_audit(store, row_id, "bad-vote", why=why)
    return TrainResult(
        row_id=row_id, action="bad-vote",
        field_changed="is_bad", before=False, after=True, why=why,
    )


def good_vote(*, store: Memory, row_id: str) -> TrainResult:
    """Reverse a bad-vote: `is_bad: false`. Idempotent on rows that
    weren't bad-voted.
    """
    existing = _find_row(store, row_id)
    if not existing.is_bad:
        return TrainResult(
            row_id=row_id, action="good-vote",
            field_changed="is_bad", before=False, after=False,
        )
    store.update_row(row_id, is_bad=False)
    _append_audit(store, row_id, "good-vote", why=None)
    return TrainResult(
        row_id=row_id, action="good-vote",
        field_changed="is_bad", before=True, after=False,
    )


def corroborate(*, store: Memory, row_id: str) -> TrainResult:
    """Bump `recurrence_count` by exactly 1 + update `last_corroborated_at`.

    Lets the operator manually push a row over the §0.6 recurrence ≥ 3
    gate when there's independent evidence the lesson generalises but
    distil hasn't seen enough capture-side corroboration yet.
    """
    existing = _find_row(store, row_id)
    before = existing.recurrence_count
    after = before + 1
    store.update_row(
        row_id,
        recurrence_count=after,
        last_corroborated_at=utcnow_iso(),
    )
    _append_audit(store, row_id, "corroborate", why=None)
    return TrainResult(
        row_id=row_id, action="corroborate",
        field_changed="recurrence_count", before=before, after=after,
    )
