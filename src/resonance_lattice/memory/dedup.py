"""Retroactive memory dedup — collapse same-text-same-workspace event rows.

Capture-time dedup (`capture._find_dup_row`, shipped in `46b74013`) prevents
new duplicates from accumulating, but it only fires forward. Memory written
before that fix carries N copies of every recurring captured event — same
text, same workspace tag, distinct transcript hashes, recurrence_count=1
each. Arrow1 finds those clusters and the LLM correctly refuses to promote
("8 identical events — noise"); the expertise primer surfaces 4 copies of
the same line; forget can only decay them via the slow age path.

This module ships the one-shot retroactive collapse: `dedup_event_rows`
groups event-level rows by `(text, workspace_tag)`, keeps the oldest of
each group, sets its `recurrence_count` to the cluster size + 1 (so the
collapsed signal isn't lost), and deletes the rest. Idempotent — running
it on already-deduped memory is a no-op.

Same `(text, workspace_tag)` invariant as the capture-time path so the two
operations agree on what counts as a duplicate.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

from ._common import utcnow_iso
from .store import Memory, Row


@dataclass(frozen=True)
class DedupResult:
    """One dedup pass outcome."""

    rows_collapsed: int  # number of rows deleted
    groups_collapsed: int  # number of (text, workspace) clusters that had >1 row
    rows_examined: int  # total event-level rows scanned


def _workspace_tag(row: Row) -> str | None:
    for tag in row.polarity:
        if tag.startswith("workspace:"):
            return tag
    return None


def _group_dups(rows: list[Row]) -> list[list[Row]]:
    """Return groups of >1 event rows sharing `(text, workspace_tag)`.

    Non-event rows are excluded — patterns/learnings/principles are
    distilled output, not raw captures, and re-running distil produces
    fresh promoted rows that should stay distinct.
    """
    buckets: dict[tuple[str, str | None], list[Row]] = collections.defaultdict(list)
    for r in rows:
        if r.level != "event":
            continue
        buckets[(r.text, _workspace_tag(r))].append(r)
    return [g for g in buckets.values() if len(g) > 1]


def dedup_event_rows(
    memory: Memory, *, dry_run: bool = False,
) -> DedupResult:
    """Collapse `(text, workspace_tag)`-equal event rows into the oldest.

    The keeper's `recurrence_count` is set to the sum of the group's
    counts (architecture: recurrence is the cumulative signal, not the
    write-count proxy it ends up as without dedup). `last_corroborated_at`
    bumps to now so retention treats the collapsed row as fresh.

    Returns counts; on `dry_run=True` the disk is untouched.
    """
    rows, _ = memory.read_all()
    rows_examined = sum(1 for r in rows if r.level == "event")
    groups = _group_dups(rows)
    if not groups:
        return DedupResult(
            rows_collapsed=0,
            groups_collapsed=0,
            rows_examined=rows_examined,
        )

    rows_collapsed = 0
    now = utcnow_iso()
    for group in groups:
        # Keep the oldest by created_at; delete the rest. Sum recurrence
        # so the cumulative signal survives the collapse — a row that
        # was captured 8 times has recurrence_count=8 after, not 1.
        ordered = sorted(group, key=lambda r: r.created_at)
        keeper = ordered[0]
        deletions = [r.row_id for r in ordered[1:]]
        bumped_recurrence = sum(r.recurrence_count for r in ordered)
        rows_collapsed += len(deletions)
        if dry_run:
            continue
        memory.update_row(
            keeper.row_id,
            recurrence_count=bumped_recurrence,
            last_corroborated_at=now,
        )
        memory.delete_rows(deletions)

    return DedupResult(
        rows_collapsed=rows_collapsed,
        groups_collapsed=len(groups),
        rows_examined=rows_examined,
    )
