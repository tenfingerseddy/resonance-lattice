"""memory_dedup — retroactive (text, workspace_tag) event-row collapse.

Pins five contracts:

  (a) Same `(text, workspace_tag)` rows collapse to the oldest, with
      recurrence_count summed across the group.

  (b) Different workspace tags do NOT collapse — same text from two
      checkouts of the same project is genuinely two events.

  (c) Non-event levels (pattern / learning / principle) are untouched.

  (d) `dry_run=True` returns the same counts as a real run but leaves
      the store unchanged.

  (e) Idempotent — re-running on already-deduped memory is a no-op.

Hermetic — no encoder, no LLM, no I/O.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


def _seed(memory, *, text: str, polarity: list[str], when: str = "2026-05-01T00:00:00Z",
          recurrence: int = 1, level: str = "event") -> str:
    """Seed memory with one row, returning its row_id."""
    return memory.add_row(
        text=text,
        polarity=polarity,
        transcript_hash=f"hash:{when}:{text[:10]}",
        embedding=np.zeros(768, dtype=np.float32),
        level=level,
    )


def _check_basic_collapse() -> int:
    """(a) 8 same-(text, workspace) rows → 1 row, recurrence_count summed."""
    from resonance_lattice.memory.dedup import dedup_event_rows
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        ids = [
            _seed(memory, text="recurring lesson",
                  polarity=["factual", "workspace:abc123"],
                  when=f"2026-05-0{i % 9 + 1}T00:00:00Z")
            for i in range(8)
        ]

        result = dedup_event_rows(memory)

        if result.rows_collapsed != 7:
            print(f"[memory_dedup] FAIL (a): collapsed={result.rows_collapsed} "
                  f"(want 7)", file=sys.stderr)
            return 1
        if result.groups_collapsed != 1:
            print(f"[memory_dedup] FAIL (a): groups={result.groups_collapsed} "
                  f"(want 1)", file=sys.stderr)
            return 1

        rows, _ = memory.read_all()
        recurring = [r for r in rows if r.text == "recurring lesson"]
        if len(recurring) != 1:
            print(f"[memory_dedup] FAIL (a): post-dedup row count "
                  f"{len(recurring)} (want 1)", file=sys.stderr)
            return 1
        if recurring[0].recurrence_count != 8:
            print(f"[memory_dedup] FAIL (a): recurrence_count "
                  f"{recurring[0].recurrence_count} (want 8)", file=sys.stderr)
            return 1
        # Keeper is the oldest by created_at — first seeded id.
        if recurring[0].row_id != ids[0]:
            print(f"[memory_dedup] FAIL (a): keeper {recurring[0].row_id!r} "
                  f"(want oldest {ids[0]!r})", file=sys.stderr)
            return 1
    print("[memory_dedup] (a) basic collapse: 8 → 1 with recurrence=8 OK",
          file=sys.stderr)
    return 0


def _check_workspace_boundary() -> int:
    """(b) Same text in two workspaces stays as two rows."""
    from resonance_lattice.memory.dedup import dedup_event_rows
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        for _ in range(3):
            _seed(memory, text="cross-workspace text",
                  polarity=["factual", "workspace:proj-A"])
        for _ in range(3):
            _seed(memory, text="cross-workspace text",
                  polarity=["factual", "workspace:proj-B"])

        result = dedup_event_rows(memory)

        if result.groups_collapsed != 2:
            print(f"[memory_dedup] FAIL (b): groups={result.groups_collapsed} "
                  f"(want 2 — one per workspace)", file=sys.stderr)
            return 1
        rows, _ = memory.read_all()
        if len(rows) != 2:
            print(f"[memory_dedup] FAIL (b): post-dedup row count "
                  f"{len(rows)} (want 2 — one per workspace)",
                  file=sys.stderr)
            return 1
    print("[memory_dedup] (b) workspace boundary preserved OK", file=sys.stderr)
    return 0


def _check_non_event_levels_untouched() -> int:
    """(c) pattern / learning / principle rows are not deduped."""
    from resonance_lattice.memory.dedup import dedup_event_rows
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        for _ in range(3):
            _seed(memory, text="distilled pattern text",
                  polarity=["prefer", "workspace:abc123"],
                  level="pattern")

        result = dedup_event_rows(memory)

        if result.rows_collapsed != 0:
            print(f"[memory_dedup] FAIL (c): pattern rows collapsed "
                  f"({result.rows_collapsed} != 0)", file=sys.stderr)
            return 1
        rows, _ = memory.read_all()
        if len(rows) != 3:
            print(f"[memory_dedup] FAIL (c): pattern rows count "
                  f"{len(rows)} (want 3)", file=sys.stderr)
            return 1
    print("[memory_dedup] (c) non-event levels untouched OK", file=sys.stderr)
    return 0


def _check_dry_run() -> int:
    """(d) dry_run reports counts but doesn't touch disk."""
    from resonance_lattice.memory.dedup import dedup_event_rows
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        for _ in range(4):
            _seed(memory, text="dry-run candidate",
                  polarity=["factual", "workspace:abc123"])

        before, _ = memory.read_all()
        before_count = len(before)
        result = dedup_event_rows(memory, dry_run=True)
        after, _ = memory.read_all()

        if len(after) != before_count:
            print(f"[memory_dedup] FAIL (d): dry-run mutated store "
                  f"({before_count} → {len(after)})", file=sys.stderr)
            return 1
        if result.rows_collapsed != 3:
            print(f"[memory_dedup] FAIL (d): dry-run count wrong "
                  f"({result.rows_collapsed} != 3)", file=sys.stderr)
            return 1
    print("[memory_dedup] (d) dry-run preserves disk OK", file=sys.stderr)
    return 0


def _check_idempotent() -> int:
    """(e) Running dedup on already-deduped memory is a no-op."""
    from resonance_lattice.memory.dedup import dedup_event_rows
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        for _ in range(5):
            _seed(memory, text="lesson",
                  polarity=["factual", "workspace:abc123"])

        first = dedup_event_rows(memory)
        second = dedup_event_rows(memory)

        if first.rows_collapsed != 4:
            print(f"[memory_dedup] FAIL (e): first pass {first!r}",
                  file=sys.stderr)
            return 1
        if second.rows_collapsed != 0 or second.groups_collapsed != 0:
            print(f"[memory_dedup] FAIL (e): second pass not no-op {second!r}",
                  file=sys.stderr)
            return 1
    print("[memory_dedup] (e) idempotent OK", file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_basic_collapse,
        _check_workspace_boundary,
        _check_non_event_levels_untouched,
        _check_dry_run,
        _check_idempotent,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_dedup] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
