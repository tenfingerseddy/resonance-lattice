"""memory_dedup — retroactive (content, workspace_tag) event-claim collapse.

Pins four contracts:

  (a) Same `(content, workspace_tag)` claims collapse to the oldest, with
      recurrence_count summed across the group.

  (b) Different workspace tags do NOT collapse — same content from two
      checkouts of the same project is genuinely two events.

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


def _vec():
    from resonance_lattice.field.encoder import DIM
    return np.zeros(DIM, dtype=np.float32)


def _seed(memory, *, text: str, polarity: list[str], when: str = "2026-05-01T00:00:00Z",
          recurrence: int = 1, level: str = "event") -> str:
    """Seed memory with one claim, returning its claim_id."""
    from resonance_lattice.state.claim import Claim, ExperienceFacts

    claim_id = f"01HZ{when}:{text[:10]}".ljust(26, "0")[:26]
    claim = Claim(
        claim_id=claim_id,
        source="experience",
        kind=level,
        content=text,
        created_at=when,
        corroboration=2.0,
        falsification=2.0,
        trust_as_of="",
        state="active",
        parent_ids=(),
        facts=ExperienceFacts(
            polarity=tuple(polarity),
            recurrence_count=recurrence,
            criticality="normal",
            created_under_intent_kind="none",
            transcript_hash=f"hash:{when}:{text[:10]}",
            origin="manual",
            last_corroborated_at=when,
        ),
    )
    memory.write(claim, embedding=_vec())
    return claim_id


def _check_basic_collapse() -> int:
    """(a) 8 same-(content, workspace) claims → 1, recurrence_count summed."""
    from resonance_lattice.memory.claim_store import ExperienceClaimStore
    from resonance_lattice.memory.dedup import dedup_event_claims

    with tempfile.TemporaryDirectory() as td:
        memory = ExperienceClaimStore(root=Path(td) / "u", encoder=None)
        ids = [
            _seed(memory, text="recurring lesson",
                  polarity=["factual", "workspace:abc123"],
                  when=f"2026-05-0{i % 9 + 1}T00:00:00Z")
            for i in range(8)
        ]

        result = dedup_event_claims(memory)

        if result.claims_collapsed != 7:
            print(f"[memory_dedup] FAIL (a): collapsed={result.claims_collapsed} "
                  f"(want 7)", file=sys.stderr)
            return 1
        if result.groups_collapsed != 1:
            print(f"[memory_dedup] FAIL (a): groups={result.groups_collapsed} "
                  f"(want 1)", file=sys.stderr)
            return 1

        claims = memory.read_all()
        recurring = [c for c in claims if c.content == "recurring lesson"]
        if len(recurring) != 1:
            print(f"[memory_dedup] FAIL (a): post-dedup claim count "
                  f"{len(recurring)} (want 1)", file=sys.stderr)
            return 1
        if recurring[0].facts.recurrence_count != 8:
            print(f"[memory_dedup] FAIL (a): recurrence_count "
                  f"{recurring[0].facts.recurrence_count} (want 8)",
                  file=sys.stderr)
            return 1
        # Keeper is the oldest by created_at — first seeded id.
        if recurring[0].claim_id != ids[0]:
            print(f"[memory_dedup] FAIL (a): keeper {recurring[0].claim_id!r} "
                  f"(want oldest {ids[0]!r})", file=sys.stderr)
            return 1
    print("[memory_dedup] (a) basic collapse: 8 → 1 with recurrence=8 OK",
          file=sys.stderr)
    return 0


def _check_workspace_boundary() -> int:
    """(b) Same content in two workspaces stays as two claims."""
    from resonance_lattice.memory.claim_store import ExperienceClaimStore
    from resonance_lattice.memory.dedup import dedup_event_claims

    with tempfile.TemporaryDirectory() as td:
        memory = ExperienceClaimStore(root=Path(td) / "u", encoder=None)
        for i in range(3):
            _seed(memory, text="cross-workspace text",
                  polarity=["factual", "workspace:proj-A"],
                  when=f"2026-05-0{i + 1}T00:00:00Z")
        for i in range(3):
            _seed(memory, text="cross-workspace text",
                  polarity=["factual", "workspace:proj-B"],
                  when=f"2026-05-0{i + 4}T00:00:00Z")

        result = dedup_event_claims(memory)

        if result.groups_collapsed != 2:
            print(f"[memory_dedup] FAIL (b): groups={result.groups_collapsed} "
                  f"(want 2 — one per workspace)", file=sys.stderr)
            return 1
        claims = memory.read_all()
        if len(claims) != 2:
            print(f"[memory_dedup] FAIL (b): post-dedup claim count "
                  f"{len(claims)} (want 2 — one per workspace)",
                  file=sys.stderr)
            return 1
    print("[memory_dedup] (b) workspace boundary preserved OK", file=sys.stderr)
    return 0


def _check_dry_run() -> int:
    """(d) dry_run reports counts but doesn't touch disk."""
    from resonance_lattice.memory.claim_store import ExperienceClaimStore
    from resonance_lattice.memory.dedup import dedup_event_claims

    with tempfile.TemporaryDirectory() as td:
        memory = ExperienceClaimStore(root=Path(td) / "u", encoder=None)
        for i in range(4):
            _seed(memory, text="dry-run candidate",
                  polarity=["factual", "workspace:abc123"],
                  when=f"2026-05-0{i + 1}T00:00:00Z")

        before = memory.read_all()
        before_count = len(before)
        result = dedup_event_claims(memory, dry_run=True)
        after = memory.read_all()

        if len(after) != before_count:
            print(f"[memory_dedup] FAIL (d): dry-run mutated store "
                  f"({before_count} → {len(after)})", file=sys.stderr)
            return 1
        if result.claims_collapsed != 3:
            print(f"[memory_dedup] FAIL (d): dry-run count wrong "
                  f"({result.claims_collapsed} != 3)", file=sys.stderr)
            return 1
    print("[memory_dedup] (d) dry-run preserves disk OK", file=sys.stderr)
    return 0


def _check_idempotent() -> int:
    """(e) Running dedup on already-deduped memory is a no-op."""
    from resonance_lattice.memory.claim_store import ExperienceClaimStore
    from resonance_lattice.memory.dedup import dedup_event_claims

    with tempfile.TemporaryDirectory() as td:
        memory = ExperienceClaimStore(root=Path(td) / "u", encoder=None)
        for i in range(5):
            _seed(memory, text="lesson",
                  polarity=["factual", "workspace:abc123"],
                  when=f"2026-05-0{i + 1}T00:00:00Z")

        first = dedup_event_claims(memory)
        second = dedup_event_claims(memory)

        if first.claims_collapsed != 4:
            print(f"[memory_dedup] FAIL (e): first pass {first!r}",
                  file=sys.stderr)
            return 1
        if second.claims_collapsed != 0 or second.groups_collapsed != 0:
            print(f"[memory_dedup] FAIL (e): second pass not no-op {second!r}",
                  file=sys.stderr)
            return 1
    print("[memory_dedup] (e) idempotent OK", file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_basic_collapse,
        _check_workspace_boundary,
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
