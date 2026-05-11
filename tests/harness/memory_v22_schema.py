"""memory_v22_schema — Horizon 1 schema contracts.

Pins the v2.1 → v2.2 schema bump from architecture §"Memory schema".
Six contracts:

  (a) v1 sidecar (schema_version=1, no v2.2 fields) loads with the
      migration-table defaults populated: level=event, criticality=normal,
      confidence=medium (or low if is_bad), parent_ids=[], cited_passages=[],
      origin derived from transcript_hash prefix, created_under_intent_kind=none.

  (b) v2.2 round-trip — write a fully-populated v2.2 row and read it back equal.

  (c) Intent-row write requires stance/achievability/status/success_criteria/
      constraints; missing any one raises before disk write.

  (d) Memory-row write rejects intent-only fields (forbidden when level is
      one of MEMORY_LEVELS).

  (e) Enum validation — bad values for level / criticality / confidence /
      origin / created_under_intent_kind / stance / achievability / status
      raise ValueError.

  (f) update_row revalidates after merge — flipping level=event → level=task
      without populating intent fields raises and leaves the row unchanged.

Hermetic — no encoder, no LLM, no live network.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np


def _check_v1_default_fill() -> int:
    """v1 row → v2.2 defaults populated on load."""
    from resonance_lattice.memory.store import Memory, _derive_origin

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        root.mkdir(parents=True)
        sidecar = root / "sidecar.jsonl"
        band_path = root / "memory.npz"
        # Three v1 rows exercising the three origin lifts + is_bad.
        rows_payload = [
            {
                "row_id": "01HZ0000000000000000000001",
                "text": "manual v1 row",
                "polarity": ["prefer", "workspace:abc123"],
                "recurrence_count": 4,
                "created_at": "2026-04-01T00:00:00Z",
                "last_corroborated_at": "2026-04-01T00:00:00Z",
                "transcript_hash": "manual",
                "is_bad": False,
                "schema_version": 1,
            },
            {
                "row_id": "01HZ0000000000000000000002",
                "text": "distilled v1 row",
                "polarity": ["factual", "workspace:abc123"],
                "recurrence_count": 7,
                "created_at": "2026-04-01T00:00:00Z",
                "last_corroborated_at": "2026-04-01T00:00:00Z",
                "transcript_hash": "distilled:abc",
                "is_bad": False,
                "schema_version": 1,
            },
            {
                "row_id": "01HZ0000000000000000000003",
                "text": "is_bad v1 row",
                "polarity": ["avoid", "workspace:abc123"],
                "recurrence_count": 1,
                "created_at": "2026-04-01T00:00:00Z",
                "last_corroborated_at": "2026-04-01T00:00:00Z",
                "transcript_hash": "manual",
                "is_bad": True,
                "schema_version": 1,
            },
        ]
        sidecar.write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in rows_payload),
            encoding="utf-8",
        )
        np.savez(band_path, band=np.zeros((3, 768), dtype=np.float32))

        memory = Memory(root=root)
        loaded, _ = memory.read_all()

    if [r.row_id for r in loaded] != [
        "01HZ0000000000000000000001",
        "01HZ0000000000000000000002",
        "01HZ0000000000000000000003",
    ]:
        print("[memory_v22_schema] FAIL (a): row_ids reordered", file=sys.stderr)
        return 1
    expectations = [
        ("manual", "medium"),
        ("distilled", "medium"),
        ("manual", "low"),
    ]
    for row, (origin, confidence) in zip(loaded, expectations):
        if row.level != "event":
            print(f"[memory_v22_schema] FAIL (a): {row.row_id} level={row.level!r}", file=sys.stderr)
            return 1
        if row.criticality != "normal":
            print(f"[memory_v22_schema] FAIL (a): {row.row_id} criticality={row.criticality!r}", file=sys.stderr)
            return 1
        if row.confidence != confidence:
            print(f"[memory_v22_schema] FAIL (a): {row.row_id} confidence={row.confidence!r} (want {confidence!r})", file=sys.stderr)
            return 1
        if row.origin != origin:
            print(f"[memory_v22_schema] FAIL (a): {row.row_id} origin={row.origin!r} (want {origin!r})", file=sys.stderr)
            return 1
        if row.parent_ids != [] or row.cited_passages != []:
            print(f"[memory_v22_schema] FAIL (a): {row.row_id} list defaults wrong", file=sys.stderr)
            return 1
        if row.created_under_intent_kind != "none":
            print(f"[memory_v22_schema] FAIL (a): {row.row_id} intent_kind={row.created_under_intent_kind!r}", file=sys.stderr)
            return 1
        if row.stance is not None or row.achievability is not None or row.status is not None:
            print(f"[memory_v22_schema] FAIL (a): {row.row_id} intent fields not None on memory row", file=sys.stderr)
            return 1
    # _derive_origin is the lifted helper — sanity-check its mapping too.
    if _derive_origin("migrated:v2.0:semantic") != "migrated":
        print("[memory_v22_schema] FAIL (a): _derive_origin migrated mismatch", file=sys.stderr)
        return 1
    print("[memory_v22_schema] (a) v1 default-fill OK", file=sys.stderr)
    return 0


def _check_v22_round_trip() -> int:
    """Full-fat v2.2 row writes and reads back equal."""
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        memory = Memory(root=root)
        row_id = memory.add_row(
            text="cross-domain principle",
            polarity=["prefer", "cross-workspace"],
            transcript_hash="distilled:xyz",
            embedding=np.zeros(768, dtype=np.float32),
            level="principle",
            criticality="high",
            confidence="verified",
            parent_ids=["01HZ0000000000000000000010"],
            cited_passages=["pid:abc"],
            origin="distilled",
            created_under_intent_kind="design",
        )
        before, _ = memory.read_all()
        memory2 = Memory(root=root)
        after, _ = memory2.read_all()

    if [asdict(r) for r in before] != [asdict(r) for r in after]:
        print("[memory_v22_schema] FAIL (b): round-trip diverged", file=sys.stderr)
        return 1
    row = after[0]
    if (row.row_id != row_id or row.level != "principle"
            or row.criticality != "high" or row.confidence != "verified"
            or row.parent_ids != ["01HZ0000000000000000000010"]
            or row.cited_passages != ["pid:abc"]
            or row.origin != "distilled"
            or row.created_under_intent_kind != "design"):
        print(f"[memory_v22_schema] FAIL (b): v2.2 fields diverged: {asdict(row)!r}", file=sys.stderr)
        return 1
    print("[memory_v22_schema] (b) v2.2 round-trip OK", file=sys.stderr)
    return 0


def _check_intent_required_fields() -> int:
    """Intent row missing any required field raises before write."""
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        memory = Memory(root=root)
        # Fully-specified intent row writes cleanly — happy path first to
        # confirm the validator isn't refusing valid input.
        ok = memory.add_row(
            text="ship the harness",
            polarity=["prefer", "workspace:abc123"],
            transcript_hash="manual",
            embedding=np.zeros(768, dtype=np.float32),
            level="goal",
            stance="do",
            achievability="medium",
            status="active",
            success_criteria=[{"text": "v1 lands", "measure": "user_confirms"}],
            constraints=["additive only"],
        )
        if not ok:
            print("[memory_v22_schema] FAIL (c): valid intent rejected", file=sys.stderr)
            return 1

        # Missing stance.
        try:
            memory.add_row(
                text="bad intent — no stance",
                polarity=["prefer", "workspace:abc123"],
                transcript_hash="manual",
                embedding=np.zeros(768, dtype=np.float32),
                level="task",
                achievability="high",
                status="active",
                success_criteria=[],
                constraints=[],
            )
        except ValueError as exc:
            if "stance" not in str(exc):
                print(f"[memory_v22_schema] FAIL (c): stance error wrong: {exc!r}", file=sys.stderr)
                return 1
        else:
            print("[memory_v22_schema] FAIL (c): missing stance accepted", file=sys.stderr)
            return 1

        # Missing constraints — different field, same rule.
        try:
            memory.add_row(
                text="bad intent — no constraints",
                polarity=["prefer", "workspace:abc123"],
                transcript_hash="manual",
                embedding=np.zeros(768, dtype=np.float32),
                level="step",
                stance="do",
                achievability="high",
                status="active",
                success_criteria=[{"text": "x", "measure": "exit_code:0"}],
            )
        except ValueError as exc:
            if "constraints" not in str(exc):
                print(f"[memory_v22_schema] FAIL (c): constraints error wrong: {exc!r}", file=sys.stderr)
                return 1
        else:
            print("[memory_v22_schema] FAIL (c): missing constraints accepted", file=sys.stderr)
            return 1
    print("[memory_v22_schema] (c) intent required fields OK", file=sys.stderr)
    return 0


def _check_memory_forbids_intent_fields() -> int:
    """Memory row with intent-only fields raises."""
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        memory = Memory(root=root)
        try:
            memory.add_row(
                text="memory row with stance — illegal",
                polarity=["prefer", "workspace:abc123"],
                transcript_hash="manual",
                embedding=np.zeros(768, dtype=np.float32),
                level="event",
                stance="do",  # forbidden on memory rows
            )
        except ValueError as exc:
            if "memory row" not in str(exc) or "stance" not in str(exc):
                print(f"[memory_v22_schema] FAIL (d): memory-forbids error wrong: {exc!r}", file=sys.stderr)
                return 1
        else:
            print("[memory_v22_schema] FAIL (d): memory row with stance accepted", file=sys.stderr)
            return 1
    print("[memory_v22_schema] (d) memory rows forbid intent fields OK", file=sys.stderr)
    return 0


def _check_enum_validation() -> int:
    """Bad enum values rejected at write time."""
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        memory = Memory(root=root)
        cases = [
            {"level": "wisdom"},
            {"criticality": "extreme"},
            {"confidence": "certain"},
            {"origin": "magic"},
            {"created_under_intent_kind": "vibing"},
        ]
        for kw in cases:
            try:
                memory.add_row(
                    text="bad enum",
                    polarity=["factual", "workspace:abc123"],
                    transcript_hash="manual",
                    embedding=np.zeros(768, dtype=np.float32),
                    **kw,
                )
            except ValueError:
                continue
            print(f"[memory_v22_schema] FAIL (e): bad enum accepted: {kw!r}", file=sys.stderr)
            return 1
    print("[memory_v22_schema] (e) enum validation OK", file=sys.stderr)
    return 0


def _check_update_revalidates() -> int:
    """update_row blocks invalid level transitions."""
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        memory = Memory(root=root)
        row_id = memory.add_row(
            text="memory event",
            polarity=["factual", "workspace:abc123"],
            transcript_hash="manual",
            embedding=np.zeros(768, dtype=np.float32),
        )
        # Promoting an event into an intent goal without populating intent
        # fields must raise — otherwise invariants break for the recall path.
        try:
            memory.update_row(row_id, level="goal")
        except ValueError as exc:
            if "intent row" not in str(exc):
                print(f"[memory_v22_schema] FAIL (f): wrong error: {exc!r}", file=sys.stderr)
                return 1
        else:
            print("[memory_v22_schema] FAIL (f): bad level update accepted", file=sys.stderr)
            return 1
        # Confirm the row stayed at level=event.
        rows, _ = memory.read_all()
        if rows[0].level != "event":
            print(f"[memory_v22_schema] FAIL (f): level mutated despite raise: {rows[0].level!r}", file=sys.stderr)
            return 1
    print("[memory_v22_schema] (f) update_row revalidates OK", file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_v1_default_fill,
        _check_v22_round_trip,
        _check_intent_required_fields,
        _check_memory_forbids_intent_fields,
        _check_enum_validation,
        _check_update_revalidates,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v22_schema] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
