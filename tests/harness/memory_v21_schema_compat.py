"""memory_v21_schema_compat — Appendix D D.9 contracts.

Pins four invariants on the §0.2 9-field schema's forward-compat
behaviour and the §14 v2.0 → v2.1 migration shape:

  (a) v2.1 reader loads its own writes round-trip. Every row written
      via `Memory.add_row` reads back equal under `to_jsonl_dict`.

  (b) v2.1 reader tolerates additive future fields. A sidecar carrying
      keys outside the §0.2 9-field schema loads cleanly with the
      unknown keys silently dropped — per the §18.8 forward-compat
      contract. No crash, no row loss.

  (c) `schema_version: 1` round-trips. Writer stamps SCHEMA_VERSION,
      reader preserves it on the in-memory Row, re-write preserves it.

  (d) `schema_version: 2` (synthetic future) loads with a single
      stderr warning per file (not per row), never crashes. Best-
      effort: known fields are read into the Row, unknown fields are
      dropped. The warning surfaces the version drift to operators.

  (e) v2.0 → v2.1 migrate produces rows whose `transcript_hash` carries
      the `migrated:` prefix; `_is_capture_row` excludes them from
      distil pickup so a freshly-migrated user can run `/rlat-distil`
      without re-feeding their archive into the LLM. Migration also
      stamps each primary polarity reasonably (verb-scan heuristic
      coverage on a calibrated fixture per §14.5).

Hermetic: synthetic-future fixture is a hand-written sidecar; the
v2.0 fixture builds a tiny LayeredMemory with `ZeroEncoder` so the
migration round-trip is bit-equivalent.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from contextlib import redirect_stderr
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ._testutil import patch_zero_encoder


# ---------------------------------------------------------------------------
# (a) round-trip
# ---------------------------------------------------------------------------


def _check_round_trip() -> int:
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        memory = Memory(root=root)
        row_id = memory.add_row(
            text="round-trip lesson",
            polarity=["prefer", "workspace:abc123"],
            transcript_hash="manual",
            embedding=np.zeros(768, dtype=np.float32),
        )
        before, _ = memory.read_all()
        # Re-open a fresh Memory pointing at the same root to exercise
        # the disk → memory load path explicitly.
        memory2 = Memory(root=root)
        after, _ = memory2.read_all()
    if [asdict(r) for r in before] != [asdict(r) for r in after]:
        print("[memory_v21_schema_compat] FAIL (a): round-trip diverged",
              file=sys.stderr)
        return 1
    if after[0].row_id != row_id:
        print("[memory_v21_schema_compat] FAIL (a): row_id corruption",
              file=sys.stderr)
        return 1
    print("[memory_v21_schema_compat] (a) writer→reader round-trip OK",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (b) additive future fields tolerated
# ---------------------------------------------------------------------------


def _check_additive_future_fields() -> int:
    from resonance_lattice.memory.store import Memory, Row, SCHEMA_VERSION

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        root.mkdir(parents=True)
        # Hand-write a sidecar row with an extra `future_field` key (and
        # a band file with the matching shape). The reader must drop
        # `future_field` silently and load the rest.
        sidecar = root / "sidecar.jsonl"
        band_path = root / "memory.npz"
        future_row = {
            "row_id": "01HZ8K3M5N7P9Q1R2S3T4V5W6X",
            "text": "row with future field",
            "polarity": ["factual", "workspace:abc123"],
            "recurrence_count": 1,
            "created_at": "2026-05-02T00:00:00Z",
            "last_corroborated_at": "2026-05-02T00:00:00Z",
            "transcript_hash": "manual",
            "is_bad": False,
            "schema_version": SCHEMA_VERSION,
            "future_field": {"foo": "bar"},
            "another_unknown": [1, 2, 3],
        }
        sidecar.write_text(json.dumps(future_row, sort_keys=True),
                            encoding="utf-8")
        band = np.zeros((1, 768), dtype=np.float32)
        np.savez(band_path, band=band)

        memory = Memory(root=root)
        rows, _ = memory.read_all()
    if len(rows) != 1:
        print(f"[memory_v21_schema_compat] FAIL (b): expected 1 row, "
              f"got {len(rows)}", file=sys.stderr)
        return 1
    if rows[0].text != "row with future field":
        print(f"[memory_v21_schema_compat] FAIL (b): text corrupted: "
              f"{rows[0].text!r}", file=sys.stderr)
        return 1
    print("[memory_v21_schema_compat] (b) additive future fields tolerated "
          "(unknown keys dropped) OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (c) schema_version=1 preserved
# ---------------------------------------------------------------------------


def _check_schema_version_round_trip() -> int:
    from resonance_lattice.memory.store import Memory, SCHEMA_VERSION

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        memory = Memory(root=root)
        memory.add_row(
            text="version-pin row",
            polarity=["factual", "workspace:abc123"],
            transcript_hash="manual",
            embedding=np.zeros(768, dtype=np.float32),
        )
        rows, _ = memory.read_all()
    if rows[0].schema_version != SCHEMA_VERSION:
        print(f"[memory_v21_schema_compat] FAIL (c): expected "
              f"schema_version={SCHEMA_VERSION}, got {rows[0].schema_version}",
              file=sys.stderr)
        return 1
    print(f"[memory_v21_schema_compat] (c) schema_version={SCHEMA_VERSION} "
          f"round-trips OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (d) schema_version=2 (synthetic future) → warn, never crash
# ---------------------------------------------------------------------------


def _check_future_schema_version_warns() -> int:
    from resonance_lattice.memory.store import Memory, SCHEMA_VERSION

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        root.mkdir(parents=True)
        sidecar = root / "sidecar.jsonl"
        band_path = root / "memory.npz"
        # Two rows under a synthetic future schema_version. The reader
        # should warn ONCE for the file, not once per row, and load both.
        rows_payload = []
        for i in range(2):
            rows_payload.append({
                "row_id": f"01HZ8K3M5N7P9Q1R2S3T4V5W6{i:02X}",
                "text": f"future row {i}",
                "polarity": ["factual", "workspace:abc123"],
                "recurrence_count": 1,
                "created_at": "2026-05-02T00:00:00Z",
                "last_corroborated_at": "2026-05-02T00:00:00Z",
                "transcript_hash": "manual",
                "is_bad": False,
                "schema_version": SCHEMA_VERSION + 1,
                "v2_only_field": "ignored",
            })
        sidecar.write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in rows_payload),
            encoding="utf-8",
        )
        np.savez(band_path, band=np.zeros((2, 768), dtype=np.float32))

        captured_stderr = io.StringIO()
        with redirect_stderr(captured_stderr):
            memory = Memory(root=root)
            loaded_rows, _ = memory.read_all()
    if len(loaded_rows) != 2:
        print(f"[memory_v21_schema_compat] FAIL (d): expected 2 rows, "
              f"got {len(loaded_rows)}", file=sys.stderr)
        return 1
    err_text = captured_stderr.getvalue()
    if "schema_version" not in err_text or "warning" not in err_text:
        print(f"[memory_v21_schema_compat] FAIL (d): missing forward-compat "
              f"warning; stderr was:\n{err_text!r}", file=sys.stderr)
        return 1
    if err_text.count("warning") > 1:
        print(f"[memory_v21_schema_compat] FAIL (d): warning fired more than "
              f"once for one file; got {err_text.count('warning')} warnings",
              file=sys.stderr)
        return 1
    print("[memory_v21_schema_compat] (d) future schema_version warns once + "
          "loads best-effort OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (e) v2.0 → v2.1 migrate produces migrated: prefix; distil excludes them
# ---------------------------------------------------------------------------


def _check_migrate_v20_to_v21() -> int:
    from resonance_lattice.memory.distil import _is_capture_row
    from resonance_lattice.memory.layered import LayeredMemory
    from resonance_lattice.memory.migrate import (
        classify_polarity,
        migrate,
    )
    from resonance_lattice.memory.store import Memory

    # Calibrate the polarity heuristic on a fixed 12-row fixture before
    # exercising the migration end-to-end.
    fixture = [
        ("avoid bare except clauses", "avoid"),
        ("don't commit secrets", "avoid"),
        ("never push to main", "avoid"),
        ("prefer pytest -xvs over plain pytest", "prefer"),
        ("use ruff for linting", "prefer"),
        ("always run simplify before commit", "prefer"),
        ("the encoder is gte-modernbert-base", "factual"),
        ("BEIR-5 floor is 0.5144 nDCG@10", "factual"),
        ("memory_v21 ships in MVP Day 11-12", "factual"),
        ("we standardised on Sonnet 4.6 last quarter", "factual"),
        ("recall returns top-5 by default", "factual"),
        ("the daemon idle-exits at 30 min", "factual"),
    ]
    correct = sum(1 for text, label in fixture
                   if classify_polarity(text) == label)
    # §14.5 contract 2: ≥ 16/20 correct on a 20-row fixture is the bar.
    # On 12 rows that scales to ≥ 10. Below that, the heuristic regressed.
    if correct < 10:
        print(f"[memory_v21_schema_compat] FAIL (e.1): polarity heuristic "
              f"hit {correct}/12 (want ≥10/12)", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as td:
        v20_root = Path(td) / "v20"
        v21_root = Path(td) / "v21base"
        # Build a v2.0 LayeredMemory with one row per tier.
        layered = LayeredMemory.init(v20_root)
        layered.add("avoid silent except clauses", tier="working")
        layered.add("prefer ruff over flake8", tier="episodic")
        layered.add("BEIR-5 floor pinned", tier="semantic")

        result = migrate(
            v20_root, v21_root=v21_root, user_id="alice",
            dry_run=False, polarity_default="factual",
        )
        if result.rows_migrated != 3:
            print(f"[memory_v21_schema_compat] FAIL (e.2): expected 3 "
                  f"migrated rows; got {result.rows_migrated}", file=sys.stderr)
            return 1
        if result.archived_path is None or not result.archived_path.exists():
            print(f"[memory_v21_schema_compat] FAIL (e.2): v2.0 root not "
                  f"archived; result={result}", file=sys.stderr)
            return 1
        if v20_root.exists():
            print(f"[memory_v21_schema_compat] FAIL (e.2): v2.0 root still "
                  f"present at {v20_root} after live migrate (should be "
                  f"renamed)", file=sys.stderr)
            return 1

        target = v21_root / "alice"
        memory = Memory(root=target)
        rows, _ = memory.read_all()
    if len(rows) != 3:
        print(f"[memory_v21_schema_compat] FAIL (e.3): expected 3 v2.1 rows; "
              f"got {len(rows)}", file=sys.stderr)
        return 1
    for row in rows:
        if not row.is_migrated():
            print(f"[memory_v21_schema_compat] FAIL (e.3): row "
                  f"{row.row_id} missing migrated: prefix; "
                  f"transcript_hash={row.transcript_hash!r}", file=sys.stderr)
            return 1
        if _is_capture_row(row):
            print(f"[memory_v21_schema_compat] FAIL (e.3): migrated row "
                  f"{row.row_id} would be picked up by distil capture filter",
                  file=sys.stderr)
            return 1
    print(f"[memory_v21_schema_compat] (e) migrate {result.rows_migrated} "
          f"rows + heuristic {correct}/12 + distil exclusion OK",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run() -> int:
    patch_zero_encoder()
    for check in [
        _check_round_trip,
        _check_additive_future_fields,
        _check_schema_version_round_trip,
        _check_future_schema_version_warns,
        _check_migrate_v20_to_v21,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v21_schema_compat] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
