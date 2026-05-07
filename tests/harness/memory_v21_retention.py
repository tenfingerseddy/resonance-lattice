"""memory_v21_retention — §0.5 + Appendix D D.4 contracts.

Pins four invariants on the retention surface (gc as the only deletion
path; recurrence-driven retention; bad-vote rows preserved for
re-distil suppression):

  (a) §0.6 recurrence gate honoured. A row with `recurrence_count <
      M` (default M=3) is dropped from `recall()` even if it clears
      every other gate. This is the same invariant memory_v21_recall
      (c) tests, framed here from the retention angle: low-recurrence
      rows naturally stay below the injection gate, no automatic
      deletion required.

  (b) `gc --max-age-days N` removes rows whose `last_corroborated_at`
      is older than N days. Per §15.2 the age clock indexes
      `last_corroborated_at` (not `created_at`) so a row that
      corroborates again resets its eligibility window — the rule is
      "things with recurrence_count == 1 that haven't seen a new
      corroboration in a month."

  (c) gc never removes is_bad rows by default (kept for re-distil
      suppression per §0.5). With `--is-bad`, gc targets only
      is_bad rows. This is the §0.5 manual-escape-hatch contract:
      every other filter (`--polarity`, `--min-recurrence`,
      `--max-age-days`) skips is_bad rows unless `--is-bad` is also
      passed.

  (d) gc with `--max-age-days` skips a row whose `last_corroborated_at`
      is recent — even if `created_at` is older than the horizon.
      Corroboration resets the clock per §15.2.

Hermetic: time-injected sidecar fixture (we hand-write `created_at` /
`last_corroborated_at` ISO strings to fast-forward / rewind the clock
without `time.sleep`).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from ._testutil import patch_zero_encoder, run_cli


def _seed_row(memory, *, text: str, primary: str = "factual",
              recurrence: int = 5,
              created_at: str = "2026-04-01T00:00:00Z",
              last_corroborated_at: str | None = None,
              is_bad: bool = False,
              transcript_hash: str = "manual") -> str:
    """Add one row, then update its mutable fields to the specified
    state. Used by every D.4 case to plant rows at hand-picked age +
    recurrence + is_bad configurations.
    """
    row_id = memory.add_row(
        text=text,
        polarity=[primary, "workspace:abc123"],
        transcript_hash=transcript_hash,
        embedding=np.zeros(768, dtype=np.float32),
    )
    fields: dict = {
        "recurrence_count": recurrence,
        "is_bad": is_bad,
        "last_corroborated_at": last_corroborated_at or created_at,
    }
    memory.update_row(row_id, **fields)
    # `created_at` is immutable on `update_row`; rewrite via direct
    # sidecar mutation since the test needs to fast-forward the clock.
    import json as _json
    sidecar = memory.root / "sidecar.jsonl"
    lines = sidecar.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for line in lines:
        obj = _json.loads(line)
        if obj["row_id"] == row_id:
            obj["created_at"] = created_at
        out.append(_json.dumps(obj, sort_keys=True))
    sidecar.write_text("\n".join(out), encoding="utf-8")
    return row_id


# ---------------------------------------------------------------------------
# (a) §0.6 recurrence gate honoured
# ---------------------------------------------------------------------------


def _check_recurrence_gate() -> int:
    from resonance_lattice.memory._common import workspace_hash
    from resonance_lattice.memory.recall import recall
    from resonance_lattice.memory.store import Memory
    from ._testutil import FixedEncoder

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        query_vec = np.zeros(768, dtype=np.float32)
        query_vec[0] = 1.0
        memory = Memory(root=root, encoder=FixedEncoder(query_vec))

        # Two rows with cosine well above floor + adequate gap; one with
        # recurrence below M, one above. Only the high-recurrence row
        # should survive the §0.6 gates.
        for i, recurrence in enumerate([1, 5]):
            emb = np.zeros(768, dtype=np.float32)
            emb[0] = 0.9 - 0.1 * i  # cosines: 0.9, 0.8 → gap ≥ 0.05
            emb[1] = float(np.sqrt(max(0.0, 1.0 - emb[0] * emb[0])))
            row_id = memory.add_row(
                text=f"row {i} recurrence={recurrence}",
                polarity=["factual", "workspace:abc123"],
                transcript_hash=f"manual-{i}",
                embedding=emb,
            )
            memory.update_row(row_id, recurrence_count=recurrence)

        hits = recall("anything", store=memory, cwd_hash="abc123", top_k=10)
    if len(hits) != 1 or hits[0].row.recurrence_count != 5:
        print(f"[memory_v21_retention] FAIL (a): expected 1 hit with "
              f"recurrence=5; got {[(h.row.recurrence_count, h.cosine) for h in hits]}",
              file=sys.stderr)
        return 1
    print("[memory_v21_retention] (a) §0.6 recurrence gate drops below-M "
          "row OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (b) gc --max-age-days removes by last_corroborated_at
# ---------------------------------------------------------------------------


def _check_gc_max_age() -> int:
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base"
        memory = Memory(root=base / "u")
        old_id = _seed_row(
            memory, text="stale row",
            created_at="2025-01-01T00:00:00Z",
            last_corroborated_at="2025-01-01T00:00:00Z",
        )
        recent_id = _seed_row(
            memory, text="recent row",
            created_at="2026-05-01T00:00:00Z",
            last_corroborated_at="2026-05-01T00:00:00Z",
        )

        rc, out, err = run_cli([
            "memory", "--memory-root", str(base), "--user", "u",
            "gc", "--max-age-days", "30",
        ])
        if rc != 0:
            print(f"[memory_v21_retention] FAIL (b): gc rc={rc}\n"
                  f"out:{out}\nerr:{err}", file=sys.stderr)
            return 1
        rows, _ = memory.read_all()
        ids_left = {r.row_id for r in rows}
    if old_id in ids_left:
        print(f"[memory_v21_retention] FAIL (b): old row {old_id} should "
              f"have been deleted by --max-age-days 30", file=sys.stderr)
        return 1
    if recent_id not in ids_left:
        print(f"[memory_v21_retention] FAIL (b): recent row {recent_id} "
              f"should have been preserved", file=sys.stderr)
        return 1
    print("[memory_v21_retention] (b) gc --max-age-days removes by "
          "last_corroborated_at OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (c) gc skips is_bad rows by default; --is-bad targets only is_bad rows
# ---------------------------------------------------------------------------


def _check_gc_isbad_preservation() -> int:
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base"
        memory = Memory(root=base / "u")
        # Both rows are old + low-recurrence + same polarity, but one
        # is_bad. A `--max-age-days 30 --polarity factual` sweep should
        # delete the not-bad row and PRESERVE the is_bad row.
        bad_id = _seed_row(
            memory, text="bad-voted row", recurrence=1, is_bad=True,
            created_at="2025-01-01T00:00:00Z",
            last_corroborated_at="2025-01-01T00:00:00Z",
        )
        normal_id = _seed_row(
            memory, text="normal stale row", recurrence=1, is_bad=False,
            created_at="2025-01-01T00:00:00Z",
            last_corroborated_at="2025-01-01T00:00:00Z",
        )

        rc, out, err = run_cli([
            "memory", "--memory-root", str(base), "--user", "u",
            "gc", "--max-age-days", "30",
        ])
        if rc != 0:
            print(f"[memory_v21_retention] FAIL (c): default gc rc={rc}\n"
                  f"out:{out}\nerr:{err}", file=sys.stderr)
            return 1
        rows, _ = memory.read_all()
        ids_after_default = {r.row_id for r in rows}
        if bad_id not in ids_after_default:
            print(f"[memory_v21_retention] FAIL (c): default gc removed "
                  f"is_bad row {bad_id}; bad rows must be preserved unless "
                  f"--is-bad is passed", file=sys.stderr)
            return 1
        if normal_id in ids_after_default:
            print(f"[memory_v21_retention] FAIL (c): default gc failed to "
                  f"remove normal stale row {normal_id}", file=sys.stderr)
            return 1

        # --is-bad now targets ONLY the is_bad row.
        rc, out, err = run_cli([
            "memory", "--memory-root", str(base), "--user", "u",
            "gc", "--is-bad",
        ])
        if rc != 0:
            print(f"[memory_v21_retention] FAIL (c): --is-bad gc rc={rc}\n"
                  f"out:{out}\nerr:{err}", file=sys.stderr)
            return 1
        rows, _ = memory.read_all()
        ids_after_isbad = {r.row_id for r in rows}
        if bad_id in ids_after_isbad:
            print(f"[memory_v21_retention] FAIL (c): --is-bad gc failed to "
                  f"remove is_bad row {bad_id}", file=sys.stderr)
            return 1
    print("[memory_v21_retention] (c) gc skips is_bad by default + "
          "--is-bad targets only is_bad OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (d) gc --max-age-days skips rows recently corroborated
# ---------------------------------------------------------------------------


def _check_gc_corroboration_resets_clock() -> int:
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base"
        memory = Memory(root=base / "u")
        # Created long ago BUT corroborated yesterday — gc must not
        # remove this row even though `created_at` is well past the
        # horizon. The clock is `last_corroborated_at` per §15.2.
        kept_id = _seed_row(
            memory, text="old but recently corroborated",
            recurrence=4,
            created_at="2025-01-01T00:00:00Z",
            last_corroborated_at="2026-05-01T00:00:00Z",
        )

        rc, _, _ = run_cli([
            "memory", "--memory-root", str(base), "--user", "u",
            "gc", "--max-age-days", "30",
        ])
        if rc != 0:
            print(f"[memory_v21_retention] FAIL (d): gc rc={rc}", file=sys.stderr)
            return 1
        rows, _ = memory.read_all()
        if kept_id not in {r.row_id for r in rows}:
            print(f"[memory_v21_retention] FAIL (d): row {kept_id} with "
                  f"recent last_corroborated_at was deleted by --max-age-days; "
                  f"corroboration must reset the clock", file=sys.stderr)
            return 1
    print("[memory_v21_retention] (d) corroboration resets the gc age clock "
          "OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run() -> int:
    patch_zero_encoder()
    for check in [
        _check_recurrence_gate,
        _check_gc_max_age,
        _check_gc_isbad_preservation,
        _check_gc_corroboration_resets_clock,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v21_retention] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
