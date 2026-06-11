"""state_attribution — recall cache + attribution + end-to-end loop closure.

Pins architecture §"Attribution — linking outcomes to memory rows" and
§"When attribution is computed". Eight contracts:

  (a) RecallCache append + read round-trip preserves entries in order.

  (b) Cache trims to cache_size when overflowing; oldest entries drop.

  (c) `read_since(iso)` returns only entries at or after the timestamp.

  (d) Tier mapping: rank 0 → primary; rank 3 → secondary; rank 6 →
      incidental.

  (e) Best-tier-wins: a row that surfaces at rank 1 in one entry and
      rank 6 in another keeps `primary` (the highest tier observed).

  (f) Empty entries → empty attribution list.

  (g) `make_turn_id` deterministic for same prompt+timestamp; differs
      across prompts.

  (h) End-to-end: simulate a prompt → cached recall → intent accept →
      OutcomeRecord lands with non-empty attribution.

Hermetic — temp dir + synthetic recall entries; no daemon, no encoder.
"""

from __future__ import annotations

import io
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


def _run(argv: list[str]) -> tuple[int, str, str]:
    from resonance_lattice.cli.app import main
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = main(argv)
    return rc, out.getvalue(), err.getvalue()


def _check_cache_round_trip() -> int:
    from resonance_lattice.state import (
        RecallCache,
        RecallEntry,
        RecallHitMetadata,
    )

    with tempfile.TemporaryDirectory() as td:
        cache = RecallCache(Path(td))
        for i in range(3):
            cache.append(RecallEntry(
                turn_id=f"t{i}",
                timestamp=f"2026-05-08T00:00:0{i}Z",
                prompt_hash=f"h{i}",
                intent_kind="implement",
                row_metadata=[
                    RecallHitMetadata(claim_id=f"r{i}", rank=0, cosine=0.9),
                ],
            ))
        loaded = cache.read_recent()
    if [e.turn_id for e in loaded] != ["t0", "t1", "t2"]:
        print(f"[state_attribution] FAIL (a): order={[e.turn_id for e in loaded]!r}",
              file=sys.stderr)
        return 1
    if loaded[0].row_metadata[0].claim_id != "r0":
        print(f"[state_attribution] FAIL (a): metadata lost",
              file=sys.stderr)
        return 1
    print("[state_attribution] (a) cache round-trip OK", file=sys.stderr)
    return 0


def _check_cache_trim() -> int:
    from resonance_lattice.state import RecallCache, RecallEntry

    with tempfile.TemporaryDirectory() as td:
        cache = RecallCache(Path(td), cache_size=3)
        for i in range(5):
            cache.append(RecallEntry(
                turn_id=f"t{i}",
                timestamp=f"2026-05-08T00:00:0{i}Z",
                prompt_hash=f"h{i}",
                intent_kind="none",
            ))
        loaded = cache.read_recent()
    if [e.turn_id for e in loaded] != ["t2", "t3", "t4"]:
        print(f"[state_attribution] FAIL (b): trim={[e.turn_id for e in loaded]!r}",
              file=sys.stderr)
        return 1
    print("[state_attribution] (b) cache trim OK", file=sys.stderr)
    return 0


def _check_read_since() -> int:
    from resonance_lattice.state import RecallCache, RecallEntry

    with tempfile.TemporaryDirectory() as td:
        cache = RecallCache(Path(td))
        for i in range(3):
            cache.append(RecallEntry(
                turn_id=f"t{i}",
                timestamp=f"2026-05-08T00:00:0{i}Z",
                prompt_hash=f"h{i}",
                intent_kind="none",
            ))
        loaded = cache.read_since("2026-05-08T00:00:01Z")
    if [e.turn_id for e in loaded] != ["t1", "t2"]:
        print(f"[state_attribution] FAIL (c): since={[e.turn_id for e in loaded]!r}",
              file=sys.stderr)
        return 1
    print("[state_attribution] (c) read_since OK", file=sys.stderr)
    return 0


def _check_tier_mapping() -> int:
    from resonance_lattice.state import (
        RecallEntry,
        RecallHitMetadata,
        attribution_from_entries,
    )

    entry = RecallEntry(
        turn_id="t",
        timestamp="2026-05-08T00:00:00Z",
        prompt_hash="h",
        intent_kind="none",
        row_metadata=[
            RecallHitMetadata(claim_id="r0", rank=0, cosine=0.9),
            RecallHitMetadata(claim_id="r3", rank=3, cosine=0.7),
            RecallHitMetadata(claim_id="r6", rank=6, cosine=0.5),
        ],
    )
    attribution = {a.claim_id: a.tier for a in attribution_from_entries([entry])}
    expected = {"r0": "primary", "r3": "secondary", "r6": "incidental"}
    if attribution != expected:
        print(f"[state_attribution] FAIL (d): {attribution!r}", file=sys.stderr)
        return 1
    print("[state_attribution] (d) tier mapping OK", file=sys.stderr)
    return 0


def _check_best_tier_wins() -> int:
    from resonance_lattice.state import (
        RecallEntry,
        RecallHitMetadata,
        attribution_from_entries,
    )

    e1 = RecallEntry(
        turn_id="t1", timestamp="t1", prompt_hash="h1", intent_kind="none",
        row_metadata=[RecallHitMetadata(claim_id="r", rank=1, cosine=0.9)],
    )
    e2 = RecallEntry(
        turn_id="t2", timestamp="t2", prompt_hash="h2", intent_kind="none",
        row_metadata=[RecallHitMetadata(claim_id="r", rank=6, cosine=0.5)],
    )
    attribution = attribution_from_entries([e1, e2])
    if len(attribution) != 1 or attribution[0].tier != "primary":
        print(f"[state_attribution] FAIL (e): {attribution!r}", file=sys.stderr)
        return 1
    print("[state_attribution] (e) best-tier wins OK", file=sys.stderr)
    return 0


def _check_empty_entries() -> int:
    from resonance_lattice.state import attribution_from_entries

    if attribution_from_entries([]) != []:
        print("[state_attribution] FAIL (f): non-empty for empty input",
              file=sys.stderr)
        return 1
    print("[state_attribution] (f) empty entries OK", file=sys.stderr)
    return 0


def _check_make_turn_id_deterministic() -> int:
    from resonance_lattice.state import make_turn_id

    a = make_turn_id("how do I X?", timestamp="2026-05-08T00:00:00Z")
    b = make_turn_id("how do I X?", timestamp="2026-05-08T00:00:00Z")
    c = make_turn_id("how do I Y?", timestamp="2026-05-08T00:00:00Z")
    if a != b or a == c:
        print(f"[state_attribution] FAIL (g): a={a!r} b={b!r} c={c!r}",
              file=sys.stderr)
        return 1
    print("[state_attribution] (g) make_turn_id deterministic OK",
          file=sys.stderr)
    return 0


def _check_end_to_end_loop() -> int:
    """Cache → intent accept → OutcomeRecord with attribution."""
    from resonance_lattice.state import (
        ClaimOutcomeLog,
        RecallCache,
        RecallEntry,
        RecallHitMetadata,
        resolve_workspace,
        state_root_for,
    )

    with tempfile.TemporaryDirectory() as td:
        # Add an intent.
        rc, out, err = _run([
            "intent", "--cwd", td, "add", "ship the harness",
            "--level", "task",
        ])
        if rc != 0:
            print(f"[state_attribution] FAIL (h): add rc={rc} err={err!r}",
                  file=sys.stderr)
            return 1
        intent_id = out.strip()

        # Resolve workspace + write a synthetic cache entry that postdates
        # the intent's creation.
        identity = resolve_workspace(Path(td))
        state_root = state_root_for(identity.root)
        cache = RecallCache(state_root)
        cache.append(RecallEntry(
            turn_id="t1",
            timestamp="9999-12-31T23:59:59Z",  # always > intent.created_at
            prompt_hash="h",
            intent_kind="implement",
            row_metadata=[
                RecallHitMetadata(claim_id="01HZ_ROW_PRIMARY", rank=0, cosine=0.9),
                RecallHitMetadata(claim_id="01HZ_ROW_SECONDARY", rank=3, cosine=0.7),
            ],
        ))

        # Accept the intent.
        rc, out, err = _run(["intent", "--cwd", td, "accept", intent_id])
        if rc != 0 or "2 attributed" not in out:
            print(f"[state_attribution] FAIL (h): accept rc={rc} out={out!r}",
                  file=sys.stderr)
            return 1

        # Read the ledger and confirm attribution landed.
        ledger = ClaimOutcomeLog(state_root)
        records = ledger.read(intent_id=intent_id)
        if len(records) != 1:
            print(f"[state_attribution] FAIL (h): records={len(records)}",
                  file=sys.stderr)
            return 1
        attribution = {a.claim_id: a.tier for a in records[0].attribution}
        if attribution != {"01HZ_ROW_PRIMARY": "primary",
                           "01HZ_ROW_SECONDARY": "secondary"}:
            print(f"[state_attribution] FAIL (h): attribution={attribution!r}",
                  file=sys.stderr)
            return 1
    print("[state_attribution] (h) end-to-end recall→outcome→attribution OK",
          file=sys.stderr)
    return 0


def _check_intent_id_round_trip() -> int:
    """(i) `intent_id` survives `RecallEntry.to_dict` → JSONL → `read_recent`.
    Backward-compatible: pre-Horizon-4 entries with no `intent_id` field
    load with `intent_id=None`."""
    from resonance_lattice.state import RecallCache, RecallEntry

    with tempfile.TemporaryDirectory() as td:
        cache = RecallCache(Path(td))
        cache.append(RecallEntry(
            turn_id="t0",
            timestamp="2026-05-09T00:00:00Z",
            prompt_hash="h0",
            intent_kind="implement",
            intent_id="01HZTASK1",
        ))
        cache.append(RecallEntry(
            turn_id="t1",
            timestamp="2026-05-09T00:00:01Z",
            prompt_hash="h1",
            intent_kind="none",
            intent_id=None,  # no live intent at recall time
        ))
        loaded = cache.read_recent()
    if [e.intent_id for e in loaded] != ["01HZTASK1", None]:
        print(f"[state_attribution] FAIL (i): intent_id round-trip "
              f"got {[e.intent_id for e in loaded]!r}", file=sys.stderr)
        return 1
    print("[state_attribution] (i) intent_id round-trip OK", file=sys.stderr)
    return 0


def _check_read_for_intent() -> int:
    """(j) `read_for_intent(intent_id)` returns only entries stamped with
    that intent_id, ignoring entries with `intent_id=None` or different
    ids. The since_iso filter further bounds the match."""
    from resonance_lattice.state import RecallCache, RecallEntry

    with tempfile.TemporaryDirectory() as td:
        cache = RecallCache(Path(td))
        cache.append(RecallEntry(
            turn_id="t0", timestamp="2026-05-09T00:00:00Z",
            prompt_hash="h0", intent_kind="implement",
            intent_id="01HZTASK1",
        ))
        cache.append(RecallEntry(
            turn_id="t1", timestamp="2026-05-09T00:00:01Z",
            prompt_hash="h1", intent_kind="implement",
            intent_id="01HZTASK2",  # different intent
        ))
        cache.append(RecallEntry(
            turn_id="t2", timestamp="2026-05-09T00:00:02Z",
            prompt_hash="h2", intent_kind="none",
            intent_id=None,  # no live intent
        ))
        cache.append(RecallEntry(
            turn_id="t3", timestamp="2026-05-09T00:00:03Z",
            prompt_hash="h3", intent_kind="implement",
            intent_id="01HZTASK1",
        ))
        only_task1 = cache.read_for_intent("01HZTASK1")
        bounded = cache.read_for_intent(
            "01HZTASK1", since_iso="2026-05-09T00:00:02Z",
        )
        unknown = cache.read_for_intent("01HZNONE")
    if [e.turn_id for e in only_task1] != ["t0", "t3"]:
        print(f"[state_attribution] FAIL (j): intent match "
              f"got {[e.turn_id for e in only_task1]!r}", file=sys.stderr)
        return 1
    if [e.turn_id for e in bounded] != ["t3"]:
        print(f"[state_attribution] FAIL (j): since filter "
              f"got {[e.turn_id for e in bounded]!r}", file=sys.stderr)
        return 1
    if unknown:
        print(f"[state_attribution] FAIL (j): unknown intent_id should "
              f"return [], got {[e.turn_id for e in unknown]!r}",
              file=sys.stderr)
        return 1
    print("[state_attribution] (j) read_for_intent filters by intent_id + "
          "since_iso OK", file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_cache_round_trip,
        _check_cache_trim,
        _check_read_since,
        _check_tier_mapping,
        _check_best_tier_wins,
        _check_empty_entries,
        _check_make_turn_id_deterministic,
        _check_end_to_end_loop,
        _check_intent_id_round_trip,
        _check_read_for_intent,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[state_attribution] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
