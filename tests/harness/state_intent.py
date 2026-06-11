"""state_intent — intent store contracts (live graph + durable store).

Pins architecture §"Live intent — in agent-state" and claim-system-design
§5 (intents leave the claim table). Contracts:

  (a) add → list round-trip — added intents surface in `list_all`.

  (b) Live levels are constrained to {step, task}; `goal` and `direction`
      raise (those live in the per-user durable store).

  (c) Status transitions write to `transitions.jsonl`; idempotent re-set
      is a no-op (no duplicate log entry).

  (d) Decomposition records append-only; empty children list is a no-op.

  (e) Active-graph corruption (malformed JSON) returns empty rather than
      crashing — defence against abrupt endings.

  (f) Sibling-block edges add idempotently — re-adding the same block is
      a no-op.

  (g) record_decomposition holds the store lock while it appends.

  (h) DurableIntentStore add → list → read round-trip.

  (i) Durable levels are constrained to {goal, direction}; `step` and
      `task` raise. A live and a durable store rooted at the same path
      do not collide (separate subdirectories).

Hermetic — temp dir + portalocker; no encoder, no LLM.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def _check_add_and_list() -> int:
    from resonance_lattice.state import LiveIntentStore

    with tempfile.TemporaryDirectory() as td:
        store = LiveIntentStore(Path(td))
        a = store.add_intent(
            level="task",
            text="ship the harness",
            stance="do",
            achievability="medium",
            success_criteria=[{"text": "v1 lands", "measure": "user_confirms"}],
            constraints=["additive only"],
            created_under_intent_kind="implement",
        )
        b = store.add_intent(
            level="step",
            text="run the harness",
            stance="do",
            achievability="high",
            success_criteria=[{"text": "rc=0", "measure": "exit_code:0"}],
            constraints=[],
            parent_ids=[a.intent_id],
        )
        loaded = store.list_all()
    if [i.intent_id for i in loaded] != [a.intent_id, b.intent_id]:
        print(f"[state_intent] FAIL (a): order/ids drifted: "
              f"{[i.intent_id for i in loaded]!r}", file=sys.stderr)
        return 1
    if loaded[1].parent_ids != [a.intent_id]:
        print(f"[state_intent] FAIL (a): parent_ids lost: "
              f"{loaded[1].parent_ids!r}", file=sys.stderr)
        return 1
    print("[state_intent] (a) add + list round-trip OK", file=sys.stderr)
    return 0


def _check_live_level_constrained() -> int:
    from resonance_lattice.state import LiveIntentStore

    with tempfile.TemporaryDirectory() as td:
        store = LiveIntentStore(Path(td))
        for bad_level in ["goal", "direction", "event", "wisdom"]:
            try:
                store.add_intent(
                    level=bad_level,
                    text="bad level",
                    stance="do",
                    achievability="medium",
                    success_criteria=[],
                    constraints=[],
                )
            except ValueError:
                continue
            print(f"[state_intent] FAIL (b): bad level {bad_level!r} accepted",
                  file=sys.stderr)
            return 1
    print("[state_intent] (b) live level constrained to {step, task} OK",
          file=sys.stderr)
    return 0


def _check_set_status_logs_transition() -> int:
    from resonance_lattice.state import LiveIntentStore

    with tempfile.TemporaryDirectory() as td:
        store = LiveIntentStore(Path(td))
        intent = store.add_intent(
            level="task",
            text="t",
            stance="do",
            achievability="medium",
            success_criteria=[],
            constraints=[],
        )
        store.set_status(intent.intent_id, "satisfied", reason="criteria met")
        # Idempotent re-set must NOT log again.
        store.set_status(intent.intent_id, "satisfied", reason="re-fire")
        log = store.read_transitions()
    if len(log) != 1:
        print(f"[state_intent] FAIL (c): log entries={len(log)}", file=sys.stderr)
        return 1
    entry = log[0]
    if (entry["intent_id"] != intent.intent_id
            or entry["from"] != "active"
            or entry["to"] != "satisfied"
            or entry["reason"] != "criteria met"):
        print(f"[state_intent] FAIL (c): bad log entry: {entry!r}",
              file=sys.stderr)
        return 1
    print("[state_intent] (c) set_status logs once + idempotent OK",
          file=sys.stderr)
    return 0


def _check_decomposition_append() -> int:
    from resonance_lattice.state import LiveIntentStore

    with tempfile.TemporaryDirectory() as td:
        store = LiveIntentStore(Path(td))
        store.record_decomposition("01HZGOAL", [], rationale="empty")
        store.record_decomposition(
            "01HZGOAL", ["01HZTASK1", "01HZTASK2"], rationale="split goal"
        )
        log = store.read_decompositions()
    if len(log) != 1:
        print(f"[state_intent] FAIL (d): log entries={len(log)}", file=sys.stderr)
        return 1
    entry = log[0]
    if (entry["parent_intent_id"] != "01HZGOAL"
            or entry["child_intent_ids"] != ["01HZTASK1", "01HZTASK2"]):
        print(f"[state_intent] FAIL (d): bad entry: {entry!r}", file=sys.stderr)
        return 1
    print("[state_intent] (d) decomposition append-only OK", file=sys.stderr)
    return 0


def _check_corrupt_active_returns_empty() -> int:
    from resonance_lattice.state import LiveIntentStore, intent_dir

    with tempfile.TemporaryDirectory() as td:
        store = LiveIntentStore(Path(td))
        # Corrupt active.json with garbage; subsequent reads must not raise.
        active = intent_dir(Path(td)) / "active.json"
        active.write_text("{not valid json", encoding="utf-8")
        loaded = store.list_all()
        # And we can still add a fresh intent on top of it (write replaces).
        added = store.add_intent(
            level="task",
            text="recovered",
            stance="do",
            achievability="medium",
            success_criteria=[],
            constraints=[],
        )
        loaded2 = store.list_all()
    if loaded != []:
        print(f"[state_intent] FAIL (e): corrupt list_all not empty: "
              f"{loaded!r}", file=sys.stderr)
        return 1
    if [i.intent_id for i in loaded2] != [added.intent_id]:
        print(f"[state_intent] FAIL (e): post-recovery state wrong: "
              f"{loaded2!r}", file=sys.stderr)
        return 1
    print("[state_intent] (e) corrupt active.json returns empty + recovers OK",
          file=sys.stderr)
    return 0


def _check_decomposition_holds_lock() -> int:
    """record_decomposition runs inside `with self._lock():` — verified by
    instrumenting `_append_log` to non-blocking-probe the same lock file.
    A blocked probe means the outer `with self._lock()` is currently held."""
    from resonance_lattice.state import LiveIntentStore

    with tempfile.TemporaryDirectory() as td:
        store = LiveIntentStore(Path(td))
        intent = store.add_intent(
            level="task", text="t", stance="do", achievability="medium",
            success_criteria=[], constraints=[],
        )

        observed_lock_held: list[bool] = []
        original_append = store._append_log

        def _instrumented(log_name: str, entry: dict) -> None:
            # If another writer were inside `with self._lock()` right now,
            # acquiring the same lock from a daughter thread would block.
            # We probe by trying a non-blocking acquire of the same path.
            import portalocker
            try:
                probe = portalocker.Lock(
                    str(store._lock_path), mode="r+b",
                    flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
                )
                probe.acquire()
                probe.release()
                observed_lock_held.append(False)
            except portalocker.exceptions.LockException:
                observed_lock_held.append(True)
            original_append(log_name, entry)

        store._append_log = _instrumented  # type: ignore[method-assign]
        store.record_decomposition(intent.intent_id, ["01HZTASK2"])

    if observed_lock_held != [True]:
        print(f"[state_intent] FAIL (g): record_decomposition held lock="
              f"{observed_lock_held!r}", file=sys.stderr)
        return 1
    print("[state_intent] (g) record_decomposition holds the store lock OK",
          file=sys.stderr)
    return 0


def _check_add_block_idempotent() -> int:
    from resonance_lattice.state import LiveIntentStore

    with tempfile.TemporaryDirectory() as td:
        store = LiveIntentStore(Path(td))
        a = store.add_intent(
            level="task", text="A", stance="do", achievability="medium",
            success_criteria=[], constraints=[],
        )
        b = store.add_intent(
            level="task", text="B", stance="do", achievability="medium",
            success_criteria=[], constraints=[],
        )
        store.add_block(a.intent_id, b.intent_id)
        store.add_block(a.intent_id, b.intent_id)  # idempotent
        intents = {i.intent_id: i for i in store.list_all()}
    if intents[a.intent_id].blocks != [b.intent_id]:
        print(f"[state_intent] FAIL (f): blocks={intents[a.intent_id].blocks!r}",
              file=sys.stderr)
        return 1
    print("[state_intent] (f) add_block idempotent OK", file=sys.stderr)
    return 0


def _check_durable_add_read() -> int:
    from resonance_lattice.state import DurableIntentStore

    with tempfile.TemporaryDirectory() as td:
        store = DurableIntentStore(Path(td))
        goal = store.add_intent(
            level="goal",
            text="ship rlat v3",
            stance="do",
            achievability="medium",
            success_criteria=[{"text": "v3 lands", "measure": "user_confirms"}],
            constraints=["additive only"],
        )
        listed = store.list_all()
        fetched = store.read(goal.intent_id)
        missing = store.read("01HZNOSUCHINTENT0000000000")
    if [i.intent_id for i in listed] != [goal.intent_id]:
        print(f"[state_intent] FAIL (h): list drifted: {listed!r}",
              file=sys.stderr)
        return 1
    if fetched is None or fetched.text != "ship rlat v3":
        print(f"[state_intent] FAIL (h): read returned {fetched!r}",
              file=sys.stderr)
        return 1
    if missing is not None:
        print(f"[state_intent] FAIL (h): read of absent id returned "
              f"{missing!r}", file=sys.stderr)
        return 1
    print("[state_intent] (h) durable add + list + read round-trip OK",
          file=sys.stderr)
    return 0


def _check_durable_level_constrained() -> int:
    from resonance_lattice.state import DurableIntentStore, LiveIntentStore

    with tempfile.TemporaryDirectory() as td:
        durable = DurableIntentStore(Path(td))
        for bad_level in ["step", "task", "event", "wisdom"]:
            try:
                durable.add_intent(
                    level=bad_level,
                    text="bad level",
                    stance="do",
                    achievability="medium",
                    success_criteria=[],
                    constraints=[],
                )
            except ValueError:
                continue
            print(f"[state_intent] FAIL (i): durable accepted {bad_level!r}",
                  file=sys.stderr)
            return 1
        # A live and a durable store rooted at the same path use distinct
        # subdirectories — neither lock nor file collides.
        live = LiveIntentStore(Path(td))
        live.add_intent(
            level="task", text="t", stance="do", achievability="medium",
            success_criteria=[], constraints=[],
        )
        durable.add_intent(
            level="goal", text="g", stance="do", achievability="medium",
            success_criteria=[], constraints=[],
        )
        if (len(live.list_all()) != 1
                or len(durable.list_all()) != 1):
            print("[state_intent] FAIL (i): co-rooted stores collided",
                  file=sys.stderr)
            return 1
    print("[state_intent] (i) durable level constrained + no collision OK",
          file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_add_and_list,
        _check_live_level_constrained,
        _check_set_status_logs_transition,
        _check_decomposition_append,
        _check_corrupt_active_returns_empty,
        _check_decomposition_holds_lock,
        _check_add_block_idempotent,
        _check_durable_add_read,
        _check_durable_level_constrained,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[state_intent] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
