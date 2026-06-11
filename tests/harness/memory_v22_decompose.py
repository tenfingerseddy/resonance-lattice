"""memory_v22_decompose — task→step decomposition contracts.

Pins architecture §"Decomposition guidance for the LLM". Six contracts:

  (a) Successful decomposition writes step children with parent_ids
      pointing at the task; records a decomposition log entry.

  (b) v1 refuses non-task levels — direction/goal/step parents return
      a structured refusal without writing any children.

  (c) Fan-out range — a stub LLM that emits 1 step is rejected (below
      min); a stub that emits 12 steps is rejected (above max).

  (d) `refuse: true` from the LLM is honoured; no children written.

  (e) Step length cap — a stub that emits a 50-word "step" is rejected.

  (f) Non-JSON LLM output is treated as refusal, not crash.

Hermetic — synthetic Intent + fake LLM client; no encoder, no network.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import namedtuple
from pathlib import Path

from resonance_lattice.memory.decompose import decompose
from resonance_lattice.state import LiveIntentStore

LLMResponse = namedtuple("LLMResponse", "text input_tokens output_tokens")


def _seed_task(store: LiveIntentStore, *, level: str = "task") -> str:
    intent = store.add_intent(
        level=level,
        text="ship the harness",
        stance="do",
        achievability="medium",
        success_criteria=[{"text": "v1 lands", "measure": "user_confirms"}],
        constraints=["additive only"],
    )
    return intent.intent_id


def _llm_with_steps(steps: list[str]):
    return lambda system, msgs, tokens: LLMResponse(
        json.dumps({"steps": steps}), 100, 50,
    )


def _check_successful_decomposition() -> int:
    with tempfile.TemporaryDirectory() as td:
        store = LiveIntentStore(Path(td))
        task_id = _seed_task(store)
        parent = next(i for i in store.list_all() if i.intent_id == task_id)
        result = decompose(
            parent,
            llm=_llm_with_steps([
                "edit foo.py to add helper",
                "run tests/harness/foo.py",
                "open a PR",
            ]),
            store=store,
        )
        if result.refused or len(result.child_intent_ids) != 3:
            print(f"[memory_v22_decompose] FAIL (a): {result!r}",
                  file=sys.stderr)
            return 1
        intents = {i.intent_id: i for i in store.list_all()}
        children = [intents[cid] for cid in result.child_intent_ids]
        if any(c.level != "step" for c in children):
            print(f"[memory_v22_decompose] FAIL (a): non-step child levels",
                  file=sys.stderr)
            return 1
        if any(task_id not in c.parent_ids for c in children):
            print(f"[memory_v22_decompose] FAIL (a): parent_ids not wired",
                  file=sys.stderr)
            return 1
        decomp_log = store.read_decompositions()
        if not decomp_log or decomp_log[0]["parent_intent_id"] != task_id:
            print(f"[memory_v22_decompose] FAIL (a): decomp log missing: "
                  f"{decomp_log!r}", file=sys.stderr)
            return 1
    print("[memory_v22_decompose] (a) successful decomposition OK",
          file=sys.stderr)
    return 0


def _check_refuses_non_task_level() -> int:
    with tempfile.TemporaryDirectory() as td:
        store = LiveIntentStore(Path(td))
        step_id = _seed_task(store, level="step")
        step_intent = next(
            i for i in store.list_all() if i.intent_id == step_id
        )
        result = decompose(
            step_intent,
            llm=_llm_with_steps(["a", "b"]),
            store=store,
        )
        if not result.refused or "task→step only" not in result.rejection_reason:
            print(f"[memory_v22_decompose] FAIL (b): {result!r}",
                  file=sys.stderr)
            return 1
        # No new children should have been written.
        if len([i for i in store.list_all() if i.parent_ids]) != 0:
            print(f"[memory_v22_decompose] FAIL (b): children written",
                  file=sys.stderr)
            return 1
    print("[memory_v22_decompose] (b) non-task level refuses OK",
          file=sys.stderr)
    return 0


def _check_fan_out_range() -> int:
    with tempfile.TemporaryDirectory() as td:
        store = LiveIntentStore(Path(td))
        task_id = _seed_task(store)
        parent = next(i for i in store.list_all() if i.intent_id == task_id)
        # Below min.
        result = decompose(
            parent, llm=_llm_with_steps(["only one"]), store=store,
        )
        if not result.refused or "fan-out 1" not in result.rejection_reason:
            print(f"[memory_v22_decompose] FAIL (c.1): {result!r}",
                  file=sys.stderr)
            return 1
        # Above max.
        result = decompose(
            parent,
            llm=_llm_with_steps([f"step {i}" for i in range(12)]),
            store=store,
        )
        if not result.refused or "fan-out 12" not in result.rejection_reason:
            print(f"[memory_v22_decompose] FAIL (c.2): {result!r}",
                  file=sys.stderr)
            return 1
    print("[memory_v22_decompose] (c) fan-out range enforced OK",
          file=sys.stderr)
    return 0


def _check_explicit_refusal() -> int:
    with tempfile.TemporaryDirectory() as td:
        store = LiveIntentStore(Path(td))
        task_id = _seed_task(store)
        parent = next(i for i in store.list_all() if i.intent_id == task_id)
        refuse_llm = lambda system, msgs, tokens: LLMResponse(
            json.dumps({"refuse": True, "reason": "too vague to decompose"}),
            50, 20,
        )
        result = decompose(parent, llm=refuse_llm, store=store)
        if (not result.refused
                or "too vague" not in result.rejection_reason):
            print(f"[memory_v22_decompose] FAIL (d): {result!r}",
                  file=sys.stderr)
            return 1
    print("[memory_v22_decompose] (d) explicit refusal honoured OK",
          file=sys.stderr)
    return 0


def _check_step_length_cap() -> int:
    with tempfile.TemporaryDirectory() as td:
        store = LiveIntentStore(Path(td))
        task_id = _seed_task(store)
        parent = next(i for i in store.list_all() if i.intent_id == task_id)
        long_step = " ".join(["word"] * 50)
        result = decompose(
            parent,
            llm=_llm_with_steps([long_step, "short"]),
            store=store,
        )
        if not result.refused or "too long" not in result.rejection_reason:
            print(f"[memory_v22_decompose] FAIL (e): {result!r}",
                  file=sys.stderr)
            return 1
    print("[memory_v22_decompose] (e) step length cap OK", file=sys.stderr)
    return 0


def _check_non_json_treated_as_refusal() -> int:
    with tempfile.TemporaryDirectory() as td:
        store = LiveIntentStore(Path(td))
        task_id = _seed_task(store)
        parent = next(i for i in store.list_all() if i.intent_id == task_id)
        garbage_llm = lambda system, msgs, tokens: LLMResponse(
            "not json at all", 10, 5,
        )
        result = decompose(parent, llm=garbage_llm, store=store)
        if not result.refused or "non-JSON" not in result.rejection_reason:
            print(f"[memory_v22_decompose] FAIL (f): {result!r}",
                  file=sys.stderr)
            return 1
    print("[memory_v22_decompose] (f) non-JSON treated as refusal OK",
          file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_successful_decomposition,
        _check_refuses_non_task_level,
        _check_fan_out_range,
        _check_explicit_refusal,
        _check_step_length_cap,
        _check_non_json_treated_as_refusal,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v22_decompose] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
