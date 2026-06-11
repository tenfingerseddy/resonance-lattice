"""cli_intent_durable — `rlat intent declare-durable` + `intent durable`.

Pins the intent-interrogation skill's CLI write seam. Durable intents
(goals + directions) live in the per-user `DurableIntentStore`, separate
from both the memory store and the per-workspace live intent graph
(claim-system-design §5). Seven contracts:

  (a) declare-durable writes a goal to the per-user durable intent store.

  (b) declare-durable writes a direction similarly.

  (c) declare-durable rejects step/task levels (those go through
      `intent add` to the live store).

  (d) Required intent fields land — stance / achievability / status /
      success_criteria / constraints all populated.

  (e) `intent durable` lists the declared goals and directions.

  (f) `intent durable --level goal` filters to goal only.

  (g) `intent path` walks live → durable in one cross-store chain.

Hermetic — temp `--memory-root` per test; no real LLM, no network.
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
        try:
            rc = main(argv)
        except SystemExit as exit_:
            # argparse calls sys.exit(2) on invalid --choice; capture so
            # the rejection-path test (c) can assert rc!=0.
            code = exit_.code
            rc = code if isinstance(code, int) else 2
    return rc, out.getvalue(), err.getvalue()


def _common_args(td: str, *more: str) -> list[str]:
    return [
        "intent", "--cwd", td, "declare-durable",
        "--memory-root", str(Path(td) / "mem"),
        "--user", "alice",
        *more,
    ]


def _open_durable(td: str):
    from resonance_lattice.memory.store import path_for_user
    from resonance_lattice.state import DurableIntentStore
    return DurableIntentStore(
        path_for_user(user_id="alice", root=Path(td) / "mem")
    )


def _check_declare_goal() -> int:
    with tempfile.TemporaryDirectory() as td:
        rc, out, err = _run(_common_args(td,
            "ship rlat v3", "--level", "goal",
            "--criterion", "user_confirms=20-session benchmark passes",
            "--constraint", "additive only",
        ))
        if rc != 0:
            print(f"[cli_intent_durable] FAIL (a): rc={rc} err={err!r}",
                  file=sys.stderr)
            return 1
        intent = _open_durable(td).read(out.strip())
    if intent is None:
        print("[cli_intent_durable] FAIL (a): intent not found",
              file=sys.stderr)
        return 1
    if intent.level != "goal" or intent.status != "active":
        print(f"[cli_intent_durable] FAIL (a): bad fields {intent!r}",
              file=sys.stderr)
        return 1
    print("[cli_intent_durable] (a) declare goal OK", file=sys.stderr)
    return 0


def _check_declare_direction() -> int:
    with tempfile.TemporaryDirectory() as td:
        rc, out, _ = _run(_common_args(td,
            "become a memory-systems specialist",
            "--level", "direction",
        ))
        if rc != 0:
            print(f"[cli_intent_durable] FAIL (b): rc={rc}", file=sys.stderr)
            return 1
        intent = _open_durable(td).read(out.strip())
    if intent is None or intent.level != "direction":
        print("[cli_intent_durable] FAIL (b): direction intent missing",
              file=sys.stderr)
        return 1
    print("[cli_intent_durable] (b) declare direction OK", file=sys.stderr)
    return 0


def _check_rejects_step_task_level() -> int:
    with tempfile.TemporaryDirectory() as td:
        # argparse rejects --level=step at the choices-list level, so the
        # CLI returns rc!=0 (typically 2 from argparse).
        rc, _, err = _run(_common_args(td, "x", "--level", "step"))
        if rc == 0:
            print("[cli_intent_durable] FAIL (c): step accepted",
                  file=sys.stderr)
            return 1
    print("[cli_intent_durable] (c) rejects step/task level OK",
          file=sys.stderr)
    return 0


def _check_intent_fields_populated() -> int:
    with tempfile.TemporaryDirectory() as td:
        rc, out, _ = _run(_common_args(td,
            "ship a thing", "--level", "goal",
            "--stance", "do", "--achievability", "high",
            "--criterion", "user_confirms=ships",
            "--constraint", "stays additive",
        ))
        intent = _open_durable(td).read(out.strip())
    if intent is None:
        print("[cli_intent_durable] FAIL (d): intent not found",
              file=sys.stderr)
        return 1
    if (intent.stance != "do" or intent.achievability != "high"
            or intent.status != "active"
            or intent.success_criteria != [{"text": "ships",
                                            "measure": "user_confirms"}]
            or intent.constraints != ["stays additive"]):
        print(f"[cli_intent_durable] FAIL (d): {intent!r}", file=sys.stderr)
        return 1
    print("[cli_intent_durable] (d) intent fields populated OK",
          file=sys.stderr)
    return 0


def _check_durable_lists_declared() -> int:
    with tempfile.TemporaryDirectory() as td:
        _run(_common_args(td, "build rlat v3", "--level", "goal"))
        _run(_common_args(td, "ship harness", "--level", "direction"))
        rc, out, _ = _run([
            "intent", "--cwd", td, "durable",
            "--memory-root", str(Path(td) / "mem"),
            "--user", "alice",
        ])
    if rc != 0:
        print(f"[cli_intent_durable] FAIL (e): rc={rc}", file=sys.stderr)
        return 1
    if "build rlat v3" not in out or "ship harness" not in out:
        print(f"[cli_intent_durable] FAIL (e): durable rows missing: {out!r}",
              file=sys.stderr)
        return 1
    print("[cli_intent_durable] (e) durable list shows declared intents OK",
          file=sys.stderr)
    return 0


def _check_durable_filter_by_level() -> int:
    with tempfile.TemporaryDirectory() as td:
        _run(_common_args(td, "build rlat v3", "--level", "goal"))
        _run(_common_args(td, "ship harness", "--level", "direction"))
        rc, out, _ = _run([
            "intent", "--cwd", td, "durable",
            "--level", "goal",
            "--memory-root", str(Path(td) / "mem"),
            "--user", "alice",
        ])
    if rc != 0:
        print(f"[cli_intent_durable] FAIL (f): rc={rc}", file=sys.stderr)
        return 1
    if "ship harness" in out:
        print(f"[cli_intent_durable] FAIL (f): direction surfaced in --level "
              f"goal: {out!r}", file=sys.stderr)
        return 1
    if "build rlat v3" not in out:
        print(f"[cli_intent_durable] FAIL (f): goal missing: {out!r}",
              file=sys.stderr)
        return 1
    print("[cli_intent_durable] (f) --level filter OK", file=sys.stderr)
    return 0


def _check_path_cross_store_chain() -> int:
    """`rlat intent path <leaf>` walks live → durable in one chain.

    Seeds a goal in the per-user durable store, then a live task whose
    parent is that goal, then a live step under the task. Walking from
    the step must surface all three rows in root → leaf order.
    """
    from resonance_lattice.state import (
        LiveIntentStore,
        resolve_workspace,
        state_root_for,
    )

    with tempfile.TemporaryDirectory() as td:
        # Goal lives in the durable store.
        rc, goal_out, _ = _run(_common_args(td,
            "ship rlat v3", "--level", "goal",
        ))
        if rc != 0:
            print("[cli_intent_durable] FAIL (g): goal seed failed",
                  file=sys.stderr)
            return 1
        goal_id = goal_out.strip()

        # Task + step land in the live store with the goal as ancestor.
        identity = resolve_workspace(Path(td))
        live = LiveIntentStore(state_root_for(identity.root))
        task = live.add_intent(
            level="task", text="harness the v3 ship",
            stance="do", achievability="medium",
            success_criteria=[], constraints=[],
            parent_ids=[goal_id],
        )
        step = live.add_intent(
            level="step", text="write the runner",
            stance="do", achievability="medium",
            success_criteria=[], constraints=[],
            parent_ids=[task.intent_id],
        )

        rc, out, err = _run([
            "intent", "--cwd", td, "path", step.intent_id,
            "--memory-root", str(Path(td) / "mem"),
            "--user", "alice",
        ])
    if rc != 0:
        print(f"[cli_intent_durable] FAIL (g): rc={rc} err={err!r}",
              file=sys.stderr)
        return 1
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if len(lines) != 3:
        print(f"[cli_intent_durable] FAIL (g): expected 3 lines, got "
              f"{len(lines)}: {out!r}", file=sys.stderr)
        return 1
    if (goal_id not in lines[0]
            or task.intent_id not in lines[1]
            or step.intent_id not in lines[2]):
        print(f"[cli_intent_durable] FAIL (g): wrong order: {out!r}",
              file=sys.stderr)
        return 1
    if "durable" not in lines[0] or "live" not in lines[1] or "live" not in lines[2]:
        print(f"[cli_intent_durable] FAIL (g): wrong source tags: {out!r}",
              file=sys.stderr)
        return 1
    print("[cli_intent_durable] (g) intent path cross-store chain OK",
          file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_declare_goal,
        _check_declare_direction,
        _check_rejects_step_task_level,
        _check_intent_fields_populated,
        _check_durable_lists_declared,
        _check_durable_filter_by_level,
        _check_path_cross_store_chain,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[cli_intent_durable] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
