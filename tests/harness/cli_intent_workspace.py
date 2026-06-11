"""cli_intent_workspace — `rlat intent` + `rlat workspace` end-to-end.

Drives the CLI dispatcher with a temp workspace and exercises:

  (a) `rlat workspace declare` writes the declaration; `rlat workspace
      status` reads it back.

  (b) `rlat intent add` writes a live intent and prints the new id.

  (c) `rlat intent list` shows the row.

  (d) `rlat intent accept <id>` flips status to satisfied + writes a
      user-source pending signal with verdict satisfied.

  (e) `rlat intent reject <id>` on a fresh intent flips to abandoned +
      writes a user-source pending signal with verdict not_satisfied.

  (f) `rlat intent accept` on an unknown id surfaces a user error.

  (i) `rlat intent capture-plan` writes a proposed task + steps.

  (j) `rlat intent activate <id>` flips a proposed intent to active;
      an unknown id surfaces a user error.

Hermetic — temp dir + `--cwd`; no encoder, no LLM, no shell.
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


def _check_workspace_declare_status() -> int:
    with tempfile.TemporaryDirectory() as td:
        rc, out, err = _run([
            "workspace", "--cwd", td, "declare", "--name", "test-ws",
            "--id", "abc123",
        ])
        if rc != 0 or "abc123" not in out:
            print(f"[cli_intent_workspace] FAIL (a): declare rc={rc} "
                  f"out={out!r}", file=sys.stderr)
            return 1
        rc, out, _ = _run(["workspace", "--cwd", td, "status"])
        if rc != 0 or "workspace_id: abc123" not in out:
            print(f"[cli_intent_workspace] FAIL (a): status out={out!r}",
                  file=sys.stderr)
            return 1
    print("[cli_intent_workspace] (a) workspace declare/status OK",
          file=sys.stderr)
    return 0


def _add_intent_in(td: str) -> str:
    rc, out, err = _run([
        "intent", "--cwd", td, "add", "ship the harness",
        "--level", "task",
        "--criterion", "user_confirms=v1 lands",
        "--constraint", "additive only",
    ])
    if rc != 0:
        raise AssertionError(f"intent add failed: rc={rc} err={err!r}")
    return out.strip()


def _check_intent_add_then_list() -> int:
    with tempfile.TemporaryDirectory() as td:
        intent_id = _add_intent_in(td)
        if not intent_id or len(intent_id) != 26:
            print(f"[cli_intent_workspace] FAIL (b): bad id={intent_id!r}",
                  file=sys.stderr)
            return 1
        rc, out, _ = _run(["intent", "--cwd", td, "list"])
        if rc != 0 or intent_id not in out or "ship the harness" not in out:
            print(f"[cli_intent_workspace] FAIL (c): list out={out!r}",
                  file=sys.stderr)
            return 1
    print("[cli_intent_workspace] (b/c) intent add + list OK", file=sys.stderr)
    return 0


def _check_accept_writes_signal() -> int:
    from resonance_lattice.state import (
        LiveIntentStore,
        PendingSignalLog,
        resolve_workspace,
        state_root_for,
    )

    with tempfile.TemporaryDirectory() as td:
        intent_id = _add_intent_in(td)
        rc, out, _ = _run(["intent", "--cwd", td, "accept", intent_id,
                           "--reason", "looks good"])
        if rc != 0 or "satisfied" not in out:
            print(f"[cli_intent_workspace] FAIL (d): accept rc={rc} "
                  f"out={out!r}", file=sys.stderr)
            return 1
        identity = resolve_workspace(Path(td))
        state_root = state_root_for(identity.root)
        store = LiveIntentStore(state_root)
        intents = {i.intent_id: i for i in store.list_all()}
        if intents[intent_id].status != "satisfied":
            print(f"[cli_intent_workspace] FAIL (d): status="
                  f"{intents[intent_id].status!r}", file=sys.stderr)
            return 1
        signals = PendingSignalLog(state_root).read(intent_id=intent_id)
        if (len(signals) != 1
                or signals[0].source != "user"
                or signals[0].value != {"verdict": "satisfied"}):
            print(f"[cli_intent_workspace] FAIL (d): signal={signals!r}",
                  file=sys.stderr)
            return 1
    print("[cli_intent_workspace] (d) accept flips status + writes signal OK",
          file=sys.stderr)
    return 0


def _check_reject_writes_signal() -> int:
    from resonance_lattice.state import (
        LiveIntentStore,
        PendingSignalLog,
        resolve_workspace,
        state_root_for,
    )

    with tempfile.TemporaryDirectory() as td:
        intent_id = _add_intent_in(td)
        rc, out, _ = _run(["intent", "--cwd", td, "reject", intent_id])
        if rc != 0 or "not_satisfied" not in out:
            print(f"[cli_intent_workspace] FAIL (e): reject rc={rc} "
                  f"out={out!r}", file=sys.stderr)
            return 1
        identity = resolve_workspace(Path(td))
        state_root = state_root_for(identity.root)
        store = LiveIntentStore(state_root)
        intents = {i.intent_id: i for i in store.list_all()}
        if intents[intent_id].status != "abandoned":
            print(f"[cli_intent_workspace] FAIL (e): status="
                  f"{intents[intent_id].status!r}", file=sys.stderr)
            return 1
        signals = PendingSignalLog(state_root).read(intent_id=intent_id)
        if signals[0].value != {"verdict": "not_satisfied"}:
            print(f"[cli_intent_workspace] FAIL (e): value="
                  f"{signals[0].value!r}", file=sys.stderr)
            return 1
    print("[cli_intent_workspace] (e) reject flips status + writes signal OK",
          file=sys.stderr)
    return 0


def _check_accept_evaluates_declared_criteria() -> int:
    """(k) The keystone join end-to-end: an intent with *declared* criteria
    (one user_confirms, one mechanical) resolves through the real synthesiser,
    not the old synthetic stub. The outcome record carries one evaluated
    `CriterionCheck` per declared criterion — the user_confirms criterion
    satisfied by the accept signal, the mechanical criterion satisfied by a
    pre-seeded mechanical signal — and the roll-up is their AND.
    """
    from resonance_lattice.state import (
        ClaimOutcomeLog,
        LiveIntentStore,
        PendingSignalLog,
        resolve_state_root,
    )

    with tempfile.TemporaryDirectory() as td:
        state_root = resolve_state_root(Path(td))
        store = LiveIntentStore(state_root)
        intent = store.add_intent(
            level="task", text="ship S4", stance="do", achievability="medium",
            success_criteria=[
                {"text": "the user is happy", "measure": "user_confirms"},
                {"text": "tests pass", "measure": "mechanical:exit_code==0"},
            ],
            constraints=[],
        )
        # A mechanical signal the PostToolUse hook would have captured.
        PendingSignalLog(state_root).append(
            source="mechanical", tool_name="bash",
            tool_payload={"exit_code": 0},
            value={"verdict": "satisfied"}, intent_id=intent.intent_id,
        )
        rc, out, err = _run(["intent", "--cwd", td, "accept", intent.intent_id,
                             "--reason", "done"])
        if rc != 0:
            print(f"[cli_intent_workspace] FAIL (k): accept rc={rc} err={err!r}",
                  file=sys.stderr)
            return 1
        records = ClaimOutcomeLog(state_root).read(intent_id=intent.intent_id)
        if len(records) != 1:
            print(f"[cli_intent_workspace] FAIL (k): {len(records)} records",
                  file=sys.stderr)
            return 1
        checks = {c.criterion_text: c for c in records[0].details.criterion_checks}
        ok = (
            records[0].roll_up_verdict == "satisfied"
            and set(checks) == {"the user is happy", "tests pass"}
            and checks["the user is happy"].verdict == "satisfied"
            and checks["the user is happy"].measure == "user_confirms"
            and checks["tests pass"].verdict == "satisfied"
            and checks["tests pass"].measure == "mechanical:exit_code==0"
        )
        if not ok:
            print(f"[cli_intent_workspace] FAIL (k): record="
                  f"{records[0].details.criterion_checks!r} "
                  f"roll={records[0].roll_up_verdict!r}", file=sys.stderr)
            return 1
    print("[cli_intent_workspace] (k) accept evaluates declared criteria OK",
          file=sys.stderr)
    return 0


def _check_unknown_intent_user_error() -> int:
    with tempfile.TemporaryDirectory() as td:
        rc, _, err = _run(["intent", "--cwd", td, "accept", "01HZNOTEXIST"])
        if rc != 1 or "01HZNOTEXIST" not in err:
            print(f"[cli_intent_workspace] FAIL (f): rc={rc} err={err!r}",
                  file=sys.stderr)
            return 1
    print("[cli_intent_workspace] (f) unknown intent → user error OK",
          file=sys.stderr)
    return 0


def _check_path_live_chain() -> int:
    """`rlat intent path <leaf>` prints the live chain root → leaf."""
    from resonance_lattice.state import (
        LiveIntentStore,
        resolve_workspace,
        state_root_for,
    )

    with tempfile.TemporaryDirectory() as td:
        identity = resolve_workspace(Path(td))
        store = LiveIntentStore(state_root_for(identity.root))
        root = store.add_intent(
            level="task", text="ship the harness", stance="do",
            achievability="medium", success_criteria=[], constraints=[],
        )
        mid = store.add_intent(
            level="step", text="write the runner", stance="do",
            achievability="medium", success_criteria=[], constraints=[],
            parent_ids=[root.intent_id],
        )
        leaf = store.add_intent(
            level="step", text="green the smoke", stance="do",
            achievability="medium", success_criteria=[], constraints=[],
            parent_ids=[mid.intent_id],
        )
        rc, out, err = _run([
            "intent", "--cwd", td, "path", leaf.intent_id,
            "--user", "alice",
            "--memory-root", str(Path(td) / "mem"),
        ])
        if rc != 0:
            print(f"[cli_intent_workspace] FAIL (g): rc={rc} err={err!r}",
                  file=sys.stderr)
            return 1
        lines = [ln for ln in out.splitlines() if ln.strip()]
        if len(lines) != 3:
            print(f"[cli_intent_workspace] FAIL (g): expected 3 lines, got "
                  f"{len(lines)}: {out!r}", file=sys.stderr)
            return 1
        if (root.intent_id not in lines[0]
                or mid.intent_id not in lines[1]
                or leaf.intent_id not in lines[2]):
            print(f"[cli_intent_workspace] FAIL (g): wrong order: {out!r}",
                  file=sys.stderr)
            return 1
        # Indentation deepens with depth (root unindented, leaf indented).
        if lines[0].startswith(" ") or not lines[2].startswith("  "):
            print(f"[cli_intent_workspace] FAIL (g): bad indent: {out!r}",
                  file=sys.stderr)
            return 1
    print("[cli_intent_workspace] (g) intent path live chain OK",
          file=sys.stderr)
    return 0


def _check_path_unknown_id_user_error() -> int:
    with tempfile.TemporaryDirectory() as td:
        rc, _, err = _run([
            "intent", "--cwd", td, "path", "01HZNOTEXIST",
            "--user", "alice",
            "--memory-root", str(Path(td) / "mem"),
        ])
        if rc != 1 or "01HZNOTEXIST" not in err:
            print(f"[cli_intent_workspace] FAIL (h): rc={rc} err={err!r}",
                  file=sys.stderr)
            return 1
    print("[cli_intent_workspace] (h) intent path unknown id → user error OK",
          file=sys.stderr)
    return 0


def _check_capture_plan_writes_proposed() -> int:
    """(i) `rlat intent capture-plan` reads a PreToolUse ExitPlanMode hook
    payload from stdin and writes one task + N steps as `proposed`.

    Plan-mode capture is the harness's bridge from Claude Code's planning
    UX to the live intent graph. Architecture's "planning-mode-listening"
    surface (manifesto §"open question 3"); the published `ExitPlanMode`
    tool call is what we hook.
    """
    import json as _json
    from resonance_lattice.cli.app import main

    payload = {
        "tool_name": "ExitPlanMode",
        "tool_input": {
            "plan": (
                "## Approach\n\n"
                "I will:\n\n"
                "1. Add the parser flag\n"
                "2. Plumb through the validator\n"
                "3. Write a harness contract\n"
                "4. Run the gates\n"
            ),
        },
    }
    with tempfile.TemporaryDirectory() as td:
        payload["cwd"] = td
        out, err = io.StringIO(), io.StringIO()
        stdin = io.StringIO(_json.dumps(payload))
        # contextlib has no `redirect_stdin`; swap directly and restore.
        original_stdin = sys.stdin
        sys.stdin = stdin
        try:
            with redirect_stdout(out), redirect_stderr(err):
                rc = main(["intent", "--cwd", td, "capture-plan"])
        finally:
            sys.stdin = original_stdin
        if rc != 0:
            print(f"[cli_intent_workspace] FAIL (i): rc={rc} err={err.getvalue()!r}",
                  file=sys.stderr)
            return 1
        task_id = out.getvalue().strip()
        if len(task_id) != 26:
            print(f"[cli_intent_workspace] FAIL (i): expected ULID task id, "
                  f"got {task_id!r}", file=sys.stderr)
            return 1

        # Now read back via list --json to verify shape.
        out2, err2 = io.StringIO(), io.StringIO()
        with redirect_stdout(out2), redirect_stderr(err2):
            rc = main(["intent", "--cwd", td, "list", "--status", "proposed",
                       "--json"])
        if rc != 0:
            print(f"[cli_intent_workspace] FAIL (i): list rc={rc}",
                  file=sys.stderr)
            return 1
        rows = _json.loads(out2.getvalue())
        # Expect 1 task + 4 steps, all proposed, all task as parent.
        if len(rows) != 5:
            print(f"[cli_intent_workspace] FAIL (i): expected 5 proposed rows, "
                  f"got {len(rows)}", file=sys.stderr)
            return 1
        tasks = [r for r in rows if r["level"] == "task"]
        steps = [r for r in rows if r["level"] == "step"]
        if len(tasks) != 1 or len(steps) != 4:
            print(f"[cli_intent_workspace] FAIL (i): wrong split — "
                  f"{len(tasks)} tasks / {len(steps)} steps", file=sys.stderr)
            return 1
        if tasks[0]["text"] != "Approach":
            print(f"[cli_intent_workspace] FAIL (i): task title "
                  f"{tasks[0]['text']!r}, expected 'Approach'", file=sys.stderr)
            return 1
        for step in steps:
            if step["status"] != "proposed":
                print(f"[cli_intent_workspace] FAIL (i): step status "
                      f"{step['status']!r}", file=sys.stderr)
                return 1
            if tasks[0]["intent_id"] not in step["parent_ids"]:
                print(f"[cli_intent_workspace] FAIL (i): step missing "
                      f"task as parent: {step!r}", file=sys.stderr)
                return 1
    print("[cli_intent_workspace] (i) capture-plan writes proposed task + "
          "steps OK", file=sys.stderr)
    return 0


def _check_activate_flips_proposed() -> int:
    """(j) `rlat intent activate <id>` flips a proposed intent to active;
    an unknown id surfaces a user error."""
    from resonance_lattice.cli.app import main
    from resonance_lattice.state import resolve_state_root
    from resonance_lattice.state.intent import LiveIntentStore

    with tempfile.TemporaryDirectory() as td:
        store = LiveIntentStore(resolve_state_root(Path(td)))
        proposed = store.add_intent(
            level="task", text="adopt me", stance="do",
            achievability="medium", success_criteria=[], constraints=[],
            status="proposed",
        )
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = main(["intent", "--cwd", td, "activate", proposed.intent_id])
        if rc != 0 or "active:" not in out.getvalue():
            print(f"[cli_intent_workspace] FAIL (j): activate rc={rc} "
                  f"out={out.getvalue()!r} err={err.getvalue()!r}",
                  file=sys.stderr)
            return 1
        live = {i.intent_id: i for i in store.list_all()}
        if live[proposed.intent_id].status != "active":
            print(f"[cli_intent_workspace] FAIL (j): status="
                  f"{live[proposed.intent_id].status!r}", file=sys.stderr)
            return 1

        out2, err2 = io.StringIO(), io.StringIO()
        with redirect_stdout(out2), redirect_stderr(err2):
            rc = main(["intent", "--cwd", td, "activate", "NO_SUCH_ID"])
        if rc != 1:
            print(f"[cli_intent_workspace] FAIL (j): unknown-id rc={rc} "
                  f"(want 1)", file=sys.stderr)
            return 1
    print("[cli_intent_workspace] (j) activate flips proposed → active OK",
          file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_workspace_declare_status,
        _check_intent_add_then_list,
        _check_accept_writes_signal,
        _check_reject_writes_signal,
        _check_accept_evaluates_declared_criteria,
        _check_unknown_intent_user_error,
        _check_path_live_chain,
        _check_path_unknown_id_user_error,
        _check_capture_plan_writes_proposed,
        _check_activate_flips_proposed,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[cli_intent_workspace] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
