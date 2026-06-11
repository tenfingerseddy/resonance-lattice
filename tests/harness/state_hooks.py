"""state_hooks — trajectory + pending-signal contracts.

Pins the SessionStart trajectory primer and PostToolUse signal-capture
behaviour. The hooks themselves are tiny shells around `rlat` package code
— this suite exercises that code; the hook scripts are integration-tested
by the dogfood loop on Claude Code, not here.

Six contracts:

  (a) Empty workspace → empty trajectory (silent SessionStart).

  (b) Single active task surfaces with status + level + text.

  (c) Active path bounded to ≤4 nodes (architecture's bounded-by-design).

  (d) Recently-resolved transitions surface in the primer.

  (e) Pending signal append → read round-trip (mechanical Bash exit_code=0
      becomes value={"verdict": "satisfied"}).

  (f) Pending signal filter by `since` returns the post-window subset.

Hermetic — temp dir, no encoder, no LLM.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def _check_empty_trajectory() -> int:
    from resonance_lattice.state import render_trajectory_primer

    with tempfile.TemporaryDirectory() as td:
        out = render_trajectory_primer(Path(td))
    if out != "":
        print(f"[state_hooks] FAIL (a): empty trajectory was {out!r}",
              file=sys.stderr)
        return 1
    print("[state_hooks] (a) empty trajectory silent OK", file=sys.stderr)
    return 0


def _check_single_active_task() -> int:
    from resonance_lattice.state import LiveIntentStore, render_trajectory_primer

    with tempfile.TemporaryDirectory() as td:
        store = LiveIntentStore(Path(td))
        store.add_intent(
            level="task",
            text="ship the harness",
            stance="do",
            achievability="medium",
            success_criteria=[],
            constraints=[],
        )
        primer = render_trajectory_primer(Path(td))
    expected_substrings = [
        "## Active intents",
        "**task**",
        "[active]",
        "ship the harness",
    ]
    for sub in expected_substrings:
        if sub not in primer:
            print(f"[state_hooks] FAIL (b): primer missing {sub!r}: {primer!r}",
                  file=sys.stderr)
            return 1
    print("[state_hooks] (b) single active task surfaces OK", file=sys.stderr)
    return 0


def _check_path_bounded() -> int:
    from resonance_lattice.state import LiveIntentStore, render_trajectory_primer

    with tempfile.TemporaryDirectory() as td:
        store = LiveIntentStore(Path(td))
        # Build a 6-deep parent chain — primer must clip to ≤ 4 nodes.
        prev_id = None
        for depth in range(6):
            level = "task" if depth < 3 else "step"
            intent = store.add_intent(
                level=level,
                text=f"depth-{depth}",
                stance="do",
                achievability="medium",
                success_criteria=[],
                constraints=[],
                parent_ids=[prev_id] if prev_id else [],
            )
            prev_id = intent.intent_id
        primer = render_trajectory_primer(Path(td))
    # Count "depth-N" hits in the primer body — should be at most 4 because
    # of _MAX_ACTIVE_PATH_NODES + the path is rendered root-to-leaf.
    matches = sum(1 for i in range(6) if f"depth-{i}" in primer)
    if matches > 4:
        print(f"[state_hooks] FAIL (c): path matches {matches} (want ≤4): "
              f"{primer!r}", file=sys.stderr)
        return 1
    if matches == 0:
        print(f"[state_hooks] FAIL (c): no depth-* in primer: {primer!r}",
              file=sys.stderr)
        return 1
    print(f"[state_hooks] (c) path bounded ({matches} depths surfaced) OK",
          file=sys.stderr)
    return 0


def _check_recently_resolved_in_primer() -> int:
    from resonance_lattice.state import LiveIntentStore, render_trajectory_primer

    with tempfile.TemporaryDirectory() as td:
        store = LiveIntentStore(Path(td))
        a = store.add_intent(
            level="task", text="active-task", stance="do",
            achievability="medium", success_criteria=[], constraints=[],
        )
        b = store.add_intent(
            level="task", text="resolved-task", stance="do",
            achievability="medium", success_criteria=[], constraints=[],
        )
        store.set_status(b.intent_id, "satisfied", reason="done")
        primer = render_trajectory_primer(Path(td))
    if "Recently resolved" not in primer:
        print(f"[state_hooks] FAIL (d): no 'Recently resolved' header: {primer!r}",
              file=sys.stderr)
        return 1
    if "satisfied" not in primer:
        print(f"[state_hooks] FAIL (d): satisfied verdict not surfaced: {primer!r}",
              file=sys.stderr)
        return 1
    print("[state_hooks] (d) recently-resolved surfaces OK", file=sys.stderr)
    return 0


def _check_pending_signal_round_trip() -> int:
    from resonance_lattice.state import PendingSignalLog

    with tempfile.TemporaryDirectory() as td:
        log = PendingSignalLog(Path(td))
        sig = log.append(
            source="mechanical",
            tool_name="Bash",
            tool_payload={"cmd": "pytest"},
            value={"verdict": "satisfied"},
            intent_id="01HZTASK1",
        )
        loaded = log.read()
    if len(loaded) != 1:
        print(f"[state_hooks] FAIL (e): rows={len(loaded)}", file=sys.stderr)
        return 1
    got = loaded[0]
    if (got.tool_name != "Bash"
            or got.value != {"verdict": "satisfied"}
            or got.intent_id != "01HZTASK1"
            or got.captured_at != sig.captured_at):
        print(f"[state_hooks] FAIL (e): round-trip drift: {got!r}",
              file=sys.stderr)
        return 1
    print("[state_hooks] (e) pending signal round-trip OK", file=sys.stderr)
    return 0


def _check_pending_signal_since_filter() -> int:
    from resonance_lattice.state import PendingSignalLog

    with tempfile.TemporaryDirectory() as td:
        log = PendingSignalLog(Path(td))
        # Three signals — write timestamps directly to test the filter.
        # Use a thin wrapper because `append` stamps `captured_at` itself.
        for _ in range(3):
            log.append(
                source="mechanical",
                tool_name="Bash",
                tool_payload={},
                value={"verdict": "satisfied"},
            )
        all_signals = log.read()
        # Filter `since` set to the second signal's timestamp — should
        # return signals 2 and 3 only.
        target = all_signals[1].captured_at
        filtered = log.read(since=target)
    if len(filtered) < 2:
        print(f"[state_hooks] FAIL (f): since filter returned {len(filtered)} "
              f"(want ≥ 2)", file=sys.stderr)
        return 1
    print(f"[state_hooks] (f) since filter OK ({len(filtered)}/{len(all_signals)} "
          f"survived)", file=sys.stderr)
    return 0


def _check_pending_signal_ring_trim() -> int:
    """(f2) the pending-signals log is a ring buffer: on overflow the file
    trims to the most recent `cache_size` entries (2026-06 review — it was
    unbounded, appended on every tool call, fully re-parsed on every read)."""
    from resonance_lattice.state import PendingSignalLog

    with tempfile.TemporaryDirectory() as td:
        log = PendingSignalLog(Path(td), cache_size=10)
        for i in range(25):
            log.append(
                source="mechanical",
                tool_name=f"Tool{i}",
                tool_payload={},
                value={"verdict": "satisfied"},
            )
        kept = log.read()
    if len(kept) != 10 or kept[-1].tool_name != "Tool24" or kept[0].tool_name != "Tool15":
        print(f"[state_hooks] FAIL (f2): ring trim wrong — kept={len(kept)} "
              f"first={kept[0].tool_name if kept else None} "
              f"last={kept[-1].tool_name if kept else None}", file=sys.stderr)
        return 1
    print("[state_hooks] (f2) pending-signal ring trim (oldest dropped) OK",
          file=sys.stderr)
    return 0


def _check_post_tool_use_payload_redaction() -> int:
    """post-tool-use.py's `_safe_payload` drops raw command bodies, edit
    content, stdout, stderr, and file paths. Persists only verdict-relevant
    metadata: first word of Bash command, exit_code; basename + error flag
    for Edit/Write tools."""
    import importlib.util

    hook_path = (Path(__file__).resolve().parents[2]
                 / ".claude" / "hooks" / "post-tool-use.py")
    spec = importlib.util.spec_from_file_location("rlat_post_tool_use", hook_path)
    if spec is None or spec.loader is None:
        print(f"[state_hooks] FAIL (g): could not load {hook_path}",
              file=sys.stderr)
        return 1
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    secret = "OPENAI_API_KEY=sk-supersecret123secret"
    bash_in = {"command": f"pytest -xvs && echo {secret}", "timeout": 30000}
    bash_out = {"stdout": secret, "stderr": secret, "exit_code": 0}
    payload = mod._safe_payload("Bash", bash_in, bash_out)
    if any(secret in str(v) for v in payload.values()):
        print(f"[state_hooks] FAIL (g): secret leaked into Bash payload: "
              f"{payload!r}", file=sys.stderr)
        return 1
    if payload.get("command") != "pytest" or payload.get("exit_code") != 0:
        print(f"[state_hooks] FAIL (g): expected verdict metadata; got "
              f"{payload!r}", file=sys.stderr)
        return 1

    edit_in = {"file_path": "/abs/path/to/secret_file.py", "new_string": secret}
    edit_out = {"is_error": False}
    payload = mod._safe_payload("Edit", edit_in, edit_out)
    if any(secret in str(v) for v in payload.values()):
        print(f"[state_hooks] FAIL (g): secret leaked into Edit payload: "
              f"{payload!r}", file=sys.stderr)
        return 1
    if payload.get("file") != "secret_file.py":
        print(f"[state_hooks] FAIL (g): expected file basename; got "
              f"{payload!r}", file=sys.stderr)
        return 1
    if "/abs/path" in str(payload):
        print(f"[state_hooks] FAIL (g): absolute path leaked: {payload!r}",
              file=sys.stderr)
        return 1
    print("[state_hooks] (g) post-tool-use payload redaction OK",
          file=sys.stderr)
    return 0


def _check_post_tool_use_verdict_extraction() -> int:
    """post-tool-use.py reads Claude Code's actual Bash response shape.
    Architecture says PostToolUse captures mechanical signals; if the
    verdict path doesn't fire on real tool responses, the closed loop's
    confidence + forget paths get no signal. The live ledger had 231
    `verdict: unknown` entries because the hook only checked `exit_code`
    while Claude Code uses `returncode`."""
    import importlib.util

    hook_path = (Path(__file__).resolve().parents[2]
                 / ".claude" / "hooks" / "post-tool-use.py")
    spec = importlib.util.spec_from_file_location(
        "rlat_post_tool_use_verdict", hook_path,
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    cases = [
        # (label, tool_name, tool_response, expected_verdict)
        ("returncode-0", "Bash", {"returncode": 0, "stdout": "ok"}, "satisfied"),
        ("returncode-1", "Bash", {"returncode": 1, "stderr": "fail"}, "not_satisfied"),
        ("exit_code legacy", "Bash", {"exit_code": 0}, "satisfied"),
        ("interrupted", "Bash", {"interrupted": True, "returncode": 0}, "not_satisfied"),
        ("missing both", "Bash", {"stdout": "no code"}, "unknown"),
        ("non-dict response", "Bash", "raw text", "unknown"),
        ("Edit OK", "Edit", {"is_error": False}, "satisfied"),
        ("Edit error", "Edit", {"is_error": True}, "not_satisfied"),
        ("Read read-only tool", "Read", {"content": "x"}, None),
    ]
    for label, tool, response, want in cases:
        got = mod._verdict_from_payload(tool, response)
        if got != want:
            print(f"[state_hooks] FAIL (h): {label!r} → {got!r} (want {want!r})",
                  file=sys.stderr)
            return 1
    # _safe_payload should also surface `exit_code` from `returncode`.
    payload = mod._safe_payload(
        "Bash", {"command": "pytest"}, {"returncode": 0},
    )
    if payload.get("exit_code") != 0:
        print(f"[state_hooks] FAIL (h): _safe_payload returncode→exit_code: "
              f"{payload!r}", file=sys.stderr)
        return 1
    print("[state_hooks] (h) Bash verdict reads returncode + interrupted OK",
          file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_empty_trajectory,
        _check_single_active_task,
        _check_path_bounded,
        _check_recently_resolved_in_primer,
        _check_pending_signal_round_trip,
        _check_pending_signal_since_filter,
        _check_pending_signal_ring_trim,
        _check_post_tool_use_payload_redaction,
        _check_post_tool_use_verdict_extraction,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[state_hooks] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
