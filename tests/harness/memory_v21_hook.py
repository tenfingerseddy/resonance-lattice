"""memory_v21_hook — v2.1 flat-memory hook + CLI surface contracts.

Pins seven invariants (Sub-MVP slice of Appendix D row D.5 in
.claude/plans/fabric-agent-flat-memory.md):

  (a) `Row.summary()` matches the §0.4 wire-format invariants used by the
      future hook — `<row_id>  [<primary>]  rec=<N>  <text>` with the
      primary polarity selected from the closed `{prefer, avoid, factual}`
      set, scope tags hidden from display.

  (c) Capture is fail-open. `capture()` against a Memory whose `add_row`
      raises `OSError` / `ValueError` / `portalocker.LockException` returns
      `CaptureResult(skip_reason="capture failed: <type>", row_id=None)`,
      never propagates the exception. The skip_reason carries the
      exception type but never the raw message — exceptions can leak
      paths or row text the redactor was protecting.

  (d) `_MAX_CAPTURED_CHARS` token-budget cap fires on a session whose
      assistant content exceeds the limit. Captured row text length is
      exactly `_MAX_CAPTURED_CHARS`; truncation is at row boundary
      (the cap point), never at encode time.

  (f) Capture pipeline is fail-open at the pipeline boundary, in addition
      to (c)'s store-failure path. The redactor and the gate are both
      synchronous-only paths and don't need their own fail-open contract;
      the test verifies the integration shape.

  (g) Manual `rlat memory add` stamps `workspace:<sha256[:6](cwd)>` by
      default. With `--scope cross-workspace`, the row carries BOTH the
      cwd workspace tag AND `cross-workspace`. Without one of them, the
      §0.6 retrieval pipeline drops the row and the row is unretrievable.

  (h) `--memory-root <base> --user alice` and `--user bob` write under
      `<base>/alice/` and `<base>/bob/` respectively, never overwriting
      one with the other.

  (i) CLI exit codes split deprecated (2) from pending-MVP (3) from
      user-error (1) from success (0). Verifies one representative of
      each across the §0.7 surface.

Sub-MVP issue: #94. Hermetic — no live encoder, no LLM calls, no real
network. Mocked encoder via the v2.0 `_testutil` pattern.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import portalocker

from ._testutil import ZeroEncoder, patch_zero_encoder, run_cli as _run_cli


# ---------------------------------------------------------------------------
# (a) Row.summary wire-format
# ---------------------------------------------------------------------------


def _check_summary_format() -> int:
    from resonance_lattice.memory.store import PRIMARY_POLARITY, Row

    row = Row(
        row_id="01HZ8K3M5N7P9Q1R2S3T4V5W6X",
        text="prefer pytest -xvs <path> for debugging",
        polarity=["prefer", "workspace:abc123"],
        recurrence_count=4,
        created_at="2026-05-01T12:00:00Z",
        last_corroborated_at="2026-05-01T12:00:00Z",
        transcript_hash="manual",
        is_bad=False,
    )
    s = row.summary()
    if row.row_id not in s:
        print(f"[memory_v21_hook] FAIL (a): row_id absent: {s!r}", file=sys.stderr)
        return 1
    if "[prefer " not in s:
        print(f"[memory_v21_hook] FAIL (a): primary polarity tag absent or "
              f"misformatted: {s!r}", file=sys.stderr)
        return 1
    if "rec=4" not in s:
        print(f"[memory_v21_hook] FAIL (a): recurrence_count absent: {s!r}",
              file=sys.stderr)
        return 1
    if "workspace:abc123" in s:
        print(f"[memory_v21_hook] FAIL (a): scope tag leaked into display: "
              f"{s!r}", file=sys.stderr)
        return 1

    bad = Row(
        row_id="01HZ8K3M5N7P9Q1R2S3T4V5W6Y",
        text="legacy noise",
        polarity=["avoid", "workspace:abc123"],
        recurrence_count=1,
        created_at="2026-05-01T12:00:00Z",
        last_corroborated_at="2026-05-01T12:00:00Z",
        transcript_hash="manual",
        is_bad=True,
    )
    if "[bad]" not in bad.summary():
        print(f"[memory_v21_hook] FAIL (a): is_bad marker absent: "
              f"{bad.summary()!r}", file=sys.stderr)
        return 1

    if PRIMARY_POLARITY != frozenset({"prefer", "avoid", "factual"}):
        print(f"[memory_v21_hook] FAIL (a): PRIMARY_POLARITY drifted from §0.3: "
              f"{sorted(PRIMARY_POLARITY)}", file=sys.stderr)
        return 1
    print("[memory_v21_hook] (a) Row.summary wire-format OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (c) Fail-open against store failures + (f) skip_reason scrubs message
# ---------------------------------------------------------------------------


def _check_fail_open() -> int:
    from resonance_lattice.memory.capture import (
        capture, Message, ToolCall, Transcript,
    )
    from resonance_lattice.memory.redaction import Redactor

    transcript = Transcript(
        session_id="ok",
        messages=[
            Message("user", "diagnose the failing build please look at recent commits"),
            Message("assistant", "x" * 300,
                    tool_calls=(ToolCall("bash", "/tmp", "ls"),)),
        ],
        cwd="/proj",
    )

    class _ExplodingStore:
        def __init__(self, exc: Exception) -> None:
            self.exc = exc

        def add_row(self, **kwargs: object) -> str:
            raise self.exc

    redactor = Redactor()
    cases = [
        OSError("disk full /home/kane/.rlat"),  # path leak vector
        ValueError("polarity rejected: 'prefer pytest -xvs' …"),  # text leak vector
        portalocker.exceptions.LockException("/tmp/.rlat/.lock timeout"),  # path leak
    ]
    for exc in cases:
        result = capture(transcript, store=_ExplodingStore(exc), redactor=redactor)
        if result.row_id is not None:
            print(f"[memory_v21_hook] FAIL (c): expected None row_id on "
                  f"{type(exc).__name__}, got {result.row_id}", file=sys.stderr)
            return 1
        if not result.skip_reason or "capture failed" not in result.skip_reason:
            print(f"[memory_v21_hook] FAIL (c): skip_reason missing prefix on "
                  f"{type(exc).__name__}: {result.skip_reason!r}",
                  file=sys.stderr)
            return 1
        if type(exc).__name__ not in result.skip_reason:
            print(f"[memory_v21_hook] FAIL (c): skip_reason missing exception "
                  f"type {type(exc).__name__}: {result.skip_reason!r}",
                  file=sys.stderr)
            return 1
        # (f) — privacy: the raw exception message MUST NOT appear in
        # skip_reason. Exceptions can attach paths, polarity strings, or
        # row text the redactor was trying to protect.
        if str(exc) in result.skip_reason:
            print(f"[memory_v21_hook] FAIL (f): exception message leaked into "
                  f"skip_reason: {result.skip_reason!r}", file=sys.stderr)
            return 1
    print("[memory_v21_hook] (c) fail-open OK across "
          "OSError + ValueError + LockException", file=sys.stderr)
    print("[memory_v21_hook] (f) skip_reason scrubs raw exc message OK",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (d) _MAX_CAPTURED_CHARS truncation
# ---------------------------------------------------------------------------


def _check_truncation_cap() -> int:
    from resonance_lattice.memory.capture import (
        capture, Message, ToolCall, Transcript, _MAX_CAPTURED_CHARS,
    )
    from resonance_lattice.memory.redaction import Redactor
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        memory = Memory(root=root, encoder=ZeroEncoder())
        redactor = Redactor()
        huge = Transcript(
            session_id="huge",
            messages=[
                Message("user", "ingest this large file please and process it carefully"),
                Message("assistant", "x" * (_MAX_CAPTURED_CHARS + 5_000),
                        tool_calls=(ToolCall("read", "/tmp/big", "ok"),)),
            ],
            cwd="/proj",
        )
        result = capture(huge, store=memory, redactor=redactor)
        if result.row_id is None:
            print(f"[memory_v21_hook] FAIL (d): expected row_id, got skip "
                  f"({result.skip_reason})", file=sys.stderr)
            return 1
        rows, _ = memory.read_all()
        captured = next(r for r in rows if r.row_id == result.row_id)
        if len(captured.text) != _MAX_CAPTURED_CHARS:
            print(f"[memory_v21_hook] FAIL (d): expected text len "
                  f"{_MAX_CAPTURED_CHARS}, got {len(captured.text)}",
                  file=sys.stderr)
            return 1
    print(f"[memory_v21_hook] (d) _MAX_CAPTURED_CHARS truncation OK "
          f"(cap={_MAX_CAPTURED_CHARS})", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (g) Workspace tag stamping default + (h) --memory-root + --user composition
# ---------------------------------------------------------------------------


def _check_scope_tag_default() -> int:
    from resonance_lattice.memory._common import workspace_tag_for_cwd
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base"
        common = ["memory", "--memory-root", str(base), "--user", "kane"]

        rc, _, err = _run_cli(common + ["add", "default scope", "--polarity", "prefer"])
        if rc != 0:
            print(f"[memory_v21_hook] FAIL (g): add rc={rc} err={err}",
                  file=sys.stderr)
            return 1
        rc, _, err = _run_cli(common + ["add", "explicit cross",
                                         "--polarity", "factual",
                                         "--scope", "cross-workspace"])
        if rc != 0:
            print(f"[memory_v21_hook] FAIL (g): add --scope rc={rc} err={err}",
                  file=sys.stderr)
            return 1

        memory = Memory(root=base / "kane")
        rows, _ = memory.read_all()
        cwd_tag = workspace_tag_for_cwd()

        default = next(r for r in rows if r.text == "default scope")
        if cwd_tag not in default.polarity:
            print(f"[memory_v21_hook] FAIL (g): default-scope row missing cwd tag "
                  f"{cwd_tag}; polarity={default.polarity}", file=sys.stderr)
            return 1
        if "cross-workspace" in default.polarity:
            print(f"[memory_v21_hook] FAIL (g): default-scope row leaked "
                  f"cross-workspace: {default.polarity}", file=sys.stderr)
            return 1

        cross = next(r for r in rows if r.text == "explicit cross")
        if cwd_tag not in cross.polarity:
            print(f"[memory_v21_hook] FAIL (g): cross-scope row missing cwd "
                  f"tag {cwd_tag}; polarity={cross.polarity}", file=sys.stderr)
            return 1
        if "cross-workspace" not in cross.polarity:
            print(f"[memory_v21_hook] FAIL (g): cross-scope row missing "
                  f"cross-workspace: {cross.polarity}", file=sys.stderr)
            return 1
    print("[memory_v21_hook] (g) cwd workspace tag stamped + cross-workspace "
          "composes alongside OK", file=sys.stderr)
    return 0


def _check_root_user_composition() -> int:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base"
        rc, _, _ = _run_cli([
            "memory", "--memory-root", str(base), "--user", "alice",
            "add", "alice row", "--polarity", "prefer",
        ])
        if rc != 0:
            print(f"[memory_v21_hook] FAIL (h): alice add rc={rc}", file=sys.stderr)
            return 1
        rc, _, _ = _run_cli([
            "memory", "--memory-root", str(base), "--user", "bob",
            "add", "bob row", "--polarity", "prefer",
        ])
        if rc != 0:
            print(f"[memory_v21_hook] FAIL (h): bob add rc={rc}", file=sys.stderr)
            return 1

        if not (base / "alice" / "sidecar.jsonl").exists():
            print(f"[memory_v21_hook] FAIL (h): alice subdir missing under "
                  f"{base}", file=sys.stderr)
            return 1
        if not (base / "bob" / "sidecar.jsonl").exists():
            print(f"[memory_v21_hook] FAIL (h): bob subdir missing under "
                  f"{base}", file=sys.stderr)
            return 1
        rc, out, _ = _run_cli([
            "memory", "--memory-root", str(base), "--user", "alice", "list",
        ])
        if "alice row" not in out or "bob row" in out:
            print(f"[memory_v21_hook] FAIL (h): alice list cross-leaked or "
                  f"empty:\n{out}", file=sys.stderr)
            return 1
    print("[memory_v21_hook] (h) --memory-root + --user composes as "
          "<base>/<user>/ OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (i) CLI exit codes
# ---------------------------------------------------------------------------


def _check_exit_codes() -> int:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base"
        common = ["memory", "--memory-root", str(base), "--user", "test"]

        rc, _, _ = _run_cli(common + ["add", "valid row", "--polarity", "prefer"])
        if rc != 0:
            print(f"[memory_v21_hook] FAIL (i): valid add rc={rc} (want 0)",
                  file=sys.stderr)
            return 1

        rc, _, _ = _run_cli(common + ["add", "  "])
        if rc != 1:
            print(f"[memory_v21_hook] FAIL (i): empty-text add rc={rc} (want 1)",
                  file=sys.stderr)
            return 1
        rc, _, _ = _run_cli(common + ["gc"])
        if rc != 1:
            print(f"[memory_v21_hook] FAIL (i): bare gc rc={rc} (want 1)",
                  file=sys.stderr)
            return 1

        for name in ["consolidate", "primer"]:
            rc, _, err = _run_cli(common + [name])
            if rc != 2:
                print(f"[memory_v21_hook] FAIL (i): `{name}` rc={rc} "
                      f"(want 2 — deprecated)", file=sys.stderr)
                return 1
            if "removed in v2.1" not in err:
                print(f"[memory_v21_hook] FAIL (i): `{name}` banner missing "
                      f"deprecation marker; stderr:\n{err}", file=sys.stderr)
                return 1

        for name in ["distil", "feedback"]:
            argv = common + [name]
            rc, _, err = _run_cli(argv)
            if rc != 3:
                print(f"[memory_v21_hook] FAIL (i): `{name}` rc={rc} "
                      f"(want 3 — pending MVP)", file=sys.stderr)
                return 1
            if "ships in v2.1 MVP" not in err:
                print(f"[memory_v21_hook] FAIL (i): `{name}` banner missing "
                      f"MVP marker; stderr:\n{err}", file=sys.stderr)
                return 1

        # `recall <query>` (one-shot) shipped Day 9-10. Empty store
        # returns rc=0 with the "(no rows pass...)" message. Bare
        # `recall` (no query, no --daemon) is rc=1 user error.
        rc, _, err = _run_cli(common + ["recall", "test query"])
        if rc != 0 or "(no rows pass" not in err:
            print(f"[memory_v21_hook] FAIL (i): `recall <query>` rc={rc} "
                  f"(want 0, empty-store gates message); stderr:\n{err}",
                  file=sys.stderr)
            return 1
        rc, _, err = _run_cli(common + ["recall"])
        if rc != 1 or "requires a <query>" not in err:
            print(f"[memory_v21_hook] FAIL (i): bare `recall` rc={rc} "
                  f"(want 1 — usage error); stderr:\n{err}",
                  file=sys.stderr)
            return 1

        # `doctor` shipped in MVP Day 7-8 — rc=0 even on partial-state
        # diagnostic output. Always returns 0 because the user is
        # asking for diagnostic info, not gating their workflow.
        rc, doctor_out, _ = _run_cli(common + ["doctor"])
        if rc != 0:
            print(f"[memory_v21_hook] FAIL (i): `doctor` rc={rc} (want 0)",
                  file=sys.stderr)
            return 1
        if "[OK] root:" not in doctor_out and "[FAIL] root:" not in doctor_out:
            print(f"[memory_v21_hook] FAIL (i): `doctor` output missing "
                  f"`root:` probe line:\n{doctor_out}", file=sys.stderr)
            return 1

        # Train ships partially in MVP Day 5-6: operator flags work
        # synchronously (rc=0); `train <task>` still points at the
        # `/rlat-train` slash command (rc=3); bare `train` is a usage
        # error (rc=1).
        rc, _, err = _run_cli(common + ["train"])
        if rc != 1:
            print(f"[memory_v21_hook] FAIL (i): bare `train` rc={rc} "
                  f"(want 1 — user error)\nstderr:\n{err}", file=sys.stderr)
            return 1
        rc, _, err = _run_cli(common + ["train", "fab_lh_001"])
        if rc != 3 or "/rlat-train" not in err:
            print(f"[memory_v21_hook] FAIL (i): `train <task>` rc={rc} "
                  f"or banner missing /rlat-train pointer:\n{err}",
                  file=sys.stderr)
            return 1
        # Operator flags require a real seeded row — add one in the
        # same CLI then exercise --bad-vote / --good-vote / --corroborate.
        rc, _, _ = _run_cli(common + ["add", "row to operate on",
                                       "--polarity", "factual"])
        if rc != 0:
            print(f"[memory_v21_hook] FAIL (i): seed add rc={rc}",
                  file=sys.stderr)
            return 1
        _, out, _ = _run_cli(common + ["list", "--format", "json"])
        seeded_id = json.loads(out)[0]["row_id"]

        for flag in ["--bad-vote", "--good-vote", "--corroborate"]:
            rc, _, err = _run_cli(common + ["train", flag, seeded_id])
            if rc != 0:
                print(f"[memory_v21_hook] FAIL (i): `train {flag} <id>` "
                      f"rc={rc} (want 0)\nstderr:\n{err}", file=sys.stderr)
                return 1

        rc, _, err = _run_cli(common + ["train", "--bad-vote", "DEADBEEF"])
        if rc != 1:
            print(f"[memory_v21_hook] FAIL (i): unknown row_id rc={rc} "
                  f"(want 1)", file=sys.stderr)
            return 1
        rc, _, err = _run_cli(common + ["train", "--bad-vote", seeded_id,
                                         "--good-vote", seeded_id])
        if rc != 1 or "mutually exclusive" not in err:
            print(f"[memory_v21_hook] FAIL (i): mutually-exclusive flags "
                  f"rc={rc} or banner missing:\n{err}", file=sys.stderr)
            return 1
    print("[memory_v21_hook] (i) CLI exit codes 0/1/2/3 distinguish "
          "ok/user-error/deprecated/pending-MVP + train operator flags "
          "OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run() -> int:
    patch_zero_encoder()
    for check in [
        _check_summary_format,
        _check_fail_open,
        _check_truncation_cap,
        _check_scope_tag_default,
        _check_root_user_composition,
        _check_exit_codes,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v21_hook] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
