"""memory_v21_hook — v2.1 flat-memory hook + CLI surface contracts.

Pins seven invariants (Sub-MVP slice of Appendix D row D.5 in
.claude/plans/fabric-agent-flat-memory.md):

  (a) `Row.summary()` matches the §0.4 wire-format invariants used by the
      future hook — `<row_id>  [<primary>]  rec=<N>  <text>` with the
      primary polarity selected from the closed `{prefer, avoid, factual}`
      set, scope tags hidden from display.

  (c) Capture is fail-open. `capture()` against a store whose `write`
      raises `OSError` / `ValueError` / `portalocker.LockException` returns
      `CaptureResult(skip_reason="capture failed: <type>", claim_ids=())`,
      never propagates the exception. The skip_reason carries the
      exception type but never the raw message — exceptions can leak
      paths or claim text the redactor was protecting.

  (d) `_MAX_CAPTURED_CHARS` token-budget cap fires on a session whose
      assistant content exceeds the limit. Captured claim text length is
      exactly `_MAX_CAPTURED_CHARS`, and it keeps the session *tail*
      (recent work) — the stale head is dropped.

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

  (i) CLI exit codes split pending-MVP (3) from user-error (1) from
      success (0). Verifies one representative of each across the §0.7
      surface.

  (j) Capture-time dedup: same `(text, workspace_tag)` from a different
      session bumps the existing row's recurrence_count instead of
      writing a new row. Different workspace with identical text produces
      a new row — the workspace boundary is what tells two checkouts of
      the same project apart. The architecture's recurrence_count tracks
      events that recur across sessions, not duplicate captures of the
      same content.

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
# (a) claim-summary wire-format
# ---------------------------------------------------------------------------


def _summary_claim(claim_id: str, *, text: str, primary: str,
                    recurrence: int, is_bad: bool):
    """Build one experience `Claim` for the summary wire-format check."""
    from resonance_lattice.state.claim import Claim, ExperienceFacts

    return Claim(
        claim_id=claim_id,
        source="experience",
        kind="event",
        content=text,
        created_at="2026-05-01T12:00:00Z",
        corroboration=2.0,
        falsification=2.0,
        trust_as_of="",
        state="active",
        parent_ids=(),
        facts=ExperienceFacts(
            polarity=(primary, "workspace:abc123"),
            recurrence_count=recurrence,
            criticality="normal",
            created_under_intent_kind="none",
            transcript_hash="manual",
            origin="manual",
            last_corroborated_at="2026-05-01T12:00:00Z",
            is_bad=is_bad,
        ),
    )


def _check_summary_format() -> int:
    # `cli.memory._claim_summary` is the successor to the old
    # `Row.summary()` — the §0.4 single-line wire-format.
    from resonance_lattice.cli.memory import _claim_summary
    from resonance_lattice.state.claim import PRIMARY_POLARITY

    claim = _summary_claim(
        "01HZ8K3M5N7P9Q1R2S3T4V5W6X",
        text="prefer pytest -xvs <path> for debugging",
        primary="prefer", recurrence=4, is_bad=False,
    )
    s = _claim_summary(claim)
    if claim.claim_id not in s:
        print(f"[memory_v21_hook] FAIL (a): claim_id absent: {s!r}", file=sys.stderr)
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

    bad = _summary_claim(
        "01HZ8K3M5N7P9Q1R2S3T4V5W6Y",
        text="legacy noise", primary="avoid", recurrence=1, is_bad=True,
    )
    if "[bad]" not in _claim_summary(bad):
        print(f"[memory_v21_hook] FAIL (a): is_bad marker absent: "
              f"{_claim_summary(bad)!r}", file=sys.stderr)
        return 1

    if PRIMARY_POLARITY != frozenset({"prefer", "avoid", "factual"}):
        print(f"[memory_v21_hook] FAIL (a): PRIMARY_POLARITY drifted from §0.3: "
              f"{sorted(PRIMARY_POLARITY)}", file=sys.stderr)
        return 1
    print("[memory_v21_hook] (a) claim-summary wire-format OK", file=sys.stderr)
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

        def read_all(self):
            # Capture's same-text dedup probe runs BEFORE write; this stub
            # returns no claims so the dedup-miss path falls through to
            # the explosive `write`, which is what the test exercises.
            return []

        def write(self, claim, **kwargs: object) -> None:
            raise self.exc

    redactor = Redactor()
    cases = [
        OSError("disk full /home/kane/.rlat"),  # path leak vector
        ValueError("polarity rejected: 'prefer pytest -xvs' …"),  # text leak vector
        portalocker.exceptions.LockException("/tmp/.rlat/.lock timeout"),  # path leak
    ]
    for exc in cases:
        result = capture(transcript, store=_ExplodingStore(exc), redactor=redactor)
        if result.claim_ids:
            print(f"[memory_v21_hook] FAIL (c): expected empty claim_ids on "
                  f"{type(exc).__name__}, got {result.claim_ids}", file=sys.stderr)
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
    from resonance_lattice.memory.claim_store import ExperienceClaimStore

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        memory = ExperienceClaimStore(root=root, encoder=ZeroEncoder())
        redactor = Redactor()
        # Distinguishable head + tail so the test proves the *tail* (the
        # session's recent work) survives and the stale head is dropped.
        head = "STALE-OPENING-MARKER"
        tail = "RECENT-WORK-MARKER"
        filler = "x" * (_MAX_CAPTURED_CHARS + 5_000 - len(head) - len(tail))
        huge = Transcript(
            session_id="huge",
            messages=[
                Message("user", "ingest this large file please and process it carefully"),
                Message("assistant", head + filler + tail,
                        tool_calls=(ToolCall("read", "/tmp/big", "ok"),)),
            ],
            cwd="/proj",
        )
        result = capture(huge, store=memory, redactor=redactor)
        if not result.claim_ids:
            print(f"[memory_v21_hook] FAIL (d): expected claim_ids, got skip "
                  f"({result.skip_reason})", file=sys.stderr)
            return 1
        claims = memory.read_all()
        captured = next(c for c in claims if c.claim_id == result.claim_ids[0])
        if len(captured.content) != _MAX_CAPTURED_CHARS:
            print(f"[memory_v21_hook] FAIL (d): expected text len "
                  f"{_MAX_CAPTURED_CHARS}, got {len(captured.content)}",
                  file=sys.stderr)
            return 1
        if not captured.content.endswith(tail):
            print("[memory_v21_hook] FAIL (d): captured content lost the "
                  "session tail (recent work)", file=sys.stderr)
            return 1
        if head in captured.content:
            print("[memory_v21_hook] FAIL (d): captured content kept the "
                  "stale head instead of truncating it", file=sys.stderr)
            return 1
    print(f"[memory_v21_hook] (d) _MAX_CAPTURED_CHARS tail truncation OK "
          f"(cap={_MAX_CAPTURED_CHARS})", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (g) Workspace tag stamping default + (h) --memory-root + --user composition
# ---------------------------------------------------------------------------


def _check_scope_tag_default() -> int:
    from resonance_lattice.memory._common import workspace_tag_for_cwd
    from resonance_lattice.memory.claim_store import ExperienceClaimStore

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

        memory = ExperienceClaimStore(root=base / "kane", encoder=None)
        claims = memory.read_all()
        cwd_tag = workspace_tag_for_cwd()

        default = next(c for c in claims if c.content == "default scope")
        if cwd_tag not in default.facts.polarity:
            print(f"[memory_v21_hook] FAIL (g): default-scope row missing cwd tag "
                  f"{cwd_tag}; polarity={default.facts.polarity}", file=sys.stderr)
            return 1
        if "cross-workspace" in default.facts.polarity:
            print(f"[memory_v21_hook] FAIL (g): default-scope row leaked "
                  f"cross-workspace: {default.facts.polarity}", file=sys.stderr)
            return 1

        cross = next(c for c in claims if c.content == "explicit cross")
        if cwd_tag not in cross.facts.polarity:
            print(f"[memory_v21_hook] FAIL (g): cross-scope row missing cwd "
                  f"tag {cwd_tag}; polarity={cross.facts.polarity}", file=sys.stderr)
            return 1
        if "cross-workspace" not in cross.facts.polarity:
            print(f"[memory_v21_hook] FAIL (g): cross-scope row missing "
                  f"cross-workspace: {cross.facts.polarity}", file=sys.stderr)
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

        if not (base / "alice" / "claims.jsonl").exists():
            print(f"[memory_v21_hook] FAIL (h): alice subdir missing under "
                  f"{base}", file=sys.stderr)
            return 1
        if not (base / "bob" / "claims.jsonl").exists():
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

        # `feedback` logs a vote with no LLM dependency — fully
        # deterministic; verify both verdicts land in feedback.log.
        from resonance_lattice.memory.feedback import feedback_log_path
        from resonance_lattice.memory.store import path_for_user
        for verdict in ["good", "bad"]:
            rc, _, err = _run_cli(common + ["feedback", verdict])
            if rc != 0:
                print(f"[memory_v21_hook] FAIL (i): `feedback {verdict}` "
                      f"rc={rc} (want 0)\n{err}", file=sys.stderr)
                return 1
        log_path = feedback_log_path(
            path_for_user(user_id="test", root=base)
        )
        votes = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if [v["verdict"] for v in votes] != ["good", "bad"]:
            print(f"[memory_v21_hook] FAIL (i): feedback.log={votes!r}",
                  file=sys.stderr)
            return 1

        # `recall <query>` (one-shot) shipped Day 9-10. Empty store
        # returns rc=0 with the "(no claims pass...)" message. Bare
        # `recall` (no query, no --daemon) is rc=1 user error.
        rc, _, err = _run_cli(common + ["recall", "test query"])
        if rc != 0 or "(no claims pass" not in err:
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
        seeded_id = json.loads(out)[0]["claim_id"]

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
    print("[memory_v21_hook] (i) CLI exit codes 0/1/3 distinguish "
          "ok/user-error/pending-MVP + train operator flags OK",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _check_capture_dedup() -> int:
    """(j) Same `(text, workspace)` capture bumps recurrence_count instead
    of writing a new row. Different workspace → new row, even with
    identical text. The architecture's `recurrence_count` is meant to
    grow when an event recurs across sessions, not when the same content
    is captured to disk N times."""
    from resonance_lattice.memory.capture import (
        capture, GateConfig, Message, ToolCall, Transcript,
    )
    from resonance_lattice.memory.redaction import Redactor
    from resonance_lattice.memory.claim_store import ExperienceClaimStore

    def _transcript(session_id: str, cwd: str) -> Transcript:
        return Transcript(
            session_id=session_id,
            messages=[
                Message("user", "diagnose the failing test please look at recent commits"),
                Message(
                    "assistant",
                    "the failing test is a regression in the encoder cache; "
                    "the fix is to bump CACHE_VERSION in field/encoder.py" + " " * 100,
                    tool_calls=(ToolCall("Read", "/proj/src/encoder.py", ""),),
                ),
            ],
            cwd=cwd,
        )

    with tempfile.TemporaryDirectory() as td:
        memory = ExperienceClaimStore(root=Path(td) / "u", encoder=None)
        redactor = Redactor()
        gate = GateConfig(require_tool_use=True, min_assistant_chars=200)

        # First capture from session A in workspace P
        r1 = capture(_transcript("sess-A", "/proj-P"),
                     store=memory, redactor=redactor, gate=gate)
        if not r1.claim_ids:
            print(f"[memory_v21_hook] FAIL (j): first capture skipped: "
                  f"{r1.skip_reason!r}", file=sys.stderr)
            return 1

        # Second capture from a DIFFERENT session but same text + same workspace
        r2 = capture(_transcript("sess-B", "/proj-P"),
                     store=memory, redactor=redactor, gate=gate)
        if r2.claim_ids != r1.claim_ids:
            print(f"[memory_v21_hook] FAIL (j): same-text-same-workspace "
                  f"dedup failed; got new row_ids {r2.claim_ids} "
                  f"vs existing {r1.claim_ids}", file=sys.stderr)
            return 1

        claims = memory.read_all()
        if len(claims) != 1:
            print(f"[memory_v21_hook] FAIL (j): expected 1 row after dedup, "
                  f"got {len(claims)}", file=sys.stderr)
            return 1
        if claims[0].facts.recurrence_count != 2:
            print(f"[memory_v21_hook] FAIL (j): expected recurrence_count=2 "
                  f"after second capture, got {claims[0].facts.recurrence_count}",
                  file=sys.stderr)
            return 1

        # Third capture: same text, DIFFERENT workspace → new row
        r3 = capture(_transcript("sess-C", "/proj-Q"),
                     store=memory, redactor=redactor, gate=gate)
        if r3.claim_ids == r1.claim_ids:
            print(f"[memory_v21_hook] FAIL (j): different-workspace capture "
                  f"deduped onto workspace-P row {r1.claim_ids}",
                  file=sys.stderr)
            return 1

        claims = memory.read_all()
        if len(claims) != 2:
            print(f"[memory_v21_hook] FAIL (j): expected 2 rows after "
                  f"different-workspace capture, got {len(claims)}",
                  file=sys.stderr)
            return 1
    print("[memory_v21_hook] (j) capture dedup bumps recurrence_count + "
          "respects workspace boundary OK", file=sys.stderr)
    return 0


def run() -> int:
    patch_zero_encoder()
    for check in [
        _check_summary_format,
        _check_fail_open,
        _check_truncation_cap,
        _check_scope_tag_default,
        _check_root_user_composition,
        _check_exit_codes,
        _check_capture_dedup,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v21_hook] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
