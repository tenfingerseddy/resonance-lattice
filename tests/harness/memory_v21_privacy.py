"""memory_v21_privacy — Layer-1 redaction + audit-log correlation contracts.

Pins six invariants (Appendix D row D.6 in
.claude/plans/fabric-agent-flat-memory.md):

  (a) Each of the 8 built-in credential patterns from §6.2 fires on a
      synthetic positive (`AKIA...`, `sk-ant-...`, `sk-...`, `ghp_...`,
      `github_pat_...`, `eyJ...JWT`, `-----BEGIN ... PRIVATE KEY-----`,
      `40+ char hex`) AND a 100-line legitimate-content corpus stays
      below the false-positive bound (`long_hex` ≤ 5%).

  (b) Tool-call reads against denylist paths (`.env*`, `.aws/credentials`,
      `*.pem`, `*.key`, etc.) replace the entire payload with
      `<REDACTED file contents>` — both POSIX and Windows separators.

  (c) Audit-log lines record `pattern=<name>  matches=<N>  row_id=<id>`
      but NEVER the offending text. Test against an Anthropic key in
      a transcript: the log line names the pattern, the secret never
      appears in the log file.

  (d) Audit log is append-only — successive `log_events` calls extend
      the file rather than overwriting it.

  (e) `capture()` correlates audit-log events with the row_id assigned
      by `store.add_row`. After capture, every event from that
      transcript appears in the log with `row_id=<the captured row_id>`.
      The no-write branch (empty after scrub) logs events under
      `transcript_hash` — covered by the implementation but not
      fixture-tested here, since contriving a gate-passing pure-
      secrets transcript is fragile.

  (f) Tool-call content with `path is None` is still pattern-scrubbed.
      A `bash` call with inline `export ANTHROPIC_API_KEY=sk-ant-...`
      and no `path` field must trip Layer-1; the captured row text
      must NOT contain the secret, the audit log must record the
      anthropic_key match.

Sub-MVP issue: #95. On the Pre-MVP safety floor per Appendix D —
non-negotiable. Hermetic — no encoder load (mocked), no LLM calls.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ._testutil import ZeroEncoder, patch_zero_encoder


# ---------------------------------------------------------------------------
# (a) Built-in patterns + false-positive bound
# ---------------------------------------------------------------------------

# Each fixture is a synthetic positive shaped like the real credential.
_PATTERN_POSITIVES: dict[str, str] = {
    "aws_access_key": "AKIAIOSFODNN7EXAMPLE",
    "anthropic_key": "sk-ant-" + "A" * 50,
    "openai_key": "sk-" + "A" * 50,
    "github_pat": "ghp_" + "A" * 36,
    "github_fine_pat": "github_pat_" + "A" * 82,
    "jwt": "eyJabcdef.eyJxyz.SflKxw-AbCdEfGhIjK",
    "pem_key": "-----BEGIN RSA PRIVATE KEY-----\nABC\n-----END RSA PRIVATE KEY-----",
    "long_hex": "0123456789abcdef0123456789abcdef",
}

# Legitimate-content corpus for the false-positive bound. Keep it varied
# so `long_hex` (the cautious pattern) gets exposed to plausible neighbours
# of true positives without containing one.
_CLEAN_CORPUS: tuple[str, ...] = (
    "the encoder revision is pinned",
    "use pytest -xvs for debugging",
    "run simplify before committing",
    "commit hash is short, e.g. 9306a9 abcdef",
    "AWS region is us-east-1, profile is dev",
    "config lives at .rlat/memory.toml",
    "the workspace path is /home/kane/code/proj",
    "passages are L2-normalised float32",
    "max-seq is 8192 tokens",
    "the daemon idle timeout defaults to 30 min",
)


def _check_builtin_patterns() -> int:
    from resonance_lattice.memory.redaction import Redactor, SECRET_PATTERNS

    r = Redactor()
    pattern_names = {name for name, _ in SECRET_PATTERNS}
    if pattern_names != set(_PATTERN_POSITIVES):
        print(f"[memory_v21_privacy] FAIL (a): pattern names drifted from "
              f"fixture coverage. patterns={pattern_names!r} "
              f"fixtures={set(_PATTERN_POSITIVES)!r}", file=sys.stderr)
        return 1

    for name, positive in _PATTERN_POSITIVES.items():
        out, events = r.scrub(positive)
        names_hit = {ev.pattern for ev in events}
        if name not in names_hit:
            print(f"[memory_v21_privacy] FAIL (a): pattern {name!r} did not "
                  f"fire on its own positive {positive!r}; events={events}",
                  file=sys.stderr)
            return 1
        if "<REDACTED>" not in out:
            print(f"[memory_v21_privacy] FAIL (a): {name!r} matched but text "
                  f"unchanged: {out!r}", file=sys.stderr)
            return 1

    # False-positive bound on the clean corpus. We tolerate ≤ 5% on
    # `long_hex` (the cautious pattern).
    fp_total = 0
    long_hex_fp = 0
    multiplied = list(_CLEAN_CORPUS) * 10  # 100 lines per fixture spec
    for line in multiplied:
        _, events = r.scrub(line)
        if events:
            fp_total += 1
            if any(ev.pattern == "long_hex" for ev in events):
                long_hex_fp += 1
    # Strict patterns (everything except long_hex) MUST not fire on the
    # clean corpus.
    strict_fp = fp_total - long_hex_fp
    if strict_fp != 0:
        print(f"[memory_v21_privacy] FAIL (a): strict patterns false-fired "
              f"on clean corpus ({strict_fp} hits)", file=sys.stderr)
        return 1
    if long_hex_fp / len(multiplied) > 0.05:
        print(f"[memory_v21_privacy] FAIL (a): long_hex false-positive rate "
              f"{long_hex_fp}/{len(multiplied)} > 5%", file=sys.stderr)
        return 1
    print(f"[memory_v21_privacy] (a) 8 patterns fire on positives; "
          f"strict FP=0, long_hex FP={long_hex_fp}/{len(multiplied)} OK",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (b) Denylist path scrub — both separator styles
# ---------------------------------------------------------------------------


def _check_denylist_paths() -> int:
    from resonance_lattice.memory.redaction import REDACTED_FILE, Redactor

    r = Redactor()
    cases = [
        "/home/kane/.env",
        "/home/kane/.envrc",
        "/etc/secrets/token.secret",
        "/home/kane/private.key",
        "/home/kane/.ssh/id_rsa",
        "/home/kane/.aws/credentials",
        r"C:\Users\kane\.aws\credentials",
        r"C:\Users\kane\private.pem",
    ]
    for path in cases:
        out, events = r.scrub_tool_call(path, "ANTHROPIC_API_KEY=sk-ant-" + "X" * 40)
        if out != REDACTED_FILE:
            print(f"[memory_v21_privacy] FAIL (b): denylist scrub miss on "
                  f"{path!r}; out={out!r}", file=sys.stderr)
            return 1
        if not any(ev.pattern == "denylist_path" for ev in events):
            print(f"[memory_v21_privacy] FAIL (b): denylist_path event missing "
                  f"on {path!r}; events={events}", file=sys.stderr)
            return 1

    safe = "/home/kane/code/main.py"
    out, _ = r.scrub_tool_call(safe, "regular content here, no secrets")
    if out == REDACTED_FILE:
        print(f"[memory_v21_privacy] FAIL (b): non-denylist {safe!r} got "
              f"file-redacted: {out!r}", file=sys.stderr)
        return 1
    print(f"[memory_v21_privacy] (b) denylist scrub fires on {len(cases)} "
          f"paths (POSIX + Windows) OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (c) + (d) Audit log: secret-free, append-only
# ---------------------------------------------------------------------------


def _check_audit_log_secret_free_and_append_only() -> int:
    from resonance_lattice.memory.redaction import Redactor

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "redaction.log"
        r = Redactor(audit_log_path=log_path)

        secret = "AKIAIOSFODNN7EXAMPLE"
        out, events = r.scrub(secret)
        r.log_events(events, row_id="ROW1")
        first_lines = log_path.read_text(encoding="utf-8").splitlines()
        if "<REDACTED>" not in out:
            print(f"[memory_v21_privacy] FAIL (c): scrub did not redact",
                  file=sys.stderr)
            return 1
        if not any("pattern=aws_access_key" in line for line in first_lines):
            print(f"[memory_v21_privacy] FAIL (c): pattern name missing from "
                  f"log:\n{first_lines}", file=sys.stderr)
            return 1
        if "AKIA" in log_path.read_text(encoding="utf-8"):
            print(f"[memory_v21_privacy] FAIL (c): audit log leaked secret",
                  file=sys.stderr)
            return 1
        if not any("row_id=ROW1" in line for line in first_lines):
            print(f"[memory_v21_privacy] FAIL (c): row_id correlation missing",
                  file=sys.stderr)
            return 1

        _, events2 = r.scrub("sk-ant-" + "B" * 50)
        r.log_events(events2, row_id="ROW2")
        second_lines = log_path.read_text(encoding="utf-8").splitlines()
        if len(second_lines) <= len(first_lines):
            print(f"[memory_v21_privacy] FAIL (d): audit log not append-only "
                  f"(len went {len(first_lines)} → {len(second_lines)})",
                  file=sys.stderr)
            return 1
        for line in first_lines:
            if line not in second_lines:
                print(f"[memory_v21_privacy] FAIL (d): pre-existing log line "
                      f"vanished after second log_events: {line!r}",
                      file=sys.stderr)
                return 1
        if "BBB" in log_path.read_text(encoding="utf-8"):
            print(f"[memory_v21_privacy] FAIL (c): audit log leaked second secret",
                  file=sys.stderr)
            return 1
    print("[memory_v21_privacy] (c) audit log records pattern + row_id but "
          "never the secret OK", file=sys.stderr)
    print("[memory_v21_privacy] (d) audit log append-only OK",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (e) Capture pipeline → audit log row_id correlation
# ---------------------------------------------------------------------------


def _check_capture_audit_correlation() -> int:
    from resonance_lattice.memory.capture import (
        Message, ToolCall, Transcript, capture,
    )
    from resonance_lattice.memory.redaction import Redactor
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        log = root / "redaction.log"
        memory = Memory(root=root, encoder=ZeroEncoder())
        redactor = Redactor(audit_log_path=log)
        leaky = Transcript(
            session_id="leak",
            messages=[
                Message("user", "why is the API call failing intermittently right now?"),
                Message("assistant",
                        ("Let me look at the request shape closely; checking config "
                         "and the request lifecycle for headers and rate limits."),
                        tool_calls=(ToolCall(
                            "read", "/code/main.py",
                            "key='sk-ant-" + "A" * 50 + "'",
                        ),)),
                Message("user", "paste the lines here"),
                Message("assistant",
                        ("Init line uses sk-ant-" + "A" * 50 +
                         " which looks fine; the failure is rate-limit cascade. "
                         "Switch to backoff with jitter — 100ms retries cause 429.")),
            ],
            cwd="/proj",
        )
        result = capture(leaky, store=memory, redactor=redactor)
        if not result.row_id or result.redactions < 1:
            print(f"[memory_v21_privacy] FAIL (e): capture skipped or no "
                  f"redactions: {result}", file=sys.stderr)
            return 1
        log_text = log.read_text(encoding="utf-8")
        rows, _ = memory.read_all()
        captured = next(r for r in rows if r.row_id == result.row_id)
        if "sk-ant-" in captured.text:
            print(f"[memory_v21_privacy] FAIL (e): row text leaked secret",
                  file=sys.stderr)
            return 1
        if "sk-ant-" in log_text:
            print(f"[memory_v21_privacy] FAIL (e): audit log leaked secret",
                  file=sys.stderr)
            return 1
        if f"row_id={result.row_id}" not in log_text:
            print(f"[memory_v21_privacy] FAIL (e): audit log missing "
                  f"row_id={result.row_id}", file=sys.stderr)
            return 1
        if "pattern=anthropic_key" not in log_text:
            print(f"[memory_v21_privacy] FAIL (e): pattern attribution missing",
                  file=sys.stderr)
            return 1

    print("[memory_v21_privacy] (e) capture → audit log row_id correlation OK",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# (f) path=None tool-call content is still pattern-scrubbed
# ---------------------------------------------------------------------------


def _check_path_none_scrub() -> int:
    from resonance_lattice.memory.capture import (
        Message, ToolCall, Transcript, capture,
    )
    from resonance_lattice.memory.redaction import Redactor
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "u"
        log = root / "redaction.log"
        memory = Memory(root=root, encoder=ZeroEncoder())
        redactor = Redactor(audit_log_path=log)

        leaky_no_path = Transcript(
            session_id="bash-no-path",
            messages=[
                Message("user", "set up the env please for the new project workspace"),
                Message("assistant", "running setup",
                        tool_calls=(ToolCall(
                            "bash", None,
                            "export ANTHROPIC_API_KEY=sk-ant-" + "X" * 50,
                        ),)),
                Message("user", "good, continue from there"),
                Message("assistant", "x" * 250,
                        tool_calls=(ToolCall("bash", None, "echo done"),)),
            ],
            cwd="/proj",
        )
        result = capture(leaky_no_path, store=memory, redactor=redactor)
        if not result.row_id or result.redactions < 1:
            print(f"[memory_v21_privacy] FAIL (f): path=None tool-call leak "
                  f"NOT scrubbed; result={result}", file=sys.stderr)
            return 1
        log_text = log.read_text(encoding="utf-8")
        if "pattern=anthropic_key" not in log_text:
            print(f"[memory_v21_privacy] FAIL (f): path=None content missed by "
                  f"pattern scrub; log:\n{log_text}", file=sys.stderr)
            return 1
        if "sk-ant-" in log_text or "XXXXXXXX" in log_text:
            print(f"[memory_v21_privacy] FAIL (f): audit log leaked secret",
                  file=sys.stderr)
            return 1
        rows, _ = memory.read_all()
        captured = next(r for r in rows if r.row_id == result.row_id)
        if "sk-ant-" in captured.text:
            print(f"[memory_v21_privacy] FAIL (f): row text leaked secret",
                  file=sys.stderr)
            return 1
    print("[memory_v21_privacy] (f) path=None tool-call content scrubbed OK",
          file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run() -> int:
    patch_zero_encoder()
    for check in [
        _check_builtin_patterns,
        _check_denylist_paths,
        _check_audit_log_secret_free_and_append_only,
        _check_capture_audit_correlation,
        _check_path_none_scrub,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v21_privacy] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
