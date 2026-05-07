"""UserPromptSubmit hook shim — `rlat memory hook` entry point.

Claude Code fires this on every user message. Pipeline (per §5.2.1):

1. Read JSON `{prompt, cwd, session_id, ...}` from stdin.
2. Try to connect to the per-user recall daemon socket.
3. On connect-fail: lazy-spawn `rlat memory recall --daemon` as a
   background subprocess, wait 100 ms, retry once.
4. Send `RecallRequest(query=prompt, cwd_hash=...)`; receive
   `RecallReply`.
5. If hits non-empty: emit `{hookSpecificOutput.additionalContext: <§0.4 block>}`
   to stdout so Claude Code prepends the `<rlat-memory>` block to the
   user's prompt context. Empty hits → emit `{}` (no injection, no
   stderr).
6. Fail-open per §16.5 / §18.5 — any uncaught exception, daemon
   timeout, or one-shot subprocess failure ends with `{}` to stdout
   and a single stderr line for operator visibility. The user's
   prompt is never blocked.

Token budget per §0.4: 1500 tokens (~6000 chars by 4-char/token
proxy), truncate at row boundary so we never emit a half-row.

Latency profile: warm-recall is well within the §0.6 p95 80ms budget
(daemon p99 ~1.5ms + IPC ~5ms + encoder embed ~12ms). Cold-spawn
first-of-session is ~800ms (200ms initial connect-fail + 100ms boot
wait + 500ms retry) — outside the warm-recall budget, but amortised
to N=1 per session and absorbed by the user's prompt-typing latency.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Heavy imports (.daemon pulls multiprocessing.connection + the encoder
# import chain, .store pulls portalocker + numpy) are deferred into
# `run_hook` after the fast-exit gates so bad-stdin / empty-prompt /
# missing-root paths don't pay the import cost.

# §0.4 token budget: 1500 tokens combined; conservative 4 chars/token
# proxy. Truncate at row boundary so callers never see a half-row.
_MAX_INJECTION_CHARS = 6000

# §5.2.1 hook → daemon retry window after lazy-spawn.
_DAEMON_BOOT_WAIT_S = 0.1
_DAEMON_RETRY_TIMEOUT_MS = 500

# §0.6 default M (recurrence threshold) — must match recall.py default.
_RECURRENCE_M = 3


def _trace(event: str) -> None:
    """Append a single line to `~/.rlat/memory/.hook_trace.log` for "did
    Claude Code invoke this command at all" diagnosis. Best-effort: any
    failure is swallowed so the trace itself can never break the hook.

    **Default off** per codex P2.3 (memory/privacy posture). Enable only
    when actively diagnosing a hook misfire by setting
    `RLAT_HOOK_TRACE=1`. Trace lines include transcript paths and
    session ids — fine for one-shot diagnosis on the operator's own
    box, but not desirable as a long-running default.
    """
    if os.environ.get("RLAT_HOOK_TRACE") != "1":
        return
    try:
        from datetime import datetime, timezone
        log_dir = Path.home() / ".rlat" / "memory"
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / ".hook_trace.log").open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()}  {event}\n")
    except Exception:
        pass


def _neutralise_boundary_tags(text: str) -> str:
    """Strip / disarm `<rlat-memory>` and `</rlat-memory>` substrings in
    row text so a malicious or accidentally-formatted row can't break out
    of the injection block delimiter or spoof a closing tag.

    Replacement uses the unicode "less-than" / "greater-than" full-width
    forms — visually similar so the row stays readable, but the literal
    `<` and `>` characters are gone so the block boundary is unforgeable.
    """
    return (
        text
        .replace("</rlat-memory>", "＜/rlat-memory＞")
        .replace("<rlat-memory>", "＜rlat-memory＞")
    )


def _format_injection(hits: list[dict], recurrence_m: int) -> tuple[str, int]:
    """Render the §0.4 `<rlat-memory>` block from RecallReply hits.

    Truncates at row boundary once the cumulative char count would
    exceed `_MAX_INJECTION_CHARS`. Returns `(block, n_rows)` so the
    caller doesn't have to re-derive the count from the string. Row
    text is run through `_neutralise_boundary_tags` so a row containing
    `</rlat-memory>` (e.g. a captured session that quoted the spec)
    can't break the delimiter and inject downstream prompt content.
    """
    from .store import Row

    body_lines: list[str] = []
    char_budget = _MAX_INJECTION_CHARS
    for hit in hits:
        row = Row(**hit["row"])
        text = _neutralise_boundary_tags(row.text.replace("\n", " ").strip())
        line = f"- *{row.primary_polarity()}* — {text}"
        if char_budget - len(line) - 1 < 0:
            break
        body_lines.append(line)
        char_budget -= len(line) + 1
    if not body_lines:
        return "", 0
    block = (
        "<rlat-memory>\n"
        f"**Memory** ({len(body_lines)} lessons, recurrence ≥{recurrence_m}):\n\n"
        + "\n".join(body_lines)
        + "\n</rlat-memory>"
    )
    return block, len(body_lines)


def _spawn_daemon(memory_root: Path) -> None:
    """Lazy-spawn the recall daemon as a detached background subprocess.

    Uses `sys.executable -m` so the hook works even when `rlat` isn't
    on PATH (common with Windows hook configs).
    """
    import subprocess

    cmd = [
        sys.executable, "-m", "resonance_lattice.cli.app",
        "memory",
        "--memory-root", str(memory_root.parent),
        "--user", memory_root.name,
        "recall", "--daemon",
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
        )
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
        )
    except (OSError, FileNotFoundError):
        return


def _recall_via_daemon_or_spawn(request, memory_root: Path):
    """Try the daemon; on connect-fail, lazy-spawn and retry once.

    Returns None on any failure — the caller treats None as "no
    injection" per the §16.5 / §18.5 fail-open contract.
    """
    import time

    from .daemon import (
        DEFAULT_TIMEOUT_MS,
        daemon_socket_address,
        load_or_create_authkey,
        request_recall,
    )

    address = daemon_socket_address(memory_root)
    authkey = load_or_create_authkey(memory_root)
    reply = request_recall(
        request, address=address, authkey=authkey,
        timeout_ms=DEFAULT_TIMEOUT_MS,
    )
    if reply is not None:
        return reply
    _spawn_daemon(memory_root)
    time.sleep(_DAEMON_BOOT_WAIT_S)
    return request_recall(
        request, address=address, authkey=authkey,
        timeout_ms=_DAEMON_RETRY_TIMEOUT_MS,
    )


def run_hook(
    *,
    stdin=sys.stdin,
    stdout=sys.stdout,
    stderr=sys.stderr,
    user_id: str | None = None,
    memory_root_base: Path | None = None,
) -> int:
    """Read the UserPromptSubmit envelope from stdin, recall, and emit
    the hook output JSON to stdout. Returns the process exit code.

    Always exits 0 — fail-open per §16.5 / §18.5. Errors surface as a
    single stderr line so operators can `tail -f ~/.rlat/memory/...`
    to debug, but they never block the prompt.
    """
    _trace("UserPromptSubmit:fired")
    try:
        payload = json.loads(stdin.read())
    except (json.JSONDecodeError, OSError):
        json.dump({}, stdout)
        return 0

    prompt = payload.get("prompt", "")
    cwd = payload.get("cwd") or os.getcwd()
    if not isinstance(prompt, str) or not prompt.strip():
        json.dump({}, stdout)
        return 0

    # Heavy imports gated behind the fast-exit checks above so bad-stdin
    # / empty-prompt fail-open paths skip the encoder + multiprocessing
    # + portalocker import chain.
    from ._common import workspace_hash
    from .daemon import RecallRequest
    from .store import path_for_user

    try:
        memory_root = path_for_user(user_id=user_id, root=memory_root_base)
    except RuntimeError:
        json.dump({}, stdout)
        return 0

    # First hook fire on a fresh install: don't spawn the daemon for an
    # empty store. Skip silently.
    if not memory_root.exists():
        json.dump({}, stdout)
        return 0

    request = RecallRequest(
        query=prompt,
        cwd_hash=workspace_hash(str(cwd)),
    )
    try:
        reply = _recall_via_daemon_or_spawn(request, memory_root)
    except Exception as exc:
        print(f"[rlat] hook recall failed: {type(exc).__name__}", file=stderr)
        json.dump({}, stdout)
        return 0

    if reply is None or not reply.hits:
        json.dump({}, stdout)
        return 0

    block, n_rows = _format_injection(reply.hits, _RECURRENCE_M)
    if not block:
        json.dump({}, stdout)
        return 0

    print(f"[rlat] Recalled {n_rows} row(s)", file=stderr)
    json.dump({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": block,
        }
    }, stdout)
    return 0


def _parse_claude_code_transcript(transcript_path: Path, session_id: str, cwd: str):
    """Best-effort parser for Claude Code's JSONL transcript shape.

    Each line is a typed envelope; we keep only `type=user` / `type=assistant`
    entries. User messages carry `message.content[].text`; assistant messages
    carry `message.content[].text` + `message.content[].input.path` /
    `.content` for `tool_use` blocks. Anything we don't recognise is dropped
    silently — fail-open is the contract per §16.5.
    """
    from .capture import Message, ToolCall, Transcript

    messages: list[Message] = []
    if not transcript_path.exists():
        return Transcript(session_id=session_id, messages=tuple(), cwd=cwd)
    for raw in transcript_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        t = obj.get("type")
        if t not in ("user", "assistant"):
            continue
        msg = obj.get("message")
        if not isinstance(msg, dict):
            continue
        content_blocks = msg.get("content")
        if isinstance(content_blocks, str):
            # Claude Code emits plain-string content for normal user turns
            # (the block-list shape is only required when tool_use / thinking
            # blocks are interleaved). Treat string content as a single text
            # block so user-only sessions aren't dropped as 0-char.
            content_blocks = [{"type": "text", "text": content_blocks}]
        elif not isinstance(content_blocks, list):
            content_blocks = []
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    text_parts.append(text)
            elif btype == "tool_use":
                name = block.get("name", "")
                bin_ = block.get("input") or {}
                path = bin_.get("path") if isinstance(bin_, dict) else None
                content = (
                    bin_.get("content") if isinstance(bin_, dict) else ""
                ) or bin_.get("command", "") if isinstance(bin_, dict) else ""
                tool_calls.append(ToolCall(
                    name=str(name),
                    path=str(path) if isinstance(path, str) else None,
                    content=str(content) if content else "",
                ))
        if text_parts or tool_calls:
            messages.append(Message(
                role=t,  # type: ignore[arg-type]
                content="\n".join(text_parts),
                tool_calls=tuple(tool_calls),
            ))
    return Transcript(session_id=session_id, messages=tuple(messages), cwd=cwd)


def run_capture_hook(
    *,
    stdin=sys.stdin,
    stdout=sys.stdout,
    stderr=sys.stderr,
    user_id: str | None = None,
    memory_root_base: Path | None = None,
) -> int:
    """SessionEnd-hook entry point. Reads Claude Code's SessionEnd envelope
    from stdin (`{session_id, transcript_path, cwd, ...}`), parses the JSONL
    transcript at `transcript_path`, runs the §5.2 capture pipeline, and
    emits `{}` to stdout.

    Always exits 0 — fail-open per §16.5 / §18.5. The SessionEnd hook fires
    when the session terminates and a memory failure must never block the
    user's session close. (The plan §5.2 calls this the "Stop hook" — but
    Claude Code's `Stop` event is per-turn, not per-session; SessionEnd
    matches the spec's once-per-session intent.)
    """
    _trace("SessionEnd:fired")
    try:
        payload = json.loads(stdin.read())
    except (json.JSONDecodeError, OSError):
        _trace("SessionEnd:bad-stdin")
        json.dump({}, stdout)
        return 0

    transcript_path_raw = payload.get("transcript_path")
    session_id = payload.get("session_id", "")
    cwd = payload.get("cwd") or os.getcwd()
    _trace(f"SessionEnd:transcript_path={transcript_path_raw!r} session={session_id!r}")
    if not transcript_path_raw:
        _trace("SessionEnd:no-transcript-path")
        json.dump({}, stdout)
        return 0

    from .capture import GateConfig, capture
    from .redaction import Redactor
    from .store import Memory, path_for_user

    try:
        memory_root = path_for_user(user_id=user_id, root=memory_root_base)
    except RuntimeError:
        json.dump({}, stdout)
        return 0
    memory_root.mkdir(parents=True, exist_ok=True)

    try:
        tp = Path(transcript_path_raw)
        if not tp.exists() and tp.parent.exists():
            # Claude Code re-mints session_id on /compact resume but keeps
            # appending to the original transcript file. The payload path
            # then references a UUID that was never written to disk. Fall
            # back to the most-recently-modified `.jsonl` in the project
            # transcripts dir (the live session's actual file).
            siblings = sorted(
                tp.parent.glob("*.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if siblings:
                _trace(f"SessionEnd:fallback transcript_path={siblings[0].name}")
                tp = siblings[0]
        size = tp.stat().st_size if tp.exists() else -1
        _trace(f"SessionEnd:transcript exists={tp.exists()} size={size}")
        transcript = _parse_claude_code_transcript(tp, session_id, cwd)
        _trace(f"SessionEnd:parsed messages={len(transcript.messages)}")
    except Exception as exc:
        _trace(f"SessionEnd:parse-failed {type(exc).__name__}: {exc}")
        print(f"[rlat] capture parse failed: {type(exc).__name__}", file=stderr)
        json.dump({}, stdout)
        return 0

    try:
        store = Memory(root=memory_root)
        redactor = Redactor(audit_log_path=memory_root / "redaction.log")
        result = capture(transcript, store=store, redactor=redactor,
                          gate=GateConfig())
    except Exception as exc:
        print(f"[rlat] capture failed: {type(exc).__name__}", file=stderr)
        json.dump({}, stdout)
        return 0

    if result.row_id:
        _trace(f"SessionEnd:captured row_id={result.row_id} redactions={result.redactions}")
        print(f"[rlat] Captured row {result.row_id} ({result.redactions} "
              f"redactions)", file=stderr)
    elif result.skip_reason:
        _trace(f"SessionEnd:skipped reason={result.skip_reason}")
        print(f"[rlat] Capture skipped: {result.skip_reason}", file=stderr)
    json.dump({}, stdout)
    return 0


if __name__ == "__main__":
    sys.exit(run_hook())
