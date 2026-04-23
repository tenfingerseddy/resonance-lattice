"""Shared helpers for the rlat memory hooks.

Lean subset of the private dogfood helpers — strips telemetry, budget
tracking, and the RLAT_DOGFOOD gate. What remains is only what the two
shipped hooks need: read stdin payload, resolve a session id, emit the
hookSpecificOutput JSON response, and swallow exceptions so a hook
failure never blocks a Claude Code session.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path


def read_hook_input() -> dict:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return {}
        return json.loads(raw)
    except Exception:
        return {}


def session_id_from_input(hook_input: dict) -> str:
    sid = hook_input.get("session_id") if isinstance(hook_input, dict) else None
    if sid:
        return str(sid)
    return os.environ.get("CLAUDE_SESSION_ID") or f"anon-{uuid.uuid4().hex[:8]}"


def emit_hook_output(payload: dict | None) -> None:
    if not payload:
        return
    try:
        sys.stdout.write(json.dumps(payload))
        sys.stdout.flush()
    except Exception:
        pass


class guard:
    """Swallow any exception and exit 0 so the session is never blocked.

    On failure, write a single line to stderr — Claude Code surfaces hook
    stderr in the session log, which is enough to debug without creating
    a telemetry directory the user didn't ask for.
    """

    def __init__(self, hook_name: str):
        self.hook_name = hook_name
        self.t0 = time.perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            try:
                elapsed_ms = round((time.perf_counter() - self.t0) * 1000.0, 1)
                sys.stderr.write(
                    f"[rlat-hook:{self.hook_name}] {type(exc).__name__}: {exc} "
                    f"(after {elapsed_ms}ms)\n"
                )
            except Exception:
                pass
        return True


def project_dir(payload: dict) -> Path:
    return Path(payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())
