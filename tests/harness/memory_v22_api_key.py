"""memory_v22_api_key — API key discovery contracts.

Pins `_anthropic.discover_api_key` against the active env-var
chain. Four contracts:

  (a) `CLAUDE_API_2` is honoured when set (Kane's active slot).

  (b) `CLAUDE_API_2` outranks `CLAUDE_API` when both are set.

  (c) `CLAUDE_API` still works on its own (legacy back-compat).

  (d) `RLAT_LLM_API_KEY_ENV` indirection still wins over named slots.

Hermetic — manipulates `os.environ` for the duration of each check;
restores the prior state in `finally`. No real LLM call.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager


@contextmanager
def _isolated_env(**overrides):
    """Set the named env vars to the override values for the with-block;
    restore everything on exit, including deletion of vars that didn't
    exist before."""
    prior = {}
    for name in (
        "CLAUDE_API_2", "CLAUDE_API", "ANTHROPIC_API_KEY",
        "RLAT_LLM_API_KEY_ENV",
    ):
        prior[name] = os.environ.get(name)
        if name in os.environ:
            del os.environ[name]
    for name, value in overrides.items():
        if value is not None:
            os.environ[name] = value
    try:
        yield
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _check_claude_api_2_honoured() -> int:
    from resonance_lattice._anthropic import discover_api_key

    with _isolated_env(CLAUDE_API_2="sk-ant-claude-api-2-key"):
        key = discover_api_key()
    if key != "sk-ant-claude-api-2-key":
        print(f"[memory_v22_api_key] FAIL (a): key={key!r}", file=sys.stderr)
        return 1
    print("[memory_v22_api_key] (a) CLAUDE_API_2 honoured OK",
          file=sys.stderr)
    return 0


def _check_claude_api_2_outranks_claude_api() -> int:
    from resonance_lattice._anthropic import discover_api_key

    with _isolated_env(
        CLAUDE_API_2="sk-claude-api-2",
        CLAUDE_API="sk-claude-api-legacy",
    ):
        key = discover_api_key()
    if key != "sk-claude-api-2":
        print(f"[memory_v22_api_key] FAIL (b): key={key!r}", file=sys.stderr)
        return 1
    print("[memory_v22_api_key] (b) CLAUDE_API_2 > CLAUDE_API OK",
          file=sys.stderr)
    return 0


def _check_claude_api_legacy_still_works() -> int:
    from resonance_lattice._anthropic import discover_api_key

    with _isolated_env(CLAUDE_API="sk-claude-api-legacy"):
        key = discover_api_key()
    if key != "sk-claude-api-legacy":
        print(f"[memory_v22_api_key] FAIL (c): key={key!r}", file=sys.stderr)
        return 1
    print("[memory_v22_api_key] (c) CLAUDE_API legacy still works OK",
          file=sys.stderr)
    return 0


def _check_indirection_wins() -> int:
    from resonance_lattice._anthropic import discover_api_key

    # Set both named slots and an indirection target; the indirected
    # value should win even though CLAUDE_API_2 also has a value.
    with _isolated_env(
        RLAT_LLM_API_KEY_ENV="MY_CUSTOM_KEY",
        CLAUDE_API_2="sk-claude-api-2",
    ):
        os.environ["MY_CUSTOM_KEY"] = "sk-via-indirection"
        try:
            key = discover_api_key()
        finally:
            del os.environ["MY_CUSTOM_KEY"]
    if key != "sk-via-indirection":
        print(f"[memory_v22_api_key] FAIL (d): key={key!r}", file=sys.stderr)
        return 1
    print("[memory_v22_api_key] (d) RLAT_LLM_API_KEY_ENV indirection wins OK",
          file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_claude_api_2_honoured,
        _check_claude_api_2_outranks_claude_api,
        _check_claude_api_legacy_still_works,
        _check_indirection_wins,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v22_api_key] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
