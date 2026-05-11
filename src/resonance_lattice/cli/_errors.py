"""CLI helpers: uniform user-error reporting.

`user_error(msg)` prints `error: <msg>` to stderr and returns
`EXIT_USER_ERROR` (rc=1). Shared so every subcommand surfaces user-facing
errors with the same prefix and exit code.
"""

from __future__ import annotations

import sys

EXIT_OK = 0
EXIT_USER_ERROR = 1


def user_error(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return EXIT_USER_ERROR
