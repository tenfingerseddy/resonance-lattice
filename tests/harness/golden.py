"""Golden-file regression — locked (corpus, query, top_k) tuples.

Stub: needs golden fixtures committed to tests/harness/fixtures/golden/.
Update via `--update-goldens` flag + reviewer sign-off (the eventual
contract). Deferred until a regression motivates the upfront fixture
investment.
"""

from __future__ import annotations

import sys


def run() -> int:
    print("[golden] SKIP — golden fixtures not committed", file=sys.stderr)
    return 0
