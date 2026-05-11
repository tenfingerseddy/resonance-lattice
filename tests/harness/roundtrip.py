"""Round-trip — build → save → load → query identical.

Stub: the substrate's incremental_refresh + incremental_sync +
optimised_reproject suites already cover the same write/read paths
end-to-end with assertions. A standalone roundtrip fixture would add
nothing those don't catch; deferred unless that changes.
"""

from __future__ import annotations

import sys


def run() -> int:
    print("[roundtrip] SKIP — covered by incremental_* + optimised_reproject",
          file=sys.stderr)
    return 0
