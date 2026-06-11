"""Verified-retrieval drift correctness.

Mutate source → freshness flags it. Revert → flag clears. --verified-only
filters drifted hits. Stub: needs a synthetic source-corpus fixture +
mutation harness. Deferred — the production audit-07 verified path is
exercised by the `freshness` CLI in real workflows.
"""

from __future__ import annotations

import sys


def run() -> int:
    print("[drift] SKIP — drift fixture + mutation harness not built",
          file=sys.stderr)
    return 2  # harness SKIP sentinel — runner reports as skipped, not passed
