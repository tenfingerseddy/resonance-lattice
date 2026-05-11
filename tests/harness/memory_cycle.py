"""LayeredMemory cycle contract — superseded.

The v2.0 LayeredMemory three-tier model was retired in favour of the
v2.1 flat-memory store. The replacement contracts ship as the
`memory_v21_*` and `memory_v22_*` suites (capture, recall, retention,
consolidate, forget, distil arrows, daemon, schema-compat). Those
already assert end-to-end add → consolidate → tier-appears + decay +
primer regen.
"""

from __future__ import annotations

import sys


def run() -> int:
    print("[memory_cycle] SKIP — superseded by memory_v21_* + memory_v22_*",
          file=sys.stderr)
    return 0
