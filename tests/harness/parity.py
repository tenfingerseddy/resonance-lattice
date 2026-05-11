"""Cross-mode parity — bundled vs local vs remote return identical top-k.

Stub: needs three KMs built across modes + top-k comparison fixture.
Deferred until a regression actually motivates it; the runtime_parity
and conversion suites already cover the underlying read paths.
"""

from __future__ import annotations

import sys


def run() -> int:
    print("[parity] SKIP — cross-mode KM fixture not built", file=sys.stderr)
    return 0
