"""Encoder determinism — same input, same output.

CPU: bit-exact. GPU: within FP16/BF16 tolerance.

Stub: the conversion to a live test is cheap (encode same text twice,
compare arrays) but adds ~30s of encoder cold-start to every field/
or install/ changeset's harness run. Deferred until that cost is paid
back by a real determinism regression risk.
"""

from __future__ import annotations

import sys


def run() -> int:
    print("[encoder_determinism] SKIP — encoder cold-start deferred",
          file=sys.stderr)
    return 0
