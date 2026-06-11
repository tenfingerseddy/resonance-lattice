"""BEIR-5 locked-floor regression gate — STUB, not wired.

The locked floor (0.5144 nDCG@10 mean) and its reproduction live in
docs/internal/BENCHMARK_GATE.md; the actual runs happen on Kaggle T4, not
in this harness. Until a local/CI reproduction is wired, this suite
reports SKIP so `--phase-gate` cannot silently claim a floor check that
never ran (2026-06 review finding: this function was `return 0`).
"""

from __future__ import annotations

import sys


def run() -> int:
    print("[benchmark_gate] SKIP — BEIR-5 floor not wired into the harness; "
          "see docs/internal/BENCHMARK_GATE.md for the manual reproduction",
          file=sys.stderr)
    return 2  # harness SKIP sentinel — runner reports as skipped, not passed
