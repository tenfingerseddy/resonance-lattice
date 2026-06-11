"""Runtime parity — ONNX vs OpenVINO vs PyTorch produce numerically
equivalent embeddings. Tolerance: cosine similarity ≥ 0.9999 between
any two runtimes for the same input.

Stub: needs ONNX + OpenVINO export fixtures committed and a triple-
runtime probe. Deferred — production recipe pins one runtime per
session via env vars; runtime swaps happen rarely.
"""

from __future__ import annotations

import sys


def run() -> int:
    print("[runtime_parity] SKIP — multi-runtime fixture not built",
          file=sys.stderr)
    return 2  # harness SKIP sentinel — runner reports as skipped, not passed
