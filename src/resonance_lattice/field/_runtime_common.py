"""Shared helpers for the field layer.

Kept module-private (`_`-prefixed filename) — outside callers go through
`field.encoder.Encoder`.
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def require_module(name: str, install_hint: str) -> ModuleType:
    """Import `name` or raise a RuntimeError with a uniform install hint."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise RuntimeError(
            f"`{name}` is not installed. {install_hint}"
        ) from exc


def require_asset(path: Path, label: str) -> None:
    """Raise a uniform missing-asset error pointing at `rlat install-encoder`."""
    if not path.exists():
        raise RuntimeError(
            f"{label} not found at {path}. Run `rlat install-encoder` to populate the cache."
        )


# Pull this many candidates from the band before dedup so the post-dedup
# top-k slice is unlikely to come up short. Tuned for the WS3 #292 observation
# that ~10-30% of nearest-neighbour pairs share (source_file, char_offset).
# Used by both field.dense and field.ann.
CANDIDATE_MULTIPLIER = 4


def cls_pool(arr: np.ndarray) -> np.ndarray:
    """Extract the CLS token from a (N, seq_len, dim) hidden-state tensor."""
    assert arr.ndim == 3, f"expected (N, seq_len, dim), got shape {arr.shape}"
    return arr[:, 0, :]


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """In-place L2 normalisation along the last axis with an ε guard.

    Mutates `x`. For a 1-D vector this divides by its norm; for a 2-D matrix
    each row is normalised. Zero-norm rows are left as-is (the ε floor only
    prevents NaN, not the all-zeros vector itself).
    """
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    np.maximum(norms, eps, out=norms)
    x /= norms
    return x


def is_intel_cpu() -> bool:
    """Best-effort vendor check used both at runtime selection (OpenVINO) and
    at install time (whether to convert ONNX → OV IR).

    OpenVINO runs on AMD/ARM x86 too, but its tuning targets Intel — we prefer
    ONNX on non-Intel CPUs to keep query latency stable until Audit 04
    measures otherwise."""
    proc = platform.processor() or ""
    if "Intel" in proc:
        return True
    if sys.platform == "win32":
        return os.environ.get("PROCESSOR_IDENTIFIER", "").startswith("Intel")
    if sys.platform.startswith("linux"):
        try:
            return "GenuineIntel" in Path("/proc/cpuinfo").read_text()
        except OSError:
            return False
    if sys.platform == "darwin":
        # Intel Macs are x86_64; Apple Silicon is arm64.
        return platform.machine() == "x86_64"
    return False
