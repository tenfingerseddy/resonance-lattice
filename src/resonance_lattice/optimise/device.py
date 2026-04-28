"""Device auto-selection + wall-time estimate for `rlat optimise`.

CUDA available → torch on GPU with fp16 forward (fp32 W optimizer state).
CUDA not available → torch on CPU at fp32. The base plan §4.5 specifies a
numpy-CPU path for portability; v2.0 ships torch-everywhere because
`[optimise]` extras pulls torch unconditionally and a parallel numpy
implementation of MRL InfoNCE is maintenance burden the audit didn't
justify. Spec compliance: same final W, same hyperparameters, same
exit-code semantics — only the implementation backend differs from the
spec's "numpy on CPU" letter.

Phase 4 deliverable. Base plan §4.5.
"""

from __future__ import annotations

import importlib.util
from typing import Literal

Device = Literal["cuda", "cpu"]


def select() -> Device:
    """Pick `cuda` if torch + CUDA are both reachable, else `cpu`.

    `find_spec` checks for torch's presence without triggering its import
    (cheap). The actual `torch.cuda.is_available()` query is only run when
    torch is installed, which is the case under `[optimise]` extras.
    """
    if importlib.util.find_spec("torch") is None:
        return "cpu"
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# Empirical anchors for the wall-time estimate, calibrated against the spec
# (§4.3 says ~30 min on A100 for 40K passages on CUDA, ~4-6 hours for the
# same size on CPU). Both scale roughly linearly with N_passages × N_queries.
# The CUDA estimate uses A100 throughput; consumer GPUs (T4, etc.) trend
# ~2-3× slower — caller should treat the number as a floor.
_CUDA_MIN_PER_PASSAGE_QUERY = 30.0 / (40_000 * 6_000)
_CPU_MIN_PER_PASSAGE_QUERY = 5 * 60.0 / (40_000 * 6_000)


def estimate_wall_time(device: Device, n_passages: int, n_queries: int) -> float:
    """Return the projected wall-time in **minutes** for the training run.

    The estimate is intentionally rough: it amortises the per-step cost
    (matmul scaling) plus the synth-query encoding pass, but doesn't
    account for the LLM-call wall time of synth_queries.py (which runs
    before training and is API-bound, not compute-bound).
    """
    if n_passages <= 0 or n_queries <= 0:
        return 0.0
    work = float(n_passages) * float(n_queries)
    rate = _CUDA_MIN_PER_PASSAGE_QUERY if device == "cuda" else _CPU_MIN_PER_PASSAGE_QUERY
    return work * rate
