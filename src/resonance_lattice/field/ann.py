"""ANN index over band embeddings.

Default: FAISS `IndexHNSWFlat` with M=32, efConstruction=200, efSearch=128.
Threshold is fixed at N > 5000 — below that, exact matmul on numpy is fast
enough that the index-build cost isn't justified.

FAISS uses METRIC_L2 over already-L2-normalised vectors: for unit vectors
||a-b||² = 2 - 2<a,b>, so L2 ranking is monotonic with cosine ranking.
FAISS HNSW + METRIC_INNER_PRODUCT has known quality issues — the L2-on-
normalised pattern is the canonical FAISS cosine recipe.

Library locked at FAISS by Audit 04 (Phase 1 #14):
- hnswlib has no precompiled wheel for Python 3.12 on Windows; source
  builds need Visual C++ Build Tools — fails the audit's tertiary
  cross-platform-wheel gate.
- ScaNN is Linux/macOS only.
- FAISS has prebuilt wheels everywhere.

efSearch=128 chosen empirically: the base plan's 32 hit ~13% recall@10 on
synthetic 50K @ 768d and ~59% at N=5K. 128 clears the 0.95 audit gate at
N=5K synthetic. Real-corpus recall (lower intrinsic dim, semantic clustering)
typically clears the gate at lower efS than synthetic random unit vectors;
exact calibration happens at Phase 1 #15 (BEIR-5 floor lock).

Phase 1 deliverable. Base plan §3.2.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import _runtime_common as common
from . import dense

if TYPE_CHECKING:
    from collections.abc import Sequence

ANN_THRESHOLD_N = 5000
HNSW_M = 32
HNSW_EFCONSTRUCTION = 200
HNSW_EFSEARCH = 128

_INSTALL_HINT = "Install with `pip install rlat[ann]`."


def should_build_ann(n_passages: int) -> bool:
    """Whether the corpus is large enough to warrant an HNSW index."""
    return n_passages > ANN_THRESHOLD_N


def build(embeddings: np.ndarray) -> Any:
    """Build a FAISS HNSW index over (N, D) L2-normalised embeddings.

    Each passage's FAISS label is its row index in `embeddings`, so search
    results map straight back to the band tensor without an indirection
    table.
    """
    faiss = common.require_module("faiss", _INSTALL_HINT)
    n, dim = embeddings.shape
    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_L2)
    # efConstruction must be set BEFORE add() — it's read during graph build.
    index.hnsw.efConstruction = HNSW_EFCONSTRUCTION
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    index.hnsw.efSearch = HNSW_EFSEARCH
    return index


def save(index: Any, path: Path) -> None:
    """Persist a FAISS index to `path`. Phase 2's store layer wires this into
    the knowledge-model `ann/` directory."""
    faiss = common.require_module("faiss", _INSTALL_HINT)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def serialize(index: Any) -> bytes:
    """Serialise a FAISS index to bytes for embedding in the .rlat archive.

    Build paths call this and hand the bytes to `archive.write(ann_blobs=...)`
    rather than reaching past `field/ann.py` into faiss directly — keeps the
    library-specific import in one place so a future ANN swap is one-file.
    """
    faiss = common.require_module("faiss", _INSTALL_HINT)
    return bytes(faiss.serialize_index(index))


def deserialize(blob: bytes) -> Any:
    """Restore a FAISS index from bytes recovered via `archive.read`.

    Inverse of `serialize`. Reapplies `efSearch=128` so a freshly-loaded
    index has the calibrated runtime knob — FAISS persists efConstruction
    in the file but efSearch is a query-time setting.
    """
    faiss = common.require_module("faiss", _INSTALL_HINT)
    arr = np.frombuffer(blob, dtype=np.uint8)
    index = faiss.deserialize_index(arr)
    index.hnsw.efSearch = HNSW_EFSEARCH
    return index


def load(path: Path, dim: int) -> Any:
    """Load a FAISS index. `dim` is unused at load time (FAISS records it in
    the file) but accepted for API symmetry with the hnswlib-era contract;
    Phase 2's store layer reads `dim` from `metadata.json` regardless."""
    faiss = common.require_module("faiss", _INSTALL_HINT)
    common.require_asset(path, "FAISS HNSW index")
    index = faiss.read_index(str(path))
    index.hnsw.efSearch = HNSW_EFSEARCH
    return index


def search(
    index: Any,
    query_embedding: np.ndarray,
    registry: "Sequence | None" = None,
    projection_matrix: np.ndarray | None = None,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Top-k cosine retrieval through a FAISS HNSW index.

    Mirrors `dense.search`'s signature so callers can route between the
    exact and ANN paths via `should_build_ann(N)` without rewiring args.

    FAISS METRIC_L2 returns squared L2 distance. For L2-normalised vectors
    the relationship is `||a-b||² = 2 - 2*cos`, so we recover cosine score
    as `1 - L2² / 2`. Higher score wins, matching `dense.search`.
    """
    if top_k <= 0:
        return []
    q = query_embedding
    if projection_matrix is not None:
        q = q @ projection_matrix.T
        common.l2_normalize(q)

    n = index.ntotal
    budget = min(top_k * common.CANDIDATE_MULTIPLIER if registry is not None else top_k, n)
    # FAISS requires efSearch >= k; bump when the candidate budget exceeds
    # HNSW_EFSEARCH and restore on exit so subsequent callers don't pay the
    # inflated cost.
    bumped_ef = False
    try:
        while True:
            if budget > HNSW_EFSEARCH:
                index.hnsw.efSearch = budget
                bumped_ef = True
            q_batch = np.ascontiguousarray(q.reshape(1, -1), dtype=np.float32)
            distances, labels = index.search(q_batch, budget)
            hits = [
                (int(i), float(1.0 - d / 2.0))
                for i, d in zip(labels[0], distances[0])
                if i >= 0  # FAISS returns -1 for empty slots
            ]
            if registry is not None:
                hits = dense.dedup_by_source(hits, registry)
            if len(hits) >= top_k or budget >= n:
                return hits[:top_k]
            budget = min(budget * 2, n)
    finally:
        if bumped_ef:
            index.hnsw.efSearch = HNSW_EFSEARCH
