"""Field — the router layer.

Single encoder: gte-modernbert-base 768d, CLS pooling, L2-normalised.
Three inference runtimes (auto-selected): ONNX (non-Intel CPU), OpenVINO
(Intel CPU), PyTorch (build only).

Phase 1 deliverable. See base plan §1, §3.
Lensed-knowledge Day 1: `retrieve_insight` adds insight-band retrieval
alongside source retrieval; composition happens at the CLI layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from . import ann, capture, dense

if TYPE_CHECKING:
    # Type-only imports to avoid circular load order: store/bands.py imports
    # field/_runtime_common.py, which triggers field/__init__.py mid-init of
    # store/archive.py — a real circular if the BandHandle / PassageCoord
    # imports are at module scope. Runtime usage in `retrieve()` doesn't
    # need them imported because they're only referenced via typed args
    # (duck-typed at runtime).
    from ..store.archive import BandHandle
    from ..store.registry import PassageCoord


def retrieve(
    query_emb: np.ndarray,
    handle: "BandHandle",
    ann_index: object | None,
    registry: "list[PassageCoord]",
    top_k: int,
) -> list[tuple[int, float]]:
    """Single retrieval entry point — ANN when an index is bound, exact dense
    cosine otherwise. Both paths return `[(passage_idx, score), ...]`
    descending by score.

    Lifted from `cli/search.py` + `cli/summary.py`; the if/else dispatch
    was duplicated across both call sites.

    The capture heart (`capture.observe`, capture.md §3): every retrieval is
    observed against its corpus (`handle.km_id`) — fingerprint + scores into the
    in-memory buffer, never the query text. `is_user_query` is inferred from
    `capture.internal_retrieval()`; an internal caller wraps its call to opt out.
    Observation never raises and never alters the result.
    """
    if ann_index is not None:
        result = ann.search(
            ann_index, query_emb,
            registry=registry,
            top_k=top_k,
        )
    else:
        result = dense.search(
            query_emb, handle.band,
            registry=registry,
            top_k=top_k,
        )
    capture.observe(getattr(handle, "km_id", None), query_emb, result, "source")
    return result


def retrieve_insight(
    query_emb: np.ndarray,
    insight_band: np.ndarray,
    ann_index: object | None,
    top_k: int,
    km_id: str | None = None,
) -> list[tuple[int, float]]:
    """Cosine top-k against the insight band.

    Parallel to `retrieve()` for the source layer, but operates on the
    `insight` band and does NOT take a registry (no source-file dedup —
    insight rows are unique by `insight_id`, and pre-promotion semantic
    duplicates are filtered by the compression test, not at query time).

    Returns `[(insight_idx, score), ...]` descending. The caller passes
    the result to `verify_insight_hits` to resolve verdict-state and
    drift, then composes with source hits at the CLI layer per the
    trust-contract foundation 3 (visible labelling at every output).

    Empty band → empty list; same shape contract as `retrieve()` on an
    empty corpus.

    `km_id` is the corpus identity for the capture heart (the insight band has
    no `BandHandle` here, so the caller passes `insight_handle.km_id`); the
    insight retrieval is observed under it (capture.md §3). Observation never
    raises and never alters the result.
    """
    if insight_band is None or insight_band.shape[0] == 0:
        return []
    if ann_index is not None:
        result = ann.search(
            ann_index, query_emb,
            registry=None,
            top_k=top_k,
        )
    else:
        result = dense.search(
            query_emb, insight_band,
            registry=None,
            top_k=top_k,
        )
    capture.observe(km_id, query_emb, result, "insight")
    return result
