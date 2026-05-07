"""Recall — §0.6 two-stage cosine retrieval over the per-user band.

Pipeline (in order, all gates active by default):

    1. Filter: drop is_bad rows; drop rows below `cosine_floor`.
    2. Workspace gate: keep rows whose polarity contains the caller's
       `cwd_hash` workspace tag OR `cross-workspace`.
    3. Confidence gate: keep only if top1 ≥ floor AND
       (top1 - top2) ≥ gap. Empty result if either fails.
    4. Recurrence gate: keep rows with recurrence_count ≥ M.
    5. Sort by cosine descending; return top_k.

Spec: `.claude/plans/fabric-agent-flat-memory.md` §0.6 + §0.4.

Two surfaces:

- `rank(query, *, rows, band, encoder, ...)` — pure algorithm; no I/O.
  The future MVP daemon caches `(rows, band)` and calls `rank()` per
  request, so the §12.4 30/80ms gate measures only the cosine + gate
  + sort cost.
- `recall(query, *, store, encoder=None, ...)` — full pipeline. Calls
  `Memory.read_all` then `rank`; what the CLI one-shot path uses.

The CLI `rlat memory recall`, the future daemon socket, and the future
UserPromptSubmit hook all delegate to one of these — they never
bypass the gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..field._runtime_common import l2_normalize
from ..field.encoder import Encoder
from ._common import workspace_hash
from .store import Memory, Row

# Defaults from §0.4 / §0.6. Callers can override per-invocation; the
# §0.6 contract says all gates run by default.
DEFAULT_COSINE_FLOOR = 0.7
DEFAULT_TOP1_TOP2_GAP = 0.05
DEFAULT_MIN_RECURRENCE = 3
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class RecallHit:
    """One row above all four §0.6 gates, with its query cosine attached.

    `cosine` is the raw value before any filter; callers that need to
    surface confidence (e.g., the `--explain` CLI flag in MVP) read it
    directly. Sort order is descending by `cosine`.
    """

    row: Row
    cosine: float


def _row_matches_cwd(row: Row, cwd_hash: str) -> bool:
    """§0.6 step 2 — row in scope iff it has the cwd's workspace tag or
    the cross-workspace bypass.
    """
    target = f"workspace:{cwd_hash}"
    for tag in row.polarity:
        if tag == target or tag == "cross-workspace":
            return True
    return False


def _encode_query(query: str, encoder: Encoder) -> np.ndarray:
    """Encode the query under the same `text + " | intent: " + intent`
    convention rows use, with intent="" since queries don't carry one.
    The L2-normalised result lets us compute cosine as a dot product.
    """
    embedding = encoder.encode([query])[0]
    l2_normalize(embedding)
    return embedding


def rank(
    query: str,
    *,
    rows: list[Row],
    band: np.ndarray,
    encoder: Encoder,
    cwd_hash: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    cosine_floor: float = DEFAULT_COSINE_FLOOR,
    top1_top2_gap: float = DEFAULT_TOP1_TOP2_GAP,
    min_recurrence: int = DEFAULT_MIN_RECURRENCE,
) -> list[RecallHit]:
    """Run the §0.6 retrieval pipeline against an already-loaded snapshot.

    Pure algorithm, no I/O. Daemon callers cache `(rows, band)` and
    invoke `rank()` per request; one-shot callers go through `recall()`.

    `cwd_hash` defaults to the caller's `Path.cwd()` so the surface is
    usable without explicit cwd plumbing. Pass an explicit value when
    serving a different working directory.
    """
    if not rows:
        return []
    if cwd_hash is None:
        cwd_hash = workspace_hash(str(Path.cwd()))

    query_emb = _encode_query(query, encoder)

    # Steps 1-2 fused: cosine threshold + is_bad + workspace gate in one
    # pass. Band is L2-normalised at write time, so a dot product equals
    # cosine. Negative cosines always fall below cosine_floor — no clamp
    # needed. At 5K rows / default cosine_floor=0.7, the early-continue
    # path eliminates ~99% of work before we touch the polarity list.
    cosines = band @ query_emb
    eligible: list[tuple[Row, float]] = []
    for row, cos in zip(rows, cosines):
        if cos < cosine_floor:
            continue
        if row.is_bad:
            continue
        if not _row_matches_cwd(row, cwd_hash):
            continue
        eligible.append((row, float(cos)))

    if not eligible:
        return []

    eligible.sort(key=lambda pair: pair[1], reverse=True)

    # Step 3: confidence gate. The gap check requires two cosines; with
    # only one eligible row it implicitly compares against floor (which
    # the row already cleared) — treat single-row as gap-passes since
    # there's no second contender to make the result ambiguous.
    top1_cosine = eligible[0][1]
    if top1_cosine < cosine_floor:
        return []
    if len(eligible) >= 2:
        top2_cosine = eligible[1][1]
        if (top1_cosine - top2_cosine) < top1_top2_gap:
            return []

    # Step 4: recurrence gate.
    above_recurrence = [
        (row, cos) for row, cos in eligible if row.recurrence_count >= min_recurrence
    ]
    if not above_recurrence:
        return []

    return [RecallHit(row=row, cosine=cos) for row, cos in above_recurrence[:top_k]]


def recall(
    query: str,
    *,
    store: Memory,
    cwd_hash: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    encoder: Encoder | None = None,
    cosine_floor: float = DEFAULT_COSINE_FLOOR,
    top1_top2_gap: float = DEFAULT_TOP1_TOP2_GAP,
    min_recurrence: int = DEFAULT_MIN_RECURRENCE,
) -> list[RecallHit]:
    """Run the §0.6 retrieval pipeline against the per-user band.

    Convenience wrapper that loads the snapshot from `store` then runs
    `rank()`. One-shot CLI / fallback path; the daemon shape calls
    `store.read_all()` once and invokes `rank()` per request.
    """
    rows, band = store.read_all()
    if encoder is None:
        encoder = store._ensure_encoder()  # type: ignore[attr-defined]
    return rank(
        query,
        rows=rows,
        band=band,
        encoder=encoder,
        cwd_hash=cwd_hash,
        top_k=top_k,
        cosine_floor=cosine_floor,
        top1_top2_gap=top1_top2_gap,
        min_recurrence=min_recurrence,
    )
