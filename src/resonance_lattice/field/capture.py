"""capture — the self-aware heart of the `.rlat`.

Every retrieval through `field.retrieve` / `field.retrieve_insight` is observed
here: the query *fingerprint* (embedding) and the per-rank scores land in an
in-memory buffer keyed by the corpus identity (`.claude/plans/insight-engine/
capture.md` §3). Observing is fused into retrieving — using a `.rlat` **is** being
seen by it; no hook, skill, or caller cooperation enforces it.

Properties (capture.md §3, resolved 2026-06-02):

- **Fingerprints, not words.** The embedding + scores only — never the query
  text (the heart has none; the text lives above it). Privacy by construction.
- **In-memory and bounded.** A per-corpus ring buffer, capped so a long-running
  process can't grow it without bound between folds. The *fold* into the `.rlat`
  lives in the store layer; this module keeps **no store/disk dependency**, so
  `field.retrieve` imports it with no cycle.
- **Raise-your-hand to be ignored.** `is_user_query` defaults True; internal
  machinery (summary probes, deep-search hops, skill-context batches, recall,
  verified retrieval) runs inside `internal_retrieval()` so it does not pollute
  the user-intent stream the head clusters. (The head is also meant to learn the
  user-vs-internal split from the stream alone — this flag is the floor that bet
  is measured against.)
- **Never breaks retrieval.** Every entry point swallows its own errors — a lost
  observation is never a failed query.

One user search produces *two* observations here (a source-band retrieval and an
insight-band retrieval); they share an identical `query_emb`, which is the
natural join key downstream.
"""

from __future__ import annotations

import os
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Iterable

_SESSION_ENV = "RLAT_DOGFOOD_SESSION"
# Per-corpus ring cap. A fold drains the buffer; this only bounds memory
# *between* folds, so a runaway loop (or a fold that never fires) can't OOM.
_MAX_BUFFERED = 4096

# One ring per corpus. The dict's key-cardinality is bounded by the number of
# distinct corpora a process queries (a handful), not by `_MAX_BUFFERED`; a fold
# drains a ring (clears its entries) but keeps the key. Not a leak in practice.
_buffers: dict[str, deque] = {}
_internal: ContextVar[bool] = ContextVar("rlat_internal_retrieval", default=False)


def session_id() -> str:
    """The session bucket for reformulation detection (capture.md §4).

    `RLAT_DOGFOOD_SESSION` wins when set (controlled multi-batch runs);
    otherwise the UTC calendar day."""
    return (
        os.environ.get(_SESSION_ENV)
        or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )


@contextmanager
def internal_retrieval():
    """Mark every retrieval inside this block as machine-internal — the
    raise-your-hand-to-be-ignored tag (capture.md §3).

    Observations made inside carry `is_user_query=False`, so summary probes /
    deep-search hops / skill-context batches / recall / verified retrieval do not
    pollute the user-intent stream. Re-entrant and exception-safe (the flag is
    always restored)."""
    token = _internal.set(True)
    try:
        yield
    finally:
        _internal.reset(token)


def observe(
    km_id: str | None,
    query_emb,
    ranked: Iterable[tuple[int, float]] | None,
    layer: str,
    *,
    is_user_query: bool | None = None,
) -> None:
    """Record one retrieval against the corpus identified by `km_id`.

    `ranked` is the `[(idx, score), ...]` the retrieval produced; `layer` is
    "source" or "insight". The query *fingerprint* (rounded embedding) and the
    scores are kept — **never** the query text. `is_user_query` defaults to "not
    inside `internal_retrieval()`". A `km_id` of None (a synthetic/in-memory
    archive with no source path) is not observable, so it is skipped. **Never
    raises** — a lost observation must not break the query that produced it."""
    try:
        if not km_id:
            return
        if is_user_query is None:
            is_user_query = not _internal.get()
        rows = sorted(list(ranked or []), key=lambda t: -t[1])
        # Buffer the fingerprint as a compact float32 array (~3 KB), not a
        # Python float list (~24 KB of objects per entry — at the 4096-entry
        # ring cap that was ~100 MB per corpus on a long-lived process, the
        # same OOM class the Fabric warm path just closed). `buffered()`
        # converts back to the rounded-list wire shape at the fold seam.
        import numpy as _np
        emb = (
            _np.asarray(query_emb, dtype=_np.float32)
            if query_emb is not None
            else None
        )
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session": session_id(),
            "layer": layer,
            "is_user_query": bool(is_user_query),
            "query_emb": emb,
            "ranked": [
                {"rank": i, "idx": int(idx), "score": round(float(s), 6)}
                for i, (idx, s) in enumerate(rows)
            ],
        }
        buf = _buffers.get(km_id)
        if buf is None:
            buf = _buffers[km_id] = deque(maxlen=_MAX_BUFFERED)
        buf.append(entry)
    except Exception:
        return  # capture never breaks retrieval


def buffered(km_id: str | None) -> list[dict]:
    """The currently-buffered (un-folded) observations for `km_id`, oldest
    first. A peek — does not drain. Never raises.

    Rows leave here in the WIRE shape (query_emb as a rounded float list,
    JSON-ready) regardless of the compact float32 storage inside the ring —
    consumers (the telemetry fold, the counters layer, tests) keep the same
    contract they always had."""
    try:
        return [_wire_row(e) for e in _buffers.get(km_id or "", ())]
    except Exception:
        return []


def drain(km_id: str | None, n: int | None = None) -> list[dict]:
    """Pop and return buffered observations for `km_id`, oldest first.

    `n=None` pops the whole buffer (the default — a full fold). `n` pops only
    the oldest `n` — exactly the snapshot a peek-then-persist fold already wrote
    — leaving any observation that arrived *during* the persist for the next
    fold, so a concurrent client's observation is never cleared unpersisted. The
    fold reads this and writes the rows into the `.rlat`; draining clears them so
    the same observation is not folded twice. Never raises."""
    try:
        buf = _buffers.get(km_id or "")
        if not buf:
            return []
        if n is None:
            out = list(buf)
            buf.clear()
        else:
            out = [buf.popleft() for _ in range(min(n, len(buf)))]
        return [_wire_row(e) for e in out]
    except Exception:
        return []


def _wire_row(entry: dict) -> dict:
    """Convert a buffered entry to the wire shape (query_emb as a rounded
    float list) — the ring stores a compact float32 array instead."""
    emb = entry.get("query_emb")
    if emb is not None and not isinstance(emb, list):
        entry = dict(entry)
        entry["query_emb"] = [round(float(x), 6) for x in emb]
    return entry
