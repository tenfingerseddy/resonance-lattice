"""telemetry — fold the capture heart's in-memory stream into the `.rlat`.

`field.capture` observes every retrieval into a bounded in-memory ring keyed by
corpus identity (the resolved `.rlat` path). This module is the **fold**: it
drains that ring, redacts each row (invariant §8), and appends it to the
`insight/telemetry.jsonl` member INSIDE the `.rlat` — so the telemetry travels
with the portable file and the loop is self-contained, no sidecar (capture.md
§3, architecture.md §7). The decide tier (`field.counters`, `curator.signals`)
reads it back with `read`, in the SAME row shape it reads the live buffer.

This is the FRONT half of the loop the back half already had — it is the fold
that `field/capture.py:134` (`drain`) was written for and that nothing called.
With it, the self-contained property flips false→true: a `.rlat` that has been
used carries the evidence of its use inside itself.

**Cadence.** The fold is a SESSION-boundary operation, not a per-query one — one
whole-archive rewrite folds a whole session's observations (architecture.md §7).
Observation is always on; *persistence* is the part the ZIP format gates: on the
one-shot CLI a "session" is signalled by `RLAT_DOGFOOD_SESSION` (or forced with
`RLAT_CAPTURE_PERSIST`), and a bare single search persists nothing — the accepted
one-shot tradeoff (capture.md §3) carried until the SQLite format re-home makes
per-query persistence cheap enough to default on. This keeps a read from
surprise-rewriting a possibly-large or read-only corpus.

**Format seam.** Everything here is format-agnostic; only the two functions in
`store.archive` (`read_telemetry` / `append_telemetry_in_place`) know it is a
ZIP. The SQLite re-home swaps those; this file, the row format, the caller in
`cli/search.py`, and every reader are unchanged.

Never raises — a failed fold (read-only corpus, disk full, lost write race) must
never break the query that produced the telemetry, exactly like capture itself.
"""

from __future__ import annotations

import os

from . import archive

# Explicit force-on/off for persistence (auto-default + override, the project
# knob rule). `RLAT_CAPTURE_PERSIST=1` forces folding on (e.g. a user who wants
# every search to teach a small corpus); `=0` forces it off even inside a
# session. Unset → fall through to the session signal below.
_PERSIST_ENV = "RLAT_CAPTURE_PERSIST"
# The same env `field.capture.session_id` reads — its presence means a capture
# session is deliberately active (a controlled multi-query run), so folding is
# wanted. Kept as a literal (not imported from capture) so this store-layer
# module carries no field dependency at import time.
_SESSION_ENV = "RLAT_DOGFOOD_SESSION"

_OFF = ("", "0", "false", "False", "no", "off")


def persistence_enabled() -> bool:
    """Whether a fold should write to disk in this process.

    Default-safe on the ZIP format: ON when persistence is explicitly forced
    (`RLAT_CAPTURE_PERSIST`) or a capture session is active
    (`RLAT_DOGFOOD_SESSION`); OFF for a bare one-shot search (no surprise
    whole-archive rewrite on a read). `RLAT_CAPTURE_PERSIST=0` forces OFF even
    inside a session. The SQLite re-home flips the default ON.
    """
    forced = os.environ.get(_PERSIST_ENV)
    if forced is not None:
        return forced not in _OFF
    return bool(os.environ.get(_SESSION_ENV))


def read(km_id: str | None, *, tail: int | None = None) -> list[dict]:
    """The persisted telemetry rows for a corpus — the `insight/telemetry.jsonl`
    member of the `.rlat` at `km_id`, in `field.capture` row shape.

    `tail` keeps only the most recent N rows — readers with superlinear cost
    over the row count (the decide tier's O(N²) intent clustering) pass it so
    a long-lived, heavily-used archive can't push them off a cliff. The parse
    is still O(file); it's the downstream math the bound protects.

    `[]` when there is no corpus path or no telemetry yet. Never raises: a
    corrupt or locked archive yields `[]` so a reader (the decide tier) degrades
    to the live buffer rather than crashing.
    """
    if not km_id:
        return []
    try:
        rows = archive.read_telemetry(km_id)
        return rows[-tail:] if tail is not None and tail > 0 else rows
    except Exception:
        return []


def flush(km_id: str | None) -> int:
    """Fold this process's buffered observations for `km_id` into the `.rlat`.

    Drains `field.capture` for the corpus, redacts each row, and appends to the
    archive's telemetry member. Returns the number of rows folded (0 when
    persistence is disabled, nothing is buffered, or anything failed). Never
    raises.

    When persistence is disabled the buffer is left intact (not drained), so a
    later flush in the same process — once a session enables it — still sees the
    observations.
    """
    if not km_id or not persistence_enabled():
        return 0
    try:
        from ..field import capture  # lazy: keep store import-time field-free

        # Peek, persist, THEN clear EXACTLY the persisted snapshot — so (1) a
        # failed write (read-only corpus, disk full) leaves the observations
        # buffered for the next fold instead of dropping them, and (2) an
        # observation that arrives during the persist (a concurrent client) is
        # not cleared unpersisted: drain(n) pops only the oldest n we just wrote.
        rows = capture.buffered(km_id)
        if not rows:
            return 0
        redacted = [_redact_row(r) for r in rows]
        n = archive.append_telemetry_in_place(km_id, redacted)
        capture.drain(km_id, len(rows))
        return n
    except Exception:
        return 0


def _redact_row(row: dict) -> dict:
    """Scrub free-text leaves of a telemetry row through the redactor
    (invariant §8). Capture rows carry no query text by construction (only the
    embedding fingerprint + scores), so this is defense-in-depth: it scrubs any
    string value (e.g. a user-set `session` id) and leaves the numeric
    fingerprint/scores untouched. Never raises — an un-redactable row degrades
    to itself rather than being dropped (telemetry is best-effort)."""
    try:
        from ..memory.redaction import Redactor

        # No audit log: telemetry lands in the portable `.rlat`, not a per-user
        # memory root, so there is no `redaction.log` home for events here.
        return _scrub_strings(row, Redactor())
    except Exception:
        return row


def _scrub_strings(value, red):
    """Recursively scrub every string leaf; pass numbers/bools through. Numbers
    (the embedding + scores) are never touched, so the secret patterns can't
    false-match the fingerprint."""
    if isinstance(value, str):
        scrubbed, _ = red.scrub(value)
        return scrubbed
    if isinstance(value, dict):
        return {k: _scrub_strings(v, red) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_strings(v, red) for v in value]
    return value
