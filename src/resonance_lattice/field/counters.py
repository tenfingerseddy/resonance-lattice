"""counters — the Tier-0 closed-form count layer (no model).

STATUS (2026-06 review, roadmap 4.7): chartered build-ahead for the
Insight Engine arm-(b) gate — no production caller yet by design (only
its harness suite exercises it). Not dead code; do not re-flag in audits.

Reads the capture stream (`field.capture` observations) and computes, per
retrieved unit, an Ebbinghaus-decayed log **access-reinforcement** — the
"count with no model" tier (`boundary.md`: trust decay + access counts +
confidence math, closed-form, never a model). This is arm (b)'s counting
substrate in the H1 §D gate (`horizon-1-capture.md`): the bar the curator head
must beat. Two count-tier pieces the spine lacks today land here —

- **access counts** — there is no `access` field on `state/claim.py`; a per-hit
  access count is new work (`capture.md` §3).
- **trust decay** — `claim_lifecycle.py` has no decay function (`boundary.md`:
  "trust decay is the count tier by design but not yet wired"); the Ebbinghaus
  recency weight below is that decay, applied to access not yet to the Beta tally.

Closed-form, pure, leaf:

- only **user-intent** observations count (`is_user_query`); machinery is ignored
  (the raise-your-hand flag, `capture.md` §3).
- a hit's weight decays with age (Ebbinghaus): `exp(-Δt / τ)`. "Now" defaults to
  the most-recent observation, so the pass is deterministic with no wall clock —
  the §D replay reproduces exactly.
- repeated hits reinforce with **diminishing returns** (log): `log1p(Σ decayed)` —
  a claim hit ten times is worth more than one hit, but not ten times more.

No model, no I/O, no persistence: the in-memory stream in, a dict out. The
buffer's fold into the `.rlat` and cross-session durability are deferred to the
format re-home (`capture.md` §3); the H1 §D proof reads the live stream
in-process. Never raises on a malformed row — a messy real stream still counts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

# Recency half-life knob. At Δt = τ a hit is worth 1/e of a fresh one; older
# hits fade smoothly rather than dropping off a cliff. 30 days is a memory-scale
# default; the §D harness may pin it for reproducibility.
_DEFAULT_TAU_DAYS = 30.0


@dataclass(frozen=True)
class ReinforceStat:
    """The closed-form count for one retrieved unit over the stream.

    `access_count` is the raw user-intent hit tally; `decayed_access` is the
    Ebbinghaus-weighted sum (recent hits count more); `reinforcement` is its
    log-damped value — the number arm (b) reads. `last_ts` is the most-recent
    hit (ISO, or None when the stream carried no parseable timestamp)."""

    access_count: int
    decayed_access: float
    reinforcement: float
    last_ts: str | None


def _parse_ts(value) -> datetime | None:
    """Parse an ISO timestamp to an aware UTC datetime; None if unparseable.

    Capture stamps `timespec="seconds"` aware ISO (`capture.observe`), but a
    real stream may carry naïve or malformed values — normalise to aware UTC so
    the age subtraction never raises on a tz mismatch."""
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def access_reinforcement(
    observations: Iterable[dict],
    *,
    now: str | datetime | None = None,
    tau_days: float = _DEFAULT_TAU_DAYS,
    layer: str | None = None,
) -> dict[tuple[str, int], ReinforceStat]:
    """Closed-form access-reinforcement over a capture stream.

    `observations` are `capture` rows (the dicts `capture.buffered`/`drain`
    return). Returns a map keyed by `(layer, idx)` — the retrieved unit — to its
    `ReinforceStat`. Pass `layer` to restrict to one band ("source" or
    "insight"); otherwise both are counted under distinct keys.

    `now` anchors the Ebbinghaus decay; when None it defaults to the latest
    observation timestamp, making the pass clock-free and reproducible. A row
    without a parseable timestamp contributes with **no decay** (weight 1.0) —
    its recency is simply unknown, not a reason to drop the hit.

    Never raises on bad input: a malformed row (bad `ranked`, non-int idx,
    missing fields) is skipped and a hostile `now`/`tau_days` (wrong type) falls
    back to its default — mirroring the hot path's never-break contract.
    """
    rows = list(observations or [])
    # A non-numeric / non-positive τ (a caller mistake) degrades to the default
    # rather than crashing — these run outside the per-row guard below.
    tau = (tau_days if isinstance(tau_days, (int, float)) and tau_days > 0
           else _DEFAULT_TAU_DAYS)

    # Resolve "now" — an explicit value wins, else the latest parseable hit ts.
    # A non-str/non-datetime "now" (e.g. an epoch int) degrades to the default
    # path rather than raising on a `.tzinfo` access.
    if isinstance(now, str):
        now_dt = _parse_ts(now)
    elif isinstance(now, datetime):
        now_dt = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    else:
        now_dt = None
    if now_dt is None:
        seen = [t for t in (_parse_ts(r.get("ts")) for r in rows if isinstance(r, dict)) if t]
        now_dt = max(seen) if seen else None

    counts: dict[tuple[str, int], int] = {}
    decayed: dict[tuple[str, int], float] = {}
    last: dict[tuple[str, int], tuple[datetime, str]] = {}

    for row in rows:
        try:
            if not isinstance(row, dict):
                continue
            if not row.get("is_user_query", True):
                continue  # machinery raised its hand — ignore (capture.md §3)
            row_layer = row.get("layer")
            if layer is not None and row_layer != layer:
                continue
            ts_raw = row.get("ts")
            ts_dt = _parse_ts(ts_raw)
            if now_dt is not None and ts_dt is not None:
                age_days = (now_dt - ts_dt).total_seconds() / 86400.0
                weight = math.exp(-max(age_days, 0.0) / tau)
            else:
                weight = 1.0  # recency unknown → undecayed, not dropped
            for hit in row.get("ranked") or []:
                idx = int(hit["idx"])
                key = (str(row_layer), idx)
                counts[key] = counts.get(key, 0) + 1
                decayed[key] = decayed.get(key, 0.0) + weight
                if ts_dt is not None and (
                    key not in last or ts_dt > last[key][0]  # compare UTC times
                ):
                    last[key] = (ts_dt, ts_raw)
        except (KeyError, TypeError, ValueError):
            continue  # one bad row never sinks the pass

    return {
        key: ReinforceStat(
            access_count=counts[key],
            decayed_access=decayed[key],
            reinforcement=math.log1p(decayed[key]),
            last_ts=last[key][1] if key in last else None,
        )
        for key in counts
    }
