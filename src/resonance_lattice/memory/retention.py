"""Retention policy — exponential decay + capacity caps.

Each tier has a half-life and a cap:

  working:  half-life 24h,    cap 200 entries
  episodic: half-life 14d,    cap 2,000 entries
  semantic: half-life ∞,      cap 20,000 entries (relevance-pruned only)

`gc(memory, tier)` recomputes effective scores via decay, drops anything below
a small floor, then enforces the cap by keeping the top-N by effective score.
Returns count removed.

Phase 5 deliverable.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass

import numpy as np

# Half-lives in seconds. `inf` for semantic — pure cap-by-relevance, never
# decays. The numbers come from v0.11 `RetentionPolicy.{WORKING,EPISODIC,
# SEMANTIC}` — the v2.0 port preserves them so a v0.11→v2.0 user's recall
# behaviour stays comparable.
_DAY = 86400.0
_PROFILES: dict[str, "RetentionPolicy"] = {}


@dataclass(frozen=True)
class RetentionPolicy:
    half_life_seconds: float
    cap_count: int
    floor_score: float = 1e-3  # drop entries whose effective score falls below


WORKING = RetentionPolicy(half_life_seconds=1.0 * _DAY, cap_count=200)
EPISODIC = RetentionPolicy(half_life_seconds=14.0 * _DAY, cap_count=2_000)
SEMANTIC = RetentionPolicy(half_life_seconds=float("inf"), cap_count=20_000)
_PROFILES.update(working=WORKING, episodic=EPISODIC, semantic=SEMANTIC)


def policy_for(tier: str) -> RetentionPolicy:
    if tier not in _PROFILES:
        raise ValueError(f"unknown tier {tier!r}; valid: {sorted(_PROFILES)}")
    return _PROFILES[tier]


def decay(score: float, age_seconds: float, half_life: float) -> float:
    """Exponential decay: score × 0.5 ** (age / half_life).

    `half_life == inf` short-circuits to no decay (semantic tier).
    """
    if math.isinf(half_life):
        return score
    if age_seconds <= 0:
        return score
    return score * (0.5 ** (age_seconds / half_life))


def _age_seconds(created_utc: str, now: _dt.datetime) -> float:
    """Parse `created_utc` and return age in seconds.

    Empty string → 0 (treat as freshly added; benign for entries that
    legitimately lack a timestamp). Malformed input → +inf, so a corrupted
    or hand-edited entry sorts to the BOTTOM of gc's effective-score
    ranking and gets cap-pruned rather than living forever at age=0.
    """
    if not created_utc:
        return 0.0
    try:
        # Strip trailing Z; fromisoformat handles '+00:00' but not 'Z' on
        # Python <3.11. ISO with 'Z' is what _utcnow_iso writes.
        s = created_utc.rstrip("Z")
        ts = _dt.datetime.fromisoformat(s).replace(tzinfo=_dt.timezone.utc)
    except ValueError:
        return float("inf")
    return max(0.0, (now - ts).total_seconds())


def gc(memory, tier: str, now: _dt.datetime | None = None) -> int:
    """Drop expired + over-cap entries from `tier`. Returns count removed.

    The `memory` argument is a `LayeredMemory` instance; cross-imports are
    avoided by duck-typing (calls `memory.all_entries(tier)` and
    `memory.replace_tier(tier, entries, embeddings)`).
    """
    policy = policy_for(tier)
    now = now or _dt.datetime.now(_dt.timezone.utc)
    entries = memory.all_entries(tier)
    if not entries:
        return 0

    scored: list[tuple[float, object]] = []
    for entry in entries:
        age = _age_seconds(entry.created_utc, now)
        eff = decay(entry.salience, age, policy.half_life_seconds)
        if eff < policy.floor_score:
            continue
        scored.append((eff, entry))

    # Cap by keeping the top-N effective-score entries. Floor + cap together
    # mean a tier never grows unboundedly even if every add lands at
    # salience=1.0.
    scored.sort(key=lambda pair: pair[0], reverse=True)
    kept = scored[:policy.cap_count]
    survivors = [e for _, e in kept]

    n_removed = len(entries) - len(survivors)
    if n_removed == 0:
        return 0

    # Rebuild the embeddings array in survivor order.
    embeddings = (
        np.vstack([e.embedding[None, :] for e in survivors])
        if survivors else
        np.zeros((0, entries[0].embedding.shape[0]), dtype=np.float32)
    )
    memory.replace_tier(tier, list(survivors), embeddings)
    return n_removed
