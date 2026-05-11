"""Manifesto recall re-rank — strength × valence × level × confidence_floor.

Architecture §"Recall — Scoring formula":

    score(row) = similarity(query, row.text)
               × strength(row)
               × valence_match(query_intent, row.valence)
               × level_match(query_intent, row.level)
               × confidence_floor(row.confidence, row.level)

The four-gate pipeline in `recall.rank` decides *which* rows are eligible to
surface; this module decides *how to order* the survivors.

> Gates stay binary. Factors do the ranking work. (architecture §"Layer
> manifesto scoring factors as a re-rank over the existing gates")

Coefficients below are the architecture's current best estimates and are
treated as engineering-spec parameters — change them without rewriting the
manifesto.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass

from ._common import parse_iso_utc
from .recall import RecallHit
from .store import (
    Confidence,
    Criticality,
    INTENT_LEVELS,
    IntentKind,
    Level,
    Row,
)

# ---------------------------------------------------------------------------
# Coefficient tables
# ---------------------------------------------------------------------------

_CRITICALITY_COEFF: dict[Criticality, float] = {
    "low": 0.7,
    "normal": 1.0,
    "high": 1.5,
    "severe": 2.0,
}

# Half-life for the recency factor, in days. Severe memories never decay
# below a high baseline; normal/low half-lives bleed faster so noise
# doesn't accumulate.
_HALF_LIFE_DAYS: dict[Criticality, float] = {
    "low": 14.0,
    "normal": 60.0,
    "high": 180.0,
    "severe": 730.0,
}

# Lower bound on `recency_factor` so a `severe avoid` row never decays into
# irrelevance — once-burned-twice-shy as a system property (architecture
# §"Field interactions worth knowing").
_SEVERE_FLOOR = 0.6

# Architecture §"New-principle protection window": newly promoted
# principles get a bounded grace period (5 sessions or 30 days, whichever
# first) at confidence_floor=medium regardless of actual confidence.
# Both arms ship as gates; either expiring drops the row to its actual
# confidence. The session arm is opt-in via `sessions_since_created` —
# callers without workspace session-ledger access pass None and only the
# 30-day clock fires. Threading the count from the recall hook through
# the daemon is a follow-up.
_NEW_PRINCIPLE_PROTECTION_DAYS = 30
_NEW_PRINCIPLE_PROTECTION_SESSIONS = 5
_PROTECTION_FLOOR_LEVEL = "medium"

# Confidence ranks for the protection floor's "treat as at least medium"
# clamp — values come from the architecture's confidence-floor matrix.
_CONFIDENCE_RANK: dict[str, int] = {
    "low": 0, "medium": 1, "high": 2, "verified": 3,
}

# IntentKind → valence-axis profile. The architecture's table mixes "about
# to act / gathering knowledge / debugging" — these are profiles agents
# enter under different intent kinds.
_VALENCE_PROFILE: dict[IntentKind, str] = {
    "debug": "debugging",
    "design": "gathering_knowledge",
    "implement": "about_to_act",
    "review": "gathering_knowledge",
    "explain": "gathering_knowledge",
    "refactor": "about_to_act",
    "none": "neutral",
}

_VALENCE_MATCH: dict[str, dict[str, float]] = {
    "about_to_act":        {"prefer": 1.0, "avoid": 1.2, "factual": 0.8},
    "gathering_knowledge": {"prefer": 0.7, "avoid": 0.7, "factual": 1.3},
    "debugging":           {"prefer": 0.9, "avoid": 1.3, "factual": 1.0},
    "neutral":             {"prefer": 1.0, "avoid": 1.0, "factual": 1.0},
}

# IntentKind → level-axis profile.
_LEVEL_PROFILE: dict[IntentKind, str] = {
    "debug": "debugging",
    "design": "designing",
    "implement": "implementing",
    "review": "explaining",
    "explain": "explaining",
    "refactor": "implementing",
    "none": "neutral",
}

_LEVEL_MATCH_MEMORY: dict[str, dict[Level, float]] = {
    "debugging":    {"event": 1.2, "pattern": 1.1, "learning": 1.0, "principle": 0.8},
    "designing":    {"event": 0.7, "pattern": 1.0, "learning": 1.1, "principle": 1.3},
    "implementing": {"event": 0.9, "pattern": 1.1, "learning": 1.2, "principle": 1.0},
    "explaining":   {"event": 0.8, "pattern": 1.0, "learning": 1.1, "principle": 1.2},
    "neutral":      {"event": 1.0, "pattern": 1.0, "learning": 1.0, "principle": 1.0},
}

# Level-aware confidence floor. Higher levels punished less so freshly-
# promoted principles get a fair shot at re-earning evidence.
_CONFIDENCE_FLOOR: dict[Confidence, dict[Level, float]] = {
    "verified": {"event": 1.0,  "pattern": 1.0,  "learning": 1.0,  "principle": 1.0},
    "high":     {"event": 0.95, "pattern": 0.95, "learning": 0.95, "principle": 0.95},
    "medium":   {"event": 0.85, "pattern": 0.9,  "learning": 0.9,  "principle": 0.95},
    "low":      {"event": 0.6,  "pattern": 0.7,  "learning": 0.8,  "principle": 0.9},
}


@dataclass(frozen=True)
class ScoreBreakdown:
    """Inspectable score components for one row.

    Returned from `score_row` so callers (debug surfaces, attribution
    rationale, the inspectable user surface) can show *why* a row ranked
    where it did. Not used in the hot path — that calls `effective_score`
    directly.
    """

    cosine: float
    strength: float
    valence_match: float
    level_match: float
    confidence_floor: float
    effective: float


# ---------------------------------------------------------------------------
# Component factors
# ---------------------------------------------------------------------------


# `_parse_iso` is the shared `_common.parse_iso_utc` — kept as a private
# alias here so `_within_new_principle_protection` can read its old name
# without churn for callers that imported it.
_parse_iso = parse_iso_utc


def recency_factor(
    last_corroborated_at: str,
    criticality: Criticality,
    *,
    now: _dt.datetime | None = None,
) -> float:
    """Half-life decay against `last_corroborated_at`.

    Architecture: *recency takes criticality as input — higher criticality
    = longer half-life; severe floor never decays below a high baseline*.
    """
    if now is None:
        now = _dt.datetime.now(_dt.timezone.utc)
    age_days = max(
        0.0,
        (now - _parse_iso(last_corroborated_at)).total_seconds() / 86400.0,
    )
    half_life = _HALF_LIFE_DAYS.get(criticality, _HALF_LIFE_DAYS["normal"])
    factor = math.pow(2.0, -age_days / half_life)
    if criticality == "severe":
        factor = max(factor, _SEVERE_FLOOR)
    return factor


def strength(row: Row, *, now: _dt.datetime | None = None) -> float:
    """Composite strength factor.

    `strength = recency × log(1 + recurrence_count) × criticality_coeff`
    (architecture §"Factor 2 — strength"). Recurrence is logarithmic so
    a row corroborated 100 times doesn't drown out a row corroborated 10
    times — diminishing returns.
    """
    rec = recency_factor(row.last_corroborated_at, row.criticality, now=now)
    recur = math.log1p(max(0, row.recurrence_count))
    crit = _CRITICALITY_COEFF.get(row.criticality, 1.0)
    return rec * recur * crit


def valence_match(intent_kind: IntentKind, row: Row) -> float:
    """Valence multiplier for `intent_kind` × `row.primary_polarity()`."""
    profile = _VALENCE_PROFILE.get(intent_kind, "neutral")
    table = _VALENCE_MATCH.get(profile, _VALENCE_MATCH["neutral"])
    return table.get(row.primary_polarity(), 1.0)


def level_match(intent_kind: IntentKind, row: Row) -> float:
    """Level multiplier for `intent_kind` × `row.level`.

    Intent-row recall is rare in practice (recall is for memory) — when
    it happens, the multiplier is neutral so the formula collapses to
    cosine × strength × valence × confidence_floor on intent rows.
    """
    if row.level in INTENT_LEVELS:
        return 1.0
    profile = _LEVEL_PROFILE.get(intent_kind, "neutral")
    table = _LEVEL_MATCH_MEMORY.get(profile, _LEVEL_MATCH_MEMORY["neutral"])
    return table.get(row.level, 1.0)  # type: ignore[arg-type]


def _within_new_principle_protection(
    row: Row,
    *,
    now: _dt.datetime | None = None,
    sessions_since_created: int | None = None,
) -> bool:
    """True if `row` is a freshly-promoted principle inside its
    architecture-mandated grace period: 5 sessions OR 30 days, whichever
    expires first. Either arm expiring drops the row to its actual
    confidence.

    The 30-day clock reads `created_at`. The 5-session clock reads the
    workspace session ledger via the caller — when `sessions_since_created`
    is None (the v1 default while daemon threading is pending), only the
    30-day arm fires.
    """
    if row.level != "principle":
        return False
    if now is None:
        now = _dt.datetime.now(_dt.timezone.utc)
    age_days = max(0.0, (now - _parse_iso(row.created_at)).total_seconds()
                   / 86400.0)
    if age_days > _NEW_PRINCIPLE_PROTECTION_DAYS:
        return False
    if (sessions_since_created is not None
            and sessions_since_created > _NEW_PRINCIPLE_PROTECTION_SESSIONS):
        return False
    return True


def confidence_floor(
    row: Row,
    *,
    now: _dt.datetime | None = None,
    sessions_since_created: int | None = None,
) -> float:
    """Level-aware confidence multiplier.

    Intent rows fall back to the `event` row of the table — they have no
    distil ladder yet, but the same "low confidence is worse than
    verified" relationship holds.

    New-principle protection (architecture §"New-principle protection
    window"): a principle within 5 sessions OR 30 days of its
    `created_at` (whichever first) is multiplied as if its confidence
    were at least `medium` regardless of actual value. Without this,
    freshly-promoted principles — diluted to `low` by Arrow 3's
    confidence rule — would be permanently buried before they could
    earn cross-domain corroboration. `sessions_since_created` opts into
    the session arm; None falls back to the 30-day clock alone.
    """
    confidence = row.confidence
    if row.level in INTENT_LEVELS:
        return _CONFIDENCE_FLOOR[confidence]["event"]
    if _within_new_principle_protection(
        row, now=now, sessions_since_created=sessions_since_created,
    ):
        if (_CONFIDENCE_RANK.get(confidence, 0)
                < _CONFIDENCE_RANK[_PROTECTION_FLOOR_LEVEL]):
            confidence = _PROTECTION_FLOOR_LEVEL  # type: ignore[assignment]
    return _CONFIDENCE_FLOOR[confidence][row.level]  # type: ignore[index]


def score_row(
    row: Row,
    cosine: float,
    *,
    intent_kind: IntentKind = "none",
    now: _dt.datetime | None = None,
) -> ScoreBreakdown:
    """Full per-row score with breakdown attached.

    Useful for debug surfaces and attribution rationale; the hot recall
    path calls `effective_score` directly which short-circuits the
    breakdown allocation.
    """
    s = strength(row, now=now)
    v = valence_match(intent_kind, row)
    lvl = level_match(intent_kind, row)
    conf = confidence_floor(row, now=now)
    effective = cosine * s * v * lvl * conf
    return ScoreBreakdown(
        cosine=cosine,
        strength=s,
        valence_match=v,
        level_match=lvl,
        confidence_floor=conf,
        effective=effective,
    )


def effective_score(
    row: Row,
    cosine: float,
    *,
    intent_kind: IntentKind = "none",
    now: _dt.datetime | None = None,
) -> float:
    """Hot-path score — no allocation, just the multiply chain."""
    return (
        cosine
        * strength(row, now=now)
        * valence_match(intent_kind, row)
        * level_match(intent_kind, row)
        * confidence_floor(row, now=now)
    )


def rerank(
    hits: list[RecallHit],
    *,
    intent_kind: IntentKind = "none",
    now: _dt.datetime | None = None,
) -> list[RecallHit]:
    """Re-order post-gate hits by the manifesto score.

    Returns the same RecallHit objects in a new order (lowest score
    drops in cosine-tied ties keep stability via the original index).
    The cosine attached to each hit is preserved — re-rank only adjusts
    *order*, not the field surfaced to callers (debug / display).
    """
    if not hits:
        return hits
    scored = [
        (effective_score(hit.row, hit.cosine, intent_kind=intent_kind, now=now), idx, hit)
        for idx, hit in enumerate(hits)
    ]
    scored.sort(key=lambda triple: (-triple[0], triple[1]))
    return [hit for _, _, hit in scored]
