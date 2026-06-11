"""Manifesto recall re-rank — strength × valence × confidence_floor.

Architecture §"Recall — Scoring formula":

    score(claim) = similarity(query, claim.content)
                 × strength(claim)
                 × valence_match(query_intent, claim.valence)
                 × confidence_floor(claim.confidence)

The four-gate pipeline in `recall.rank` decides *which* claims are eligible
to surface; this module decides *how to order* the survivors.

> Gates stay binary. Factors do the ranking work. (architecture §"Layer
> manifesto scoring factors as a re-rank over the existing gates")

Coefficients below are the architecture's current best estimates and are
treated as engineering-spec parameters — change them without rewriting the
manifesto. Experience-only — corpus ranking has its own scorer in
`store.verified`.
"""

from __future__ import annotations

import datetime as _dt
import math

from ..state.claim import Claim
from ._common import parse_iso_utc
from .recall import RecallHit
from .store import Confidence, Criticality, IntentKind

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

# Lower bound on `recency_factor` so a `severe avoid` claim never decays
# into irrelevance — once-burned-twice-shy as a system property
# (architecture §"Field interactions worth knowing").
_SEVERE_FLOOR = 0.6

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

# Confidence floor — every survivor is an `event`, so the per-kind axis
# collapses to a single multiplier per confidence rung.
_CONFIDENCE_FLOOR: dict[Confidence, float] = {
    "verified": 1.0,
    "high":     0.95,
    "medium":   0.85,
    "low":      0.6,
}


# ---------------------------------------------------------------------------
# Component factors
# ---------------------------------------------------------------------------


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
        (now - parse_iso_utc(last_corroborated_at)).total_seconds() / 86400.0,
    )
    half_life = _HALF_LIFE_DAYS.get(criticality, _HALF_LIFE_DAYS["normal"])
    factor = math.pow(2.0, -age_days / half_life)
    if criticality == "severe":
        factor = max(factor, _SEVERE_FLOOR)
    return factor


def strength(claim: Claim, *, now: _dt.datetime | None = None) -> float:
    """Composite strength factor.

    `strength = recency × log(1 + recurrence_count) × criticality_coeff`
    (architecture §"Factor 2 — strength"). Recurrence is logarithmic so
    a claim corroborated 100 times doesn't drown out one corroborated 10
    times — diminishing returns.
    """
    facts = claim.facts
    rec = recency_factor(facts.last_corroborated_at, facts.criticality, now=now)
    recur = math.log1p(max(0, facts.recurrence_count))
    crit = _CRITICALITY_COEFF.get(facts.criticality, 1.0)
    return rec * recur * crit


def valence_match(intent_kind: IntentKind, claim: Claim) -> float:
    """Valence multiplier for `intent_kind` × the claim's primary polarity."""
    profile = _VALENCE_PROFILE.get(intent_kind, "neutral")
    table = _VALENCE_MATCH.get(profile, _VALENCE_MATCH["neutral"])
    return table.get(claim.facts.primary_polarity(), 1.0)


def confidence_floor(claim: Claim) -> float:
    """Confidence multiplier for an experience claim."""
    return _CONFIDENCE_FLOOR[claim.confidence]  # type: ignore[index]


def effective_score(
    claim: Claim,
    cosine: float,
    *,
    intent_kind: IntentKind = "none",
    now: _dt.datetime | None = None,
) -> float:
    """Hot-path score — no allocation, just the multiply chain.

    `cosine × strength × valence_match × confidence_floor`.
    """
    return (
        cosine
        * strength(claim, now=now)
        * valence_match(intent_kind, claim)
        * confidence_floor(claim)
    )


def rerank(
    hits: list[RecallHit],
    *,
    intent_kind: IntentKind = "none",
    now: _dt.datetime | None = None,
) -> list[RecallHit]:
    """Re-order post-gate hits by `effective_score`.

    Returns the same RecallHit objects in a new order (cosine-tied ties
    keep stability via the original index). The cosine attached to each
    hit is preserved — re-rank only adjusts *order*, not the field
    surfaced to callers (debug / display).
    """
    if not hits:
        return hits
    scored = [
        (
            effective_score(hit.claim, hit.cosine,
                            intent_kind=intent_kind, now=now),
            idx, hit,
        )
        for idx, hit in enumerate(hits)
    ]
    scored.sort(key=lambda triple: (-triple[0], triple[1]))
    return [hit for _, _, hit in scored]
