"""Attribution reducer — resolved-intent outcome log → per-insight Beta weight.

docs/internal/GROUNDING_MODEL.md §"Confidence & attribution": attribution is a
pure function from the resolved-intent outcome log to per-insight corroboration
/ falsification weight, poison-guarded by verdict_confidence × source ×
provenance. The reducer output feeds `insight_lifecycle.accumulate_outcome`,
which adds the weight to an insight's Beta tallies. This module is pure — it
neither reads a ledger nor writes an archive; callers supply the outcome log and
apply the weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..state.claim_outcome import TIER_WEIGHTS

if TYPE_CHECKING:  # element type of a record's criterion_checks
    from ..state.claim_outcome import CriterionCheck


@dataclass(frozen=True)
class InsightWeight:
    """A reducer's verdict for one insight — Beta weight on each side."""

    corroboration: float = 0.0
    falsification: float = 0.0


def _collect_weights(
    corr: dict[str, float], fals: dict[str, float],
) -> dict[str, InsightWeight]:
    """Pair the two tally dicts into the reducer return shape. The reducer
    writes both dicts together, so they always share a key set."""
    return {
        insight_id: InsightWeight(
            corroboration=corr[insight_id],
            falsification=fals[insight_id],
        )
        for insight_id in corr
    }


# ---------------------------------------------------------------------------
# Criterion-level reducer + poison guard (S4 d3 + d4)
# ---------------------------------------------------------------------------
#
# The keystone join folds a resolved intent's per-criterion verdicts
# (state/measure.py) into trust. A criterion verdict carries a
# `verdict_confidence` and a `verdict_source`, and the intent a provenance —
# the three the **poison guard** scales by, so a weak or machine-asserted
# verdict cannot move a corpus claim's trust at full weight (architecture §5,
# invariant 6). This is the guard against S5's auto-harvested criteria eroding
# good claims; the scaling happens *here*, in the reducer, before the weight
# ever reaches `accumulate_outcome` — the trust seam stays untouched.

# verdict_confidence → trust-weight multiplier. A low-confidence (typically
# llm-only) verdict moves trust a fifth as far as a high-confidence one — so a
# lone low-confidence `not_satisfied` cannot retire a well-corroborated claim.
_CONFIDENCE_WEIGHT: dict[str, float] = {"high": 1.0, "medium": 0.5, "low": 0.2}

# verdict_source → multiplier. A `user_override` is a user vouch for a
# criterion whose declared check never ran (state/measure.py) — real, but not
# an independent measurement, so it carries less than a measured `signal`.
_SOURCE_WEIGHT: dict[str, float] = {"signal": 1.0, "user_override": 0.7}

# Intent provenance → multiplier. A user-declared intent's criteria are
# trusted; auto-harvested criteria (S5) are quarantined low until corroborated.
_PROVENANCE_WEIGHT: dict[str, float] = {"user": 1.0, "harvested": 0.3}

# Confidence ordering for picking a record's decisive verdict_confidence.
_CONFIDENCE_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


def poison_guard_scale(
    verdict_confidence: str, verdict_source: str, provenance: str,
) -> float:
    """The poison guard (S4 d4): the multiplier on a criterion outcome's
    trust weight, from `verdict_confidence × verdict_source × provenance`.
    Unknown keys fall to the most conservative (lowest) factor, so an
    unrecognised value attenuates rather than amplifies."""
    return (
        _CONFIDENCE_WEIGHT.get(verdict_confidence, 0.2)
        * _SOURCE_WEIGHT.get(verdict_source, 0.7)
        * _PROVENANCE_WEIGHT.get(provenance, 0.3)
    )


@dataclass(frozen=True)
class CriterionOutcome:
    """One resolved intent's criterion-level outcome — the intent-record
    projection the criterion reducer consumes.

    `attributed` is the intent's `(claim_id, tier)` attribution; `roll_up` is
    the AND across its criterion checks; `verdict_confidence` / `verdict_source`
    are the record's *decisive* values (`decisive_verdict`), and `provenance`
    is how the intent was captured (`user` until S5 harvests).
    """

    attributed: tuple[tuple[str, str], ...]
    roll_up: str
    verdict_confidence: str
    verdict_source: str
    provenance: str = "user"


def decisive_verdict(
    checks: "list[CriterionCheck]", roll_up: str,
) -> tuple[str, str]:
    """Reduce a record's criterion checks to the `(verdict_confidence,
    verdict_source)` that should scale its trust weight.

    For a `satisfied` roll-up, the satisfaction is only as strong as its
    **weakest** criterion, so the minimum confidence governs (and any
    `user_override` among them marks the whole vouch unmeasured — the source
    discount applies). For a `not_satisfied` roll-up, the **strongest**
    falsifying check governs its confidence — a high-confidence *measured*
    failure is the deliberately-**un-attenuated** path (it should falsify
    fully; do not "harden" it), while a low-confidence (e.g. llm-only) failure
    is attenuated by the guard. The source of a failure is always `"signal"`:
    `user_override` is an accept-side concept (a vouch *to satisfied* with no
    measured check), so it must never discount a falsification — a user reject
    is authoritative. Empty checks → `("low", "signal")`, the most
    conservative pairing."""
    if not checks:
        return "low", "signal"
    if roll_up == "not_satisfied":
        fails = [c for c in checks if c.verdict == "not_satisfied"] or checks
        strongest = max(fails, key=lambda c: _CONFIDENCE_ORDER.get(
            c.verdict_confidence, 0))
        return strongest.verdict_confidence, "signal"
    weakest = min(checks, key=lambda c: _CONFIDENCE_ORDER.get(
        c.verdict_confidence, 0))
    source = (
        "user_override"
        if any(c.verdict_source == "user_override" for c in checks)
        else "signal"
    )
    return weakest.verdict_confidence, source


def criterion_weighted(
    outcomes: list[CriterionOutcome],
) -> dict[str, InsightWeight]:
    """Criterion-level reducer (S4 d3): fold resolved-intent criterion
    outcomes into per-claim Beta weight, **poison-guarded**.

    Each outcome's roll-up signs the weight — `satisfied` corroborates,
    `not_satisfied` falsifies, `unknown` is skipped — and the
    `poison_guard_scale` for its decisive verdict attenuates it. The signed,
    scaled weight is spread across the attributed claims by tier (load-bearing
    `primary` claims earn the most), so the trust seam stays untouched."""
    corr: dict[str, float] = {}
    fals: dict[str, float] = {}
    for o in outcomes:
        if o.roll_up not in ("satisfied", "not_satisfied"):
            continue  # unknown / pending carry no signal
        scale = poison_guard_scale(
            o.verdict_confidence, o.verdict_source, o.provenance)
        corr_share = scale if o.roll_up == "satisfied" else 0.0
        fals_share = scale if o.roll_up == "not_satisfied" else 0.0
        for claim_id, tier in o.attributed:
            w = TIER_WEIGHTS.get(tier, 0.0)
            corr[claim_id] = corr.get(claim_id, 0.0) + w * corr_share
            fals[claim_id] = fals.get(claim_id, 0.0) + w * fals_share
    return _collect_weights(corr, fals)
