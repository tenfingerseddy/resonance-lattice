"""Memory store primitives — the shared enums, rung seeds, and per-user path.

The categorical enums of an experience claim, the Beta-tally rung seeds,
and the per-user directory resolver. The store itself is
`memory.claim_store.ExperienceClaimStore` over `state.claim.Claim`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

Criticality = Literal["low", "normal", "high", "severe"]
CRITICALITY_VALUES: frozenset[str] = frozenset(
    {"low", "normal", "high", "severe"}
)

Confidence = Literal["low", "medium", "high", "verified"]
CONFIDENCE_VALUES: frozenset[str] = frozenset(
    {"low", "medium", "high", "verified"}
)

Origin = Literal[
    "manual", "distilled", "migrated", "outcome_derived", "intent_resolution",
]

# The agent's intent-kind context at capture time, used by the manifesto
# recall re-rank's `level_match` factor. `none` is the default for claims
# captured outside an active intent.
IntentKind = Literal[
    "debug", "design", "implement", "review", "explain", "refactor", "none",
]
INTENT_KIND_VALUES: frozenset[str] = frozenset(
    {"debug", "design", "implement", "review", "explain", "refactor", "none"}
)

# Beta-tally seeds per confidence rung — the inverse of `store.insight`'s
# `confidence_band` mapping. A claim seeded this way reads back at the same
# rung; the confidence pass re-derives tallies from the ledger thereafter.
# `confidence.py` asserts at import that each seed bands back to its own
# rung, so this table can't drift out of step with the band cuts.
_RUNG_TALLY_SEED: dict[str, tuple[float, float]] = {
    "low": (1.0, 3.0),       # beta mean 0.25
    "medium": (2.0, 2.0),    # beta mean 0.50
    "high": (3.0, 1.0),      # beta mean 0.75
    "verified": (4.0, 1.0),  # beta mean 0.80
}


def seed_tallies_for_rung(confidence: str) -> tuple[float, float]:
    """`(corroboration, falsification)` Beta tallies for a confidence rung.

    The returned tallies' Beta mean falls inside `confidence`'s band, so a
    claim seeded this way reads back at the same rung. An unknown rung
    falls back to `medium`.
    """
    return _RUNG_TALLY_SEED.get(confidence, _RUNG_TALLY_SEED["medium"])


def path_for_user(user_id: str | None = None, root: Path | None = None) -> Path:
    """Resolve `~/.rlat/memory/<user-id>/`. Falls back through
    `RLAT_MEMORY_USER` → `USER` → `USERNAME`.
    """
    if user_id is None:
        user_id = (
            os.environ.get("RLAT_MEMORY_USER")
            or os.environ.get("USER")
            or os.environ.get("USERNAME")
        )
    if not user_id:
        raise RuntimeError(
            "could not derive user_id from RLAT_MEMORY_USER / USER / USERNAME — "
            "pass --user explicitly"
        )
    base = Path(root) if root is not None else Path.home() / ".rlat" / "memory"
    return base / user_id
