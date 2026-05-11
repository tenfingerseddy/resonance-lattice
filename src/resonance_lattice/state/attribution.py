"""Attribution — pair recall results with outcomes for credit assignment.

Architecture §"Attribution — linking outcomes to memory rows":

> When an outcome is recorded, attribution identifies which memory rows
> contributed to the action that produced it. Without attribution, outcomes
> are floating signal — they happened, but the system can't trace them
> back to specific beliefs to credit or debit.

Three signals available for weighting:
  1. Recall rank + score    — free; computed at recall time
  2. Prompt position         — implicit in rank-ordered injection
  3. Action alignment        — cosine between row text + action content

v1 ships rank-only attribution: alignment refinement requires either a
cheap heuristic (per-tool body cosine probe) or LLM judgement, both
deferred behind the rank-based first cut. The architecture's tier rule
collapses cleanly when alignment is unknown:

  rank ≤ 2  → primary    (architecture: AND alignment high)
  rank ≤ 5  → secondary  (architecture: AND alignment non-trivial)
  else      → incidental

Without alignment evidence the attribution is "best-effort upper bound" —
a row that ranked high but didn't shape the action gets primary credit
it doesn't deserve. Mechanism 1 of confidence raising compensates because
the row also has to appear in *multiple* successful outcomes to clear the
threshold; one misattributed primary doesn't move the needle.
"""

from __future__ import annotations

from .ledger import Attribution
from .recall_cache import RecallEntry

# Tier cutoffs — rank-based v1; engineering-spec tunable.
PRIMARY_RANK_CUTOFF = 2
SECONDARY_RANK_CUTOFF = 5


def _tier_for_rank(rank: int) -> str:
    if rank < PRIMARY_RANK_CUTOFF:
        return "primary"
    if rank < SECONDARY_RANK_CUTOFF:
        return "secondary"
    return "incidental"


def attribution_from_entries(
    entries: list[RecallEntry],
) -> list[Attribution]:
    """Collapse recall entries into one attribution list per row.

    A row that surfaced in multiple recalls keeps its *highest* tier —
    a row that hit primary in one recall and incidental in another stays
    primary because the "shaped at least one action" signal dominates.
    """
    if not entries:
        return []
    best_tier: dict[str, str] = {}
    best_rank: dict[str, int] = {}
    best_cosine: dict[str, float] = {}
    tier_priority = {"primary": 2, "secondary": 1, "incidental": 0}
    for entry in entries:
        for hit in entry.row_metadata:
            tier = _tier_for_rank(hit.rank)
            current = best_tier.get(hit.row_id)
            if current is None or tier_priority[tier] > tier_priority[current]:
                best_tier[hit.row_id] = tier
                best_rank[hit.row_id] = hit.rank
                best_cosine[hit.row_id] = hit.cosine
    return [
        Attribution(
            row_id=row_id,
            tier=best_tier[row_id],
            recall_rank=best_rank[row_id],
            cosine=best_cosine[row_id],
        )
        for row_id in best_tier
    ]
