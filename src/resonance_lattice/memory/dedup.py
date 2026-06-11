"""Retroactive memory dedup — collapse same-text-same-workspace event claims.

Capture-time dedup (`capture._find_dup_in`) prevents new duplicates from
accumulating, but it only fires forward.
Memory written before that fix carries N copies of every recurring
captured event — same text, same workspace tag, distinct transcript
hashes, recurrence_count=1 each. Arrow1 finds those clusters and the LLM
correctly refuses to promote ("8 identical events — noise"); the
expertise primer surfaces 4 copies of the same line; forget can only
decay them via the slow age path.

This module ships the one-shot retroactive collapse: `dedup_event_claims`
groups event-level claims by `(content, workspace_tag)`, keeps the oldest
of each group, sets its `recurrence_count` to the cluster size + 1 (so the
collapsed signal isn't lost), and deletes the rest. Idempotent — running
it on already-deduped memory is a no-op.

Same `(content, workspace_tag)` invariant as the capture-time path so the
two operations agree on what counts as a duplicate.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

from ..state.claim import Claim, evolve
from ._common import utcnow_iso
from .claim_store import ExperienceClaimStore


@dataclass(frozen=True)
class DedupResult:
    """One dedup pass outcome."""

    claims_collapsed: int  # number of claims deleted
    groups_collapsed: int  # number of (content, workspace) clusters that had >1 claim
    claims_examined: int  # total event-level claims scanned


def _workspace_tag(claim: Claim) -> str | None:
    for tag in claim.facts.polarity:
        if tag.startswith("workspace:"):
            return tag
    return None


def _group_dups(claims: list[Claim]) -> list[list[Claim]]:
    """Return groups of >1 event claims sharing `(content, workspace_tag)`.

    Non-event claims are excluded — patterns/learnings/principles are
    distilled output, not raw captures, and re-running distil produces
    fresh promoted claims that should stay distinct.
    """
    buckets: dict[tuple[str, str | None], list[Claim]] = collections.defaultdict(list)
    for c in claims:
        if c.kind != "event":
            continue
        buckets[(c.content, _workspace_tag(c))].append(c)
    return [g for g in buckets.values() if len(g) > 1]


def dedup_event_claims(
    memory: ExperienceClaimStore, *, dry_run: bool = False,
) -> DedupResult:
    """Collapse `(content, workspace_tag)`-equal event claims into the oldest.

    The keeper's `recurrence_count` is set to the sum of the group's
    counts (architecture: recurrence is the cumulative signal, not the
    write-count proxy it ends up as without dedup). `last_corroborated_at`
    bumps to now so retention treats the collapsed claim as fresh.

    Returns counts; on `dry_run=True` the disk is untouched.
    """
    claims = memory.read_all()
    claims_examined = sum(1 for c in claims if c.kind == "event")
    groups = _group_dups(claims)
    if not groups:
        return DedupResult(
            claims_collapsed=0,
            groups_collapsed=0,
            claims_examined=claims_examined,
        )

    claims_collapsed = 0
    now = utcnow_iso()
    keepers: list[Claim] = []
    deletions: list[str] = []
    for group in groups:
        # Keep the oldest by created_at; delete the rest. Sum recurrence
        # so the cumulative signal survives the collapse — a claim that
        # was captured 8 times has recurrence_count=8 after, not 1.
        ordered = sorted(group, key=lambda c: c.created_at)
        keeper = ordered[0]
        group_deletions = [c.claim_id for c in ordered[1:]]
        bumped_recurrence = sum(c.facts.recurrence_count for c in ordered)
        claims_collapsed += len(group_deletions)
        keepers.append(evolve(
            keeper,
            recurrence_count=bumped_recurrence,
            last_corroborated_at=now,
        ))
        deletions.extend(group_deletions)
    # One batched write + one batched delete — O(groups), not O(groups²).
    if not dry_run:
        memory.write_many(keepers)
        memory.delete(deletions)

    return DedupResult(
        claims_collapsed=claims_collapsed,
        groups_collapsed=len(groups),
        claims_examined=claims_examined,
    )
