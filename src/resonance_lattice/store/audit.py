"""Audit + trace — the trust contract's queryable surface.

The architecture's foundation 5 says every output must be auditable back
to source. This module provides the read-only interface that exposes
the full provenance state of a knowledge model:

  audit_summary(km)        -> layer sizes, drift counts, stale count
  audit_stale(km)          -> list of stale corpus claims with reasons
  audit_orphans(km)        -> corpus claims whose source has been removed
  trace_insight(km, id)    -> full citation chain for one corpus claim
  trace_source(km, id)     -> corpus claims that cite a given source

These functions are the substrate for the `rlat audit` / `rlat trace`
CLI commands and for any future inspector UI. Read-only — no mutation,
no hidden side-effects.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..state.claim import FINAL_STATES, Claim
from . import archive


@dataclass(frozen=True)
class AuditSummary:
    """High-level archive audit. One per archive."""
    source_passages: int
    insight_total: int
    insight_active: int
    insight_candidate: int
    insight_stale: int
    insight_retired: int
    source_drift_status_counts: dict[str, int]   # verified / drifted / missing
    insight_orphans: int                          # cited source no longer exists


@dataclass(frozen=True)
class InsightTrace:
    """Full provenance trail for one corpus claim."""
    insight: Claim
    source_passages: list[dict]      # resolved citations (passage_id, source_file, content_hash, drift)
    source_orphans: list[str]        # cited passage_ids that no longer exist
    lineage_chain: list[Claim]       # parent claims walked via `parent_ids`


def audit_summary(contents: archive.ArchiveContents) -> AuditSummary:
    """Compute the archive's audit summary in one pass."""
    state_counts = {"active": 0, "candidate": 0, "stale": 0, "retired": 0}
    source_ids = {c.passage_id for c in contents.registry}
    orphans = 0
    for ins in contents.insights:
        state_counts[ins.state] = state_counts.get(ins.state, 0) + 1
        # Only corpus claims carry CorpusFacts.citations — an experience/attribute claim in the insight list has
        # no citations to orphan, and reading `.facts.citations` on it would AttributeError.
        if ins.source == "corpus" and not all(c.passage_id in source_ids for c in ins.facts.citations):
            orphans += 1

    # `content_hash` is build-time and live drift is a retrieval-time
    # concern surfaced via VerifiedHit.drift_status. The static audit
    # reflects post-refresh state — refresh updates hashes inline, so
    # every row in the registry is verified-against-itself by definition.
    drift_counts = {"verified": len(contents.registry)}

    return AuditSummary(
        source_passages=len(contents.registry),
        insight_total=len(contents.insights),
        insight_active=state_counts.get("active", 0),
        insight_candidate=state_counts.get("candidate", 0),
        insight_stale=state_counts.get("stale", 0),
        insight_retired=state_counts.get("retired", 0),
        source_drift_status_counts=drift_counts,
        insight_orphans=orphans,
    )


def audit_stale(contents: archive.ArchiveContents) -> list[Claim]:
    """All corpus claims currently in the `stale` state. Returns
    input-order list."""
    return [ins for ins in contents.insights if ins.state == "stale"]


def audit_orphans(contents: archive.ArchiveContents) -> list[Claim]:
    """Corpus claims whose citations point to passage_ids that no longer
    exist in the source registry. Caused by source deletion + refresh.
    """
    source_ids = {c.passage_id for c in contents.registry}
    out: list[Claim] = []
    for ins in contents.insights:
        if ins.state in FINAL_STATES or ins.source != "corpus":
            continue
        if any(c.passage_id not in source_ids for c in ins.facts.citations):
            out.append(ins)
    return out


def trace_insight(
    contents: archive.ArchiveContents, claim_id: str,
) -> InsightTrace:
    """Full provenance chain for one corpus claim_id.

    Resolves every citation against the source registry; flags orphans
    (cited passages that no longer exist). Lineage chain follows
    `parent_ids` — claim→claim provenance.

    Raises KeyError if claim_id is not in the archive.
    """
    target = next(
        (i for i in contents.insights if i.claim_id == claim_id), None
    )
    if target is None:
        raise KeyError(f"claim_id {claim_id!r} not in this archive")

    coords_by_id = {c.passage_id: c for c in contents.registry}
    source_passages: list[dict] = []
    orphans: list[str] = []
    for cit in target.facts.citations:
        coord = coords_by_id.get(cit.passage_id)
        if coord is None:
            orphans.append(cit.passage_id)
            continue
        source_passages.append({
            "passage_id": coord.passage_id,
            "source_file": coord.source_file,
            "char_offset": coord.char_offset,
            "char_length": coord.char_length,
            "content_hash": coord.content_hash,
            "citation_confidence": cit.confidence,
            "citation_char_span": cit.char_span,
        })

    # Lineage chain: walk parent claims via `parent_ids`.
    lineage: list[Claim] = []
    seen: set[str] = {target.claim_id}
    parents = list(target.parent_ids)
    while parents:
        pid = parents.pop(0)
        if pid in seen:
            continue
        seen.add(pid)
        parent = next(
            (i for i in contents.insights if i.claim_id == pid), None
        )
        if parent is None:
            continue
        lineage.append(parent)
        parents.extend(parent.parent_ids)

    return InsightTrace(
        insight=target,
        source_passages=source_passages,
        source_orphans=orphans,
        lineage_chain=lineage,
    )


def trace_source(
    contents: archive.ArchiveContents, source_passage_id: str,
) -> list[Claim]:
    """Reverse trace — every corpus claim that cites a given source passage.

    Used by the audit to answer 'what claims depend on this passage?'
    before editing it.
    """
    out: list[Claim] = []
    for ins in contents.insights:
        if ins.source != "corpus":
            continue  # experience/attribute claims have no corpus citations to reverse-trace
        if any(c.passage_id == source_passage_id
               for c in ins.facts.citations):
            out.append(ins)
    return out
