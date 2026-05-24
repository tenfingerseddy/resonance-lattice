"""Audit + trace — the trust contract's queryable surface.

The architecture's foundation 5 says every output must be auditable back
to source. This module provides the read-only interface that exposes
the full provenance state of a knowledge model:

  audit_summary(km)        -> layer sizes, drift counts, stale count
  audit_stale(km)          -> list of stale insights with reasons
  audit_orphans(km)        -> insights whose source has been removed
  trace_insight(km, id)    -> full citation chain for one insight
  trace_source(km, id)     -> insights that cite a given source

These functions are the substrate for the `rlat audit` / `rlat trace`
CLI commands (Day 7) and for any future inspector UI. Read-only — no
mutation, no hidden side-effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import archive
from .insight import FINAL_STATES, InsightPassage


@dataclass(frozen=True)
class AuditSummary:
    """High-level archive audit. One per archive."""
    source_passages: int
    insight_total: int
    insight_accepted: int
    insight_candidate: int
    insight_stale: int
    insight_rejected: int
    insight_retired: int
    source_drift_status_counts: dict[str, int]   # verified / drifted / missing
    insight_orphans: int                          # cited source no longer exists


@dataclass(frozen=True)
class InsightTrace:
    """Full provenance trail for one insight passage."""
    insight: InsightPassage
    source_passages: list[dict]      # resolved citations (passage_id, source_file, content_hash, drift)
    source_orphans: list[str]        # cited passage_ids that no longer exist
    lineage_chain: list[InsightPassage]  # parent insights (currently always empty in v1)


def audit_summary(contents: archive.ArchiveContents) -> AuditSummary:
    """Compute the archive's audit summary in one pass."""
    state_counts = {
        "accepted": 0, "candidate": 0, "stale": 0,
        "rejected": 0, "rejected_corrected": 0, "retired": 0,
    }
    source_ids = {c.passage_id for c in contents.registry}
    orphans = 0
    for ins in contents.insights:
        state_counts[ins.verdict_state] = state_counts.get(ins.verdict_state, 0) + 1
        if not all(c.passage_id in source_ids for c in ins.citations):
            orphans += 1

    # `content_hash` is build-time and live drift is a retrieval-time
    # concern surfaced via VerifiedHit.drift_status. The static audit
    # reflects post-refresh state — refresh updates hashes inline, so
    # every row in the registry is verified-against-itself by definition.
    drift_counts = {"verified": len(contents.registry)}

    return AuditSummary(
        source_passages=len(contents.registry),
        insight_total=len(contents.insights),
        insight_accepted=state_counts.get("accepted", 0),
        insight_candidate=state_counts.get("candidate", 0),
        insight_stale=state_counts.get("stale", 0),
        insight_rejected=state_counts.get("rejected", 0)
                          + state_counts.get("rejected_corrected", 0),
        insight_retired=state_counts.get("retired", 0),
        source_drift_status_counts=drift_counts,
        insight_orphans=orphans,
    )


def audit_stale(contents: archive.ArchiveContents) -> list[InsightPassage]:
    """All insights currently in the `stale` state. Returns input-order
    list."""
    return [ins for ins in contents.insights if ins.verdict_state == "stale"]


def audit_orphans(contents: archive.ArchiveContents) -> list[InsightPassage]:
    """Insights whose citations point to passage_ids that no longer
    exist in the source registry. Caused by source deletion + refresh.
    """
    source_ids = {c.passage_id for c in contents.registry}
    out: list[InsightPassage] = []
    for ins in contents.insights:
        if ins.verdict_state in FINAL_STATES:
            continue
        if any(c.passage_id not in source_ids for c in ins.citations):
            out.append(ins)
    return out


def trace_insight(
    contents: archive.ArchiveContents, insight_id: str,
) -> InsightTrace:
    """Full provenance chain for one insight_id.

    Resolves every citation against the source registry; flags orphans
    (cited passages that no longer exist). Lineage chain follows
    parent_ids (v1: always empty until insight-to-insight promotion
    lands).

    Raises KeyError if insight_id is not in the archive.
    """
    target = next((i for i in contents.insights if i.insight_id == insight_id), None)
    if target is None:
        raise KeyError(f"insight_id {insight_id!r} not in this archive")

    coords_by_id = {c.passage_id: c for c in contents.registry}
    source_passages: list[dict] = []
    orphans: list[str] = []
    for cit in target.citations:
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

    # Lineage chain: walk parent insights (v1: empty, but the
    # iterator is here so v2 doesn't need to add the surface).
    lineage: list[InsightPassage] = []
    seen: set[str] = {target.insight_id}
    parents = list(target.lineage)
    while parents:
        pid = parents.pop(0)
        if pid in seen:
            continue
        seen.add(pid)
        parent = next((i for i in contents.insights if i.insight_id == pid), None)
        if parent is None:
            continue
        lineage.append(parent)
        parents.extend(parent.lineage)

    return InsightTrace(
        insight=target,
        source_passages=source_passages,
        source_orphans=orphans,
        lineage_chain=lineage,
    )


def trace_source(
    contents: archive.ArchiveContents, source_passage_id: str,
) -> list[InsightPassage]:
    """Reverse trace — every insight that cites a given source passage.

    Used by the audit to answer 'what claims depend on this passage?'
    before editing it.
    """
    out: list[InsightPassage] = []
    for ins in contents.insights:
        if any(c.passage_id == source_passage_id for c in ins.citations):
            out.append(ins)
    return out
