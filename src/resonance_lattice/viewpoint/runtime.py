"""Viewpoint deliberation runtime.

The query-time pipeline that turns `(query, lens, corpus)` into a
`ViewpointPosition` with full provenance. Architecture §6.

Pipeline:

  1. Encode query
  2. Retrieve top-K from source band (using existing field.retrieve)
  3. Retrieve top-K from insight band (using existing field.retrieve_insight)
  4. Retrieve top-K from lens.private_insights (when present)
  5. Apply lens re-ranking:
       - source scores ×= lens.trust_for_source(source_file)
       - insight scores ×= lens.preference_for_insight(insight_id)
  6. Merge, sort by adjusted score, take top-K
  7. Construct ProvenanceGraph from cited insights → source passages
  8. Compute source_only_alternative (the answer-shaped baseline)
  9. Return ViewpointPosition

Heavy LLM deliberation (contradiction detection, boundary identification,
stance application) is the v2 surface; v1 ships the structural pipeline
with `deliberation_depth=0`. The architecture's `deliberation_depth`
parameter is wired but doesn't activate the LLM-driven pass until v2 —
the substrate is ready when the prompts and the cost budget are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..field import ann, retrieve, retrieve_insight
from ..field.dense import topk_indices
from ..store import archive
from ..store.insight import InsightCitation
from ..store.verified import (
    InsightHit,
    VerifiedHit,
    verify_hits,
    verify_insight_hits,
)


@dataclass(frozen=True)
class ProvenanceNode:
    """One node in a viewpoint position's provenance graph.

    Each node references either a source passage (`layer="source"`,
    `id=passage_id`), an insight passage (`layer="insight"`,
    `id=insight_id`), or the answer itself (`layer="viewpoint"`,
    `id=position_id`).

    `cites` is the list of node ids this node depends on. The full graph
    is a DAG with the viewpoint at the root and source passages at the
    leaves.
    """
    layer: Literal["source", "insight", "lens-private", "viewpoint"]
    id: str
    score: float
    drift_status: str
    cites: tuple[str, ...] = ()


@dataclass(frozen=True)
class ViewpointPosition:
    """A lens speaking — the query-time output of a viewpoint.

    `answer`: synthesised position (text; v1 returns the highest-score
    hit's text concatenated under layer labels — full LLM synthesis is
    v2 work). `source_only_alternative` is the answer you would have
    gotten without the lens overlay or insight layer — the
    trust-contract foundation 5 surface that lets a user always
    compare.
    """
    query: str
    answer: str
    hits: tuple[Any, ...]               # tuple[VerifiedHit | InsightHit, ...]
    provenance: tuple[ProvenanceNode, ...]
    deliberation_trace: tuple[str, ...]
    confidence: float
    source_only_alternative: str
    lens_id: str | None


def _retrieve_lens_private(
    query_emb: np.ndarray,
    lens,
    top_k: int,
) -> list[tuple[int, float]]:
    """Cosine top-k over a lens's private insight band. Returns empty
    when the lens has no private insights."""
    if lens is None:
        return []
    band = lens.private_insights_band
    if band is None or band.shape[0] == 0:
        return []
    scores = band @ query_emb
    top_idx = topk_indices(scores, min(top_k, band.shape[0]))
    return [(int(i), float(scores[int(i)])) for i in top_idx]


def _apply_lens_reranking(
    source_hits: list[VerifiedHit],
    insight_hits: list[InsightHit],
    lens,
) -> tuple[list[VerifiedHit], list[InsightHit]]:
    """Multiply source scores by trust_weights, insight scores by
    insight_preferences. Returns new lists with adjusted scores.

    A lens of `None` is the identity transform — the v1 source-only-
    style behaviour where no lens is loaded.
    """
    if lens is None:
        return source_hits, insight_hits
    from dataclasses import replace
    adjusted_source = [
        replace(h, score=h.score * lens.trust_for_source(h.source_file))
        for h in source_hits
    ]
    adjusted_insight = [
        replace(h, score=h.score * lens.preference_for_insight(h.insight_id))
        for h in insight_hits
    ]
    return adjusted_source, adjusted_insight


def _format_position_answer(hits: list) -> str:
    """v1 answer composition: concatenate hits under layer headings.

    A full LLM synthesis lives at v2 (architecture §6.2 step 5); v1's
    job is to assemble the structural surface so the harness's existing
    LLM call sites (deep-search) can consume it. The composition is
    deterministic so tests can assert against it.
    """
    if not hits:
        return "[VIEWPOINT] no hits returned"
    lines: list[str] = []
    for h in hits:
        if h.layer == "insight":
            lines.append(f"[INSIGHT {h.insight_id} verdict={h.verdict_state} "
                         f"confidence={h.confidence:.2f}] {h.content}")
        else:
            lines.append(f"[SOURCE {h.source_file}:{h.char_offset}+{h.char_length} "
                         f"drift={h.drift_status}] {h.text}")
    return "\n\n".join(lines)


def _build_provenance(hits: list) -> tuple[ProvenanceNode, ...]:
    """Construct the provenance graph as a flat node list.

    Insight hits expand their citations into source nodes; source hits
    contribute themselves. Duplicate source nodes (cited by multiple
    insights) are deduplicated by id.
    """
    nodes: dict[str, ProvenanceNode] = {}
    for h in hits:
        if h.layer == "insight":
            cites = tuple(c.passage_id for c in h.citations)
            nodes.setdefault(h.insight_id, ProvenanceNode(
                layer="insight",
                id=h.insight_id,
                score=h.score,
                drift_status=h.drift_status,
                cites=cites,
            ))
            for c in h.citations:
                nodes.setdefault(c.passage_id, ProvenanceNode(
                    layer="source",
                    id=c.passage_id,
                    score=0.0,        # cited by insight, not retrieved directly
                    drift_status="verified",  # citation-time; live drift checked separately
                ))
        else:
            nodes.setdefault(
                f"{h.source_file}:{h.char_offset}",
                ProvenanceNode(
                    layer="source",
                    id=f"{h.source_file}:{h.char_offset}",
                    score=h.score,
                    drift_status=h.drift_status,
                ),
            )
    return tuple(nodes.values())


def deliberate(
    query: str,
    query_emb: np.ndarray,
    contents: archive.ArchiveContents,
    *,
    lens=None,
    source_store=None,
    top_k: int = 10,
    deliberation_depth: int = 0,
    include_stale: bool = False,
) -> ViewpointPosition:
    """Run the viewpoint pipeline. Returns a ViewpointPosition.

    `query_emb` is the encoded query in the corpus's encoder space.
    `source_store` is the resolved Store (local/bundled/remote) needed
    to fetch source-passage text — pass the result of
    `store.open_store(km_path, contents, ...)`.

    `deliberation_depth=0` (v1 default): retrieval + lens re-ranking,
    no LLM deliberation. Higher values are reserved for v2 contradiction
    + boundary analysis.

    Trust contract: every position carries its `source_only_alternative`
    — the answer that source-only retrieval would have given. The user
    can always compare.
    """
    trace: list[str] = []

    # ---- Source retrieval ----
    source_handle = contents.select_band()
    source_ann = ann.deserialize(source_handle.ann_blob) if source_handle.ann_blob else None
    source_raw = retrieve(
        query_emb, source_handle, source_ann, contents.registry, top_k,
    )
    source_hits = verify_hits(source_raw, source_store, contents.registry) \
                  if source_store is not None else []
    trace.append(f"source: retrieved {len(source_hits)} via {source_handle.name}")

    # ---- Insight retrieval ----
    insight_hits: list[InsightHit] = []
    insight_handle = contents.insight_band() if contents.insights else None
    if insight_handle is not None:
        insight_ann_index = (
            ann.deserialize(insight_handle.ann_blob)
            if insight_handle.ann_blob else None
        )
        raw = retrieve_insight(
            query_emb, insight_handle.band, insight_ann_index, top_k,
        )
        insight_hits = verify_insight_hits(
            raw, contents.insights, include_stale=include_stale,
        )
        trace.append(f"insight: retrieved {len(insight_hits)} (include_stale={include_stale})")

    # ---- Lens-private retrieval ----
    lens_private_hits: list[InsightHit] = []
    if lens is not None and lens.private_insights:
        raw_lp = _retrieve_lens_private(query_emb, lens, top_k)
        # Treat private insights the same shape as shared — same verdict
        # state machine applies. The lens's promotion to shared has not
        # yet happened, but at query time we surface them to the lens
        # owner.
        lens_private_hits = verify_insight_hits(
            raw_lp, lens.private_insights, include_stale=include_stale,
        )
        trace.append(f"lens-private: retrieved {len(lens_private_hits)}")

    # Capture the un-reranked source hits BEFORE the lens overlay
    # mutates scores — this is the source-only-alternative baseline.
    source_hits_baseline = list(source_hits)

    # ---- Lens re-ranking ----
    source_hits, insight_hits = _apply_lens_reranking(
        source_hits, insight_hits, lens,
    )
    # Private hits also get the insight-preference treatment.
    if lens is not None:
        from dataclasses import replace
        lens_private_hits = [
            replace(h, score=h.score * lens.preference_for_insight(h.insight_id))
            for h in lens_private_hits
        ]
    if lens is not None:
        trace.append(f"lens applied: {lens.manifest.lens_id}")

    # ---- Merge & sort & top-K ----
    all_hits = list(source_hits) + list(insight_hits) + list(lens_private_hits)
    all_hits.sort(key=lambda h: -h.score)
    all_hits = all_hits[:top_k]

    # ---- Source-only alternative (trust contract foundation 5) ----
    # Use the pre-rerank baseline captured above — no redundant verify.
    source_only_text = _format_position_answer(source_hits_baseline[:top_k])
    trace.append(
        f"source-only alternative computed ({len(source_hits_baseline)} hits)"
    )

    # ---- v1 answer composition ----
    answer = _format_position_answer(all_hits)
    provenance = _build_provenance(all_hits)

    # ---- v1 confidence: weighted average of top-K insight confidence ----
    insight_confidences = [
        h.confidence for h in all_hits
        if h.layer == "insight" and h.confidence > 0
    ]
    overall_conf = (
        sum(insight_confidences) / len(insight_confidences)
        if insight_confidences else 0.0
    )

    return ViewpointPosition(
        query=query,
        answer=answer,
        hits=tuple(all_hits),
        provenance=provenance,
        deliberation_trace=tuple(trace),
        confidence=overall_conf,
        source_only_alternative=source_only_text,
        lens_id=lens.manifest.lens_id if lens is not None else None,
    )
