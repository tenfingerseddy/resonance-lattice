"""Promotion pipeline — run candidates through the compression test, write
survivors back to the .rlat insight layer.

The bridge between memory (where synthesis_candidate rows accumulate) and
the corpus insight layer (where promoted rows live). At consolidation
cadence — `Stop` / `SessionEnd` hooks, or explicit
`rlat memory promote-to-corpus` — the harness calls `promote_candidates`
with the eligible memory rows plus a query history; the function decides
which graduate.

The memory module owns *which* rows are eligible (semantic tier, has
citations, verdict-positive, scope=shared). This module owns the
gating + writeback. Single concern per layer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from . import archive
from .compression_test import (
    CompressionTestResult,
    QueryRecord,
    run_compression_test,
)
from .faithfulness import FaithfulnessReport, assess_faithfulness
from .insight import (
    InsightCitation,
    InsightPassage,
    compute_insight_id,
    seed_confidence,
)


@dataclass(frozen=True)
class SynthesisCandidate:
    """Memory-row-shaped input to the promotion pipeline.

    Lightweight DTO so the promotion module doesn't depend on the memory
    module's internals. Consumers (the consolidator hook, the CLI) build
    these from memory rows.
    """
    candidate_id: str               # memory row id (so we can mark it promoted)
    content: str
    citations: tuple[InsightCitation, ...]
    source_passage_hashes: tuple[str, ...]
    source_model_hash: str
    query: str | None
    intent_context: str | None
    encoder_version: str
    faithfulness: float | None = None   # gate score; seeds the Beta prior
    kind: str = "synthesis"


@dataclass(frozen=True)
class PromotionOutcome:
    """One candidate's verdict from the promotion run."""
    candidate_id: str
    promoted: bool
    promoted_insight_id: str | None
    test_result: CompressionTestResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _build_insight_passage(
    candidate: SynthesisCandidate, idx: int,
) -> InsightPassage:
    insight_id = compute_insight_id(
        candidate.content,
        list(candidate.source_passage_hashes),
        candidate.source_model_hash,
    )
    corroboration, falsification = seed_confidence(candidate.faithfulness)
    return InsightPassage(
        insight_idx=idx,
        insight_id=insight_id,
        kind=candidate.kind,         # type: ignore[arg-type]
        content=candidate.content,
        citations=candidate.citations,
        query=candidate.query,
        generated_at=_now(),
        source_model_hash=candidate.source_model_hash,
        source_passage_hashes=candidate.source_passage_hashes,
        verdict_state="accepted",    # promotion implies the test passed
        verdict_signals=(),
        lineage=(),
        intent_context=candidate.intent_context,
        stale_if_sources_drift=True,
        encoder_version=candidate.encoder_version,
        corroboration=corroboration,
        falsification=falsification,
    )


def promote_candidates(
    km_path: str | Path,
    candidates: list[SynthesisCandidate],
    candidate_embeddings: np.ndarray,
    queries: list[QueryRecord],
    *,
    contents: archive.ArchiveContents | None = None,
) -> list[PromotionOutcome]:
    """Run each candidate through the compression test; promote survivors.

    Single atomic writeback at the end — either all survivors land
    together or none (a write failure leaves the archive untouched, per
    `write_insight_layer_in_place`'s atomic-replace contract).

    `candidate_embeddings`: shape `(N_candidates, D)` L2-normalised in the
    corpus's encoder space.

    `contents`: optional pre-loaded ArchiveContents; passed by callers
    (the consolidator) that already have it in hand to avoid a redundant
    full read.

    Returns one `PromotionOutcome` per candidate, in input order. The
    caller is responsible for marking promoted rows in the memory store
    (so they don't re-promote on the next cycle).
    """
    if candidate_embeddings.shape[0] != len(candidates):
        raise ValueError(
            f"candidate_embeddings has {candidate_embeddings.shape[0]} rows "
            f"but candidates list has {len(candidates)}"
        )

    p = Path(km_path)
    if contents is None:
        contents = archive.read(p)

    source_band = contents.bands["base"]
    source_registry = contents.registry
    insight_band = contents.bands.get(archive.INSIGHT_BAND_NAME)
    insights = list(contents.insights)

    outcomes: list[PromotionOutcome] = []
    new_insights: list[InsightPassage] = []
    existing_ids = {i.insight_id for i in insights}
    # Running insight band: starts as the existing band (or an empty (0, D)
    # shell), grows by one row each time a candidate promotes. Eliminates
    # the O(N²) per-iteration vstack — each promotion does a single append.
    if insight_band is not None:
        running_band = insight_band.copy()
    else:
        running_band = np.zeros(
            (0, candidate_embeddings.shape[1]), dtype="float32",
        )

    for i, candidate in enumerate(candidates):
        # Build a temporary InsightPassage to feed the test (test reads
        # citations + content_hashes; doesn't care about insight_idx yet).
        candidate_idx = len(insights) + len(new_insights)
        provisional = _build_insight_passage(candidate, candidate_idx)
        emb = candidate_embeddings[i].astype("float32", copy=False)

        # Pre-test: same insight_id already in the layer or queued this
        # cycle? Skip the compression-test compute and return a clean
        # 'idempotent' outcome. Mirrors memory.dedup's corroborate-or-add
        # idempotency — running the consolidator twice with the same
        # candidate is a no-op rather than a duplicate-reject.
        if provisional.insight_id in existing_ids:
            from .compression_test import CompressionTestResult
            outcomes.append(PromotionOutcome(
                candidate_id=candidate.candidate_id,
                promoted=False,
                promoted_insight_id=None,
                test_result=CompressionTestResult(
                    passed=False, reason="idempotent",
                    coverage_with=0.0, coverage_without=0.0, coverage_delta=0.0,
                    distinct_sources=len({c.passage_id for c in candidate.citations}),
                    nearest_duplicate_score=1.0,
                    growth_ratio=1.0,
                ),
            ))
            continue

        # The running insight layer (existing + already-promoted this cycle)
        # is maintained incrementally below; no per-iteration vstack.
        running_insights = insights + new_insights

        result = run_compression_test(
            provisional, emb,
            source_band, source_registry,
            running_band if running_band.shape[0] > 0 else None,
            running_insights,
            queries,
        )

        if result.passed:
            running_band = np.vstack([running_band, emb.reshape(1, -1)])
            new_insights.append(provisional)
            existing_ids.add(provisional.insight_id)
            outcomes.append(PromotionOutcome(
                candidate_id=candidate.candidate_id,
                promoted=True,
                promoted_insight_id=provisional.insight_id,
                test_result=result,
            ))
        else:
            outcomes.append(PromotionOutcome(
                candidate_id=candidate.candidate_id,
                promoted=False,
                promoted_insight_id=None,
                test_result=result,
            ))

    # No survivors → no write.
    if not new_insights:
        return outcomes

    # Compose final insight layer + atomic writeback. Re-stamp insight_idx
    # to match the running_band's row order.
    from dataclasses import replace as _replace
    final_insights = [
        _replace(ins, insight_idx=i)
        for i, ins in enumerate(insights + new_insights)
    ]
    archive.write_insight_layer_in_place(p, final_insights, running_band)
    return outcomes


def _candidate_from_answer(
    question: str, answer: str, cited: list[dict],
    faithfulness: float | None = None,
) -> SynthesisCandidate:
    """Build a SynthesisCandidate from a deep-search answer + its cited
    passages. `cited` rows must carry `passage_id` + `content_hash`.
    `faithfulness` is the gate score — it seeds the Beta confidence prior."""
    citations = tuple(
        InsightCitation(
            passage_id=p["passage_id"], char_span=None,
            confidence=float(p.get("score", 0.5)),
        )
        for p in cited
    )
    hashes = tuple(p["content_hash"] for p in cited)
    h = hashlib.sha256()
    h.update(answer.encode("utf-8", errors="replace"))
    for ph in sorted(hashes):
        h.update(ph.encode("ascii"))
    return SynthesisCandidate(
        candidate_id=h.hexdigest()[:16],
        content=answer,
        citations=citations,
        source_passage_hashes=hashes,
        source_model_hash="deep-search-default",
        query=question,
        intent_context=None,
        encoder_version="gte-mb-768",
        faithfulness=faithfulness,
    )


def promote_if_faithful(
    km_path: str | Path,
    *,
    question: str,
    answer: str,
    evidence_passages: list[dict],
    client,
    model: str | None = None,
    contents: archive.ArchiveContents | None = None,
) -> tuple[FaithfulnessReport, list[PromotionOutcome]]:
    """Faithfulness-gate a deep-search answer, then promote if it passes.

    The autonomous entry to the confidence lifecycle — no user verdict
    (docs/internal/GROUNDING_MODEL.md). Faithful → the answer is built into
    a SynthesisCandidate and run through `promote_candidates` (the
    compression test still gates the write). Not faithful → returns the
    report with no outcomes; the archive is untouched.

    `client` is an `anthropic.Anthropic`-shaped client for the faithfulness
    judge; inject a stub for tests.
    """
    from ..field.encoder import encode

    report = assess_faithfulness(
        question, answer, evidence_passages, client, model=model,
    )
    if not report.faithful:
        return report, []

    # Candidate citations must carry passage_id + content_hash for a real
    # provenance binding; a projection without the registry omits these.
    cited = [
        p for p in evidence_passages
        if p.get("passage_id") and p.get("content_hash")
    ]
    if not cited:
        return report, []

    candidate = _candidate_from_answer(
        question, answer, cited, faithfulness=report.score,
    )
    embedding = encode([answer]).astype("float32")
    outcomes = promote_candidates(
        km_path, [candidate], embedding, queries=[], contents=contents,
    )
    return report, outcomes


def candidates_from_memory_rows(rows: Iterable[dict]) -> list[SynthesisCandidate]:
    """Adapter: turn memory-row dicts into SynthesisCandidate DTOs.

    Expected memory row shape (kind="synthesis_candidate"):
        {
          "id": str,
          "content": str,
          "citations": [{"passage_id": str, "char_span": [int, int]|None,
                          "confidence": float}, ...],
          "source_passage_hashes": [str, ...],
          "source_model_hash": str,
          "query": str | None,
          "intent_context": str | None,
          "encoder_version": str,
        }

    Memory rows missing required fields are skipped silently (the caller
    upstream is responsible for sanity-checking; we don't crash a
    consolidation cycle on one malformed row).
    """
    out: list[SynthesisCandidate] = []
    for r in rows:
        try:
            citations = tuple(
                InsightCitation(
                    passage_id=c["passage_id"],
                    char_span=(tuple(c["char_span"]) if c.get("char_span") else None),
                    confidence=float(c.get("confidence", 0.9)),
                )
                for c in r["citations"]
            )
            out.append(SynthesisCandidate(
                candidate_id=r["id"],
                content=r["content"],
                citations=citations,
                source_passage_hashes=tuple(r["source_passage_hashes"]),
                source_model_hash=r["source_model_hash"],
                query=r.get("query"),
                intent_context=r.get("intent_context"),
                encoder_version=r.get("encoder_version", ""),
                faithfulness=r.get("faithfulness"),
            ))
        except (KeyError, ValueError, TypeError):
            continue
    return out
