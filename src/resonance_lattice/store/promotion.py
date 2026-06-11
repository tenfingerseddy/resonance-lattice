"""Promotion pipeline — run candidates through the compression test, write
survivors back to the .rlat insight layer.

The bridge between authored answers (deep-search, curator fills, the
agent/human caller-verified path) and the corpus insight layer (where
promoted rows live). Callers funnel through `promote_if_faithful`, which
builds the verdict-anchored query history and calls `promote_candidates`;
the function decides which graduate.

The memory module owns *which* rows are eligible (semantic tier, has
citations, verdict-positive, scope=shared). This module owns the
gating + writeback. Single concern per layer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..state.claim import Claim
from ..state.claim_lifecycle import GateSignals, consolidate_corpus
from . import archive
from .compression_test import (
    CompressionTestResult,
    QueryRecord,
    run_compression_test,
)
from .corpus_claim_io import new_corpus_claim
from .faithfulness import FaithfulnessReport, assess_faithfulness
from .insight import InsightCitation

# Most-recent previously-promoted queries the verdict-anchored coverage
# gate re-checks per landing (one batched encode; bounds cost on bands
# with long promotion histories).
_VERDICT_QUERY_WINDOW = 32


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
    provenance: str | None = None       # trust tier; None auto-derives from citations (user must be explicit)


@dataclass(frozen=True)
class PromotionOutcome:
    """One candidate's verdict from the promotion run."""
    candidate_id: str
    promoted: bool
    promoted_claim_id: str | None
    test_result: CompressionTestResult


def _build_corpus_claim(candidate: SynthesisCandidate) -> Claim:
    """Build the provisional corpus `Claim` for a candidate — born
    `candidate` (the `new_corpus_claim` default). The compression test
    reads it; a passing candidate is then transitioned to `active` by
    the lifecycle spine (`consolidate_corpus`), not born active here."""
    return new_corpus_claim(
        content=candidate.content,
        kind=candidate.kind,
        citations=candidate.citations,
        source_model_hash=candidate.source_model_hash,
        source_passage_hashes=candidate.source_passage_hashes,
        faithfulness=candidate.faithfulness,
        query=candidate.query,
        intent_context=candidate.intent_context,
        encoder_version=candidate.encoder_version,
        provenance=candidate.provenance,  # None → auto-derive the tier from the citations
    )


def promote_candidates(
    km_path: str | Path,
    candidates: list[SynthesisCandidate],
    candidate_embeddings: np.ndarray,
    queries: list[QueryRecord],
    *,
    contents: archive.ArchiveContents | None = None,
    require_lift: bool = True,
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
    new_insights: list[Claim] = []
    existing_ids = {c.claim_id for c in insights}
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
        # Build the provisional corpus claim to feed the test (test reads
        # citations + content_hashes; the band-row join is positional).
        provisional = _build_corpus_claim(candidate)
        emb = candidate_embeddings[i].astype("float32", copy=False)

        # Pre-test: same content fingerprint already in the layer or queued
        # this cycle? Skip the compression-test compute and return a clean
        # 'idempotent' outcome. Mirrors memory.dedup's corroborate-or-add
        # idempotency — running the consolidator twice with the same
        # candidate is a no-op rather than a duplicate-reject.
        if provisional.claim_id in existing_ids:
            from .compression_test import CompressionTestResult
            outcomes.append(PromotionOutcome(
                candidate_id=candidate.candidate_id,
                promoted=False,
                promoted_claim_id=None,
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
            require_lift=require_lift,
        )

        promoted = (
            consolidate_corpus(
                provisional,
                signals=GateSignals(compression_test_pass=True),
            )
            if result.passed else None
        )
        if promoted is not None and promoted.state == "active":
            # Compression test passed AND the spine committed the
            # candidate→active transition. Only an `active` claim is
            # written + reported promoted — a passing candidate the
            # spine declined (trust below PROMOTE_CONFIDENCE_THRESHOLD,
            # or below RETIRE_FLOOR) is NOT written, so its content
            # fingerprint stays free for a later, better candidate.
            running_band = np.vstack([running_band, emb.reshape(1, -1)])
            new_insights.append(promoted)
            existing_ids.add(promoted.claim_id)
            outcomes.append(PromotionOutcome(
                candidate_id=candidate.candidate_id,
                promoted=True,
                promoted_claim_id=promoted.claim_id,
                test_result=result,
            ))
        else:
            # Either the compression test failed, or it passed but the
            # spine declined the transition. `test_result` carries the
            # distinction; `promoted` is False either way.
            outcomes.append(PromotionOutcome(
                candidate_id=candidate.candidate_id,
                promoted=False,
                promoted_claim_id=None,
                test_result=result,
            ))

    # No survivors → no write. The claim↔band-row join is positional, so
    # the final layer is just existing + newly-promoted in order.
    if not new_insights:
        return outcomes

    archive.write_insight_layer_in_place(
        p, insights + new_insights, running_band,
    )
    return outcomes


def _candidate_from_answer(
    question: str, answer: str, cited: list[dict],
    faithfulness: float | None = None,
    provenance: str | None = None,
) -> SynthesisCandidate:
    """Build a SynthesisCandidate from a deep-search answer + its cited
    passages. `cited` rows must carry `passage_id` + `content_hash`.
    `faithfulness` is the gate score — it seeds the Beta confidence prior.
    `provenance` overrides the trust tier (e.g. "user" when the user vouched);
    None auto-derives it from the citations."""
    citations = tuple(
        InsightCitation(
            passage_id=p["passage_id"], char_span=None,
            confidence=float(p.get("score", 0.5)),
            source_url=p.get("source_url"),  # carries external provenance; None for corpus
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
        provenance=provenance,
    )


def promote_if_faithful(
    km_path: str | Path,
    *,
    question: str,
    answer: str,
    evidence_passages: list[dict],
    client=None,
    model: str | None = None,
    contents: archive.ArchiveContents | None = None,
    provenance: str | None = None,
    faithfulness: float | None = None,
) -> tuple[FaithfulnessReport, list[PromotionOutcome]]:
    """Faithfulness-gate a deep-search answer, then promote if it passes.

    The autonomous entry to the confidence lifecycle — no user verdict
    (docs/internal/GROUNDING_MODEL.md). Faithful → the answer is built into
    a SynthesisCandidate and run through `promote_candidates` (the
    compression test still gates the write). Not faithful → returns the
    report with no outcomes; the archive is untouched.

    `client` is an `anthropic.Anthropic`-shaped client for the faithfulness
    judge; inject a stub for tests.

    CALLER-VERIFIED landing (`client=None`): the FREE agent/human path. When
    no `client` is given, the LLM faithfulness gate is SKIPPED because the
    caller asserts it already verified the claim traces to its sources
    (agent-as-judge in a skill, or human curation in `rlat-curate`). An
    explicit `faithfulness` score (0..1, the caller's verification
    confidence) is then REQUIRED — without it the call refuses to land
    (the safety floor: no silent unverified write). Everything downstream
    still applies: the citations must carry `passage_id`+`content_hash`, the
    compression test gates the write, and the lifecycle spine still requires
    ≥2 distinct citations + trust ≥ 0.5. So `client=None` moves the
    *grounding* judgement to the caller; it does not drop the other guards.
    """
    from ..field.encoder import encode

    if client is not None:
        report = assess_faithfulness(
            question, answer, evidence_passages, client, model=model,
        )
        if not report.faithful:
            return report, []
    else:
        # No judge client → the caller asserts the claim is verified. Require an explicit confidence; refuse to
        # land a caller-verified claim with no stated faithfulness (silence beats an unverified write).
        if faithfulness is None:
            return FaithfulnessReport(
                claim_support=0.0, question_relevance=0.0, faithful=False, claims=(),
                reason="caller-verified landing requires an explicit faithfulness score"), []
        f = max(0.0, min(1.0, float(faithfulness)))
        report = FaithfulnessReport(
            claim_support=f, question_relevance=f, faithful=True, claims=(),
            reason="caller-asserted (verified by agent/human, LLM gate skipped)")

    # Candidate citations must carry passage_id + content_hash for a real
    # provenance binding; a projection without the registry omits these.
    cited = [
        p for p in evidence_passages
        if p.get("passage_id") and p.get("content_hash")
    ]
    if not cited:
        return report, []

    candidate = _candidate_from_answer(
        question, answer, cited, faithfulness=report.score, provenance=provenance,
    )

    # Verdict-anchored REGRESSION gate (2026-06 review, roadmap 4.3): the
    # compression test's coverage guards ran with queries=[] in every
    # production promotion — inert. The query set is the verdict-anchored
    # prior demand: questions of previously-promoted ACTIVE claims (each
    # passed the faithfulness gate when it landed), with their cited
    # passage_ids as the coverage targets. A new landing must not REGRESS
    # that covered demand (delta < 0 fails). Lift is deliberately NOT
    # required (`require_lift=False`): a novel-topic fill scores delta == 0
    # against prior demand by construction, and demanding lift would also
    # reject legitimate synthesis whose sources are already retrievable
    # (synthesis value ≠ routing value — the measured +0.082 authoring lift
    # exists even when sources rank in top-K). One batched encode covers
    # the answer + every query text. Best-effort: failure degrades to
    # queries=[] (the pre-wire behaviour) — bookkeeping must never brick
    # a landing.
    if contents is None:
        contents = archive.read(Path(km_path))
    queries: list[QueryRecord] = []
    try:
        prior = [
            c for c in (contents.insights or [])
            if c.state == "active"
            and getattr(c.facts, "query", None)
            and c.facts.citations
        ][-_VERDICT_QUERY_WINDOW:]
        texts = [answer] + [c.facts.query for c in prior]
        embs = encode(texts).astype("float32")
        embedding = embs[:1]
        for emb, c in zip(embs[1:], prior):
            ids = frozenset(
                cit.passage_id for cit in c.facts.citations if cit.passage_id
            )
            if ids:
                queries.append(QueryRecord(emb, ids))
    except Exception:  # noqa: BLE001 — coverage bookkeeping must not brick a landing
        queries = []
        embedding = encode([answer]).astype("float32")

    outcomes = promote_candidates(
        km_path, [candidate], embedding, queries=queries, contents=contents,
        require_lift=False,
    )
    return report, outcomes
