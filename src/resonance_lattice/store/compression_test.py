"""Compression test — the universal promotion gate for the insight layer.

The transformer-inspired rule: a candidate insight earns promotion only if
it would *improve* coverage of historical verdict-positive queries — and
only if it's not a paraphrase, not a semantic duplicate of an existing
insight, and doesn't bloat the layer past the growth cap.

Architecture §7. Applied at every promotion interface:

  memory synthesis_candidate → corpus insight (this module)
  lens private_insights → shared corpus insight (this module)
  insight kind → higher kind (this module, future)

This module is a pure-function test. The promotion pipeline (sibling
`promotion.py`) is the caller that runs the test for every candidate and
writes the survivors back to the .rlat via `write_insight_layer_in_place`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from ..field.dense import topk_indices
from .insight import InsightPassage
from .registry import PassageCoord

# Default parameters — tunable from the engineering spec once empirical
# data from the validation wave informs the right values.
DEFAULT_TOP_K = 10
DEFAULT_GROWTH_CAP = 0.001          # ≤ 0.1% layer growth per promotion
DEFAULT_MIN_DISTINCT_SOURCES = 2    # anti-paraphrase guard
DEFAULT_DUPLICATE_THRESHOLD = 0.95  # semantic-duplicate cosine cutoff

# Outcome reasons returned in CompressionTestResult.reason. Centralised so
# callers can match on them without typo risk; the promotion pipeline's
# pre-test idempotent guard adds one more value ("idempotent").
CompressionReason = Literal[
    "passed", "paraphrase", "duplicate", "bloat", "regression", "no_lift",
    "idempotent",
]


@dataclass(frozen=True)
class QueryRecord:
    """One historical verdict-positive query.

    `query_embedding` is L2-normalised float32 in the corpus's encoder
    space. `expected_passage_ids` is the set of source passage_ids that
    appeared in the accepted answer — coverage is measured as
    `expected_passage_ids ∩ top_K_retrieved_ids` proportion.
    """
    query_embedding: np.ndarray
    expected_passage_ids: frozenset[str]


@dataclass(frozen=True)
class CompressionTestResult:
    """Outcome of running the test on one candidate."""
    passed: bool
    reason: CompressionReason
    coverage_with: float            # mean recall@K when candidate included
    coverage_without: float         # mean recall@K when candidate excluded
    coverage_delta: float           # with - without
    distinct_sources: int
    nearest_duplicate_score: float  # max cosine vs existing insight band; -inf if none
    growth_ratio: float             # |layer_with| / max(1, |layer_without|)


def _recall_at_k(
    expected: frozenset[str],
    retrieved_ids: Sequence[str],
) -> float:
    """Recall@K — proportion of expected ids that appear in retrieved.

    Returns 0.0 when `expected` is empty (no signal to credit).
    """
    if not expected:
        return 0.0
    hit = sum(1 for pid in retrieved_ids if pid in expected)
    return hit / len(expected)


def _top_k_source(
    query: np.ndarray, band: np.ndarray, registry: list[PassageCoord], top_k: int,
) -> list[str]:
    """Top-K source passage_ids by cosine. Used by both with/without paths."""
    if band.shape[0] == 0:
        return []
    scores = band @ query
    top_idx = topk_indices(scores, min(top_k, band.shape[0]))
    return [registry[int(i)].passage_id for i in top_idx]


def _top_k_insight_supporting_ids(
    query: np.ndarray, insight_band: np.ndarray | None,
    insights: list[InsightPassage], top_k: int,
) -> frozenset[str]:
    """Top-K insight rows' supporting source passage_ids.

    An insight contributes the union of its cited source passages —
    that's how it "covers" a query. Used to merge insight reach into the
    coverage calculation.
    """
    if insight_band is None or insight_band.shape[0] == 0:
        return frozenset()
    scores = insight_band @ query
    top_idx = topk_indices(scores, min(top_k, insight_band.shape[0]))
    out: set[str] = set()
    for i in top_idx:
        for c in insights[int(i)].citations:
            out.add(c.passage_id)
    return frozenset(out)


def _measure_coverage(
    queries: list[QueryRecord],
    source_band: np.ndarray,
    source_registry: list[PassageCoord],
    insight_band: np.ndarray | None,
    insights: list[InsightPassage],
    top_k: int,
) -> float:
    """Mean recall@K over all queries — source top-K ∪ insight-citation reach."""
    if not queries:
        return 0.0
    total = 0.0
    for q in queries:
        src_ids = _top_k_source(q.query_embedding, source_band, source_registry, top_k)
        insight_ids = _top_k_insight_supporting_ids(
            q.query_embedding, insight_band, insights, top_k,
        )
        retrieved = set(src_ids) | insight_ids
        total += _recall_at_k(q.expected_passage_ids, list(retrieved))
    return total / len(queries)


def run_compression_test(
    candidate: InsightPassage,
    candidate_embedding: np.ndarray,
    source_band: np.ndarray,
    source_registry: list[PassageCoord],
    insight_band: np.ndarray | None,
    insights: list[InsightPassage],
    queries: list[QueryRecord],
    *,
    top_k: int = DEFAULT_TOP_K,
    growth_cap: float = DEFAULT_GROWTH_CAP,
    min_distinct_sources: int = DEFAULT_MIN_DISTINCT_SOURCES,
    duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
) -> CompressionTestResult:
    """Decide whether a candidate insight earns promotion.

    Five guards run in order; the first failure short-circuits with a
    descriptive `reason`:

    1. citation diversity — ≥ `min_distinct_sources` distinct cited
       source passage_ids (anti-paraphrase)
    2. semantic-duplicate — `candidate_embedding` cosine vs every
       existing insight in the layer must stay below
       `duplicate_threshold`
    3. growth cap — layer growth ratio after add must be ≤ 1 + growth_cap
    4. coverage delta — coverage_with ≥ coverage_without on the
       provided query records
    5. signed coverage — coverage_delta ≥ 0 (no regression)

    Pure — does not mutate inputs. Returns a structured result so the
    caller (the promotion pipeline) can log reasons for rejection.
    """
    distinct_sources = len({c.passage_id for c in candidate.citations})

    # Guard 1 — anti-paraphrase
    if distinct_sources < min_distinct_sources:
        return CompressionTestResult(
            passed=False, reason="paraphrase",
            coverage_with=0.0, coverage_without=0.0, coverage_delta=0.0,
            distinct_sources=distinct_sources,
            nearest_duplicate_score=-float("inf"),
            growth_ratio=1.0,
        )

    # Guard 2 — semantic duplicate
    nearest = -float("inf")
    if insight_band is not None and insight_band.shape[0] > 0:
        sims = insight_band @ candidate_embedding
        nearest = float(sims.max())
        if nearest >= duplicate_threshold:
            return CompressionTestResult(
                passed=False, reason="duplicate",
                coverage_with=0.0, coverage_without=0.0, coverage_delta=0.0,
                distinct_sources=distinct_sources,
                nearest_duplicate_score=nearest,
                growth_ratio=1.0,
            )

    # Guard 3 — growth cap
    existing_count = insight_band.shape[0] if insight_band is not None else 0
    growth = (existing_count + 1) / max(1, existing_count)
    # A first-ever promotion has growth ratio 2.0 (1/1). Treat the empty
    # layer as a special case so the cap doesn't reject the first row.
    if existing_count > 0 and (growth - 1.0) > growth_cap:
        return CompressionTestResult(
            passed=False, reason="bloat",
            coverage_with=0.0, coverage_without=0.0, coverage_delta=0.0,
            distinct_sources=distinct_sources,
            nearest_duplicate_score=nearest,
            growth_ratio=growth,
        )

    # Guards 4 + 5 — coverage measurement
    cov_without = _measure_coverage(
        queries, source_band, source_registry, insight_band, insights, top_k,
    )
    # Construct the "with" insight layer by appending candidate.
    if insight_band is None or insight_band.shape[0] == 0:
        with_band = candidate_embedding.reshape(1, -1)
        with_insights = [candidate]
    else:
        with_band = np.vstack([insight_band, candidate_embedding.reshape(1, -1)])
        with_insights = insights + [candidate]
    cov_with = _measure_coverage(
        queries, source_band, source_registry, with_band, with_insights, top_k,
    )
    delta = cov_with - cov_without

    if delta < 0:
        return CompressionTestResult(
            passed=False, reason="regression",
            coverage_with=cov_with, coverage_without=cov_without,
            coverage_delta=delta,
            distinct_sources=distinct_sources,
            nearest_duplicate_score=nearest,
            growth_ratio=growth,
        )

    # When there are no historical queries the delta is 0 by construction.
    # We treat that as a "pass on diversity alone" — the test can't reject
    # for lack of coverage data, but the candidate has cleared the
    # paraphrase, duplicate, and bloat guards.
    if delta == 0 and queries:
        return CompressionTestResult(
            passed=False, reason="no_lift",
            coverage_with=cov_with, coverage_without=cov_without,
            coverage_delta=delta,
            distinct_sources=distinct_sources,
            nearest_duplicate_score=nearest,
            growth_ratio=growth,
        )

    return CompressionTestResult(
        passed=True, reason="passed",
        coverage_with=cov_with, coverage_without=cov_without,
        coverage_delta=delta,
        distinct_sources=distinct_sources,
        nearest_duplicate_score=nearest,
        growth_ratio=growth,
    )
