"""Typed result objects for RQL ops.

Every op returns a typed result, never a raw ndarray or untyped tuple.

`Citation` is the pure-location type: passage_idx → source coordinates +
build-time content_hash. To resolve drift status against the live source,
call `store.verify(c.source_file, c.char_offset, c.char_length,
c.content_hash)` — RQL deliberately decouples from `Store` so callers that
have a path-only handle (e.g. profile rendering) don't pay the verify cost.

Phase 6 deliverable.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..store.base import DriftStatus
from ..store.registry import PassageCoord


@dataclass(frozen=True)
class Citation:
    """Pure location reference: passage_idx → (source_file, char_span, hash).

    Carries the BUILD-TIME content_hash. Drift detection requires a live
    `Store.verify(...)` call against the same fields; do not assume this
    citation is verified — it's the address, not the verification.

    Field-identical to `store.registry.PassageCoord` today, but kept as a
    distinct type so the RQL public API doesn't leak store/registry types
    to consumers — and so future RQL surface fields (band_name, corpus_id
    from `compose`) can extend `Citation` without touching the on-disk
    `PassageCoord` schema.
    """
    passage_idx: int
    source_file: str
    char_offset: int
    char_length: int
    content_hash: str

    @classmethod
    def from_coord(cls, coord: PassageCoord) -> "Citation":
        """Lift a registry-level `PassageCoord` to the RQL `Citation` type.

        The single conversion point — every RQL op that constructs Citations
        from `ArchiveContents.registry` should go through here so a future
        Citation field add (e.g. corpus_id from `compose`) is one edit.
        """
        return cls(
            passage_idx=coord.passage_idx,
            source_file=coord.source_file,
            char_offset=coord.char_offset,
            char_length=coord.char_length,
            content_hash=coord.content_hash,
        )


@dataclass(frozen=True)
class CitationHit:
    """Citation + retrieval-derived score + materialised text."""
    citation: Citation
    score: float
    text: str


@dataclass(frozen=True)
class NeighborHit:
    """A `neighbors` op result: a passage's nearest neighbour + cosine."""
    citation: Citation
    cosine: float


@dataclass(frozen=True)
class DuplicateCluster:
    """A `near_duplicates` cluster: 2+ passages with pairwise cosine ≥ threshold.

    `mean_cosine` is the arithmetic mean of all distinct pairs in the cluster
    (NOT just the seed→member cosines). For singletons mean_cosine is 1.0,
    but `near_duplicates` filters singletons out — every cluster has ≥ 2
    members.
    """
    citations: list[Citation]
    mean_cosine: float


@dataclass(frozen=True)
class ComposedHit:
    """A search hit from a `ComposedKnowledgeModel` — Citation + score + which
    corpus it came from.

    `corpus_label` is the key under which the source knowledge model was
    registered with `compose(...)` — preserves attribution across federated
    queries.
    """
    corpus_label: str
    citation: Citation
    score: float


@dataclass(frozen=True)
class PassagePair:
    """A passage from KM A paired with a passage from KM B + their cosine.

    Used by `compare.intersect` to surface cross-corpus consensus: pairs of
    passages from two knowledge models with cosine ≥ threshold on the
    base band. Both citations carry full provenance — caller can resolve
    each back to its source independently.
    """
    citation_a: Citation
    citation_b: Citation
    cosine: float


@dataclass(frozen=True)
class CompareResult:
    """Cross-knowledge-model comparison metrics — all from base bands.

    Field meanings:
    - `a_passage_count` / `b_passage_count`: registry sizes.
    - `a_backbone_revision` / `b_backbone_revision`: pinned encoder revision.
    - `revision_match`: True when both KMs were built at the same encoder
      revision. False means the cosine numbers are still ordinally meaningful
      (both bands are unit vectors) but magnitudes are across different
      embedding distributions.
    - `centroid_cosine`: cos(mean(A_band), mean(B_band)) — single thematic
      alignment number.
    - `coverage_a_in_b` / `coverage_b_in_a`: mean max-cos of a `sample_size`
      sample. Asymmetric — A-into-B and B-into-A differ when corpora are
      different sizes or one is a strict superset.
    - `overlap_score`: symmetric (avg of both coverages). Single number for
      "how much do these corpora overlap?"
    - `sample_size`: actual sample size used (capped by min corpus size).
    """
    a_passage_count: int
    b_passage_count: int
    a_backbone_revision: str
    b_backbone_revision: str
    revision_match: bool
    centroid_cosine: float
    coverage_a_in_b: float
    coverage_b_in_a: float
    overlap_score: float
    sample_size: int


@dataclass(frozen=True)
class ContradictionPair:
    """A `contradictions` op result — high-cosine, low-lexical pair.

    Heuristic: pairs with `cosine ≥ cosine_threshold` AND `jaccard ≤
    lexical_threshold` (Jaccard over token-3-grams). The intuition:
    semantically-near but lexically-disjoint pairs may be paraphrases
    of opposing claims. Will produce noise — ship as experimental.
    """
    citation_a: Citation
    citation_b: Citation
    cosine: float
    jaccard: float


@dataclass(frozen=True)
class AuditReport:
    """`audit` op result: supporting + potentially-contradicting evidence
    for a claim, with corpus-level confidence indicators.

    `supporting`: top-K passages with cosine ≥ support_threshold to the
        claim. These DIRECTLY back the claim.
    `contradicting`: passages with high cosine to a supporting passage
        AND low lexical overlap (potentially a paraphrased disagreement).
    `source_count`: distinct source files in `supporting` — high count
        means multiple independent sources back the claim; low count
        means the supporting evidence may be a single source repeated.
    `drift_fraction`: fraction of supporting passages with non-verified
        drift_status. High → the supporting evidence is stale.
    """
    supporting: list["CitationHit"]
    contradicting: list["CitationHit"]
    source_count: int
    drift_fraction: float


@dataclass(frozen=True)
class ConfidenceMetrics:
    """Calibration metrics for an `evidence` retrieval — beyond raw scores.

    - `top1_score`: cosine of the top-1 hit. The single strongest signal
      for "is retrieval working at all on this query." High = a relevant
      passage exists; low = the query is off-corpus or under-represented.
    - `top1_top2_gap`: top-1 score minus top-2 score. Wide gap → confident
      single-best answer; narrow gap → top result is one of several
      plausible matches (often a paraphrase cluster — same fact stated
      multiple ways). Reported, not gated on, because paraphrase
      clustering inverts its meaning.
    - `source_diversity`: unique source_files in the top-K divided by K.
      1.0 = every hit from a different source (broad evidence); low =
      hits cluster around one source (potential corpus bias).
    - `drift_fraction`: fraction of top-K hits with drift_status != "verified".
      High → the evidence base is stale relative to its sources.
    - `band_used`: "base" or "optimised" — surfaces which band actually
      ran the retrieval.
    """
    top1_score: float
    top1_top2_gap: float
    source_diversity: float
    drift_fraction: float
    band_used: str

    @classmethod
    def from_verified(
        cls, verified: list, band_name: str
    ) -> "ConfidenceMetrics":
        """Compute calibration metrics over a list of `VerifiedHit`s.

        Single computation site for the metric formulae — both
        `rql.navigate.evidence` (typed-result API) and
        `cli.skill_context` (markdown-header-printing CLI) call this so
        the metric definitions can never drift between surfaces.
        """
        if not verified:
            return cls(
                top1_score=0.0,
                top1_top2_gap=0.0,
                source_diversity=0.0,
                drift_fraction=0.0,
                band_used=band_name,
            )
        scores = [v.score for v in verified]
        top1_top2_gap = scores[0] - scores[1] if len(scores) >= 2 else 1.0
        sources = {v.source_file for v in verified}
        drift_count = sum(1 for v in verified if v.drift_status != "verified")
        return cls(
            top1_score=scores[0],
            top1_top2_gap=top1_top2_gap,
            source_diversity=len(sources) / len(verified),
            drift_fraction=drift_count / len(verified),
            band_used=band_name,
        )


@dataclass(frozen=True)
class EvidenceReport:
    """`evidence` op result: top-K hits + calibrated confidence metrics.

    The flagship op for "search me X with citations and confidence." Each
    hit carries full source attribution (Citation) + the materialised
    passage text + the retrieval score; the report-level `confidence`
    surfaces calibration the user can act on (e.g. abstain when
    `top1_top2_gap < 0.05`).
    """
    hits: list["CitationHit"]
    confidence: ConfidenceMetrics


@dataclass(frozen=True)
class DriftRecord:
    """A passage with non-clean drift_status. Used in `DriftReport`."""
    citation: Citation
    drift_status: DriftStatus


@dataclass(frozen=True)
class DriftReport:
    """Within-KM drift: passages whose recorded content_hash no longer matches
    the live source. `drifted` = source exists but hash mismatches;
    `missing` = source file no longer exists. `verified_count` is the
    count of clean passages — useful for "X out of N drifted" framing.
    """
    drifted: list[DriftRecord]
    missing: list[DriftRecord]
    verified_count: int


@dataclass(frozen=True)
class CorpusDiff:
    """Cross-KM corpus diff: passages added / removed between two snapshots.

    `added` and `removed` are determined semantically (cosine ≥ threshold
    against the other corpus). `unchanged_count` is old rows with a
    near-match in new — gives a "X of N passages unchanged" framing.
    """
    added: list[Citation]
    removed: list[Citation]
    unchanged_count: int


@dataclass(frozen=True)
class Profile:
    """Typed corpus snapshot: structural + verified-retrieval statistics.

    `drift_counts` is populated by `inspect.profile(..., store=...)`; absent
    (empty dict) when called without a store. Verifying drift requires
    reading every source file once, which `inspect.profile` will only do
    when explicitly asked.
    """
    n_passages: int
    n_source_files: int
    source_distribution: dict[str, int]
    drift_counts: dict[DriftStatus, int]
    bands: dict[str, int]
    encoder_id: str
    encoder_revision: str
    format_version: int
