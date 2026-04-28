"""RQL — Resonance Query Language (curated v2.0 surface).

The product surface for power users + integrators who want more than `rlat
search`. RQL is the Python API; CLI subcommands wrap a subset of it.

The 14 ops are organised into 5 groups:

  Foundation (4):     locate, profile, neighbors, near_duplicates
  Comparison (3):     compare, unique, intersect
  Composition (2):    compose, merge
  Evidence + drift (3): evidence, drift, corpus_diff
  Experimental (2):   contradictions, audit

Cross-knowledge-model ops (compare, unique, intersect, compose, merge,
corpus_diff) ALWAYS use the base band — the cross-model rule that makes
these ops work at all relies on every knowledge model carrying a
comparable base band. `corpus_diff` and `merge` additionally require the
backbone revisions to match exactly (a threshold-based classification
becomes meaningless across distinct embedding distributions); `compare`
warns on revision mismatch but proceeds because cosine ordering still
holds.

Every op returns a typed result (Citation, Hit, Profile, etc.); never a raw
ndarray or untyped tuple. See `types.py` for the full type set.

Phase 6 deliverable. Per-op rationale: docs/internal/RQL.md.
"""

from .compare import compare, intersect, unique
from .compose import ComposedKnowledgeModel, compose, merge
from .experimental import audit, contradictions
from .inspect import locate, near_duplicates, profile
from .navigate import corpus_diff, drift, evidence, neighbors
from .types import (
    AuditReport,
    Citation,
    CitationHit,
    CompareResult,
    ComposedHit,
    ConfidenceMetrics,
    ContradictionPair,
    CorpusDiff,
    DriftRecord,
    DriftReport,
    DuplicateCluster,
    EvidenceReport,
    NeighborHit,
    PassagePair,
    Profile,
)

__all__ = [
    # Foundation ops (commit 1)
    "locate",
    "near_duplicates",
    "neighbors",
    "profile",
    # Comparison ops (commit 2) — cross-KM, base-band only
    "compare",
    "intersect",
    "unique",
    # Composition ops (commit 3) — multi-KM federated + merge
    "compose",
    "merge",
    "ComposedKnowledgeModel",
    # Evidence + drift (commit 4) — flagship + change tracking
    "evidence",
    "drift",
    "corpus_diff",
    # Experimental ops (commit 5) — heuristic surfaces, may produce noise
    "contradictions",
    "audit",
    # Typed results
    "AuditReport",
    "Citation",
    "CitationHit",
    "CompareResult",
    "ComposedHit",
    "ConfidenceMetrics",
    "ContradictionPair",
    "CorpusDiff",
    "DriftRecord",
    "DriftReport",
    "DuplicateCluster",
    "EvidenceReport",
    "NeighborHit",
    "PassagePair",
    "Profile",
]
