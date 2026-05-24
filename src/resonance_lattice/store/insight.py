"""Insight layer — earned, cited derived passages.

The second of the three user-facing layers in a knowledge model (source / insight /
lens). Insight passages are LLM-synthesised compressions of the source layer,
written back to the .rlat after passing the compression-test promotion gate.
Every insight passage carries:

  - A citation chain back to source passages (by passage_id + content_hash)
  - A verdict-state machine (candidate → accepted → stale → retired)
  - Provenance: source_model_hash, generated_at, intent_context, lineage
  - Drift sensitivity: source_passage_hashes for cascade detection

On disk: `insight.jsonl` alongside `passages.jsonl`, one row per insight,
`insight_idx` implied by line position. Band: `bands/insight.npz` with the
same encoder + dim + L2-normalisation as the source band, so retrieval uses
one substrate.

See `.claude/plans/lensed-knowledge-architecture.md` §4 for the full
specification. This module owns the schema + JSONL I/O; the archive module
owns the ZIP-layout side; the compression-test module (Day 4) owns
promotion gating.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal

# ----------------------------------------------------------------------------
# Enums (Literal-typed so they cross JSON boundaries as strings)
# ----------------------------------------------------------------------------

InsightKind = Literal[
    "synthesis",   # earned compression from a deep-search answer
    "faq",         # promoted from a repeated query
    "pattern",     # cross-event regularity
    "principle",   # cross-context generalisation
    "mechanism",   # causal claim ("X → Y under Z")
    "boundary",    # domain-of-validity annotation
    "negation",    # what is reliably not true
    "gap",         # known coverage gap (failed grounding)
]

VerdictState = Literal[
    "candidate",            # synthesised, awaiting compression test + verdict
    "accepted",             # passed test + verdict-positive; live in retrieval
    "rejected",             # /reject or failed compression test
    "rejected_corrected",   # /correct provided replacement; this row retiring
    "stale",                # source drift detected; awaiting re-verification
    "retired",              # final state; excluded from retrieval permanently
]

VerdictSource = Literal["user", "llm", "mechanical"]
VerdictPolarity = Literal["accept", "reject", "neutral"]

# State-set constants — single source of truth for membership checks
# across retrieval, lifecycle, and verdict routing.
RETRIEVABLE_STATES: frozenset[VerdictState] = frozenset({"accepted"})
# Final / absorbing states: rejected, rejected_corrected, retired never
# leave their state regardless of subsequent signals, and never surface
# in retrieval.
FINAL_STATES: frozenset[VerdictState] = frozenset({
    "rejected", "rejected_corrected", "retired",
})
# Pending states: surface only with --include-stale; on the path to
# accepted or retired but not yet authoritative.
PENDING_STATES: frozenset[VerdictState] = frozenset({"candidate", "stale"})


# ----------------------------------------------------------------------------
# Component dataclasses (frozen — file representation is immutable per version)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class InsightCitation:
    """Back-reference from an insight passage to a source passage.

    `passage_id` matches the stable id in `passages.jsonl` (see
    `store.registry.compute_id`). `char_span` is optional; when present it
    narrows the citation to a sub-range within the source passage's
    char_offset..char_offset+char_length window. `confidence` is the LLM's
    judgement of how strongly this source supports the insight (0..1).
    """
    passage_id: str
    char_span: tuple[int, int] | None
    confidence: float


@dataclass(frozen=True)
class VerdictSignal:
    """One verdict event tied to an insight passage.

    Appended to `InsightPassage.verdict_signals` whenever `/accept`,
    `/reject`, `/correct`, or a mechanical PostToolUse signal references
    this insight. `lens_id` records which lens emitted the verdict —
    cross-lens verdict consensus is what gates promotion from lens-private
    to shared insight.
    """
    source: VerdictSource
    polarity: VerdictPolarity
    timestamp: str          # ISO 8601 UTC
    lens_id: str | None     # None = workspace-default scope


# ----------------------------------------------------------------------------
# Main insight passage
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class InsightPassage:
    """One insight passage — earned, cited derived content.

    `insight_idx` is implied by line position in `insight.jsonl`; not stored.
    `insight_id` is a stable content-derived id that survives
    refresh/sync deltas (mirrors source `passage_id` semantics).
    """
    insight_idx: int
    insight_id: str
    kind: InsightKind
    content: str
    citations: tuple[InsightCitation, ...]
    query: str | None
    generated_at: str
    source_model_hash: str
    source_passage_hashes: tuple[str, ...]
    verdict_state: VerdictState
    verdict_signals: tuple[VerdictSignal, ...]
    lineage: tuple[str, ...]
    intent_context: str | None
    stale_if_sources_drift: bool
    encoder_version: str
    # Beta-accumulation tallies. `confidence` is derived from them, never
    # stored — see the `confidence` property. Defaulted so bare fixtures
    # still build; the JSONL loader seeds legacy rows from stored confidence.
    corroboration: float = 0.0
    falsification: float = 0.0

    @property
    def confidence(self) -> float:
        """Beta-mean confidence: corroboration / (corroboration + falsification).

        Derived, single source of truth — the tallies are the state, this
        is the read. An unseeded row (both tallies 0 — only a bare fixture
        built without `seed_confidence`) reads as a neutral 0.5.
        """
        total = self.corroboration + self.falsification
        if total <= 0.0:
            return 0.5
        return self.corroboration / total

    def is_retrievable(self) -> bool:
        """Whether this insight surfaces in default retrieval. `stale`
        rows opt in via `--include-stale`; `retired` and `rejected*` never
        surface."""
        return self.verdict_state in RETRIEVABLE_STATES


# ----------------------------------------------------------------------------
# Stable identity
# ----------------------------------------------------------------------------

def compute_insight_id(
    content: str,
    source_passage_hashes: Iterable[str],
    source_model_hash: str,
) -> str:
    """Stable insight id derived from content + provenance.

    Mirrors `store.registry.compute_id` — a 16-char hex slice of SHA-256
    over the canonicalised inputs. Two synthesis runs that produce
    identical content over identical sources via identical model+prompt
    get the same id (deterministic), which lets the compression test's
    semantic-duplicate guard short-circuit on hash match before doing a
    cosine similarity check.

    Deliberately NOT keyed on `generated_at` or `query` — those are
    provenance metadata, not identity.
    """
    h = hashlib.sha256()
    h.update(content.encode("utf-8", errors="replace"))
    h.update(b"\x1f")
    h.update(source_model_hash.encode("ascii"))
    h.update(b"\x1f")
    for ph in sorted(source_passage_hashes):
        h.update(ph.encode("ascii"))
        h.update(b"\x1e")
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------
# Confidence seeding (Beta prior — docs/internal/GROUNDING_MODEL.md)
# ----------------------------------------------------------------------------

# Prior pseudocounts: a neutral Beta(1, 1) base, plus the faithfulness score
# entering as `_PRIOR_SEED_WEIGHT` of pseudo-evidence — so a faithful insight
# starts modestly positive, not pinned high. Outcome reducers add weight to
# the tallies from there; see insight_lifecycle.accumulate_outcome.
_PRIOR_BASE = 1.0
_PRIOR_SEED_WEIGHT = 2.0


def seed_confidence(faithfulness: float | None) -> tuple[float, float]:
    """Prior `(corroboration, falsification)` pseudocounts for a new insight.

    `faithfulness` is the gate score that admitted the insight (0..1), or
    None when promoted by a path that didn't run the gate (→ neutral).
    """
    f = 0.5 if faithfulness is None else max(0.0, min(1.0, faithfulness))
    corroboration = _PRIOR_BASE + _PRIOR_SEED_WEIGHT * f
    falsification = _PRIOR_BASE + _PRIOR_SEED_WEIGHT * (1.0 - f)
    return corroboration, falsification


# ----------------------------------------------------------------------------
# JSONL I/O
# ----------------------------------------------------------------------------

def _citation_to_dict(c: InsightCitation) -> dict[str, Any]:
    return {
        "passage_id": c.passage_id,
        "char_span": list(c.char_span) if c.char_span is not None else None,
        "confidence": c.confidence,
    }


def _citation_from_dict(d: dict[str, Any]) -> InsightCitation:
    raw_span = d.get("char_span")
    span: tuple[int, int] | None = None
    if raw_span is not None:
        span = (int(raw_span[0]), int(raw_span[1]))
    return InsightCitation(
        passage_id=d["passage_id"],
        char_span=span,
        confidence=float(d.get("confidence", 1.0)),
    )


def _verdict_to_dict(v: VerdictSignal) -> dict[str, Any]:
    return {
        "source": v.source,
        "polarity": v.polarity,
        "timestamp": v.timestamp,
        "lens_id": v.lens_id,
    }


def _verdict_from_dict(d: dict[str, Any]) -> VerdictSignal:
    return VerdictSignal(
        source=d["source"],
        polarity=d["polarity"],
        timestamp=d["timestamp"],
        lens_id=d.get("lens_id"),
    )


def load_jsonl(text_lines: Iterable[str]) -> list[InsightPassage]:
    """Parse insight.jsonl lines into InsightPassage rows.

    `insight_idx` is assigned from line position (matches
    `store.registry.load_jsonl`'s convention). Blank lines are rejected
    rather than skipped because that would renumber later rows and break
    the `(insight_idx ↔ insight band row)` join.

    Raises `json.JSONDecodeError` on malformed input. Missing optional
    fields fall back to safe defaults; missing required fields raise
    `KeyError`.
    """
    rows: list[InsightPassage] = []
    for line in text_lines:
        obj = json.loads(line)
        citations = tuple(
            _citation_from_dict(c) for c in obj.get("citations", [])
        )
        verdict_signals = tuple(
            _verdict_from_dict(v) for v in obj.get("verdict_signals", [])
        )
        corroboration = obj.get("corroboration")
        falsification = obj.get("falsification")
        if corroboration is None or falsification is None:
            # Legacy row predating the Beta model — reconstruct the tallies
            # from the stored confidence so every loaded row has a valid
            # Beta state. The `confidence` property derives from them.
            corroboration, falsification = seed_confidence(
                float(obj.get("confidence", 0.5))
            )
        rows.append(InsightPassage(
            insight_idx=len(rows),
            insight_id=obj["id"],
            kind=obj["kind"],
            content=obj["content"],
            citations=citations,
            query=obj.get("query"),
            generated_at=obj["generated_at"],
            source_model_hash=obj["source_model_hash"],
            source_passage_hashes=tuple(obj.get("source_passage_hashes", [])),
            verdict_state=obj.get("verdict_state", "candidate"),
            verdict_signals=verdict_signals,
            lineage=tuple(obj.get("lineage", [])),
            intent_context=obj.get("intent_context"),
            stale_if_sources_drift=bool(obj.get("stale_if_sources_drift", True)),
            encoder_version=obj.get("encoder_version", ""),
            corroboration=float(corroboration),
            falsification=float(falsification),
        ))
    return rows


def write_jsonl(rows: list[InsightPassage]) -> str:
    """Serialise to JSONL in `insight_idx` order, one row per line.

    `insight_idx` is omitted (recoverable from line position); `insight_id`
    is emitted under key `id` for diff-stable compactness, mirroring
    `store.registry.write_jsonl`. Raises if the list isn't in contiguous
    0..N order — the band row join depends on it.
    """
    for i, r in enumerate(rows):
        if r.insight_idx != i:
            raise ValueError(
                f"InsightPassage list must be in contiguous insight_idx order; "
                f"position {i} has insight_idx={r.insight_idx}"
            )
    parts: list[str] = []
    for r in rows:
        d: dict[str, Any] = {
            "id": r.insight_id,
            "kind": r.kind,
            "content": r.content,
            "citations": [_citation_to_dict(c) for c in r.citations],
            "query": r.query,
            "generated_at": r.generated_at,
            "source_model_hash": r.source_model_hash,
            "source_passage_hashes": list(r.source_passage_hashes),
            "verdict_state": r.verdict_state,
            "verdict_signals": [_verdict_to_dict(v) for v in r.verdict_signals],
            "confidence": r.confidence,
            "corroboration": r.corroboration,
            "falsification": r.falsification,
            "lineage": list(r.lineage),
            "intent_context": r.intent_context,
            "stale_if_sources_drift": r.stale_if_sources_drift,
            "encoder_version": r.encoder_version,
        }
        parts.append(json.dumps(d, sort_keys=True))
    return "\n".join(parts)


# ----------------------------------------------------------------------------
# Verdict / state-transition helpers
# ----------------------------------------------------------------------------

def append_verdict(
    insight: InsightPassage, signal: VerdictSignal,
) -> InsightPassage:
    """Return a new InsightPassage with `signal` appended to verdict_signals.

    Pure — does not mutate the input. State transitions are NOT computed
    here (state machine lives in the verdict-lifecycle module, Day 2). This
    helper exists so the verdict-routing path can stay one-liner clear:
    `insight = append_verdict(insight, signal)`.
    """
    from dataclasses import replace
    return replace(insight, verdict_signals=insight.verdict_signals + (signal,))


def mark_stale(insight: InsightPassage) -> InsightPassage:
    """Flip verdict_state to 'stale'. Used by the drift-cascade pass when
    any cited source passage's content_hash changes (Day 2 wiring)."""
    from dataclasses import replace
    return replace(insight, verdict_state="stale")
