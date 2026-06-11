"""Corpus-claim primitives — citations, verdict signals, trust math.

The corpus insight layer of a knowledge model is a set of earned, cited
`Claim`s (`source="corpus"`); the record itself is the unified
`state.claim.Claim`, serialised by `store.corpus_claim_io`. This module
owns the small shared primitives both the record and its serialiser
depend on:

  - `InsightCitation` / `VerdictSignal` — the nested-tuple fields a
    `CorpusFacts` carries.
  - `compute_insight_id` — the stable content-derived fingerprint.
  - `seed_confidence` / `beta_mean` / `confidence_band` — the Beta trust
    math shared by experience and corpus claims.
  - the citation / verdict (de)serialisers `corpus_claim_io` uses for
    the nested `CorpusFacts` fields.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Literal

# ----------------------------------------------------------------------------
# Enums (Literal-typed so they cross JSON boundaries as strings)
# ----------------------------------------------------------------------------

VerdictSource = Literal["user", "llm", "mechanical"]
VerdictPolarity = Literal["accept", "reject", "neutral"]


# ----------------------------------------------------------------------------
# Component dataclasses (frozen — file representation is immutable per version)
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class InsightCitation:
    """Back-reference from a corpus claim to its source.

    For a CORPUS source, `passage_id` matches the stable id in `passages.jsonl`
    (see `store.registry.compute_id`) and `source_url` is None. For an EXTERNAL
    source (a verified web/credible-source fill — `external_fill`), `passage_id`
    is a synthetic `external:<hash>` id (not in the corpus registry) and
    `source_url` carries the citable URL. An external claim SERVES via its claim
    STATE like any insight row (`verified.InsightHit.drift_status` derives from
    `active`/`stale`/… — there is no citation-level drift check at retrieval). The
    CORPUS-drift machinery skips external citations (`claim_lifecycle.detect_drift`
    + `reverification`) so a missing registry lookup is not mistaken for a vanished
    source — otherwise every external fill would be falsely staled on the next
    `rlat refresh`. `char_span` optionally narrows a corpus citation to a sub-range.
    `confidence` is how strongly this source supports the claim (0..1).

    `source_url` is optional + defaulted + last so existing claims (corpus-only)
    deserialise unchanged and re-serialise byte-identically (it is omitted from the
    dict when None).
    """
    passage_id: str
    char_span: tuple[int, int] | None
    confidence: float
    source_url: str | None = None

    @property
    def is_external(self) -> bool:
        """A non-corpus citation — carries a URL / synthetic external passage id."""
        return self.source_url is not None or self.passage_id.startswith("external:")


@dataclass(frozen=True)
class VerdictSignal:
    """One verdict event tied to a corpus claim.

    Appended to `CorpusFacts.verdict_signals` whenever `/accept`,
    `/reject`, `/correct`, or a mechanical PostToolUse signal references
    this claim. `lens_id` records which lens emitted the verdict —
    cross-lens verdict consensus is what gates promotion from lens-private
    to shared.
    """
    source: VerdictSource
    polarity: VerdictPolarity
    timestamp: str          # ISO 8601 UTC
    lens_id: str | None     # None = workspace-default scope


# ----------------------------------------------------------------------------
# Stable identity
# ----------------------------------------------------------------------------

def compute_insight_id(
    content: str,
    source_passage_hashes: Iterable[str],
    source_model_hash: str,
) -> str:
    """Stable content-derived id for a corpus claim — its
    `content_fingerprint`, and the `claim_id` of a freshly-minted one.

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
# entering as `_PRIOR_SEED_WEIGHT` of pseudo-evidence — so a faithful claim
# starts modestly positive, not pinned high. Outcome reducers add weight to
# the tallies from there; see insight_lifecycle.accumulate_outcome.
_PRIOR_BASE = 1.0
_PRIOR_SEED_WEIGHT = 2.0


# Provenance-tier prior — a higher-trust SOURCE seeds a higher starting confidence: a user-vouched fact > a
# cross-source-verified external fact > a single web source ≈ a corpus-synthesised claim. Added as extra
# corroboration pseudo-evidence on top of the faithfulness seed. "corpus" (the default) adds nothing, so every
# existing caller is byte-identical. The OUTCOME side already weights provenance (`insight_attribution`); this is
# the missing SEED side — the keystone of the explicit, auditable trust model (user ≥ verified-external ≥
# single-external ≥ corpus). Tuned against the bands (medium 0.40 / high 0.70 / verified 0.76): at faithfulness
# 0.8, corpus seeds ~0.65 (medium), verified-external ~0.72 (high), user ~0.77 (verified) — the ordering the model
# requires, while a fresh fill still earns its final rank by outcomes.
PROVENANCE_TIERS = ("user", "verified_external", "single_external", "corpus")
_PROVENANCE_SEED_BOOST = {
    "user": 2.0,
    "verified_external": 1.0,
    "single_external": 0.0,
    "corpus": 0.0,
}


def seed_confidence(
    faithfulness: float | None, *, provenance: str = "corpus",
) -> tuple[float, float]:
    """Prior `(corroboration, falsification)` pseudocounts for a new corpus
    claim.

    `faithfulness` is the gate score that admitted the claim (0..1), or
    None when promoted by a path that didn't run the gate (→ neutral).
    `provenance` lifts the corroboration prior by source tier (see
    `_PROVENANCE_SEED_BOOST`); the default "corpus" adds nothing, so an
    unset caller is unchanged. An unknown tier is treated as no boost.
    """
    f = 0.5 if faithfulness is None else max(0.0, min(1.0, faithfulness))
    boost = _PROVENANCE_SEED_BOOST.get(provenance, 0.0)
    corroboration = _PRIOR_BASE + _PRIOR_SEED_WEIGHT * f + boost
    falsification = _PRIOR_BASE + _PRIOR_SEED_WEIGHT * (1.0 - f)
    return corroboration, falsification


def all_external(citations) -> bool:
    """True iff `citations` is non-empty and EVERY citation is external (a web/source fill, not a corpus passage).

    The one definition of "this claim is an external fill", shared by provenance, the compression gate, drift
    re-verification, and the world-freshness enumerator (so the predicate can't drift between them)."""
    cits = tuple(citations or ())
    return bool(cits) and all(c.is_external for c in cits)


def provenance_tier(citations) -> str:
    """Derive a claim's default provenance tier from its citations (when not set explicitly).

    All-external citations with ≥ 2 DISTINCT sources → "verified_external" (the cross-source-agreement an external
    fill enforces); a single external source → "single_external"; anything corpus-anchored → "corpus". The "user"
    tier is NEVER inferred — only the user vouching makes a fact user-sourced, so a caller must set it explicitly."""
    cits = tuple(citations or ())
    if all_external(cits):
        distinct = {(c.source_url or c.passage_id) for c in cits}
        return "verified_external" if len(distinct) >= 2 else "single_external"
    return "corpus"


def beta_mean(corroboration: float, falsification: float) -> float:
    """The Beta-distribution mean — `corroboration / (corroboration +
    falsification)`. The single definition of a confidence value; an
    unweighted pair (both 0) reads as the neutral 0.5."""
    total = corroboration + falsification
    return corroboration / total if total > 0.0 else 0.5


# Beta-mean confidence bands — the 4-rung label over `beta_mean`. Tuned so
# the observable calibration holds: 2 clean wins → medium, 3 → high,
# 5 → verified (the significance benchmark's restraint test).
CONFIDENCE_MEDIUM_BAND = 0.40
CONFIDENCE_HIGH_BAND = 0.70
CONFIDENCE_VERIFIED_BAND = 0.76


def confidence_band(trust: float) -> str:
    """Label a Beta-mean `trust` value with its 4-rung confidence — the
    one derived view of trust shared by experience and corpus claims."""
    if trust >= CONFIDENCE_VERIFIED_BAND:
        return "verified"
    if trust >= CONFIDENCE_HIGH_BAND:
        return "high"
    if trust >= CONFIDENCE_MEDIUM_BAND:
        return "medium"
    return "low"


def confidence_factor(trust: float) -> float:
    """Trust → retrieval-score multiplier, centred on the neutral Beta mean.

    A claim with no net evidence (`trust` 0.5) leaves the cosine unchanged;
    corroboration lifts it, falsification sinks it. Centred (`0.5 + trust`)
    rather than a raw `× trust` so a corpus claim stays cross-comparable
    with a source hit in a merged ranking — a raw multiply by [0, 1] would
    sink every claim below every source. Gentle (≈0.8..1.5 across the live
    confidence range) so relevance leads and confidence only breaks
    near-ties. Shared by `store.verified` and the corpus branch of
    `memory.rerank`."""
    return 0.5 + trust


# ----------------------------------------------------------------------------
# Citation / verdict (de)serialisers — used by `corpus_claim_io` for the
# nested-tuple `CorpusFacts` fields.
# ----------------------------------------------------------------------------

def _citation_to_dict(c: InsightCitation) -> dict[str, Any]:
    row: dict[str, Any] = {
        "passage_id": c.passage_id,
        "char_span": list(c.char_span) if c.char_span is not None else None,
        "confidence": c.confidence,
    }
    # Omit when None so corpus-only claims re-serialise byte-identically (forward-
    # compatible: an older reader ignores the key; a newer one reads it back).
    if c.source_url is not None:
        row["source_url"] = c.source_url
    return row


def _citation_from_dict(d: dict[str, Any]) -> InsightCitation:
    raw_span = d.get("char_span")
    span: tuple[int, int] | None = None
    if raw_span is not None:
        span = (int(raw_span[0]), int(raw_span[1]))
    return InsightCitation(
        passage_id=d["passage_id"],
        char_span=span,
        confidence=float(d.get("confidence", 1.0)),
        source_url=d.get("source_url"),  # None for legacy / corpus-only citations
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
