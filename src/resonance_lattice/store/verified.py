"""Verified-retrieval surface (WS3 #292).

Every search hit carries: source_file, char_offset, char_length, content_hash,
drift_status. Users can pass --verified-only to filter out drifted chunks.

`DriftStatus` and `compute_hash` are owned by `store.base` (single source of
truth — `Store.verify` consumes both); re-exported here so callers can keep
importing from `store.verified` without knowing the internal split.

Drift status:
- "verified" — content_hash matches authoritative source.
- "drifted"  — source exists but hash mismatches.
- "missing"  — source file no longer exists.

InsightHit is the sibling type for the lensed-knowledge insight layer. The
two stay structurally distinct on purpose (trust-contract foundation 5:
source and insight must be visibly different at every output surface).
Code that consumes hits dispatches on type / `layer` field.

Phase 2 deliverable. Ports the v0.11 surface verbatim. Day 1 of the
lensed-knowledge build adds InsightHit + composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..state.claim import FINAL_STATES, PENDING_STATES, Claim, ClaimKind, ClaimState
from .base import DriftStatus, RemoteShaMismatch, Store, compute_hash
from .insight import InsightCitation, confidence_factor
from .registry import PassageCoord

__all__ = [
    "AttributeServeHit",
    "BandRecallHit",
    "ConstraintServeHit",
    "DriftStatus",
    "InsightHit",
    "VerifiedHit",
    "compute_hash",
    "filter_verified",
    "rank_insight_band",
    "serve_band_attributes",
    "serve_band_constraints",
    "superseded_experience_rows",
    "verify_hits",
    "verify_insight_hits",
]


@dataclass(frozen=True)
class VerifiedHit:
    passage_idx: int
    source_file: str
    char_offset: int
    char_length: int
    content_hash: str
    drift_status: DriftStatus
    score: float
    text: str
    # Per-row business key — present only for row-mode (semantic-slicer)
    # knowledge models; None for ordinary chunked corpora. Carries the
    # caller's domain identity (e.g. an Airbnb listing id) so a slicer
    # consumer can build a TREATAS key set straight from the hit list.
    key: str | None = None
    # Layer discriminator — kept on both hit types so a mixed list of
    # SourceHit-shaped VerifiedHits and InsightHits can be filtered by
    # `[h for h in hits if h.layer == "source"]` without isinstance checks.
    layer: Literal["source"] = "source"


# Corpus-claim `state` → display-layer drift_status mapping. Source-only
# rendering paths can treat InsightHit.drift_status uniformly with VerifiedHit's
# even though the underlying semantics differ.
_INSIGHT_STATE_TO_DRIFT: dict[ClaimState, DriftStatus] = {
    "active": "verified",
    "stale": "drifted",
    "candidate": "drifted",
    "retired": "missing",
}


@dataclass(frozen=True)
class InsightHit:
    """One insight-layer retrieval hit. Sibling of VerifiedHit.

    `insight_idx` joins the insight band to the row. `claim_id` is the
    corpus claim's identity; `content_fingerprint` is the stable
    content-derived key a portable lens keys preferences on.

    `state` is the canonical `ClaimState`; `drift_status` is derived from
    it via `_INSIGHT_STATE_TO_DRIFT` so callers can apply `filter_verified`
    uniformly across source and insight hits.

    `score` is a *rank score*, not a raw cosine: it is the cosine scaled
    by a confidence factor (see `verify_insight_hits`), so it can exceed
    1.0 for a strongly-corroborated claim. `confidence` carries the
    unmodulated Beta mean (`Claim.trust`) for display + decomposition.
    """
    insight_idx: int
    claim_id: str
    content_fingerprint: str
    kind: ClaimKind
    content: str
    citations: tuple[InsightCitation, ...]
    source_passage_hashes: tuple[str, ...]
    state: ClaimState
    confidence: float
    created_at: str
    intent_context: str | None
    score: float
    layer: Literal["insight"] = "insight"

    @property
    def drift_status(self) -> DriftStatus:
        return _INSIGHT_STATE_TO_DRIFT[self.state]


def filter_verified(hits: "list[VerifiedHit] | list[InsightHit] | list") -> list:
    """Drop drifted/missing hits across either source or insight layers.

    Mixed lists are supported — VerifiedHit and InsightHit both carry
    `drift_status` (InsightHit's is derived from verdict_state) so the
    filter is uniform.
    """
    return [h for h in hits if h.drift_status == "verified"]


def verify_hits(
    hits: list[tuple[int, float]],
    store: Store,
    registry: list[PassageCoord],
) -> list[VerifiedHit]:
    """Resolve `(passage_idx, score)` hits into `VerifiedHit`s with drift status.

    Looks up each hit's coordinate via `registry[passage_idx]`, calls
    `store.verify` to detect drift against the build-time content_hash, and
    `store.fetch` to retrieve the current authoritative text — skipped on
    `"missing"` and `"drifted"` status (text becomes empty), since fetch
    would raise `FileNotFoundError` for a missing source and
    `RemoteShaMismatch` for a remote-mode SHA-pin mismatch. The base
    class's per-instance text cache makes verify+fetch amortise to a single
    full-file read per source file regardless of hit count.

    `RemoteShaMismatch` from `store.fetch` is *also* caught explicitly
    (and demoted to drifted+empty) for the case where `verify` returned
    "verified" against the recorded `content_hash` but the SHA-pin guard
    fired during the actual byte fetch — that combination is rare but
    representable, e.g. a remote-mode entry whose recorded passage hash
    matches the cached bytes but whose cached bytes' file-level SHA no
    longer matches the manifest pin (cache corruption). Without this
    catch, `rlat search`, `rlat skill-context`, and `rql.evidence` would
    crash on remote drift instead of surfacing it as a drift status.

    Output is in input order; sorting and `--verified-only` filtering are
    the caller's job (see `filter_verified`).

    Raises `IndexError` if any `passage_idx` is out of range — that means
    the caller passed hits from a different knowledge model's registry,
    which is a programming error, not a runtime drift.
    """
    out: list[VerifiedHit] = []
    for passage_idx, score in hits:
        coord = registry[passage_idx]
        drift_status = store.verify(
            coord.source_file,
            coord.char_offset,
            coord.char_length,
            coord.content_hash,
        )
        if drift_status == "missing" or drift_status == "drifted":
            # Skip fetch on missing (would FileNotFoundError) and on drifted
            # (text doesn't match recorded content_hash; returning the live
            # bytes would be a footgun for citation paths). Empty text keeps
            # the row in output so callers see the drift status rather than
            # silently dropped hits.
            text = ""
        else:
            try:
                text = store.fetch(
                    coord.source_file,
                    coord.char_offset,
                    coord.char_length,
                )
            except RemoteShaMismatch:
                # Remote SHA-pin mismatch fired during fetch (e.g. cache
                # corruption surfaced after `verify` returned verified
                # against the per-passage hash). Demote to drifted+empty
                # rather than crashing the retrieval path.
                drift_status = "drifted"
                text = ""
        out.append(VerifiedHit(
            passage_idx=passage_idx,
            source_file=coord.source_file,
            char_offset=coord.char_offset,
            char_length=coord.char_length,
            content_hash=coord.content_hash,
            drift_status=drift_status,
            score=float(score),
            text=text,
            key=coord.key,
        ))
    return out


def superseded_experience_rows(insights: list[Claim]) -> frozenset[int]:
    """Band indices of keyed experience world rows superseded by a newer value.

    The serve-time newest-wins rule every render path shares: per
    (`kind`, `facts.attribute_key`) group of retrievable EXPERIENCE claims,
    only the newest (`created_at`, then append order) serves; the rest are
    superseded — still on disk, never rendered. Unkeyed rows ("") are never
    superseded (a keyless fact can't suppress another); corpus rows never
    participate. Pure; no I/O.
    """
    newest: dict[tuple[str, str], int] = {}
    losers: set[int] = set()
    for idx, claim in enumerate(insights):
        if claim.source != "experience" or not claim.is_retrievable():
            continue
        key = claim.facts.attribute_key
        if not key:
            continue
        group = (claim.kind, key)
        prior = newest.get(group)
        if prior is None:
            newest[group] = idx
        elif (claim.created_at, idx) > (insights[prior].created_at, prior):
            losers.add(prior)
            newest[group] = idx
        else:
            losers.add(idx)
    return frozenset(losers)


def verify_insight_hits(
    hits: list[tuple[int, float]],
    insights: list[Claim],
    *,
    include_stale: bool = False,
) -> list[InsightHit]:
    """Resolve `(insight_idx, score)` hits into `InsightHit`s — both band
    sources (corpus claims AND experience world claims).

    The claim's `state` drives the derived `drift_status` for
    uniform rendering: active → verified, stale/candidate → drifted,
    retired → missing. `include_stale=False` (the default) filters
    stale/candidate rows out silently — the same way `--verified-only`
    filters drifted source hits.

    Retired rows are always excluded regardless of `include_stale` —
    a final state with no path back into retrieval.

    `InsightHit.score` is the raw cosine scaled by a confidence factor
    (docs/internal/GROUNDING_MODEL.md §"What confidence does") — a
    corroborated claim floats above a provisional one at similar
    relevance. The result is sorted by that modulated score, so callers no
    longer get input order. The unmodulated Beta mean (`Claim.trust`)
    stays on `InsightHit.confidence` for display + decomposition.

    Raises `IndexError` if any `insight_idx` is out of range — programming
    error, same as `verify_hits`.
    """
    out: list[InsightHit] = []
    superseded = superseded_experience_rows(insights)
    for insight_idx, score in hits:
        row = insights[insight_idx]
        if row.state in FINAL_STATES:
            continue
        if row.state in PENDING_STATES and not include_stale:
            continue
        # Serve-time recency filter: a keyed world value with a newer
        # sibling never renders (same newest-wins rule the framed
        # serve channels apply) — kept on disk, dropped from serve.
        if insight_idx in superseded:
            continue
        # Both band sources render. A corpus claim carries citations and a
        # content_fingerprint; an experience claim (ExperienceFacts — a world
        # attribute, constraint, or falsified finding) has neither, so its
        # corpus-only fields render empty and its `kind` carries the meaning.
        # Before v3 the experience rows were skipped here, which made band
        # claims invisible to `rlat search` (v3 S2 fixed that).
        is_corpus = row.source == "corpus"
        out.append(InsightHit(
            insight_idx=insight_idx,
            claim_id=row.claim_id,
            content_fingerprint=row.facts.content_fingerprint if is_corpus else "",
            kind=row.kind,
            content=row.content,
            citations=row.facts.citations if is_corpus else (),
            source_passage_hashes=(
                row.facts.source_passage_hashes if is_corpus else ()),
            state=row.state,
            confidence=row.trust,
            created_at=row.created_at,
            intent_context=row.facts.intent_context if is_corpus else None,
            score=float(score) * confidence_factor(row.trust),
        ))
    out.sort(key=lambda h: h.score, reverse=True)
    return out


@dataclass(frozen=True)
class BandRecallHit:
    """One ranked insight-band claim for recall-time attribution.

    Source-agnostic: built from a claim's CORE fields only (`claim_id`,
    `source`, `state`, `trust`), never `facts`, so it is safe over a band
    holding either source. `rank` is the 0-based position in THIS ranking — it
    seeds the attribution tier (`state.attribution._tier_for_rank`), so it must
    be the corpus ranking's own 0-based index, never appended to the experience
    hits' ranks. `cosine` is the raw similarity; `score` is the trust-modulated
    ordering key.
    """

    claim_id: str
    source: str
    rank: int
    cosine: float
    score: float


def rank_insight_band(
    query_emb: np.ndarray,
    insights: list[Claim],
    insight_band: np.ndarray,
    *,
    top_k: int,
    cosine_floor: float = 0.0,
) -> list[BandRecallHit]:
    """Rank the insight band for recall-time attribution — source-agnostic.

    Cosine top-k over the band; keep only **retrievable** (`active`) claims —
    `candidate`/`stale`/`retired` are never surfaced into attribution, so trust
    cannot move on a non-retrievable claim; score each
    `cosine × confidence_factor(trust)` (the same modulation
    `verify_insight_hits` uses); sort by score; take `top_k`; stamp each with
    its OWN 0-based `rank`.

    The cosine FLOOR is applied to the RAW cosine (before modulation). Reads
    only core `Claim` fields (`claim_id`, `source`, `is_retrievable()`,
    `trust`) — never `facts` — so a mixed-source band cannot AttributeError
    here. Pure; no I/O.
    """
    from ..field import retrieve_insight

    if insight_band is None or insight_band.shape[0] == 0 or not insights:
        return []
    # Rank the whole band, then floor + active-filter + truncate — so the
    # top-k is taken AFTER non-retrievable / sub-floor claims are dropped, not
    # before. The insight band is small (earned claims only), so a full rank is
    # cheap. No ANN index on the light-read path → dense cosine.
    raw = retrieve_insight(
        query_emb, insight_band, None, top_k=insight_band.shape[0],
    )
    ranked: list[tuple[Claim, float, float]] = []
    for idx, cos in raw:
        if cos < cosine_floor:
            continue
        claim = insights[idx]
        if not claim.is_retrievable():
            continue
        ranked.append((claim, float(cos), float(cos) * confidence_factor(claim.trust)))
    ranked.sort(key=lambda t: t[2], reverse=True)
    return [
        BandRecallHit(
            claim_id=c.claim_id, source=c.source, rank=i, cosine=cos, score=score,
        )
        for i, (c, cos, score) in enumerate(ranked[:top_k])
    ]


@dataclass(frozen=True)
class AttributeServeHit:
    """One user-world ATTRIBUTE claim selected for serving into the agent's context.

    The content-bearing serve channel: an `attribute`-kind experience claim
    carries a plain world fact (SKU/role/version/corpus size), served as its
    own `content` for prompt injection. `attribute_key` is the normalized
    subject the newest-wins dedup grouped by ("" = unkeyed, never deduped);
    `created_at` is the immutable mint time that decided newest; `score` is the
    trust-modulated cosine that orders the served set.
    """

    content: str
    attribute_key: str
    created_at: str
    score: float


def serve_band_attributes(
    query_emb: np.ndarray,
    insights: list[Claim],
    insight_band: np.ndarray,
    *,
    top_k: int,
    cosine_floor: float = 0.0,
) -> list[AttributeServeHit]:
    """Select user-world ATTRIBUTE claims to serve — newest value per subject.

    The cosine-ranked content-serve sibling to `verify_insight_hits` (the
    ranked render across both band sources): it surfaces `active`
    `attribute`-kind EXPERIENCE claims as their own content. Cosine-ranks the whole band, keeps only retrievable
    attribute claims, then DEDUPS by `facts.attribute_key` — when the band holds
    more than one value for the same changing subject, only the NEWEST
    (`created_at`) is served. The older values stay on disk untouched: this is a
    serve-time filter, never a delete. An unkeyed attribute ("") is its own
    subject and is never deduped (a keyless fact can't suppress another).

    The `cosine_floor` is applied to the served representative's OWN cosine
    AFTER the newest-wins pick — so an older value that out-ranks the current
    one can never be served in its place (the precise bug this fixes), and a
    subject whose newest value is irrelevant to the query drops out entirely
    rather than falling back to a stale value. Scores
    `cosine × confidence_factor(trust)` (the modulation the other band readers
    share), sorts by score, takes `top_k`. Reads `facts` (the attribute key), so
    it cannot live in source-agnostic `rank_insight_band`. Pure; no I/O.
    """
    from ..field import retrieve_insight

    if insight_band is None or insight_band.shape[0] == 0 or not insights:
        return []
    raw = retrieve_insight(
        query_emb, insight_band, None, top_k=insight_band.shape[0],
    )
    # Newest-wins dedup keyed by attribute_key. Unkeyed ("") claims are never
    # grouped — each survives as its own hit. Pick by created_at (ISO strings
    # sort chronologically); on an exact same-second tie, the claim appended
    # LATER — higher band index, since capture only ever appends — is the more
    # recent capture, so the tie-break is band index, never cosine. The choice
    # never depends on similarity, so an older value out-ranking the newer can't
    # win the slot.
    best: dict[str, tuple[Claim, float, int]] = {}
    unkeyed: list[tuple[Claim, float]] = []
    for idx, cos in raw:
        claim = insights[idx]
        if claim.source != "experience" or claim.kind != "attribute":
            continue
        if not claim.is_retrievable():
            continue
        key = claim.facts.attribute_key
        if not key:
            unkeyed.append((claim, float(cos)))
            continue
        prior = best.get(key)
        if prior is None or (claim.created_at, idx) > (prior[0].created_at, prior[2]):
            best[key] = (claim, float(cos), idx)
    hits: list[AttributeServeHit] = []
    kept = [(c, cos) for c, cos, _ in best.values()] + unkeyed
    for claim, cos in kept:
        if cos < cosine_floor:
            continue
        hits.append(AttributeServeHit(
            content=claim.content,
            attribute_key=claim.facts.attribute_key,
            created_at=claim.created_at,
            score=cos * confidence_factor(claim.trust),
        ))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


@dataclass(frozen=True)
class ConstraintServeHit:
    """One standing-constraint or tried-and-falsified claim selected for
    serving. No score: the serve is ALL-always (see `serve_band_constraints`),
    so there is no ranking to carry. `kind` ("constraint" | "negation") picks
    the framed section (`store.serve_framing`); `attribute_key` is the subject
    the newest-wins dedup grouped by; `created_at` decided newest; `claim_id`
    keeps the served row attributable (lens / future outcome accounting).
    """

    claim_id: str
    content: str
    kind: str
    attribute_key: str
    created_at: str


def serve_band_constraints(insights: list[Claim]) -> list[ConstraintServeHit]:
    """Serve ALL standing constraints + falsified findings — no retrieval gate.

    The R1-proven design: constraints were served all-always (no cosine floor,
    no top-k, no selection) and measured zero over-blocking on collateral
    questions — a hard rule of the world applies whether or not the query is
    about it, so relevance ranking is the wrong filter. Query-independent on
    purpose: callers need no embedding.

    Keeps retrievable (`active`) EXPERIENCE claims of kind `constraint` or
    `negation`; drops values superseded by a newer sibling
    (`superseded_experience_rows` — the shared newest-wins rule, unkeyed
    ("") claims never deduped); returns them in band (capture) order.
    Pure; no I/O.
    """
    superseded = superseded_experience_rows(insights)
    return [
        ConstraintServeHit(
            claim_id=c.claim_id,
            content=c.content,
            kind=c.kind,
            attribute_key=c.facts.attribute_key,
            created_at=c.created_at,
        )
        for idx, c in enumerate(insights)
        if (c.source == "experience" and c.kind in ("constraint", "negation")
            and c.is_retrievable() and idx not in superseded)
    ]
