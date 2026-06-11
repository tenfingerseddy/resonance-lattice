"""Corpus claim ↔ JSONL serialisation, factory, and legacy migration.

Corpus claims (`source="corpus"`, `facts` a `CorpusFacts`) are earned,
cited compressions of the source layer. They live *inside* one `.rlat`
archive: `insight.jsonl` (a flattened `Claim`+`CorpusFacts` JSONL) plus
the parallel `bands/insight.npz`. The archive read/write path
(`store.archive`) and the corpus lifecycle (`promotion`,
`insight_lifecycle`, `reverification`) operate on raw `list[Claim]` from
`ArchiveContents.insights`; this module owns the serialisation, the
`new_corpus_claim` factory, and the one-shot on-read migration of a
pre-Stage-5 `InsightPassage`-shaped `insight.jsonl`.

The core-field serialisation lives in `state/claim_io.py`, shared with
`ExperienceClaimStore`.

**Unified band (S3).** The band-seam functions `claims_to_jsonl` /
`rows_to_claims` are *source-discriminated*: a corpus claim serialises via the
`CorpusFacts` path here; an experience claim serialises via the shared
`experience_claim_*` helpers on the spine. One band, both sources — the
discriminator on read is the row's `source` core field. The corpus
`_claim_to_row` / `_row_to_claim` stay corpus-only; `_claim_to_row` still
raises on a non-`CorpusFacts` claim — a defensive invariant for the corpus
branch.
"""

from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime, timezone
from typing import Iterable

from ..state.claim import Claim, CorpusFacts
from ..state.claim_io import (
    core_from_row,
    core_to_row,
    experience_claim_from_row,
    experience_claim_to_row,
)
from .insight import (
    InsightCitation,
    _citation_from_dict,
    _citation_to_dict,
    _verdict_from_dict,
    _verdict_to_dict,
    compute_insight_id,
    provenance_tier,
    seed_confidence,
)

_FACTS_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(CorpusFacts))

# Plain tuple-typed `CorpusFacts` fields — JSON has no tuple; these
# serialise as lists. (`citations` and `verdict_signals` are nested-tuple
# fields whose elements need per-element (de)serialisers; the core
# tuple-typed field set lives in `claim_io._TUPLE_CORE`.)
_TUPLE_FACTS_PLAIN: frozenset[str] = frozenset({"source_passage_hashes"})

# `CorpusFacts` fields with a default that a pre-S3-idempotency row may omit —
# read them with `.get` (falling back to the dataclass default) instead of the
# strict `row[name]` the position-keyed band join uses for required fields.
_OPTIONAL_FACTS: dict[str, object] = {
    "seed_corroboration": -1.0,
    "seed_falsification": -1.0,
}

# Legacy `verdict_state` → `ClaimState`. A reject and a correction both
# land in `retired`; the distinction is carried by `parent_ids` (a
# correction's replacement names the retired original) plus the preserved
# `verdict_signals`. Used only by the one-shot legacy migration.
_VERDICT_STATE_TO_CLAIM_STATE: dict[str, str] = {
    "candidate": "candidate",
    "accepted": "active",
    "stale": "stale",
    "rejected": "retired",
    "rejected_corrected": "retired",
    "retired": "retired",
}


# ---------------------------------------------------------------------------
# Claim ↔ JSONL serialisers (also imported by `archive`)
# ---------------------------------------------------------------------------


def _claim_to_row(claim: Claim) -> dict:
    """Flatten a corpus `Claim` (core + `CorpusFacts`) to one JSON-ready dict.

    The two nested-tuple `facts` fields — `citations` and `verdict_signals`
    — serialise via the shared `insight` (de)serialisers."""
    facts = claim.facts
    if not isinstance(facts, CorpusFacts):
        raise TypeError(
            f"corpus_claim_io serialises corpus claims only; got facts "
            f"{type(facts).__name__}"
        )
    row = core_to_row(claim)
    for name in _FACTS_FIELDS:
        value = getattr(facts, name)
        if name == "citations":
            row[name] = [_citation_to_dict(c) for c in value]
        elif name == "verdict_signals":
            row[name] = [_verdict_to_dict(v) for v in value]
        elif name in _TUPLE_FACTS_PLAIN:
            row[name] = list(value)
        else:
            row[name] = value
    return row


def _row_to_claim(row: dict) -> Claim:
    """Rebuild a corpus `Claim` from one flattened Stage-5 row dict."""
    facts_kwargs: dict = {}
    for name in _FACTS_FIELDS:
        if name in _OPTIONAL_FACTS:
            facts_kwargs[name] = row.get(name, _OPTIONAL_FACTS[name])
            continue
        value = row[name]
        if name == "citations":
            facts_kwargs[name] = tuple(_citation_from_dict(c) for c in value)
        elif name == "verdict_signals":
            facts_kwargs[name] = tuple(_verdict_from_dict(v) for v in value)
        elif name in _TUPLE_FACTS_PLAIN:
            facts_kwargs[name] = tuple(value)
        else:
            facts_kwargs[name] = value
    return Claim(facts=CorpusFacts(**facts_kwargs), **core_from_row(row))


def _migrate_legacy_row(d: dict) -> Claim:
    """Reshape one pre-Stage-5 `InsightPassage`-shaped `insight.jsonl` row
    into a corpus `Claim`. Detected by the caller via `"verdict_state" in d`.

    `claim_id` keeps the old content-derived `insight_id` so attribution
    and lens references stay valid (the ULID rekey is Stage 6); the same
    value is the `content_fingerprint`. The old `lineage` is claim→claim
    provenance — it becomes `parent_ids`."""
    corroboration = d.get("corroboration")
    falsification = d.get("falsification")
    if corroboration is None or falsification is None:
        # Row predating the Beta model — reconstruct tallies from the
        # stored confidence (mirrors the deleted `insight.load_jsonl`).
        corroboration, falsification = seed_confidence(
            float(d.get("confidence", 0.5))
        )
    insight_id = d["id"]
    return Claim(
        claim_id=insight_id,
        source="corpus",
        kind=d["kind"],
        content=d["content"],
        created_at=d["generated_at"],
        corroboration=float(corroboration),
        falsification=float(falsification),
        trust_as_of="",
        state=_VERDICT_STATE_TO_CLAIM_STATE[d.get("verdict_state", "candidate")],
        parent_ids=tuple(d.get("lineage", [])),
        facts=CorpusFacts(
            citations=tuple(
                _citation_from_dict(c) for c in d.get("citations", [])
            ),
            content_fingerprint=insight_id,
            source_model_hash=d["source_model_hash"],
            source_passage_hashes=tuple(d.get("source_passage_hashes", [])),
            verdict_signals=tuple(
                _verdict_from_dict(v) for v in d.get("verdict_signals", [])
            ),
            query=d.get("query"),
            intent_context=d.get("intent_context"),
            stale_if_sources_drift=bool(d.get("stale_if_sources_drift", True)),
            encoder_version=d.get("encoder_version", ""),
            # Treat the migration point as the born baseline so the attribution
            # apply re-derives idempotently — a migrated `active` claim left
            # unseeded would fall back to additive and re-fold the ledger on
            # every consolidate (the original §B drift, for legacy archives).
            seed_corroboration=float(corroboration),
            seed_falsification=float(falsification),
        ),
    )


def rows_to_claims(text_lines: Iterable[str]) -> list[Claim]:
    """Parse `insight.jsonl` lines into `Claim`s — corpus *or* experience.

    Source-discriminated (S3): a pre-Stage-5 row (an `InsightPassage`,
    detected by `"verdict_state"`) is always corpus and migrated in place; a
    row tagged `source="experience"` is rebuilt via the shared experience
    helper; everything else is a corpus row. Blank lines are not skipped —
    they would desync the position-keyed band-row join, so a malformed input
    raises rather than silently renumbering."""
    claims: list[Claim] = []
    for line in text_lines:
        d = json.loads(line)
        if "verdict_state" in d:
            claims.append(_migrate_legacy_row(d))
        elif d.get("source") == "experience":
            claims.append(experience_claim_from_row(d))
        else:
            claims.append(_row_to_claim(d))
    return claims


def claims_to_jsonl(claims: list[Claim]) -> str:
    """Serialise `Claim`s to `insight.jsonl` text, one row per line, in list
    order — the band-row join is positional.

    Source-discriminated (S3): an experience claim serialises via the shared
    `experience_claim_to_row`; a corpus claim via `_claim_to_row` (which still
    raises on a non-`CorpusFacts` claim — the corpus-branch invariant)."""
    return "\n".join(
        json.dumps(
            experience_claim_to_row(c) if c.source == "experience"
            else _claim_to_row(c),
            sort_keys=True,
        )
        for c in claims
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def new_corpus_claim(
    *,
    content: str,
    kind: str,
    citations: tuple[InsightCitation, ...],
    source_model_hash: str,
    source_passage_hashes: tuple[str, ...],
    faithfulness: float | None = None,
    query: str | None = None,
    intent_context: str | None = None,
    stale_if_sources_drift: bool = True,
    encoder_version: str = "",
    parent_ids: tuple[str, ...] = (),
    state: str = "candidate",
    provenance: str | None = None,
) -> Claim:
    """Mint a fresh corpus `Claim` — the one place the `source="corpus"` /
    `content_fingerprint` / Beta-seed / `created_at` plumbing lives.

    `claim_id` is the content-derived `content_fingerprint`
    (`compute_insight_id`) — stable across rebuilds, so a portable lens
    can key preferences on it. `faithfulness` seeds the Beta prior via
    `seed_confidence`. `provenance` lifts that seed by source tier (user >
    verified-external > single-external > corpus); None auto-derives the
    tier from the citations (`provenance_tier`), so an external fill seeds
    higher trust than a corpus synthesis without the caller saying so. A
    freshly-synthesised claim is born `candidate`; the lifecycle spine
    (`claim_lifecycle.consolidate_corpus`) commits the candidate→active
    transition once the compression test passes."""
    fingerprint = compute_insight_id(
        content, source_passage_hashes, source_model_hash
    )
    tier = provenance if provenance is not None else provenance_tier(citations)
    corroboration, falsification = seed_confidence(faithfulness, provenance=tier)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return Claim(
        claim_id=fingerprint,
        source="corpus",
        kind=kind,
        content=content,
        created_at=created_at,
        corroboration=corroboration,
        falsification=falsification,
        trust_as_of="",
        state=state,
        parent_ids=tuple(parent_ids),
        facts=CorpusFacts(
            citations=tuple(citations),
            content_fingerprint=fingerprint,
            source_model_hash=source_model_hash,
            source_passage_hashes=tuple(source_passage_hashes),
            verdict_signals=(),
            query=query,
            intent_context=intent_context,
            stale_if_sources_drift=stale_if_sources_drift,
            encoder_version=encoder_version,
            # Record the born prior so the attribution apply can re-derive the
            # absolute tally idempotently (§B): seed == the initial tally here.
            seed_corroboration=corroboration,
            seed_falsification=falsification,
        ),
    )
