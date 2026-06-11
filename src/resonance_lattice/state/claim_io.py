"""Shared serialisation and merge plumbing for claim storage.

`core_to_row` / `core_from_row` flatten and rebuild a `Claim`'s core
fields (everything but `facts`) for JSONL persistence; both the
experience store (`memory.claim_store.ExperienceClaimStore`) and the
corpus serialisers (`store.corpus_claim_io`) wrap their own `facts`
layer around these helpers.

`merge_claims` / `delete_claims` are the read-merge-write bodies the
experience store wraps under its portalocker lock. The encoder is
passed in lazily via `encoder_provider` so the call site does not
allocate one for a no-encode batch.
"""

from __future__ import annotations

from dataclasses import MISSING, fields
from typing import Callable

import numpy as np

from ..field._runtime_common import l2_normalize
from ..field.encoder import DIM, Encoder
from .claim import Claim, ExperienceFacts


_CORE_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(Claim) if f.name != "facts"
)
_TUPLE_CORE: frozenset[str] = frozenset({"parent_ids"})

# Core fields carrying a dataclass default — an older row that predates the
# field (e.g. `writer`, added in S1.5) loads with the default instead of
# KeyError-ing. Required (no-default) core fields still demand their key.
_CORE_DEFAULTS: dict[str, object] = {
    f.name: f.default
    for f in fields(Claim)
    if f.name != "facts" and f.default is not MISSING
}


def core_to_row(claim: Claim) -> dict:
    """Flatten a claim's core fields (excluding `facts`) to one JSON-ready
    dict. Tuple-typed core fields serialise as lists; the inverse round-trip
    lives in `core_from_row`."""
    return {
        name: (list(getattr(claim, name))
               if name in _TUPLE_CORE else getattr(claim, name))
        for name in _CORE_FIELDS
    }


def core_from_row(row: dict) -> dict:
    """Read a row's core-field kwargs for `Claim(**...)`. Tuple-typed
    core fields are restored from their list serialisation. A core field
    absent from the row falls back to its dataclass default when it has one
    (forward/backward compat — e.g. `writer` on pre-S1.5 rows); a required
    core field still raises `KeyError` if missing."""
    out: dict = {}
    for name in _CORE_FIELDS:
        if name not in row and name in _CORE_DEFAULTS:
            out[name] = _CORE_DEFAULTS[name]
        elif name in _TUPLE_CORE:
            out[name] = tuple(row[name])
        else:
            out[name] = row[name]
    return out


# ---------------------------------------------------------------------------
# Experience-facts row layer — shared by the per-user `ExperienceClaimStore`
# and the unified insight band (S3). Corpus-facts serialisation needs the
# citation/verdict (de)serialisers in `store.insight`, so it stays in
# `store.corpus_claim_io`; the experience facts layer touches only plain
# `ExperienceFacts` fields, so it lives here on the shared spine — importable
# by both `store` and `memory` with no layering cycle.
# ---------------------------------------------------------------------------

_EXPERIENCE_FACTS_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(ExperienceFacts)
)
# `ExperienceFacts` tuple-typed fields — JSON has no tuple, so they serialise
# as lists and restore to tuples on load.
_TUPLE_EXPERIENCE_FACTS: frozenset[str] = frozenset({"polarity"})


def experience_claim_to_row(claim: Claim) -> dict:
    """Flatten an experience `Claim` (core + `ExperienceFacts`) to one row."""
    row = core_to_row(claim)
    for name in _EXPERIENCE_FACTS_FIELDS:
        value = getattr(claim.facts, name)
        row[name] = list(value) if name in _TUPLE_EXPERIENCE_FACTS else value
    return row


def experience_claim_from_row(row: dict) -> Claim:
    """Rebuild an experience `Claim` from one flattened row dict."""
    facts = ExperienceFacts(**{
        name: (tuple(row[name]) if name in _TUPLE_EXPERIENCE_FACTS else row[name])
        for name in _EXPERIENCE_FACTS_FIELDS
        if name in row  # tolerate older rows missing newer defaulted fields (e.g. attribute_key)
    })
    return Claim(facts=facts, **core_from_row(row))


def merge_claims(
    existing: list[Claim],
    band: np.ndarray,
    new_claims: list[Claim],
    *,
    embeddings: np.ndarray | None,
    encoder_provider: Callable[[], Encoder],
) -> tuple[list[Claim], np.ndarray]:
    """Apply an insert-or-replace batch over `(existing, band)`, returning
    the new `(claims, band)`.

    `embeddings` is the optional `(N, DIM)` matrix in `new_claims` order;
    omitted, each claim's band vector is reused if its `content` is
    unchanged and otherwise encoded from `content` in one batch.
    `encoder_provider` is called only when at least one row needs
    encoding — the lazy-allocate path the stores both rely on."""
    if embeddings is not None and embeddings.shape != (len(new_claims), DIM):
        raise ValueError(
            f"embeddings must be ({len(new_claims)}, {DIM}); "
            f"got {embeddings.shape}"
        )
    if len({c.claim_id for c in new_claims}) != len(new_claims):
        raise ValueError("duplicate claim_id in batch")

    index = {c.claim_id: i for i, c in enumerate(existing)}

    vectors: list[np.ndarray | None] = [None] * len(new_claims)
    to_encode: list[int] = []
    for j, claim in enumerate(new_claims):
        if embeddings is not None:
            vectors[j] = embeddings[j]
            continue
        old = index.get(claim.claim_id)
        if old is not None and existing[old].content == claim.content:
            vectors[j] = band[old]
        else:
            to_encode.append(j)
    if to_encode:
        encoder = encoder_provider()
        encoded = encoder.encode([new_claims[j].content for j in to_encode])
        for k, j in enumerate(to_encode):
            vec = encoded[k]
            l2_normalize(vec)
            vectors[j] = vec

    rows = list(existing)
    band_rows = list(band)
    for j, claim in enumerate(new_claims):
        i = index.get(claim.claim_id)
        if i is None:
            index[claim.claim_id] = len(rows)
            rows.append(claim)
            band_rows.append(vectors[j])
        else:
            rows[i] = claim
            band_rows[i] = vectors[j]
    new_band = (
        np.vstack(band_rows) if band_rows
        else np.zeros((0, DIM), dtype=np.float32)
    )
    return rows, new_band


def delete_claims(
    existing: list[Claim],
    band: np.ndarray,
    claim_ids: list[str],
) -> tuple[list[Claim], np.ndarray, int]:
    """Remove `claim_ids` from `(existing, band)`. Returns
    `(kept, kept_band, removed_count)`; `removed_count == 0` is the
    caller's signal to skip the atomic rewrite entirely."""
    targets = set(claim_ids)
    if not targets:
        return existing, band, 0
    keep = [i for i, c in enumerate(existing) if c.claim_id not in targets]
    removed = len(existing) - len(keep)
    if removed == 0:
        return existing, band, 0
    kept = [existing[i] for i in keep]
    new_band = band[keep] if band.size else band
    return kept, new_band, removed
