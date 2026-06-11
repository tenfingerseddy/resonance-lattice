"""Single-shot capture of WORLD ATTRIBUTES into a `.rlat` insight band.

The proven band content (the outcome program + R1/R1-X) is a stable attribute of the WORLD the
knowledge model covers — a tenant's capacity or region policy, a garden's water restrictions, a
practice's jurisdiction rule — that no corpus document holds and that lifts answers when served.
Scope contract (Kane, 2026-06-10): the band is shareable, so only facts true for anyone using the
knowledge model belong here; facts about an individual person do not. The passive miner enforces
that via its GATE 4 and drops person-facts; THIS module is also the EXPLICIT capture surface
(`rlat capture-attribute` / `rlat capture-env` on a named archive), where putting a fact into the
file is the user's own deliberate act on a file they chose.

Unlike experience PATTERNS (which earn retrievability through recurrence), an attribute is a
single-mention stable fact: recurrence is structurally dead for attributes (0/259 recur ≥2×
across sessions), so capture is gated by CRITICALITY + TRUST on FIRST mention, not recurrence.

This is the capture half of the world-fact loop: it lands an `attribute`-kind claim physically
inside the archive's insight layer (`insight.jsonl` + `bands/insight.npz`) via the shipped
`write_insight_layer_in_place`, so the no-LLM, cloud-off `retrieve_insight` serve path surfaces
it. Born `active` — a user-asserted world fact is trusted on capture (the user is the authority
on their world), not earned through the recurrence/consolidation gate that patterns pass.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from ..field.encoder import Encoder
from ..state.claim import Claim
from ..store import archive
from .claim_store import new_experience_claim

# criticality → Beta-seed rung (trust the fact carries on capture; attributes have no recurrence
# signal, so this + criticality is the whole importance signal — cf. the memory importance model).
_RUNG_FOR_CRITICALITY = {"low": "low", "normal": "medium", "high": "high", "critical": "high"}


def _l2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype="float32")
    return v / (np.linalg.norm(v) + 1e-9)


# The world-claim kinds this surface mints — the three serve-proven content
# classes. `attribute` = stable world fact; `constraint` = standing hard rule,
# served all-always (`store.verified.serve_band_constraints`); `negation` =
# tried-and-falsified finding — by convention its content CARRIES the evidence
# pointer ("Tried X; falsified by <benchmark/record>"), since the verdict plus
# its receipt is the measured active ingredient (R2).
WORLD_CLAIM_KINDS = ("attribute", "constraint", "negation")


def make_attribute_claim(content: str, *, criticality: str = "high",
                         attribute_key: str = "", kind: str = "attribute") -> Claim:
    """Mint one `active` single-shot world claim (no recurrence gate).

    `attribute_key` is the normalized subject (e.g. "powershell version") the
    serve-time newest-wins dedup groups by; "" leaves the claim un-keyed and so
    never deduped (a keyless fact can't suppress another). `kind` picks the
    world content class (`WORLD_CLAIM_KINDS`) — same mint, same gates."""
    if kind not in WORLD_CLAIM_KINDS:
        raise ValueError(f"kind must be one of {WORLD_CLAIM_KINDS}, got {kind!r}")
    rung = _RUNG_FOR_CRITICALITY.get(criticality, "high")
    claim = new_experience_claim(
        content=content.strip(),
        polarity=("factual",),
        transcript_hash="manual",          # single-shot manual capture → origin="manual"
        kind=kind,
        rung=rung,
        recurrence_count=1,                 # single mention — recurrence does NOT gate attributes
        criticality=criticality,
        attribute_key=attribute_key,
    )
    # Born active: an asserted user attribute is trusted on capture, not earned via the
    # recurrence/consolidation gate that experience PATTERNS must pass to become retrievable.
    return dataclasses.replace(claim, state="active")


def capture_attributes(
    km_path: str | Path,
    contents: list[str],
    *,
    keys: list[str] | None = None,
    criticality: str = "high",
    encoder: Encoder | None = None,
    kind: str = "attribute",
) -> list[Claim]:
    """Append user-world attribute claims to the archive's insight band (single atomic writeback).

    Reads the existing insight layer (so prior insights — corpus claims or earlier attributes — are
    preserved), encodes each attribute in the corpus encoder space (L2-normalised, the band
    contract), appends row-aligned, and writes via the shipped atomic in-place writer. Returns the
    newly-minted claims. Empty `contents` is a no-op.
    """
    # Pair each content with its attribute key BEFORE the empty-strip reindexes,
    # so a dropped (empty) content takes its key with it and the rest stay aligned.
    pairs = [(c.strip(), (keys[i] if keys and i < len(keys) else ""))
             for i, c in enumerate(contents) if c and c.strip()]
    if not pairs:
        return []
    texts = [c for c, _ in pairs]
    enc = encoder or Encoder()
    new_claims = [make_attribute_claim(c, criticality=criticality, attribute_key=k,
                                       kind=kind)
                  for c, k in pairs]
    new_rows = np.vstack([_l2(v) for v in enc.encode_batched(texts, batch_size=8)]).astype("float32")

    existing = archive.read_insight_layer(km_path)
    if existing is not None:
        prior_claims, prior_band = existing
        claims = list(prior_claims) + new_claims
        band = np.vstack([prior_band, new_rows]).astype("float32")
    else:
        claims, band = new_claims, new_rows
    archive.write_insight_layer_in_place(km_path, claims, band)
    return new_claims


def capture_attribute(
    km_path: str | Path, content: str, *, criticality: str = "high",
    attribute_key: str = "", encoder: Encoder | None = None,
    kind: str = "attribute",
) -> Claim:
    """Single-shot capture of ONE world claim on first mention. See `capture_attributes`."""
    out = capture_attributes(km_path, [content], keys=[attribute_key],
                             criticality=criticality, encoder=encoder, kind=kind)
    return out[0]
