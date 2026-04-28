"""metadata.json schema for v4 knowledge models.

Schema per base plan §2.3. Backbone revision is pinned (HF commit hash) so a
knowledge model built at revision A does not silently misbehave if the install
later upgrades to B.

Phase 2 deliverable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# v4 is the on-disk format version. Bumped only on breaking layout changes.
# `Metadata.format_version` defaults to this; `archive.py` imports it for
# version-mismatch checks so there's exactly one source of truth.
FORMAT_VERSION = 4


@dataclass
class BackboneInfo:
    name: str = "Alibaba-NLP/gte-modernbert-base"
    revision: str = ""              # filled at build time
    dim: int = 768
    pool: Literal["cls"] = "cls"
    max_seq_length: int = 8192


BandRole = Literal["retrieval_default", "in_corpus_retrieval"]


@dataclass
class BandInfo:
    role: BandRole
    dim: int                        # 768 for base, 512 for optimised
    l2_norm: bool = True
    passage_count: int = 0
    # Optimised-only:
    dim_native: int | None = None
    w_shape: tuple[int, int] | None = None
    nested_mrl_dims: list[int] | None = None
    trained_from: str | None = None
    # Unknown per-band keys captured by `from_json` for round-trip preservation
    # (forward-compat: future band slots like quantisation / lexical metadata
    # that older readers shouldn't drop or fail on). Not part of the public
    # schema.
    _extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Metadata:
    format_version: int = FORMAT_VERSION
    # `kind` and `store_mode` Literals must stay in sync with the Kind and
    # StoreMode enums in resonance_lattice.config.
    kind: Literal["corpus", "intent"] = "corpus"
    backbone: BackboneInfo = field(default_factory=BackboneInfo)
    bands: dict[str, BandInfo] = field(default_factory=dict)
    store_mode: Literal["bundled", "local", "remote"] = "local"
    ann: dict[str, Any] = field(default_factory=dict)
    build_config: dict[str, Any] = field(default_factory=dict)
    created_utc: str = ""
    rlat_version: str = "2.0.0a1"
    # Unknown top-level keys captured by `from_json` for round-trip preservation
    # (forward-compat for newer rlat versions). Not part of the public schema.
    _extras: dict[str, Any] = field(default_factory=dict)


_KNOWN_TOP_LEVEL = frozenset({
    "format_version", "kind", "backbone", "bands", "store_mode",
    "ann", "build_config", "created_utc", "rlat_version",
})

_KNOWN_BAND_KEYS = frozenset({
    "role", "dim", "l2_norm", "passage_count",
    "dim_native", "w_shape", "nested_mrl_dims", "trained_from",
})


def to_json(meta: Metadata) -> str:
    """Serialise to a stable, human-readable JSON string.

    `sort_keys=True` keeps diffs reviewable across rebuilds; `indent=2` is the
    on-disk format inside the .rlat ZIP. Tuples (w_shape) serialise as JSON
    arrays via dataclasses.asdict + the default JSONEncoder. Unknown keys
    captured by `from_json` under `_extras` (top-level and per-band) are
    merged back at their respective levels so a future-format file round-trips
    without losing data.
    """
    payload = asdict(meta)
    extras = payload.pop("_extras", None) or {}
    payload.update(extras)
    for name, b in payload.get("bands", {}).items():
        b_extras = b.pop("_extras", None) or {}
        b.update(b_extras)
    return json.dumps(payload, indent=2, sort_keys=True)


def from_json(text: str) -> Metadata:
    """Parse metadata.json text into a Metadata.

    Unknown top-level fields are stashed under `Metadata._extras`; unknown
    per-band fields are stashed under each `BandInfo._extras`. Round-trip
    preserves both — newer rlat versions can ship additional keys (top-level
    or under `bands.<name>`) and an older reader won't drop them or hard-fail
    on `BandInfo(**raw)`. Unknown keys nested under `build_config` / `ann`
    are kept verbatim (those are `dict[str, Any]`).
    Raises `json.JSONDecodeError` on malformed input.
    """
    raw = json.loads(text)

    backbone = BackboneInfo(**raw.get("backbone", {}))

    bands_raw = raw.get("bands", {}) or {}
    bands: dict[str, BandInfo] = {}
    for name, b_raw in bands_raw.items():
        b_known = {k: v for k, v in b_raw.items() if k in _KNOWN_BAND_KEYS}
        b_extras = {k: v for k, v in b_raw.items() if k not in _KNOWN_BAND_KEYS}
        # JSON arrays come back as lists; restore the tuple invariant for
        # w_shape so callers can rely on its shape contract.
        if b_known.get("w_shape") is not None:
            b_known["w_shape"] = tuple(b_known["w_shape"])
        bands[name] = BandInfo(**b_known, _extras=b_extras)

    extras = {k: v for k, v in raw.items() if k not in _KNOWN_TOP_LEVEL}

    return Metadata(
        format_version=raw.get("format_version", FORMAT_VERSION),
        kind=raw.get("kind", "corpus"),
        backbone=backbone,
        bands=bands,
        store_mode=raw.get("store_mode", "local"),
        ann=raw.get("ann", {}) or {},
        build_config=raw.get("build_config", {}) or {},
        created_utc=raw.get("created_utc", ""),
        rlat_version=raw.get("rlat_version", "2.0.0a1"),
        _extras=extras,
    )
