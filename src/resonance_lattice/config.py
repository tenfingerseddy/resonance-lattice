"""Configuration types for v2.0.0.

Single-recipe default (gte-modernbert-base 768d) collapses ~80% of v0.11's
preset registries to constants. The surviving config surface is two enums
(StoreMode, Kind) and two frozen dataclasses (MaterialiserConfig, BuildConfig).

No retrieval knobs. No encoder knobs. No reranker / lexical / sparsify config.
See docs/internal/audits/02_deps_and_presets.md for the audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StoreMode(str, Enum):
    """How source files are resolved at query time.

    See docs/user/STORAGE_MODES.md (will be written in Phase 2).
    """

    BUNDLED = "bundled"  # zstd-framed source/ inside the .rlat ZIP archive
    LOCAL = "local"      # FS resolution via --source-root (default at build)
    REMOTE = "remote"    # HTTP-backed, SHA-pinned, lockfile sync


class Kind(str, Enum):
    """Knowledge-model kind tag (base plan §9). v2.0 ships only the tag;
    intent operators are deferred to v2.1+.
    """

    CORPUS = "corpus"
    INTENT = "intent"


@dataclass(frozen=True)
class MaterialiserConfig:
    """Token budgets for context assembly (`rlat search --format context`).

    UX-level knobs, not retrieval knobs. Consumers can override on a per-call basis.
    `chars_per_token` is the conservative under-estimate used to convert
    `token_budget` to a character cap when building the materialised context;
    English averages ~3.7, code averages ~3.0–3.5, so 4 keeps headroom.
    """

    token_budget: int = 3000
    sections_landscape: int = 600
    sections_structure: int = 800
    sections_evidence: int = 1600
    chars_per_token: int = 4


@dataclass(frozen=True)
class BuildConfig:
    """Build-time configuration. Defaults pending Audit 5 (chunking) lock.

    See docs/internal/audits/03_format_ann_chunking.md.
    """

    chunker: str = "passage_v1"
    min_chars: int = 200
    max_chars: int = 3200
    store_mode: StoreMode = StoreMode.LOCAL
    kind: Kind = Kind.CORPUS
