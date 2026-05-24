"""Lens — the portable user/team perspective overlay.

The third user-facing layer in lensed knowledge (source / insight / lens).
A lens carries memory, intent history, verdicts, trust preferences, and
an optional declared editorial stance — and it travels across corpora.

The schema is **designed-portable**: no field carries an identifier
specific to one corpus. Trust weights are keyed by source-pattern globs;
insight preferences are keyed by content hash. Loading a lens against a
different `.rlat` re-resolves preferences by hash without breaking.

See `.claude/plans/lensed-knowledge-architecture.md` §5.
"""

from __future__ import annotations

from .schema import (
    InsightPreference,
    Lens,
    LensManifest,
    LensScope,
    TrustWeight,
    compose,
    load,
    save,
)

__all__ = [
    "InsightPreference",
    "Lens",
    "LensManifest",
    "LensScope",
    "TrustWeight",
    "compose",
    "load",
    "save",
]
