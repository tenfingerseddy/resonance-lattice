"""Helpers shared across the v2.1 flat-memory modules.

These live here (not in `store.py` / `capture.py`) because the future
MVP migrate + daemon recall paths derive workspace + transcript hashes
the same way as capture, and share the timestamp shape with every row
write.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
from pathlib import Path
from typing import Iterable


def utcnow_iso() -> str:
    """ISO-8601 UTC timestamp with second precision and trailing Z.

    Locked to the v2.1 sidecar `created_at` / `last_corroborated_at` shape.
    """
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalise_cwd(cwd: str) -> str:
    """Canonicalise a cwd string before hashing.

    Windows paths are case-insensitive at the filesystem level, but Claude
    Code's UserPromptSubmit envelope sometimes passes a lowercased drive
    letter (`c:\\Users\\...`) while `os.getcwd()` returns uppercase
    (`C:\\Users\\...`) — the same workspace would hash to two different
    `workspace:<hash>` tags and the §0.6 workspace gate would drop every
    hit. `os.path.normcase` lowercases on Windows and is a no-op on POSIX,
    so we route every workspace_hash input through it.
    """
    return os.path.normcase(cwd)


def workspace_hash(cwd: str) -> str:
    """sha256[:6] of normalised cwd, used as the `workspace:<hash>` scope.

    Six hex chars = 24 bits ≈ 16M-way collision space; collisions matter
    only for cross-workspace bleed risk and §18.3 mitigations are wired at
    the retrieval layer (D.8 harness suite checks intentional collisions).
    Path normalisation is via `os.path.normcase` — case-folds on Windows,
    no-op on POSIX. See `_normalise_cwd` for the rationale.
    """
    return hashlib.sha256(_normalise_cwd(cwd).encode("utf-8")).hexdigest()[:6]


def workspace_tag_for_cwd(cwd: str | Path | None = None) -> str:
    """Build the `workspace:<hash>` scope-tag string for `cwd`.

    Defaults to `Path.cwd()`. Callers (manual CLI add, Stop-hook capture,
    future MVP migrate) share this single derivation so the harness has
    one mock point for the §18.3 cwd-collision contract test.
    """
    target = str(cwd) if cwd is not None else str(Path.cwd())
    return f"workspace:{workspace_hash(target)}"


def stable_hash(parts: Iterable[bytes | str]) -> str:
    """Stable SHA-256 hex over a sequence of byte/string parts.

    Uses NUL separators between parts so concatenation can't collide
    (`"ab" + "c"` and `"a" + "bc"` produce different hashes). Fed by the
    Stop-hook capture path to derive `transcript_hash`, and by the future
    daemon recall path to dedup query bodies.
    """
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, str):
            part = part.encode("utf-8")
        h.update(part)
        h.update(b"\x00")
    return h.hexdigest()
