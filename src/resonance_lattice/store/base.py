"""Store base — concrete `fetch`/`verify`/cache + abstract `_read_full_text_uncached`.

A Store resolves `(source_file, char_offset, char_length)` back to authoritative
text. Three subclasses implement only the per-mode read primitive; the
shared shape (slice, hash-compare-on-verify, per-instance text cache) lives
here.

Per-passage (not per-file) hashing is intentional: a one-line edit elsewhere
in a 5K-line file shouldn't mark every passage in the file as drifted; only
the spans whose char-range actually changed.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Literal

DriftStatus = Literal["verified", "drifted", "missing"]


class RemoteShaMismatch(Exception):
    """Raised by `RemoteStore` when cached or upstream bytes for a source file
    don't match the SHA-pinned manifest entry. Distinct from `ValueError` so
    drift-detection paths (`Store.verify`, `verify_hits`) can map this to
    `DriftStatus="drifted"` without also swallowing local validation errors
    (path-traversal guards, malformed argparse input) that genuinely raise
    plain `ValueError`.

    Carries `source_file`, `expected_sha`, `actual_sha` so callers can render
    a useful error or surface drift detail without re-parsing the message.
    """

    def __init__(self, source_file: str, expected_sha: str, actual_sha: str):
        super().__init__(
            f"SHA-pin mismatch for {source_file!r}: cache={actual_sha[:12]} "
            f"manifest={expected_sha[:12]}"
        )
        self.source_file = source_file
        self.expected_sha = expected_sha
        self.actual_sha = actual_sha


def sha256_hex(data: bytes | str) -> str:
    """Bare-hex sha256 digest — no prefix. The remote-mode manifest contract
    uses bare hex (`{"sha256": "<64-hex>"}`); `compute_hash` adds the
    `sha256:` registry prefix for the verified-retrieval contract. Both
    callers go through this helper so the hash convention is single-sourced.

    Accepts str (utf-8 encoded) or bytes (used directly).
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def compute_hash(text: str) -> str:
    """Canonical content hash — `sha256:<64-hex>` of utf-8-encoded text.

    Used at build time to populate `PassageCoord.content_hash` and at query
    time by `Store.verify` to detect drift. The `sha256:` prefix is part of
    the recorded value so a future algorithm swap (blake3, sha3) is a
    one-line registry entry, not a metadata-format break.
    """
    return f"sha256:{sha256_hex(text)}"


class Store(ABC):
    """Resolves passage coordinates to authoritative text.

    Subclasses implement `_read_full_text_uncached` (the per-mode read
    primitive). The base class wraps it with a per-instance text cache and
    provides concrete `fetch` / `verify` so all three stores share the same
    slice + hash-compare logic — single source of truth, no drift between
    `bundled` / `local` / `remote` semantics.

    Cache lifetime equals the Store instance's lifetime, which in v2.0 is one
    CLI query. No invalidation logic; re-issue the query for fresh reads.
    """

    def __init__(self) -> None:
        self._text_cache: dict[str, str] = {}

    @abstractmethod
    def _read_full_text_uncached(self, source_file: str) -> str:
        """Read the full source file as utf-8 text.

        Raise `FileNotFoundError` if the source is unavailable; `verify`
        catches and reports as `drift_status == "missing"`. Subclasses may
        raise `ValueError` for tamper-detection (e.g. SHA-pin mismatch in
        the remote store) — those propagate up; they're not "missing".
        """
        ...

    def _read_full_text(self, source_file: str) -> str:
        """Cached wrapper. A typical search returns multiple hits in the
        same source file, each calling fetch + verify; without the cache
        each file is read 2N times for N hits in one file. The cache is
        populated lazily on first read."""
        cached = self._text_cache.get(source_file)
        if cached is not None:
            return cached
        text = self._read_full_text_uncached(source_file)
        self._text_cache[source_file] = text
        return text

    def fetch(self, source_file: str, char_offset: int, char_length: int) -> str:
        """Return the passage text. Raises `FileNotFoundError` if the source
        is unavailable."""
        return self._read_full_text(source_file)[char_offset:char_offset + char_length]

    def fetch_all(self, source_files: "list[str] | set[str]") -> dict[str, str]:
        """Materialise every requested source file as full text.

        Used by `rlat convert` (Audit 08) to bulk-read every file behind a
        registry before re-emitting in a different storage mode. Returns
        `{source_file: full_text}` — LF-normalised by whichever
        `_read_full_text_uncached` implementation the subclass uses (local
        reads in universal-newlines mode, remote replaces `\\r\\n` and
        `\\r`, bundled inherits whatever the build wrote).

        Default implementation walks via the cached `_read_full_text` so
        that subclasses that benefit from per-mode bulk fetch (parallel
        downloads, ZIP-batch reads) override this with a faster path; the
        default is correct for any subclass that hasn't.

        Raises whatever `_read_full_text_uncached` raises on the first
        failing entry — `FileNotFoundError` for missing sources,
        `RemoteShaMismatch` for SHA-pin failures. Callers (convert)
        validate every passage against `content_hash` after fetch_all
        returns; that's where Audit 07 invariant 2 is enforced.
        """
        out: dict[str, str] = {}
        for sf in source_files:
            if sf in out:
                continue
            out[sf] = self._read_full_text(sf)
        return out

    def verify(
        self,
        source_file: str,
        char_offset: int,
        char_length: int,
        expected_hash: str,
    ) -> DriftStatus:
        """Re-hash the slice and return whether it still matches
        `expected_hash` (the value recorded in `PassageCoord.content_hash` at
        build time).

        Failure modes are normalised to `DriftStatus` so callers (search,
        skill-context, profile) can reason uniformly:

        - `FileNotFoundError` (local/remote: missing source) → `"missing"`.
        - `RemoteShaMismatch` (remote SHA-pin mismatch raised by
          `RemoteStore` when cached/upstream bytes don't match the manifest
          hash) → `"drifted"`. Typed exception, not bare `ValueError`,
          so local-store path-traversal guards (which raise `ValueError`)
          continue to propagate as fatal — they're not drift, they're
          validation failures.
        """
        try:
            text = self._read_full_text(source_file)
        except FileNotFoundError:
            return "missing"
        except RemoteShaMismatch:
            return "drifted"
        slice_text = text[char_offset:char_offset + char_length]
        return "verified" if compute_hash(slice_text) == expected_hash else "drifted"
