"""Remote store — HTTP-backed, SHA-pinned read primitive.

The .rlat carries a **manifest** (`{source_file: {url, sha256}}`) recorded at
build time; query never trusts the network unconditionally. `RemoteStore`
downloads on first access to a persistent on-disk cache and SHA-pin-verifies
against the manifest before returning. Subsequent reads hit the cache;
tampered or partial downloads raise `RemoteShaMismatch` (mapped to
`DriftStatus="drifted"` by `Store.verify` and `verify_hits`) rather than
silently serving wrong text.

`fetch` / `verify` / per-instance text cache are inherited from
`store.base.Store`; this file owns the per-mode read primitive plus the
SHA-pin check, plus `freshness()` (read-only upstream-vs-pin poll).

The reconciliation path lives in `cli/maintain.py:cmd_sync` +
`store/remote_index.py` — incremental delta-apply on the same
`store/incremental.py` pipeline as `rlat refresh`. Audit 07 is the
design source of truth.

Phase 2 deliverable (read path). Phase 7 wiring (freshness impl + open_store
factory + cli/build remote manifest emission). Audit 07 (incremental sync).
"""

from __future__ import annotations

import concurrent.futures
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, IO

from .base import DriftStatus, RemoteShaMismatch, Store, sha256_hex

# Worker count for parallel `fetch_all`. HTTP-bound, so threads (not
# processes) — the GIL releases on the socket read inside urllib. 8 mirrors
# the optimise pipeline's synth-query concurrency; well within the
# rate-limit headroom of typical CI/CDN endpoints.
_FETCH_ALL_WORKERS = 8

# Network-error classes that `freshness` and `sync` collapse to "missing"
# (or drop, in sync's case). Narrow on purpose: a KeyError on a malformed
# manifest entry, a MemoryError on huge downloads, or a programming error
# in `_download` should NOT silently mark an entry "missing" — those are
# real bugs that need to surface.
_NETWORK_ERRORS = (urllib.error.URLError, TimeoutError, ConnectionError, OSError)

_DEFAULT_TIMEOUT_SECONDS = 30

# Injectable for tests — production calls `_default_opener` which wraps
# `urllib.request.urlopen` with a default timeout. Stays stdlib (no
# `requests` dep). Returns a context-manager handle with `.read()`. Tests
# substitute a callable returning BytesIO.
UrlOpener = Callable[[str], IO[bytes]]


def compose_manifest(
    source_files: dict[str, str], url_base: str,
) -> dict[str, dict[str, str]]:
    """Build a `{source_file: {url, sha256}}` manifest for remote-mode
    archives.

    Single owner of the URL-join + per-file sha256 contract — both
    `cli/build.py` (fresh remote-mode build) and `store/conversion.py`
    (`rlat convert --to remote`) call this so the manifest format
    can never drift between code paths.

    Trailing slashes on `url_base` are normalised so both
    `https://x/y` and `https://x/y/` produce identical entries.
    Relative paths are per-segment URL-quoted (`safe='/'`) so source
    paths containing spaces, `#`, `?`, `%`, etc. produce well-formed
    URLs that the upstream HTTP server resolves unambiguously.

    Returns the dict in the exact shape `archive.write(remote_manifest=...)`
    expects.
    """
    prefix = url_base.rstrip("/")
    return {
        rel: {
            "url": f"{prefix}/{urllib.parse.quote(rel, safe='/')}",
            "sha256": sha256_hex(text),
        }
        for rel, text in source_files.items()
    }


def _default_opener(url: str) -> IO[bytes]:
    # Default timeout prevents `rlat search` from hanging indefinitely on a
    # stalled upstream — without it, a single slow remote source blocks the
    # whole query. URL provenance is the in-archive manifest; tampered URLs
    # still trip SHA-pin verification downstream so noqa is justified by
    # that downstream check rather than any signature on the URL itself.
    return urllib.request.urlopen(url, timeout=_DEFAULT_TIMEOUT_SECONDS)  # noqa: S310


class RemoteStore(Store):
    """HTTP-backed source resolver with on-disk cache and SHA-pin verification.

    `manifest` is the parsed `{source_file: {"url": str, "sha256": str}}`
    mapping — typically `ArchiveContents.remote_manifest` from the loaded
    .rlat. `cache_dir` is a per-knowledge-model persistent directory; cache
    keys are derived from `source_file` so the layout is stable across URL
    changes.
    """

    def __init__(
        self,
        manifest: dict[str, dict[str, str]],
        cache_dir: str | Path,
        opener: UrlOpener | None = None,
    ):
        super().__init__()
        self.manifest: dict[str, dict[str, str]] = dict(manifest)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Late lookup so test fixtures can monkey-patch
        # `remote._default_opener` between import time and call time.
        self._opener = opener if opener is not None else _default_opener

    def _cache_path(self, source_file: str, expected_sha: str) -> Path:
        # Key on (source_file, expected_sha) so a `rlat sync` that re-pins
        # a path to a new SHA naturally falls through to a fresh download
        # — old cache files for the prior SHA are inert (different filename)
        # and never trusted. Without this, post-sync reads SHA-mismatch the
        # OLD cached bytes against the NEW pinned SHA and surface as drift.
        # Hashing the source_file portion also keeps the layout stable
        # regardless of path separators / case / traversal characters.
        return (
            self.cache_dir
            / f"{sha256_hex(source_file)[:16]}.{expected_sha[:16]}.bytes"
        )

    def _download(self, url: str) -> bytes:
        with self._opener(url) as resp:
            return resp.read()

    def _fetch_text(self, source_file: str) -> str:
        """Download (or cache-hit) and return decoded + LF-normalized text.

        Build hashes the LF-normalized text (Python's `read_text` opens in
        universal-newlines mode, collapsing CRLF→LF on read). Remote MUST
        normalize identically before hashing, otherwise CRLF source files
        always SHA-mismatch even when the upstream content is unchanged.
        Char offsets in the registry are also positions in the LF-normalized
        text — slicing raw bytes at those offsets returns the wrong content.
        """
        spec = self.manifest.get(source_file)
        if spec is None:
            raise FileNotFoundError(
                f"{source_file!r} not in remote manifest; either store_mode != "
                f"'remote' or build dropped this file"
            )
        expected_sha = spec["sha256"]
        cache_file = self._cache_path(source_file, expected_sha)
        if cache_file.exists():
            data = cache_file.read_bytes()
        else:
            data = self._download(spec["url"])
            # Atomic cache write so a crash mid-download doesn't leave a
            # half-file that future reads would trust.
            tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(cache_file)
        text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        actual_sha = sha256_hex(text)
        if actual_sha != expected_sha:
            # Typed exception (not bare ValueError) so drift-detection paths
            # (`Store.verify`, `verify_hits`) can map this to drifted-status
            # without swallowing unrelated `ValueError`s from local stores
            # (path-traversal guards in particular).
            raise RemoteShaMismatch(
                source_file=source_file,
                expected_sha=expected_sha,
                actual_sha=actual_sha,
            )
        return text

    def _read_full_text_uncached(self, source_file: str) -> str:
        return self._fetch_text(source_file)

    def fetch_all(self, source_files: "list[str] | set[str]") -> dict[str, str]:
        """Override the ABC default with parallel-fetch semantics.

        `rlat convert --to {bundled|local}` against a remote-mode archive
        materialises every source file via this method. The default
        implementation walks `_read_full_text` sequentially — for a
        100-file corpus that's 100 serial HTTP round-trips, each blocking
        the next.

        Threads (not processes) because the GIL releases on socket reads
        inside `urllib.request.urlopen` — HTTP-bound workloads scale well
        with `_FETCH_ALL_WORKERS = 8`. Per-file SHA-pin verification still
        runs inside `_fetch_text`, so a bit-rotted cache or shifted
        upstream still raises `RemoteShaMismatch` on the first failing
        entry.

        Cache hits short-circuit the network; the per-instance text cache
        is populated on the first successful read of each path, so a
        repeat conversion (e.g. dry-run + real run in the same process)
        pays network only once.
        """
        unique = list(dict.fromkeys(source_files))
        if not unique:
            return {}
        # Cache hits — handled serially since they don't touch the network.
        # Network-bound work only goes through the executor.
        out: dict[str, str] = {}
        to_fetch: list[str] = []
        for sf in unique:
            cached = self._text_cache.get(sf)
            if cached is not None:
                out[sf] = cached
            else:
                to_fetch.append(sf)
        if not to_fetch:
            return out
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=_FETCH_ALL_WORKERS,
        ) as pool:
            results = list(pool.map(self._read_full_text, to_fetch))
        for sf, text in zip(to_fetch, results):
            out[sf] = text
        return out

    def freshness(self) -> dict[str, DriftStatus]:
        """Read-only "is upstream still serving the SHAs we pinned?"

        Walks every manifest entry, downloads the upstream bytes, hashes
        them, and compares to the pinned `sha256`. Returns a per-entry
        DriftStatus map: `verified` (hash matches), `drifted` (hash
        mismatches — upstream file changed since build), `missing`
        (network error — connection refused, 404, 5xx, timeout).

        Read-only: does NOT touch the on-disk cache, does NOT mutate the
        manifest. Use `sync()` to actually pull updates. Network-bound;
        progress is the caller's job (CLI prints per-entry status).
        """
        out: dict[str, DriftStatus] = {}
        for source_file, spec in self.manifest.items():
            try:
                data = self._download(spec["url"])
            except _NETWORK_ERRORS:
                out[source_file] = "missing"
                continue
            text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
            out[source_file] = (
                "verified" if sha256_hex(text) == spec["sha256"] else "drifted"
            )
        return out

    # `rlat sync` is the reconciliation primitive — it lives in
    # `cli/maintain.cmd_sync` and lands on `store/incremental.apply_delta`
    # (the same delta-apply pipeline `rlat refresh` uses). That path
    # bucketises on stable `passage_id`, re-encodes only updated + added
    # passages, re-projects the optimised band from the new base for free,
    # and writes atomically — so the manifest, bands, and registry stay
    # internally consistent. The codex P0 manifest-only mode is statically
    # impossible: `apply_delta` requires the encoder by signature.
