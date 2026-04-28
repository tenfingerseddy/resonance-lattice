"""RemoteIndex — upstream-state oracle for `rlat sync`.

The `RemoteIndex` Protocol abstracts how `rlat sync` discovers what changed
upstream since the archive was built. Concrete implementations:

  HttpManifestIndex  — generic HTTP corpora. Two modes:
                       - `from_url(manifest_url)`: upstream serves a stable
                         JSON manifest endpoint (`{source_file: {url, sha256}}`)
                         that lists ALL current files. Detects added + modified
                         + removed.
                       - `from_existing(existing_manifest)`: poll every URL in
                         the archive's recorded manifest. Detects modified +
                         removed only — cannot discover NEW files.

  GitHubCompareIndex — v2.1+. Uses the GitHub compare API to get the 3-bucket
                       delta between two refs without re-downloading every file.
                       Skeleton present; full impl deferred to v2.1.

Audit 07 commit 5/8.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, IO, Protocol

from .remote import _NETWORK_ERRORS, UrlOpener, _default_opener
from .base import sha256_hex


@dataclass(frozen=True)
class RemoteDelta:
    """The 4-bucket upstream delta. Mirrors v0.11's compare output plus an
    `unavailable` bucket for transient network errors that must not be
    silently classified as removals.

    - `added`       — paths in upstream catalog but not in existing manifest.
    - `modified`    — paths whose upstream sha256 differs from the existing
                      manifest pin.
    - `removed`     — paths in existing manifest but not in upstream catalog
                      (catalog mode only — catalog is authoritative).
    - `unavailable` — paths whose upstream URL was unreachable (timeout,
                      connection refused, 5xx). Poll mode appends here
                      rather than to `removed` because a network blip
                      shouldn't delete corpus content. `cmd_sync` aborts
                      by default if this bucket is non-empty; the caller
                      can opt in to treating unavailable as removed.
    """
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    head_ref: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.modified or self.removed or self.unavailable)


class RemoteIndex(Protocol):
    """Upstream-state oracle for `rlat sync`. Implementations decide which
    primitive to use (manifest-poll, GitHub compare, S3 versioned listing,
    git protocol) — the consumer just sees the 3-bucket delta and a
    `fetch(path) → bytes` accessor for the changed paths."""

    def head_ref(self) -> str:
        """Current upstream ref identifier (commit SHA / version tag /
        manifest etag). Recorded into the archive after a successful sync
        so the next sync can pivot off it."""

    def changed_files_since(self, pinned_ref: str) -> RemoteDelta:
        """Compute the 3-bucket delta between `pinned_ref` and `head_ref()`.
        For implementations without ref semantics (HttpManifest), `pinned_ref`
        may be ignored in favour of comparing against the implementation's
        internal `existing_manifest`."""

    def fetch(self, path: str) -> bytes:
        """Fetch the live bytes of one upstream path. Called only for paths
        in `delta.added + delta.modified` — unchanged files are not
        re-fetched."""


# ---------------------------------------------------------------------------
# HttpManifestIndex
# ---------------------------------------------------------------------------

@dataclass
class _ManifestEntry:
    url: str
    sha256: str


def _spec(url: str, sha: str) -> dict[str, str]:
    return {"url": url, "sha256": sha}


class HttpManifestIndex:
    """Upstream-state oracle backed by an HTTP-served manifest.

    Two construction modes:

      from_url(manifest_url, opener=...)
        Fetch `manifest_url` to get the upstream's authoritative listing
        of `{source_file: {url, sha256}}`. Discovers added + modified +
        removed paths. Use when upstream maintains a stable manifest
        endpoint (e.g. CI-built artefact, GitHub raw).

      from_existing(existing_manifest, opener=...)
        Poll every URL in the archive's recorded manifest, compute SHAs
        from the live bytes. Discovers modified + removed only — cannot
        enumerate NEW files because there's no upstream catalog. Use as
        a fallback for corpora that don't serve a manifest endpoint.

    Either way, the returned RemoteDelta feeds incremental.bucketise via
    the cli/sync wiring.
    """

    def __init__(
        self,
        existing_manifest: dict[str, dict[str, str]],
        upstream_manifest: dict[str, dict[str, str]] | None,
        head_ref_value: str,
        opener: UrlOpener | None = None,
    ):
        self._existing = dict(existing_manifest)
        self._upstream = (
            dict(upstream_manifest) if upstream_manifest is not None else None
        )
        self._head_ref = head_ref_value
        # Late lookup so test fixtures can monkey-patch
        # `remote_index._default_opener` between import time and call time.
        self._opener = opener if opener is not None else _default_opener
        # Per-path bytes cache populated during poll-mode delta detection
        # (each URL is fetched once to compute its sha; the same bytes are
        # what `fetch()` would re-download). Without this cache, poll mode
        # round-trips every URL twice — once for delta detection, once for
        # fetch — doubling network usage on every sync.
        self._poll_byte_cache: dict[str, bytes] = {}

    @classmethod
    def from_url(
        cls,
        existing_manifest: dict[str, dict[str, str]],
        manifest_url: str,
        opener: UrlOpener | None = None,
    ) -> "HttpManifestIndex":
        """Fetch a manifest from `manifest_url` and use it as the upstream
        catalog. The fetched manifest's hash becomes `head_ref`."""
        effective_opener = opener if opener is not None else _default_opener
        with effective_opener(manifest_url) as resp:
            data: bytes = resp.read()
        text = data.decode("utf-8")
        upstream = json.loads(text)
        head_ref = sha256_hex(text)[:16]
        return cls(
            existing_manifest=existing_manifest,
            upstream_manifest=upstream,
            head_ref_value=head_ref,
            opener=effective_opener,
        )

    @classmethod
    def from_existing(
        cls,
        existing_manifest: dict[str, dict[str, str]],
        opener: UrlOpener | None = None,
    ) -> "HttpManifestIndex":
        """Poll-mode: no upstream catalog, just re-fetch every URL in the
        existing manifest and detect modified/removed. Cannot discover
        added files (there's no source for that signal)."""
        return cls(
            existing_manifest=existing_manifest,
            upstream_manifest=None,
            head_ref_value="poll-" + sha256_hex(
                json.dumps(existing_manifest, sort_keys=True)
            )[:12],
            opener=opener,
        )

    def head_ref(self) -> str:
        return self._head_ref

    def changed_files_since(self, pinned_ref: str) -> RemoteDelta:
        """Compute the 3-bucket delta. `pinned_ref` is informational here —
        the comparison is between `existing_manifest` and either the
        upstream catalog (catalog mode) or the live URL-poll results
        (poll mode)."""
        if self._upstream is not None:
            return self._delta_from_catalog()
        return self._delta_from_poll()

    def _delta_from_catalog(self) -> RemoteDelta:
        # Diff the FULL spec (url + sha256), not just sha256: a file moved
        # to a new URL with unchanged content still needs a manifest update
        # so future fetches go to the right place. The bytes-unchanged case
        # is handled on the cmd_sync side — a path with the same sha256 but
        # a different URL is bucketed as "modified" here, but cmd_sync's
        # bucketise (on stable passage_id) yields zero passage updates for
        # it, so no re-encoding happens; the only effect is the manifest
        # rewrite. That's the correct semantics.
        assert self._upstream is not None
        added: list[str] = []
        modified: list[str] = []
        removed: list[str] = []
        upstream_paths = set(self._upstream.keys())
        existing_paths = set(self._existing.keys())
        for path in sorted(upstream_paths - existing_paths):
            added.append(path)
        for path in sorted(existing_paths - upstream_paths):
            removed.append(path)
        for path in sorted(upstream_paths & existing_paths):
            up = self._upstream[path]
            ex = self._existing[path]
            if up.get("sha256") != ex.get("sha256") or up.get("url") != ex.get("url"):
                modified.append(path)
        return RemoteDelta(
            added=added, modified=modified, removed=removed,
            unavailable=[],
            head_ref=self._head_ref,
        )

    def _delta_from_poll(self) -> RemoteDelta:
        # Poll mode has no upstream catalog, so a network error on any URL
        # is genuinely ambiguous — the file might be removed upstream OR
        # the request might have transient-failed (timeout, 5xx, DNS blip,
        # CDN flap). We append to `unavailable`, not `removed`, so the
        # caller decides — `cmd_sync` aborts by default rather than
        # silently deleting corpus content.
        #
        # The fetched bytes are cached on the instance so `fetch()` reuses
        # them — without this, sync round-trips every modified URL twice
        # (once here, once when cmd_sync calls index.fetch()).
        modified: list[str] = []
        unavailable: list[str] = []
        for path, spec in self._existing.items():
            try:
                with self._opener(spec["url"]) as resp:
                    data = resp.read()
            except _NETWORK_ERRORS:
                unavailable.append(path)
                continue
            self._poll_byte_cache[path] = data
            text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
            if sha256_hex(text) != spec["sha256"]:
                modified.append(path)
        return RemoteDelta(
            added=[], modified=sorted(modified), removed=[],
            unavailable=sorted(unavailable),
            head_ref=self._head_ref,
        )

    def fetch(self, path: str) -> bytes:
        """Fetch live bytes for one path. Looks up the URL in the upstream
        catalog (catalog mode) or in the existing manifest (poll mode —
        same URL applies, only the SHA changed). Poll mode reuses the
        bytes cached during delta detection rather than re-fetching."""
        cached = self._poll_byte_cache.get(path)
        if cached is not None:
            return cached
        spec = (
            (self._upstream and self._upstream.get(path))
            or self._existing.get(path)
        )
        if spec is None:
            raise KeyError(
                f"{path!r} not in upstream catalog or existing manifest"
            )
        with self._opener(spec["url"]) as resp:
            return resp.read()

    def upstream_spec(self, path: str) -> dict[str, str] | None:
        """Return the upstream spec for a path. The shape differs by mode:

        - **catalog mode**: returns `{"url", "sha256"}` straight from the
          upstream-served catalog. The sha256 IS authoritative — the
          caller (cmd_sync) validates that the fetched bytes hash to the
          same value.

        - **poll mode**: returns `{"url"}` only — sha256 is intentionally
          omitted because we don't have an authoritative upstream value
          to validate against; we only have whatever bytes the URL
          returned, and the caller should write the live-bytes hash into
          the new manifest. Including the existing manifest's pinned sha
          here would falsely look like an upstream assertion.

        Returns None if the path isn't known to either source.
        """
        if self._upstream is not None:
            return self._upstream.get(path)
        existing = self._existing.get(path)
        if existing is None:
            return None
        return {"url": existing["url"]}


# ---------------------------------------------------------------------------
# GitHubCompareIndex (v2.1+)
# ---------------------------------------------------------------------------

class GitHubCompareIndex:
    """Skeleton — full implementation deferred to v2.1.

    Will use the GitHub compare API
    (`/repos/{org}/{repo}/compare/{base}...{head}`) to get the 3-bucket
    delta without re-downloading every file. Auth via `GITHUB_TOKEN`.
    Rate-limit aware (5K/h authenticated, 60/h anonymous).
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "GitHubCompareIndex ships in v2.1. Use HttpManifestIndex for now."
        )
