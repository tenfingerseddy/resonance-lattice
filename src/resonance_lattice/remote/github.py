# SPDX-License-Identifier: BUSL-1.1
"""GitHub-backed Fetcher for remote-mode knowledge models.

Resolves a GitHub repo URL (optionally qualified with ``#branch``,
``@sha``, or ``@tag``) to:

    - a list of tracked file paths at a pinned commit SHA (build time)
    - raw file bytes at that SHA (query time, via raw.githubusercontent.com)
    - a diff of changed files between two SHAs (sync time)

All three use stdlib ``urllib.request`` with an optional ``GITHUB_TOKEN``
env var for higher rate limits. No new runtime dependencies.

URL forms accepted
    https://github.com/OWNER/REPO
    https://github.com/OWNER/REPO#branch-or-tag
    https://github.com/OWNER/REPO@commitsha
    git@github.com:OWNER/REPO.git  (canonicalised)
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

API_BASE = "https://api.github.com"
RAW_BASE = "https://raw.githubusercontent.com"
USER_AGENT = "resonance-lattice/rlat"
DEFAULT_TIMEOUT = 20.0

# Matches https URLs and the SSH "git@github.com:org/repo.git" shorthand.
_GITHUB_RE = re.compile(
    r"""
    ^
    (?:https?://github\.com/ | git@github\.com: )
    (?P<org>[^/\s]+)
    /
    (?P<repo>[^/\s#@]+?)
    (?:\.git)?
    (?: (?P<sep>[#@]) (?P<ref>[^\s]+) )?
    /?
    $
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class GithubOrigin:
    """Parsed identity of a GitHub repo + optional ref."""

    org: str
    repo: str
    ref: str | None = None  # branch | tag | sha, None = "default branch"

    @property
    def key(self) -> str:
        """Filesystem-safe cache key (``github__OWNER__REPO``)."""
        return f"github__{self.org}__{self.repo}"

    @property
    def base_url(self) -> str:
        return f"https://github.com/{self.org}/{self.repo}"


def parse_origin(url: str) -> GithubOrigin:
    """Parse ``url`` into a GithubOrigin. Raises ValueError on bad shapes."""
    url = url.strip()
    m = _GITHUB_RE.match(url)
    if not m:
        raise ValueError(
            f"Not a recognised GitHub URL: {url!r}. "
            f"Expected forms: https://github.com/OWNER/REPO "
            f"[#branch | @sha]."
        )
    return GithubOrigin(
        org=m.group("org"),
        repo=m.group("repo"),
        ref=m.group("ref"),
    )


def _http_json(url: str, token: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> dict | list:
    """GET ``url`` and decode JSON. Raises on HTTP error."""
    req = urllib.request.Request(url, headers=_headers(token, accept="application/vnd.github+json"))
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — URL sanitised by caller
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def _http_bytes(url: str, token: str | None = None, timeout: float = DEFAULT_TIMEOUT) -> bytes:
    """GET ``url`` and return raw bytes."""
    req = urllib.request.Request(url, headers=_headers(token))
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def _headers(token: str | None, accept: str = "*/*") -> dict[str, str]:
    h = {"User-Agent": USER_AGENT, "Accept": accept}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


class GithubFetcher:
    """Fetcher implementation against github.com public APIs.

    Thread-safe for reads (stateless aside from ``token``). Construct
    once per origin; reuse across queries.
    """

    def __init__(
        self,
        origin: GithubOrigin | str,
        token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.origin = parse_origin(origin) if isinstance(origin, str) else origin
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.timeout = timeout

    # ── reference resolution ───────────────────────────────────────

    def resolve_sha(self, ref: str | None = None) -> str:
        """Turn a ref (branch | tag | sha | None) into a 40-char commit SHA.

        ``None`` resolves to the repo's default branch HEAD. A 40-char
        hex ref is returned as-is (already a SHA). Anything else is
        resolved via the GitHub commits API.
        """
        ref = ref if ref is not None else self.origin.ref
        if ref and re.fullmatch(r"[0-9a-f]{40}", ref):
            return ref
        if ref is None:
            # Look up default branch.
            info = _http_json(
                f"{API_BASE}/repos/{self.origin.org}/{self.origin.repo}",
                token=self.token, timeout=self.timeout,
            )
            assert isinstance(info, dict)
            ref = info.get("default_branch", "main")
        # Resolve branch / tag / short sha -> full sha.
        data = _http_json(
            f"{API_BASE}/repos/{self.origin.org}/{self.origin.repo}/commits/{urllib.parse.quote(ref, safe='')}",
            token=self.token, timeout=self.timeout,
        )
        assert isinstance(data, dict)
        sha = data.get("sha")
        if not sha or not isinstance(sha, str):
            raise ValueError(f"GitHub commits API returned no sha for ref {ref!r}")
        return sha

    # ── Fetcher protocol ───────────────────────────────────────────

    def list_files(
        self, ref: str | None = None, path_prefix: str | None = None,
    ) -> tuple[str, list[str]]:
        """Return ``(sha, rel_paths)`` — file list at the resolved sha.

        Uses the git-trees API with ``recursive=1``. Handles "truncated"
        responses (repos with > ~100k paths) by raising — callers can
        branch into pagination if that ever matters for rlat use cases.

        If ``path_prefix`` is given, only paths under that subdirectory
        are returned. The prefix is compared as ``<prefix>/`` so
        ``Doc`` matches ``Doc/tutorial/...`` but not ``Doctor/...``.
        """
        sha = self.resolve_sha(ref)
        tree = _http_json(
            f"{API_BASE}/repos/{self.origin.org}/{self.origin.repo}/git/trees/{sha}?recursive=1",
            token=self.token, timeout=self.timeout,
        )
        assert isinstance(tree, dict)
        if tree.get("truncated"):
            raise RuntimeError(
                f"GitHub tree listing truncated at {sha}: the repo has too "
                f"many files for a single git-trees response. Narrow the "
                f"build scope or add pagination to GithubFetcher."
            )
        paths = sorted(
            node["path"]
            for node in tree.get("tree", [])
            if node.get("type") == "blob" and "path" in node
        )
        if path_prefix:
            prefix = path_prefix.strip("/") + "/"
            paths = [p for p in paths if p.startswith(prefix)]
        return sha, paths

    def fetch(self, sha: str, rel_path: str) -> bytes:
        """GET raw file bytes via raw.githubusercontent.com."""
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError(
                f"fetch() requires a 40-char commit SHA, got {sha!r}. "
                f"Pin with resolve_sha() first."
            )
        rel = rel_path.lstrip("/")
        url = (
            f"{RAW_BASE}/{self.origin.org}/{self.origin.repo}/{sha}/"
            + "/".join(urllib.parse.quote(p, safe="") for p in rel.split("/"))
        )
        return _http_bytes(url, token=self.token, timeout=self.timeout)

    def compare(
        self, base_sha: str, head_sha: str, path_prefix: str | None = None,
    ) -> dict:
        """Return ``{added, modified, removed, head_sha}`` between two SHAs.

        Uses the GitHub compare API, which caps at 300 files per response.
        For larger diffs callers should fall back to re-listing the whole
        tree at head_sha and reconciling against the knowledge model manifest.

        If ``path_prefix`` is given, only file entries under that
        subdirectory are reported; upstream changes outside the scope
        are silently dropped so scoped knowledge models don't pick up
        drift they never indexed.
        """
        url = (
            f"{API_BASE}/repos/{self.origin.org}/{self.origin.repo}"
            f"/compare/{urllib.parse.quote(base_sha, safe='')}..."
            f"{urllib.parse.quote(head_sha, safe='')}"
        )
        data = _http_json(url, token=self.token, timeout=self.timeout)
        assert isinstance(data, dict)
        scope = (path_prefix.strip("/") + "/") if path_prefix else None
        def _in_scope(p: str) -> bool:
            return scope is None or p.startswith(scope)
        added: list[str] = []
        modified: list[str] = []
        removed: list[str] = []
        for f in data.get("files", []):
            status = f.get("status", "")
            path = f.get("filename", "")
            if not path or not _in_scope(path):
                continue
            if status == "added":
                added.append(path)
            elif status == "removed":
                removed.append(path)
            elif status in ("modified", "changed", "renamed"):
                modified.append(path)
                # 'renamed' status also carries a 'previous_filename'
                # which the caller may want to treat as removed; keep
                # that logic in sync when it matters.
        return {
            "added": sorted(added),
            "modified": sorted(modified),
            "removed": sorted(removed),
            "head_sha": data.get("merge_base_commit", {}).get("sha", head_sha) or head_sha,
        }
