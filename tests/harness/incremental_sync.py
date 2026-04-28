"""incremental_sync — `rlat sync` is correct under remote edits.

Six guarantees:

  1. Build → sync against unchanged upstream is a no-op.
  2. Modify one upstream file → sync re-encodes only that file's
     passages. Other rows are byte-identical lifts. Poll-mode-also:
     the manifest entry for the modified file holds the live-bytes
     sha, not the stale pinned sha (regression guard for the
     `spec.get("sha256") or sha256_hex(text)` writeback bug).
  3. Catalog mode (`--upstream-manifest`) detects an added file →
     sync appends new passages.
  4. Catalog mode detects a removed file → sync drops its passages.
  5. After sync, manifest's pinned sha256 advances to the new content
     hashes — a re-poll is internally consistent.
  6. Poll-mode network errors → unavailable bucket; sync ABORTS
     (rc=2) by default rather than silently deleting corpus content.
     `--treat-unreachable-as-removed` migrates them into removed.

Uses an in-process fake `RemoteIndex` that returns canned bytes via a
`UrlOpener` to keep the suite hermetic (no live network).

Audit 07 commit 7/8.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Callable, IO

import numpy as np


from ._testutil import Args as _Args, build_corpus


def _build_remote(root: Path, manifest_files: dict[str, str]) -> Path:
    """Wrapper around the shared `build_corpus` test helper that pins this
    suite's upstream-URL fixture. Builds a remote-mode archive whose
    manifest URLs land at `https://upstream.test/corpus/<rel>`; the actual
    fetch path is exercised through fixture openers at sync time, not at
    build time."""
    return build_corpus(
        root, manifest_files,
        mode="remote",
        remote_url_base="https://upstream.test/corpus",
    )


def _make_opener(byte_responses: dict[str, bytes]) -> Callable[[str], IO[bytes]]:
    """Construct a UrlOpener that returns bytes for known URLs and raises
    ConnectionError for unknown ones."""
    def opener(url: str) -> IO[bytes]:
        if url not in byte_responses:
            raise ConnectionError(f"no fixture for {url}")
        return io.BytesIO(byte_responses[url])
    return opener


def _flaky_opener(byte_responses: dict[str, bytes], unreachable_url: str) -> Callable[[str], IO[bytes]]:
    """Like _make_opener but raises ConnectionError on `unreachable_url` to
    simulate a transient network failure on one entry."""
    def opener(url: str) -> IO[bytes]:
        if url == unreachable_url:
            raise ConnectionError(f"simulated network failure for {url}")
        if url not in byte_responses:
            raise ConnectionError(f"no fixture for {url}")
        return io.BytesIO(byte_responses[url])
    return opener


def _read(km: Path):
    from resonance_lattice.store import archive
    return archive.read(km)


def _sha(text: str) -> str:
    from resonance_lattice.store.base import sha256_hex
    return sha256_hex(text)


def run() -> int:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        contents_initial = {
            "a.md": "# Alpha\n\nFirst doc about authentication and login flows. "
                    "Sessions persist for 24 hours by default.",
            "b.md": "# Beta\n\nSecond doc about credentials and tokens. "
                    "Tokens rotate weekly.",
        }
        km = _build_remote(root, contents_initial)
        c0 = _read(km)
        n0 = len(c0.registry)
        manifest0 = dict(c0.remote_manifest)

        # ---- Guarantee 1: sync against unchanged upstream is a no-op ----
        unchanged_responses = {
            spec["url"]: contents_initial[path].encode("utf-8")
            for path, spec in manifest0.items()
        }
        from resonance_lattice.cli.maintain import cmd_sync
        # Inject our fake opener via direct construction of the index then
        # patching `cmd_sync` is awkward; cleaner is to call the underlying
        # sync logic with a fixture index. But cmd_sync constructs its own
        # index. We patch `_default_opener` for the sync test.
        import resonance_lattice.store.remote_index as ri_mod
        original_opener = ri_mod._default_opener
        ri_mod._default_opener = _make_opener(unchanged_responses)
        try:
            rc = cmd_sync(_Args(
                knowledge_model=str(km),
                upstream_manifest=None, batch_size=4,
                discard_optimised=False, dry_run=False,
                treat_unreachable_as_removed=False,
            ))
        finally:
            ri_mod._default_opener = original_opener
        if rc != 0:
            print(f"[incremental_sync] FAIL guarantee 1: sync rc={rc}",
                  file=sys.stderr)
            return 1
        c1 = _read(km)
        if len(c1.registry) != n0:
            print(f"[incremental_sync] FAIL guarantee 1: passage count "
                  f"changed {n0} → {len(c1.registry)}", file=sys.stderr)
            return 1
        print("[incremental_sync] guarantee 1 (no-op sync) OK", file=sys.stderr)

        # ---- Guarantee 2: modify one file → only that file re-encoded ----
        a_idxs_old = [c.passage_idx for c in c1.registry if c.source_file == "a.md"]
        a_rows_before = c1.bands["base"][a_idxs_old].copy()

        contents_modified = dict(contents_initial)
        contents_modified["b.md"] = (
            "# Beta v2\n\nRewritten content about API keys, secret rotation, "
            "and revocation. New auth flow forces logout on every revoke."
        )
        modified_responses = {
            spec["url"]: contents_modified[path].encode("utf-8")
            for path, spec in manifest0.items()
        }
        ri_mod._default_opener = _make_opener(modified_responses)
        try:
            rc = cmd_sync(_Args(
                knowledge_model=str(km),
                upstream_manifest=None, batch_size=4,
                discard_optimised=False, dry_run=False,
                treat_unreachable_as_removed=False,
            ))
        finally:
            ri_mod._default_opener = original_opener
        if rc != 0:
            print(f"[incremental_sync] FAIL guarantee 2: sync rc={rc}",
                  file=sys.stderr)
            return 1
        c2 = _read(km)
        a_idxs_new = [c.passage_idx for c in c2.registry if c.source_file == "a.md"]
        a_rows_after = c2.bands["base"][a_idxs_new]
        if not np.array_equal(a_rows_before, a_rows_after):
            print(f"[incremental_sync] FAIL guarantee 2: a.md rows changed "
                  f"(should be byte-identical lift)", file=sys.stderr)
            return 1
        # Poll-mode SHA writeback regression guard: after a poll-mode sync
        # that modified b.md, the new manifest entry for b.md MUST hold the
        # sha256 of the new bytes, not the old pinned sha. The original
        # `spec.get("sha256") or sha256_hex(text)` writeback would silently
        # keep the old (truthy) sha here and produce an inconsistent
        # archive: new bands + new registry + stale manifest pin.
        new_b_sha = c2.remote_manifest["b.md"]["sha256"]
        if new_b_sha != _sha(contents_modified["b.md"]):
            print(f"[incremental_sync] FAIL guarantee 2 (poll-mode sha "
                  f"writeback): manifest b.md sha={new_b_sha[:12]} but "
                  f"live-bytes sha={_sha(contents_modified['b.md'])[:12]}",
                  file=sys.stderr)
            return 1
        print("[incremental_sync] guarantee 2 (selective re-encode + poll-mode sha) OK",
              file=sys.stderr)

        # ---- Guarantee 3 + 4: catalog mode discovers add + remove ----
        added_path = "c.md"
        added_text = (
            "# Gamma\n\nThird doc about session storage in Redis. "
            "Each session has a TTL.")
        # Build a synthetic upstream catalog with: a.md unchanged,
        # b.md modified again, c.md NEW, b.md still present.
        # (Remove a.md from upstream → removed bucket fires.)
        catalog: dict[str, dict[str, str]] = {
            "b.md": {
                "url": "https://upstream.test/corpus/b.md",
                "sha256": _sha(contents_modified["b.md"]),
            },
            added_path: {
                "url": f"https://upstream.test/corpus/{added_path}",
                "sha256": _sha(added_text),
            },
        }
        catalog_url = "https://upstream.test/manifest.json"
        catalog_bytes = json.dumps(catalog).encode("utf-8")
        catalog_responses = {
            catalog_url: catalog_bytes,
            "https://upstream.test/corpus/b.md": contents_modified["b.md"].encode("utf-8"),
            f"https://upstream.test/corpus/{added_path}": added_text.encode("utf-8"),
        }
        ri_mod._default_opener = _make_opener(catalog_responses)
        try:
            rc = cmd_sync(_Args(
                knowledge_model=str(km),
                upstream_manifest=catalog_url, batch_size=4,
                discard_optimised=False, dry_run=False,
            ))
        finally:
            ri_mod._default_opener = original_opener
        if rc != 0:
            print(f"[incremental_sync] FAIL guarantees 3+4: sync rc={rc}",
                  file=sys.stderr)
            return 1
        c3 = _read(km)
        sources = {c.source_file for c in c3.registry}
        if added_path not in sources:
            print(f"[incremental_sync] FAIL guarantee 3: {added_path} missing "
                  f"from registry; sources={sources}", file=sys.stderr)
            return 1
        if "a.md" in sources:
            print(f"[incremental_sync] FAIL guarantee 4: a.md still in "
                  f"registry after upstream removal; sources={sources}",
                  file=sys.stderr)
            return 1
        print("[incremental_sync] guarantee 3 (added) OK", file=sys.stderr)
        print("[incremental_sync] guarantee 4 (removed) OK", file=sys.stderr)

        # ---- Guarantee 5: manifest sha pin advances ----
        b_pin = c3.remote_manifest["b.md"]["sha256"]
        if b_pin != _sha(contents_modified["b.md"]):
            print(f"[incremental_sync] FAIL guarantee 5: b.md sha pin "
                  f"didn't advance to new content hash. got={b_pin[:12]} "
                  f"expected={_sha(contents_modified['b.md'])[:12]}",
                  file=sys.stderr)
            return 1
        c_pin = c3.remote_manifest[added_path]["sha256"]
        if c_pin != _sha(added_text):
            print(f"[incremental_sync] FAIL guarantee 5: {added_path} sha "
                  f"pin missing or wrong. got={c_pin[:12] if c_pin else None}",
                  file=sys.stderr)
            return 1
        if "a.md" in c3.remote_manifest:
            print(f"[incremental_sync] FAIL guarantee 5: a.md still in "
                  f"manifest after upstream removal", file=sys.stderr)
            return 1
        print("[incremental_sync] guarantee 5 (manifest pin advances) OK",
              file=sys.stderr)

        # ---- Guarantee 6: poll-mode network errors abort by default;
        # --treat-unreachable-as-removed migrates them into removed ----
        # Build a fresh remote archive in a sub-corpus so we don't depend
        # on the previous mutated state.
        unreach_root = root / "unreach"
        unreach_files = {
            "x.md": "Doc X. The quick brown fox jumps over the lazy dog. "
                    "Patterns of authentication need to be carefully tested.",
            "y.md": "Doc Y. Lorem ipsum dolor sit amet about session tokens "
                    "and how the auth flow works in this corpus.",
        }
        km_unreach = _build_remote(unreach_root, unreach_files)
        c_pre = _read(km_unreach)
        manifest_pre = dict(c_pre.remote_manifest)
        n_pre = len(c_pre.registry)
        # Make y.md's URL unreachable; x.md still resolves with unchanged bytes.
        flaky_responses = {
            manifest_pre["x.md"]["url"]: unreach_files["x.md"].encode("utf-8"),
        }
        ri_mod._default_opener = _flaky_opener(
            flaky_responses, manifest_pre["y.md"]["url"],
        )
        try:
            rc_abort = cmd_sync(_Args(
                knowledge_model=str(km_unreach),
                upstream_manifest=None, batch_size=4,
                discard_optimised=False, dry_run=False,
                treat_unreachable_as_removed=False,
            ))
        finally:
            ri_mod._default_opener = original_opener
        if rc_abort == 0:
            print(f"[incremental_sync] FAIL guarantee 6: sync returned 0 "
                  f"despite an unavailable upstream path (should rc=2)",
                  file=sys.stderr)
            return 1
        # Archive must be untouched after the abort.
        c_post_abort = _read(km_unreach)
        if len(c_post_abort.registry) != n_pre:
            print(f"[incremental_sync] FAIL guarantee 6: archive mutated "
                  f"despite abort ({n_pre} → {len(c_post_abort.registry)})",
                  file=sys.stderr)
            return 1

        # With the opt-in flag, sync proceeds and y.md's passages drop.
        ri_mod._default_opener = _flaky_opener(
            flaky_responses, manifest_pre["y.md"]["url"],
        )
        try:
            rc_proceed = cmd_sync(_Args(
                knowledge_model=str(km_unreach),
                upstream_manifest=None, batch_size=4,
                discard_optimised=False, dry_run=False,
                treat_unreachable_as_removed=True,
            ))
        finally:
            ri_mod._default_opener = original_opener
        if rc_proceed != 0:
            print(f"[incremental_sync] FAIL guarantee 6: "
                  f"--treat-unreachable-as-removed sync rc={rc_proceed}",
                  file=sys.stderr)
            return 1
        c_post = _read(km_unreach)
        sources_post = {c.source_file for c in c_post.registry}
        if "y.md" in sources_post:
            print(f"[incremental_sync] FAIL guarantee 6: y.md still in "
                  f"registry after --treat-unreachable-as-removed sync",
                  file=sys.stderr)
            return 1
        print("[incremental_sync] guarantee 6 (unavailable-bucket safety) OK",
              file=sys.stderr)

    print("[incremental_sync] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
