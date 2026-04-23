# SPDX-License-Identifier: BUSL-1.1
"""rlat refresh: re-index drifted files in an external-mode knowledge model.

Preserves the field tensor where chunk hashes still match — only
drifted / missing / newly-added chunks trigger forget+superpose cycles.
Unchanged chunks stay byte-identical in the field.

The workflow:
  1. Load the knowledge model with source_root so we get LocalStore.
  2. Group manifest entries by source_file.
  3. For each file:
     - Missing → forget every chunk bound to that file (file deleted).
     - Exists → re-chunk, compare content_hash per chunk.
       * Match → skip (field preserved).
       * Mismatch → update(old_sid, new_phase) — atomic forget+superpose
         that keeps the source_id stable so downstream references survive.
       * New chunk (file grew) → superpose with a predictable source_id
         derived from the existing id prefix for that file.
       * Stale chunk (file shrunk) → remove.
  4. Save the knowledge model back in external mode with the updated manifest.

Requires an encoder. For v1.0.0 the caller passes one in (typically the
same one used to build the knowledge model); the CLI resolves it from the
knowledge model's embedded encoder config or from --encoder flags.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from resonance_lattice.chunker import auto_chunk
from resonance_lattice.encoder import Encoder
from resonance_lattice.lattice import Lattice
from resonance_lattice.serialise import RlatHeader
from resonance_lattice.store import LocalStore, SourceContent

if TYPE_CHECKING:
    from resonance_lattice.remote.github import GithubFetcher


@dataclass
class RefreshReport:
    """Summary of what changed during a refresh."""

    files_checked: int = 0
    files_clean: int = 0          # no drift detected
    files_drifted: int = 0        # had at least one changed/added/removed chunk
    files_missing: int = 0        # file no longer exists on disk
    chunks_preserved: int = 0     # content_hash matched → no field change
    chunks_updated: int = 0       # content_hash mismatched → forget + superpose
    chunks_added: int = 0         # new chunks from grown files
    chunks_removed: int = 0       # stale chunks from shrunk or missing files
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"files checked:   {self.files_checked}",
            f"  clean:         {self.files_clean}",
            f"  drifted:       {self.files_drifted}",
            f"  missing:       {self.files_missing}",
            f"chunks preserved: {self.chunks_preserved}  (field untouched)",
            f"chunks updated:  {self.chunks_updated}  (forget + superpose)",
            f"chunks added:    {self.chunks_added}",
            f"chunks removed:  {self.chunks_removed}",
        ]
        if self.warnings:
            lines.append("warnings:")
            lines.extend(f"  - {w}" for w in self.warnings[:10])
            if len(self.warnings) > 10:
                lines.append(f"  ... and {len(self.warnings) - 10} more")
        return "\n".join(lines)


def _extract_manifest(lattice: Lattice) -> dict[str, Any]:
    """Pull the __source_manifest__ JSON out of the store, regardless of
    whether the store is an LocalStore (where the manifest lives in
    meta_store) or a regular SourceStore."""
    if isinstance(lattice.store, LocalStore):
        return dict(lattice.store._manifest)
    entry = lattice.store.retrieve("__source_manifest__")
    if entry and entry.full_text:
        try:
            return json.loads(entry.full_text)
        except Exception:
            return {}
    return {}


def _id_prefix_for_file(
    file_source_ids: list[str], fallback_stem: str
) -> str:
    """Derive the source_id prefix used for a given file from the existing
    ids bound to it. Existing build pipelines use patterns like
    `{stem}_{slug}_{path_hash}_{idx:04d}` or `{stem}_{idx:06d}` — we keep
    the part before the trailing `_NNNN` so new chunks appended during
    refresh slot in predictably.

    Empty list → fall back to the file stem, which matches the simpler
    build variant."""
    if not file_source_ids:
        return fallback_stem
    sample = file_source_ids[0]
    parts = sample.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return fallback_stem


def _make_content(source_id: str, chunk, source_file_display: str) -> SourceContent:
    """Build a SourceContent that carries everything Lattice.save() will
    need to emit a correct manifest entry for this chunk.

    ``source_file_display`` is whatever string should appear in the
    knowledge model's manifest for this source — ``str(abs_path)`` during
    local refresh, the posix-relative path during remote sync.
    """
    return SourceContent(
        source_id=source_id,
        summary=chunk.text[:300],
        full_text=chunk.text,
        metadata={
            "source_file": source_file_display,
            "heading": chunk.heading,
            "chunk_type": chunk.chunk_type,
            "char_offset": chunk.char_offset,
            "content_hash": chunk.content_hash,
        },
    )


def _reconcile_file_chunks(
    *,
    lattice: Lattice,
    encoder: Encoder,
    source_file_display: str,
    entries: list[tuple[str, dict]],
    live_chunks: list | None,
    report: RefreshReport,
) -> bool:
    """Reconcile one file's manifest entries against live chunks.

    Shared by ``refresh_cartridge`` (chunks read from disk) and
    ``sync_remote_cartridge`` (chunks fetched from HTTP). The caller is
    responsible for obtaining ``live_chunks`` — ``None`` means the file
    is no longer part of the corpus and every chunk bound to it should
    be removed.

    Returns True when the file had any chunk-level changes.
    """
    if live_chunks is None:
        had_changes = False
        for sid, _entry in entries:
            if lattice.remove(sid):
                report.chunks_removed += 1
                had_changes = True
        return had_changes

    # Index existing entries by chunk_index for fast lookup. Entries
    # without a chunk_index (pre-A2 builds) can't be aligned positionally
    # — fall back to hash-based matching.
    by_idx: dict[int, tuple[str, dict]] = {}
    orphans: list[tuple[str, dict]] = []
    for sid, entry in entries:
        idx = entry.get("chunk_index")
        if isinstance(idx, int):
            by_idx[idx] = (sid, entry)
        else:
            orphans.append((sid, entry))

    stem_fallback = Path(source_file_display).stem
    id_prefix = _id_prefix_for_file(
        [sid for sid, _ in entries], fallback_stem=stem_fallback
    )
    file_had_changes = False

    for i, chunk in enumerate(live_chunks):
        if i in by_idx:
            sid, entry = by_idx[i]
            stored_hash = entry.get("content_hash")
            if stored_hash and stored_hash == chunk.content_hash:
                # Field stays byte-identical — this is the whole point
                # of refresh/sync-over-rebuild.
                report.chunks_preserved += 1
                continue
            phase = encoder.encode_passage(chunk.text)
            content = _make_content(sid, chunk, source_file_display)
            lattice.update(sid, phase, new_salience=1.0, new_content=content)
            report.chunks_updated += 1
            file_had_changes = True
        else:
            new_sid = (
                f"{id_prefix}_{i:04d}" if id_prefix else f"{stem_fallback}_{i:04d}"
            )
            # Collision guard: duplicate stems / concurrent runs.
            suffix = 0
            candidate = new_sid
            while candidate in lattice._phase_cache:
                suffix += 1
                candidate = f"{new_sid}_add{suffix}"
            new_sid = candidate
            phase = encoder.encode_passage(chunk.text)
            content = _make_content(new_sid, chunk, source_file_display)
            lattice.superpose(
                phase_spectrum=phase, source_id=new_sid, content=content,
            )
            report.chunks_added += 1
            file_had_changes = True

    for idx, (sid, _entry) in by_idx.items():
        if idx >= len(live_chunks):
            if lattice.remove(sid):
                report.chunks_removed += 1
                file_had_changes = True

    for sid, entry in orphans:
        stored_hash = entry.get("content_hash")
        if stored_hash and any(c.content_hash == stored_hash for c in live_chunks):
            report.chunks_preserved += 1
        else:
            if lattice.remove(sid):
                report.chunks_removed += 1
                file_had_changes = True

    return file_had_changes


def refresh_cartridge(
    cartridge_path: str | Path,
    source_root: str | Path,
    encoder: Encoder | None = None,
    output_path: str | Path | None = None,
) -> RefreshReport:
    """Refresh an external-mode knowledge model against the live source tree.

    Args:
        cartridge_path: Input .rlat file. Must have been built with
            `store_mode="external"` — embedded knowledge models don't have a
            manifest to diff against.
        source_root: Directory the manifest paths resolve under. This is
            typically the same root recorded in `__source_root_hint__` but
            the caller can override when the corpus has moved.
        encoder: Encoder used to re-encode drifted / new chunks. Must
            produce phase vectors compatible with the knowledge model's field
            (same bands × dim). Pass None to have the knowledge model restore
            its embedded encoder config; raises if no encoder is available.
        output_path: Where to write the refreshed knowledge model. Defaults to
            overwriting cartridge_path in place.

    Returns:
        RefreshReport summarising what changed.
    """
    cartridge_path = Path(cartridge_path)
    source_root = Path(source_root).resolve()
    output_path = Path(output_path) if output_path else cartridge_path

    report = RefreshReport()

    # Embedded cartridges carry no manifest to diff against, so refresh
    # can't operate on them. Detect this from the on-disk header *before*
    # loading — passing source_root at load time would otherwise construct
    # an LocalStore around an embedded cartridge and silently "work"
    # but with no entries, producing a confusing empty report.
    with open(cartridge_path, "rb") as _f:
        _hdr = RlatHeader.from_bytes(_f.read(RlatHeader.SIZE))
    if _hdr.store_mode != "external":
        raise ValueError(
            f"rlat refresh requires an external-mode cartridge; "
            f"{cartridge_path} has store_mode={_hdr.store_mode!r}. "
            f"Rebuild with `--store-mode external` first."
        )

    # restore_encoder when the caller didn't supply one — refresh needs an
    # encoder to re-embed drifted/new chunks, and the cartridge's own
    # encoder config is the most reliable fallback (guarantees build/query
    # parity).
    lattice = Lattice.load(
        cartridge_path,
        source_root=source_root,
        restore_encoder=(encoder is None),
    )
    if not isinstance(lattice.store, LocalStore):
        # Defensive: header said external, load path should've given us
        # an LocalStore. If not, something's off — fail loudly.
        raise ValueError(
            f"rlat refresh: cartridge header says external but load "
            f"produced {type(lattice.store).__name__}. Likely a stale "
            f"cartridge built against a pre-A1 codebase."
        )
    if encoder is not None:
        lattice.encoder = encoder
    if lattice.encoder is None:
        raise ValueError(
            "refresh_cartridge could not resolve an encoder. Pass one "
            "explicitly or ensure the knowledge model has an embedded encoder "
            "config (built with a non-None lattice.encoder)."
        )
    encoder = lattice.encoder

    manifest = _extract_manifest(lattice)

    # Bucket manifest entries by their source_file (skip reserved __ keys).
    by_file: dict[str, list[tuple[str, dict]]] = {}
    for sid, entry in manifest.items():
        if sid.startswith("__"):
            continue
        if isinstance(entry, str):
            entry = {"source_file": entry}
        elif not isinstance(entry, dict):
            continue
        sf = entry.get("source_file", "")
        if not sf:
            continue
        by_file.setdefault(sf, []).append((sid, entry))

    for source_file_rel, entries in by_file.items():
        report.files_checked += 1
        abs_path = (source_root / source_file_rel).resolve()
        try:
            exists = abs_path.exists() and abs_path.is_file()
        except OSError:
            exists = False

        if not exists:
            report.files_missing += 1
            _reconcile_file_chunks(
                lattice=lattice, encoder=encoder,
                source_file_display=str(abs_path),
                entries=entries, live_chunks=None, report=report,
            )
            continue

        try:
            text = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            report.warnings.append(f"read failed for {abs_path}: {exc}")
            continue

        live_chunks = auto_chunk(text, source_file=str(abs_path))
        file_had_changes = _reconcile_file_chunks(
            lattice=lattice, encoder=encoder,
            source_file_display=str(abs_path),
            entries=entries, live_chunks=live_chunks, report=report,
        )
        if file_had_changes:
            report.files_drifted += 1
        else:
            report.files_clean += 1

    # Persist back. We save in external mode (A6 will flip this as the
    # default; for now the refresh path keeps whatever mode the caller
    # used at build time — external is the only supported input here).
    lattice.save(output_path, store_mode="external")

    return report


# ── Remote mode: freshness + sync ──────────────────────────────────────
#
# These mirror ``refresh_cartridge``'s shape for remote-origin
# cartridges (store_mode == "remote"). Freshness is read-only and
# reports upstream drift without touching the cartridge; sync applies
# the diff through the same ``_reconcile_file_chunks`` helper as local
# refresh, so both code paths evolve together.

from dataclasses import (
    dataclass as _dc_remote,  # noqa: E402  (late import keeps refresh module self-contained)
)


@_dc_remote
class RemoteFreshnessReport:
    """Read-only drift summary between a knowledge model's pinned SHA and upstream HEAD."""

    pinned_sha: str
    head_sha: str
    commits_behind: int
    added: list[str]
    modified: list[str]
    removed: list[str]

    def is_stale(self) -> bool:
        return self.pinned_sha != self.head_sha and (
            bool(self.added) or bool(self.modified) or bool(self.removed) or self.commits_behind > 0
        )

    def __str__(self) -> str:
        short_pin = self.pinned_sha[:10] if self.pinned_sha else "<unknown>"
        short_head = self.head_sha[:10] if self.head_sha else "<unknown>"
        if not self.is_stale():
            return f"pinned at {short_pin} — up to date with upstream {short_head}"
        lines = [
            f"pinned at  {short_pin}",
            f"upstream   {short_head}",
            f"diff       +{len(self.added)} added, ~{len(self.modified)} modified, -{len(self.removed)} removed",
            "run `rlat sync <cart>` to apply.",
        ]
        return "\n".join(lines)


def _load_remote_origin(lattice: Lattice) -> dict:
    """Pull __remote_origin__ out of the knowledge model, as a dict.

    Accepts both full_text-encoded JSON (how CLI build writes it) and
    metadata-based entries (defensive — older builds). Empty dict means
    the knowledge model isn't remote-mode.
    """
    sc = lattice.store.retrieve("__remote_origin__")
    if sc is None:
        return {}
    if sc.full_text:
        try:
            return json.loads(sc.full_text)
        except Exception:
            pass
    return dict(sc.metadata or {})


def _require_remote_origin(
    lattice: Lattice, cartridge_path: Path,
) -> tuple[dict, GithubFetcher]:
    """Validate ``lattice`` is remote-mode and return (origin_meta, fetcher)."""
    from resonance_lattice.remote.github import (
        GithubFetcher as _GH,
    )
    from resonance_lattice.remote.github import (
        GithubOrigin as _GO,
    )
    origin_meta = _load_remote_origin(lattice)
    if not origin_meta or origin_meta.get("type") != "github":
        raise ValueError(
            f"{cartridge_path} is not a remote-mode cartridge (no "
            f"__remote_origin__ with type='github'). Build with "
            f"`rlat build <github-url> ...` to create one."
        )
    origin = _GO(
        org=origin_meta["org"],
        repo=origin_meta["repo"],
        ref=origin_meta.get("ref"),
    )
    return origin_meta, _GH(origin=origin)


def check_remote_freshness(cartridge_path: str | Path) -> RemoteFreshnessReport:
    """Read-only upstream drift check. One GitHub API call.

    Never mutates the knowledge model; never touches the local disk cache.
    Returns a ``RemoteFreshnessReport`` describing the diff between the
    knowledge model's pinned SHA and the origin's HEAD on the pinned ref (or
    default branch if ref is None).
    """
    cartridge_path = Path(cartridge_path)
    with open(cartridge_path, "rb") as _f:
        _hdr = RlatHeader.from_bytes(_f.read(RlatHeader.SIZE))
    if _hdr.store_mode != "remote":
        raise ValueError(
            f"rlat freshness needs a remote-mode cartridge; "
            f"{cartridge_path} has store_mode={_hdr.store_mode!r}."
        )

    lattice = Lattice.load(cartridge_path, restore_encoder=False)
    origin_meta, fetcher = _require_remote_origin(lattice, cartridge_path)
    pinned = origin_meta.get("commit_sha", "")
    head = fetcher.resolve_sha(origin_meta.get("ref"))
    if pinned == head:
        return RemoteFreshnessReport(
            pinned_sha=pinned, head_sha=head, commits_behind=0,
            added=[], modified=[], removed=[],
        )
    diff = fetcher.compare(pinned, head, path_prefix=origin_meta.get("path_prefix"))
    # GitHub's compare API returns a commit count under "total_commits"
    # when we request the full compare body; our minimal _http_json in
    # phase 3 doesn't carry that through, so derive a lower bound from
    # the changed-file count until we extend the fetcher.
    commits_behind = max(
        len(diff.get("added", [])) + len(diff.get("modified", [])) + len(diff.get("removed", [])),
        1,
    )
    return RemoteFreshnessReport(
        pinned_sha=pinned, head_sha=head,
        commits_behind=commits_behind,
        added=list(diff.get("added", [])),
        modified=list(diff.get("modified", [])),
        removed=list(diff.get("removed", [])),
    )


def sync_remote_cartridge(
    cartridge_path: str | Path,
    encoder: Encoder | None = None,
    output_path: str | Path | None = None,
) -> RefreshReport:
    """Pull the upstream diff, apply it through the shared reconciler.

    Only changed files get refetched — O(changed), not O(corpus). The
    knowledge model's ``__remote_origin__.commit_sha`` is bumped to the new
    head on success; a failed sync leaves the knowledge model at its original
    pinned SHA so queries continue to work.
    """
    cartridge_path = Path(cartridge_path)
    output_path = Path(output_path) if output_path else cartridge_path

    with open(cartridge_path, "rb") as _f:
        _hdr = RlatHeader.from_bytes(_f.read(RlatHeader.SIZE))
    if _hdr.store_mode != "remote":
        raise ValueError(
            f"rlat sync needs a remote-mode cartridge; "
            f"{cartridge_path} has store_mode={_hdr.store_mode!r}."
        )

    lattice = Lattice.load(
        cartridge_path, restore_encoder=(encoder is None),
    )
    if encoder is not None:
        lattice.encoder = encoder
    if lattice.encoder is None:
        raise ValueError(
            "sync_remote_cartridge could not resolve an encoder. Pass "
            "one explicitly or ensure the knowledge model has an embedded "
            "encoder config."
        )
    encoder = lattice.encoder

    origin_meta, fetcher = _require_remote_origin(lattice, cartridge_path)
    pinned = origin_meta.get("commit_sha", "")
    head = fetcher.resolve_sha(origin_meta.get("ref"))
    report = RefreshReport()
    if pinned == head:
        return report  # already current

    path_prefix_raw = origin_meta.get("path_prefix")
    path_prefix = (path_prefix_raw.strip("/") + "/") if path_prefix_raw else ""
    diff = fetcher.compare(pinned, head, path_prefix=path_prefix_raw)
    manifest = _extract_manifest(lattice)

    # Bucket existing manifest entries by source_file so we can apply
    # per-file reconciliation the same way refresh_cartridge does.
    # Manifest paths are scope-relative for scoped builds; diff paths
    # from the GitHub compare API are always repo-relative — so we strip
    # the prefix from diff paths before lookup and re-apply it when
    # fetching.
    by_file: dict[str, list[tuple[str, dict]]] = {}
    for sid, entry in manifest.items():
        if sid.startswith("__"):
            continue
        if isinstance(entry, str):
            entry = {"source_file": entry}
        elif not isinstance(entry, dict):
            continue
        sf = entry.get("source_file", "")
        if sf:
            by_file.setdefault(sf, []).append((sid, entry))

    def _strip_prefix(upstream_rel: str) -> str:
        if path_prefix and upstream_rel.startswith(path_prefix):
            return upstream_rel[len(path_prefix):]
        return upstream_rel

    def _fetch_chunks(upstream_rel: str, manifest_rel: str) -> list | None:
        try:
            raw = fetcher.fetch(head, upstream_rel)
        except Exception as exc:
            report.warnings.append(f"fetch failed for {upstream_rel}: {exc}")
            return None
        text = raw.decode("utf-8", errors="replace")
        return auto_chunk(text, source_file=manifest_rel)

    # Removed files: drop every chunk bound to them.
    for upstream_rel in diff.get("removed", []):
        manifest_rel = _strip_prefix(upstream_rel)
        entries = by_file.get(manifest_rel, [])
        if not entries:
            continue
        report.files_checked += 1
        report.files_missing += 1
        _reconcile_file_chunks(
            lattice=lattice, encoder=encoder,
            source_file_display=manifest_rel,
            entries=entries, live_chunks=None, report=report,
        )

    # Modified files: refetch, re-chunk, reconcile against existing entries.
    for upstream_rel in diff.get("modified", []):
        manifest_rel = _strip_prefix(upstream_rel)
        entries = by_file.get(manifest_rel, [])
        report.files_checked += 1
        chunks = _fetch_chunks(upstream_rel, manifest_rel)
        if chunks is None:
            continue
        changed = _reconcile_file_chunks(
            lattice=lattice, encoder=encoder,
            source_file_display=manifest_rel,
            entries=entries, live_chunks=chunks, report=report,
        )
        if changed:
            report.files_drifted += 1
        else:
            report.files_clean += 1

    # Added files: fetch, chunk, superpose (no existing entries to match).
    for upstream_rel in diff.get("added", []):
        manifest_rel = _strip_prefix(upstream_rel)
        report.files_checked += 1
        chunks = _fetch_chunks(upstream_rel, manifest_rel)
        if chunks is None:
            continue
        _reconcile_file_chunks(
            lattice=lattice, encoder=encoder,
            source_file_display=manifest_rel,
            entries=[], live_chunks=chunks, report=report,
        )
        report.files_drifted += 1

    # Bump the pinned SHA before save so the cartridge reflects the new origin.
    new_origin = dict(origin_meta)
    new_origin["commit_sha"] = head
    lattice.store.store(SourceContent(
        source_id="__remote_origin__",
        summary=f"pinned remote origin ({new_origin.get('org')}/{new_origin.get('repo')})",
        full_text=json.dumps(new_origin),
        metadata=new_origin,
    ))

    lattice.save(output_path, store_mode="remote")
    return report
