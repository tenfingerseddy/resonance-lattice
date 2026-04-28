"""Storage-mode conversion — `rlat convert` pipeline (Audit 08).

Converts a knowledge model between any pair of `bundled` / `local` /
`remote` modes without rebuilding embeddings. The bands, registry, ANN
index, and optimised W matrix are all storage-mode-independent — they
describe the corpus content, not how source bytes resolve at query time.
What changes between modes is just `metadata.store_mode` plus the
supporting payload (zstd-framed `source/` for bundled, `manifest.json`
for remote, neither for local).

Three correctness invariants (Audit 07 pattern, baked in):

  1. Every kept passage's `content_hash` is re-validated against live
     bytes resolved via the SOURCE mode's Store before write. Drift
     aborts; no silent stale carry-forward into the new mode.
  2. Bands, registry, ANN, optimised W byte-identical pre/post.
     Conversion preserves corpus identity; `passage_id` stable.
  3. metadata.store_mode advances atomically with the payload swap
     (`tmp + os.replace` via `archive.write`). No window where the
     metadata says one mode but payloads still hold the other.

Six pairwise transitions, all routed through one function. The branching
is small: source mode determines how `Store.fetch_all()` retrieves bytes,
target mode determines what supporting payload to compose for
`archive.write()`. Bands + registry + projections + ann_blobs flow
through unchanged.

Audit 08 commit 3/6.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import archive
from .base import RemoteShaMismatch, compute_hash
from .bundled import pack_source_files
from .local import _resolve_safe
from .remote import compose_manifest


@dataclass(frozen=True)
class ConversionResult:
    """Outcome of a conversion. Counts let callers render a one-line
    status banner without re-reading the archive."""
    archive_path: Path
    from_mode: str
    to_mode: str
    n_passages: int
    n_files: int
    n_drifted: int          # always 0 on success — non-zero means we aborted
    output_bytes: int


class ConversionDriftError(Exception):
    """Raised when one or more passages' live `content_hash` doesn't match
    the registry's recorded hash during conversion. Conversion does NOT
    write a new archive in this case — the user must run `rlat refresh`
    (local) or `rlat sync` (remote) to reconcile drift first.

    Carries the per-passage drift report so callers can render a useful
    error or surface paths to fix.
    """

    def __init__(self, drifted_paths: list[str], n_total_passages: int):
        super().__init__(
            f"{len(drifted_paths)} of {n_total_passages} passages "
            f"have drifted source content; convert refuses to carry "
            f"stale bytes into the new mode. Run `rlat refresh` or "
            f"`rlat sync` first to reconcile, then re-run convert."
        )
        self.drifted_paths = drifted_paths
        self.n_total_passages = n_total_passages


def convert(
    archive_path: Path,
    target_mode: str,
    *,
    source_root: Path | None = None,
    remote_url_base: str | None = None,
    output_path: Path | None = None,
) -> ConversionResult:
    """Convert a knowledge model between storage modes.

    `target_mode` is one of `"bundled"`, `"local"`, `"remote"`. `source_root`
    is required when the target is `local` (where to materialise the source
    files) and when the SOURCE is `bundled` and target is anything else
    (where to extract). `remote_url_base` is required when the target is
    `remote`.

    Output defaults to in-place atomic replacement of `archive_path`.

    Raises:
      - `ValueError` on missing required flags or unknown target_mode.
      - `ValueError` on idempotent no-op (`target_mode == current_mode`).
        Caller can catch + surface as a friendly "already in mode X."
      - `ConversionDriftError` if any kept passage's content_hash doesn't
        match the live bytes — the new archive is NOT written.
    """
    if target_mode not in ("bundled", "local", "remote"):
        raise ValueError(
            f"unknown target_mode {target_mode!r}; "
            f"choose from bundled/local/remote"
        )

    archive_path = Path(archive_path)
    output_path = Path(output_path) if output_path else archive_path

    # Read the source archive in full — we need every payload so the write
    # below can re-compose with the source-mode-specific payload swapped.
    contents = archive.read(archive_path)
    from_mode = contents.metadata.store_mode

    if from_mode == target_mode:
        raise ValueError(
            f"{archive_path} is already in mode {target_mode!r}; "
            f"nothing to convert"
        )
    if target_mode == "remote" and not remote_url_base:
        raise ValueError(
            "convert --to remote requires --remote-url-base <url-prefix>"
        )
    # source_root requirement is mode-pair-dependent; resolved per branch
    # below so the error message can be specific.

    # Open a Store for the SOURCE mode so we can bulk-read every file the
    # registry references. open_store() resolves the mode from the loaded
    # contents.metadata.store_mode automatically.
    # Local import: store/__init__.py imports from each subclass module;
    # if conversion.py imported open_store at module-top, any
    # `from store import conversion` would trigger the chain
    # store → bundled/local/remote → archive → conversion (back into
    # conversion mid-load). Late binding inside the function avoids it.
    from . import open_store
    src_store = open_store(archive_path, contents, source_root=source_root)
    unique_sources = sorted({c.source_file for c in contents.registry})
    # Remote SHA-pin failure during fetch_all surfaces here as
    # RemoteShaMismatch — wrap as ConversionDriftError so the CLI's friendly
    # error path (rc=2 + drifted-paths listing) catches it instead of the
    # user seeing an unhandled traceback.
    try:
        src_texts: dict[str, str] = src_store.fetch_all(unique_sources)
    except RemoteShaMismatch as exc:
        raise ConversionDriftError(
            [exc.source_file], len(contents.registry),
        ) from exc

    # Invariant 1: every kept passage's content_hash MUST match the live
    # bytes resolved via the source store. Conversion refuses to carry
    # stale bytes into a new mode; the user reconciles first via
    # refresh/sync.
    drifted: list[str] = []
    for c in contents.registry:
        text = src_texts.get(c.source_file)
        if text is None:
            drifted.append(c.source_file)
            continue
        slice_text = text[c.char_offset:c.char_offset + c.char_length]
        if compute_hash(slice_text) != c.content_hash:
            drifted.append(f"{c.source_file}:{c.char_offset}+{c.char_length}")
    if drifted:
        raise ConversionDriftError(drifted, len(contents.registry))

    # Compose target-mode supporting payloads. Bands, registry,
    # projections, ann_blobs flow through unchanged.
    new_remote_manifest: dict[str, dict[str, str]] = {}
    new_source_files: dict[str, bytes] = {}

    if target_mode == "bundled":
        # Pack every source file as a zstd frame. Same primitive cli/build
        # uses for fresh bundled archives.
        new_source_files = pack_source_files(src_texts)

    elif target_mode == "local":
        if source_root is None:
            raise ValueError(
                "convert --to local requires --source-root <dir> "
                "where the source files should be materialised"
            )
        source_root = Path(source_root)
        source_root.mkdir(parents=True, exist_ok=True)
        materialised: list[str] = []
        for rel_posix, text in src_texts.items():
            # Path-traversal guard: a tampered .rlat could carry source_file
            # keys like `"../../etc/passwd"` or absolute paths. Resolve via
            # the same guard LocalStore reads through, so writes can never
            # escape source_root.
            dst = _resolve_safe(source_root, rel_posix)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text, encoding="utf-8")
            materialised.append(str(dst))
        # Update build_config so future `rlat refresh` runs walk the
        # materialised mirror, not the original source paths recorded by
        # the source-mode build (which may be remote URLs or absent).
        contents.metadata.build_config["source_root"] = str(source_root)
        contents.metadata.build_config["source_paths"] = sorted(materialised)

    elif target_mode == "remote":
        # Single source of truth for url+sha256 manifest composition; same
        # helper feeds fresh remote-mode `rlat build`. The user is
        # responsible for actually publishing those bytes at the asserted
        # URLs; convert records what the manifest claims.
        new_remote_manifest = compose_manifest(src_texts, remote_url_base)
        contents.metadata.build_config["upstream_url_base"] = (
            remote_url_base.rstrip("/")
        )
        # Reset pinned_ref — the new manifest defines a fresh pin baseline.
        contents.metadata.build_config["pinned_ref"] = ""

    # Update metadata.store_mode and write atomically. Bands, registry,
    # projections, and ann_blobs are passed through unchanged — same
    # bytes, just under a different mode declaration.
    contents.metadata.store_mode = target_mode
    archive.write(
        output_path,
        metadata=contents.metadata,
        bands=contents.bands,
        registry=contents.registry,
        projections=contents.projections,
        ann_blobs=contents.ann_blobs,
        source_files=new_source_files or None,
        remote_manifest=new_remote_manifest or None,
    )

    return ConversionResult(
        archive_path=output_path,
        from_mode=from_mode,
        to_mode=target_mode,
        n_passages=len(contents.registry),
        n_files=len(unique_sources),
        n_drifted=0,
        output_bytes=output_path.stat().st_size,
    )
