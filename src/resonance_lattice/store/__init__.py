"""Store — the authoritative content layer.

Three storage modes (orthogonal to retrieval):
- bundled: source files packed inside the .rlat ZIP under source/.
- local:   source resolved from disk via --source-root (default).
- remote:  HTTP-backed, SHA-pinned, lockfile-managed.

Knowledge-model file format v4: ZIP archive containing metadata.json, NPZ
band(s), optional ANN index, optional bundled source.

`open_store(km_path, contents, source_root)` is the canonical mode→Store
factory used by every CLI subcommand that needs to resolve passages back
to text. Single source of truth so `rlat search`, `rlat profile`,
`rlat compare`, and `rlat refresh` don't each re-implement the dispatch.

Phase 2 deliverable. See base plan §2, §5.
"""

from __future__ import annotations

from pathlib import Path

from .._paths import xdg_cache_root
from ..config import StoreMode
from .archive import ArchiveContents
from .base import Store, sha256_hex
from .bundled import BundledStore
from .local import LocalStore
from .remote import RemoteStore


def _default_remote_cache_dir(km_path: str | Path) -> Path:
    """Per-knowledge-model on-disk cache for downloaded source files.

    `<XDG_CACHE_HOME or ~/.cache>/rlat/remote/<sha>/`. The sha is hashed
    from the resolved .rlat path so two knowledge models at different
    paths don't share a cache (avoids one's `rlat sync` silently
    invalidating the other).
    """
    digest = sha256_hex(str(Path(km_path).resolve()))[:16]
    return xdg_cache_root() / "remote" / digest

__all__ = [
    "ArchiveContents",
    "BundledStore",
    "LocalStore",
    "RemoteStore",
    "Store",
    "open_store",
]


def open_store(
    km_path: str | Path,
    contents: ArchiveContents,
    source_root: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> Store:
    """Construct the right Store from the archive's recorded mode.

    - `local` (default): uses `--source-root` or the build-time root
      recorded in `metadata.build_config`. Raises `ValueError` if neither.
    - `bundled`: reads from the .rlat itself.
    - `remote`: SHA-pinned HTTP fetcher backed by an on-disk cache. Pulls
      the manifest from `contents.remote_manifest` (the loaded `.rlat`
      writes it under `manifest.json` in remote builds). Cache dir
      defaults to `~/.cache/rlat/remote/<km-sha>/` — pass `cache_dir=`
      to override (useful for tests + multi-machine deployments).
    """
    mode = StoreMode(contents.metadata.store_mode)
    if mode is StoreMode.BUNDLED:
        return BundledStore(km_path)
    if mode is StoreMode.LOCAL:
        root = source_root or contents.metadata.build_config.get("source_root")
        if not root:
            raise ValueError(
                f"local-mode knowledge model has no recorded source_root and "
                f"--source-root was not supplied; pass --source-root <dir>"
            )
        return LocalStore(root)
    # Remote mode
    if not contents.remote_manifest:
        raise ValueError(
            f"{km_path} declares store_mode='remote' but the remote_manifest "
            f"is empty — archive is corrupt or built without --remote-url-base"
        )
    return RemoteStore(
        manifest=contents.remote_manifest,
        cache_dir=cache_dir or _default_remote_cache_dir(km_path),
    )
