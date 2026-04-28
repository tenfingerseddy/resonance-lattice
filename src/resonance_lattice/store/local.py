"""Local store — source files resolved from disk via `--source-root`.

The default storage mode for the live-edit workflow: the corpus stays on
disk, the knowledge model only records pointers
`(source_file, char_offset, char_length)`. Drift is detected by re-hashing
the slice against `PassageCoord.content_hash`.

`fetch` / `verify` / per-instance text cache are inherited from
`store.base.Store`; this file owns the per-mode read primitive plus the
path-traversal guard.

`verify` returns `"missing"` when the file no longer exists, `"drifted"` when
the on-disk content has changed since build, `"verified"` otherwise.
`rlat refresh` (Phase 3 #30) is the recovery path for drift.

Path-traversal hazard: `source_file` keys are written by build but
deserialised from a JSONL inside the .rlat — a tampered archive could
inject `"../../etc/passwd"`. `_resolve_safe` rejects any resolved path that
escapes `source_root`.

Phase 2 deliverable. Base plan §5.
"""

from __future__ import annotations

from pathlib import Path

from .base import Store


def _resolve_safe(source_root: Path, source_file: str) -> Path:
    """Resolve `source_root / source_file` and reject path traversal.

    Raises `ValueError` if the resolved target sits outside `source_root` —
    prevents a tampered registry entry like `"../../etc/passwd"` from being
    served as a fetch result. `resolve(strict=False)` so the check still
    fires for files that don't exist yet (the existence error is for `read`
    to raise, not this guard). Symlinks are followed; an in-root symlink
    pointing outside the root is correctly rejected.
    """
    target = (source_root / source_file).resolve(strict=False)
    root = source_root.resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"source_file {source_file!r} escapes source_root {source_root}; "
            f"refusing to read {target}"
        ) from exc
    return target


class LocalStore(Store):
    """Read source from disk relative to a source_root path."""

    def __init__(self, source_root: str | Path):
        super().__init__()
        self.source_root = Path(source_root)

    def _read_full_text_uncached(self, source_file: str) -> str:
        path = _resolve_safe(self.source_root, source_file)
        # `Path.read_text()` uses universal-newlines text mode, normalising
        # `\r\n` and `\r` to `\n` on read. Drift is cross-platform stable
        # ONLY because the build-time chunker reads the same way; if any
        # build path ever switches to `read_bytes()` or
        # `open(..., newline="")`, recorded hashes diverge across platforms.
        # UTF-8 is explicit because the build commits to it.
        return path.read_text(encoding="utf-8")
