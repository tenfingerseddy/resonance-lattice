"""CLI helpers: load a knowledge model / open a store with friendly errors.

The triplet — `is_file()` check + `archive.read` try/except + `print(error,
file=sys.stderr); sys.exit(1)` — is shared across `search`, `profile`,
`compare`, `summary`, `refresh`. Lifted here so adding a new subcommand
doesn't recopy the same six lines, and so the error wording stays
consistent across the surface.

`open_store_or_exit` is the same lift for `open_store(...)` — the
`(ValueError, NotImplementedError) -> friendly stderr exit(1)` pattern
appeared in `search`, `summary`, and the optimise + skill-context paths.

Lives in `cli/` (not `store/`) because `sys.exit` and stderr-print are
CLI-layer concerns; the store layer must stay library-callable.
"""

from __future__ import annotations

import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..store import archive, open_store
from ..store.base import Store


def load_or_exit(km_path: str | Path) -> archive.ArchiveContents:
    """Read a v4 .rlat archive or `sys.exit(1)` with a friendly message.

    Catches the three documented failure modes (path not a file, ZIP corrupt,
    metadata or band schema rejected) and routes each to a stderr line that
    names what to fix. On success returns `ArchiveContents` directly.
    """
    p = Path(km_path)
    if not p.is_file():
        print(f"error: {p} is not a file", file=sys.stderr)
        sys.exit(1)
    try:
        return archive.read(p)
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        print(f"error: {p} is not a valid v4 knowledge model: {exc}",
              file=sys.stderr)
        sys.exit(1)


@dataclass(frozen=True)
class BuildSpec:
    """Effective build inputs replayed by `rlat refresh` / `rlat watch`.

    `source_paths` and `extensions` come from the recorded provenance
    (`build_config.source_paths`, `build_config.extensions` — Audit 07).
    `min_chars` / `max_chars` fall back to the build-time defaults
    (`_DEFAULT_MIN_CHARS` / `_DEFAULT_MAX_CHARS`) when an older archive
    didn't record them. CLI overrides are applied by the caller via the
    keyword arguments of `load_build_spec`.
    """
    source_root: Path
    source_paths: list[Path]
    extensions: frozenset[str]
    min_chars: int
    max_chars: int


def load_build_spec(
    contents: archive.ArchiveContents,
    *,
    source_paths_override: list[str] | None = None,
    source_root_override: str | None = None,
    extensions_override: list[str] | None = None,
) -> BuildSpec | None:
    """Read the build provenance recorded by `rlat build` and return the
    effective `BuildSpec` for a refresh / watch.

    Returns `None` if neither the override nor the recorded `source_root`
    is available — caller surfaces an actionable error and exits.
    """
    # Lazy import to keep `cli/_load.py` decoupled from `cli/build.py` at
    # module load time (build.py is the heavy entry point that pulls
    # encoder constants on import).
    from .build import _DEFAULT_MAX_CHARS, _DEFAULT_MIN_CHARS, _DEFAULT_TEXT_EXTS

    bc = contents.metadata.build_config
    source_root_str = source_root_override or bc.get("source_root")
    if not source_root_str:
        return None
    source_root = Path(source_root_str)

    if source_paths_override:
        sources = [Path(s) for s in source_paths_override]
    elif bc.get("source_paths"):
        sources = [Path(p) for p in bc["source_paths"]]
    else:
        sources = [source_root]

    if extensions_override is not None:
        extensions = frozenset(
            ("." + e.lstrip(".")).lower() for e in extensions_override
        )
    elif bc.get("extensions"):
        extensions = frozenset(bc["extensions"])
    else:
        extensions = _DEFAULT_TEXT_EXTS

    return BuildSpec(
        source_root=source_root,
        source_paths=sources,
        extensions=extensions,
        min_chars=int(bc.get("min_chars", _DEFAULT_MIN_CHARS)),
        max_chars=int(bc.get("max_chars", _DEFAULT_MAX_CHARS)),
    )


def open_store_or_exit(
    km_path: Path,
    contents: archive.ArchiveContents,
    source_root: str | None,
) -> Store:
    """Open the store backing a knowledge model or `sys.exit(1)` with a
    friendly message. Catches the two documented mode-dispatch failure modes
    (`ValueError` for unknown/bad mode, `NotImplementedError` for a mode
    whose backend isn't wired up yet — currently `remote`)."""
    try:
        return open_store(km_path, contents, source_root)
    except (ValueError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
