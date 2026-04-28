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
