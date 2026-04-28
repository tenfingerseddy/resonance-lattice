"""Shared filesystem-path helpers — XDG-aware cache root etc.

Single source of truth so a future cache-root reorganisation is one edit
instead of grepping through every consumer. Used by `install.encoder`
(encoder weights cache) and `store.__init__._default_remote_cache_dir`
(remote-mode source cache).
"""

from __future__ import annotations

import os
from pathlib import Path


def xdg_cache_root() -> Path:
    """Return `<XDG_CACHE_HOME or ~/.cache>/rlat/`. Honours the XDG spec
    on Linux/Mac and degrades cleanly on Windows (where `~/.cache` is the
    convention rlat picked — Windows has no canonical user cache spec)."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "rlat"
