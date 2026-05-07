"""Resonance Lattice — knowledge-model retrieval for AI assistants.

v2.0.0 base-first architecture:
- Field (router):      gte-modernbert-base 768d, dense cosine, opt-in MRL optimised d=512.
- Store (authority):   bundled / local / remote, knowledge-model format v4 (ZIP+JSON+NPZ).
- Reader:              not in v2.0 — consumer composes synthesis on retrieved passages.

See `.claude/plans/base-first-rebuild.md` for the technical spec and
`.claude/plans/yes-i-want-to-glowing-lynx.md` for the process plan.
"""

__version__ = "2.1.0a1"

# The public Python flow for v2.0 is `archive.read(path)` + `open_store(...)`;
# RQL ops compose on top. See `docs/internal/RQL.md`.
from resonance_lattice.config import StoreMode, Kind, MaterialiserConfig, BuildConfig

__all__ = [
    "__version__",
    "StoreMode",
    "Kind",
    "MaterialiserConfig",
    "BuildConfig",
]
