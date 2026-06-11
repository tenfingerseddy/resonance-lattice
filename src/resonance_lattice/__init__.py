"""Resonance Lattice — a knowledge model that knows its own world.

Three layers:
- Field (router):      gte-modernbert-base 768d, dense cosine.
- Store (authority):   bundled / local / remote, knowledge-model format v4 (ZIP+JSON+NPZ),
                       plus the insight band — earned claims about the corpus and its world
                       (facts, standing constraints, tried-and-falsified findings).
- Grounded synthesis:  passages and, via deep-search, faithfully-grounded answers —
                       every claim traces to the corpus (docs/internal/GROUNDING_MODEL.md).

Spec: docs/internal/ARCHITECTURE.md. The version below must match
pyproject.toml's `[project] version` (cli_smoke guarantee K6 enforces parity).
"""

__version__ = "3.0.0"

# The public Python flow for v2.0 is `archive.read(path)` + `open_store(...)`;
# RQL ops compose on top. See `docs/internal/RQL.md`. The `build.*` symbols
# expose the pipeline used by `rlat build`, the Fabric UDF build/refresh
# handlers, and embedded library callers.
from resonance_lattice.build.pipeline import (
    BuildError,
    BuildResult,
    RefreshError,
    RefreshResult,
    build_rlat,
    refresh_rlat,
)
from resonance_lattice.build.walker import (
    FilesystemSourceWalker,
    SourceWalker,
)
from resonance_lattice.config import BuildConfig, Kind, MaterialiserConfig, StoreMode

__all__ = [
    "__version__",
    "BuildConfig",
    "BuildError",
    "BuildResult",
    "FilesystemSourceWalker",
    "Kind",
    "MaterialiserConfig",
    "RefreshError",
    "RefreshResult",
    "SourceWalker",
    "StoreMode",
    "build_rlat",
    "refresh_rlat",
]
