"""Build pipeline — pure Python API for producing and refreshing `.rlat`s.

`cli/build.py` and `cli/maintain.py` are thin argparse wrappers over the
functions in this package. External callers (the Fabric UDF, anything
embedding rlat as a library) import from here directly.
"""

from .pipeline import (
    BuildError,
    BuildResult,
    RefreshError,
    RefreshResult,
    build_rlat,
    refresh_rlat,
)
from .walker import (
    DEFAULT_TEXT_EXTENSIONS,
    FilesystemSourceWalker,
    SourceWalker,
    common_root,
)

__all__ = [
    "DEFAULT_TEXT_EXTENSIONS",
    "BuildError",
    "BuildResult",
    "FilesystemSourceWalker",
    "RefreshError",
    "RefreshResult",
    "SourceWalker",
    "build_rlat",
    "common_root",
    "refresh_rlat",
]
