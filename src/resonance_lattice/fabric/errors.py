"""Typed errors raised by the Fabric UDF runtime helpers."""

from __future__ import annotations


class FabricSetupError(RuntimeError):
    """Bound Lakehouse layout doesn't match the UDF's expectations.

    Raised when a referenced .rlat or its parent directory is missing —
    i.e. the maintainer forgot to upload `Files/rlat/<km>.rlat`, or the
    UDF was bound to the wrong Lakehouse alias.
    """
