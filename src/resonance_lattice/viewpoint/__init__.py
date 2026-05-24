"""Viewpoint — the lens, in act.

A lens is a persistent file. A viewpoint is what happens when the lens
actually answers a query: retrieve from source + insight + lens-private,
re-rank by lens trust weights + insight preferences, optionally
deliberate over the candidate set, and emit a position with full
provenance.

See `.claude/plans/lensed-knowledge-architecture.md` §6.
"""

from __future__ import annotations

from .runtime import (
    ProvenanceNode,
    ViewpointPosition,
    deliberate,
)

__all__ = [
    "ProvenanceNode",
    "ViewpointPosition",
    "deliberate",
]
