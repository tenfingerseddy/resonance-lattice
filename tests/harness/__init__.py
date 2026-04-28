"""Custom testing harness for the v2.0.0 rebuild.

Layered on top of pytest. Enforces invariants at every commit and prevents
silent drift across storage modes, band variants, and inference runtimes.

See docs/internal/REBUILD_PLAN.md and the rebuild plan §Testing Harness for
the full specification.
"""
