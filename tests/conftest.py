"""Pytest config.

The contract harness (tests/harness/, run via `python -m tests.harness.runner`)
is the primary gate; pytest currently collects only tests/unit/. The 2026-06
roadmap carries an open decision: bridge the harness suites into pytest or
retire the pytest config.
"""

from __future__ import annotations
