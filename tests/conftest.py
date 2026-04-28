"""Pytest config + common fixtures.

Phase 0 scaffold. Real fixtures land per phase as their suites populate.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def harness_fixtures_dir() -> Path:
    return Path(__file__).parent / "harness" / "fixtures"


@pytest.fixture(scope="session")
def harness_large_fixtures_dir() -> Path:
    return Path(__file__).parent / "harness" / "fixtures_large"
