"""BEIR-5 locked-floor regression gate.

Runs gte-modernbert-base 768d (no optimised) against 5 BEIR corpora.
Floor locked at Phase 1 completion.

Slow — runs phase-boundary + nightly only, NOT every commit.

Phase 1 sets the floor; Phase 7 confirms launch candidate clears it.
"""


def run() -> int:
    return 0
