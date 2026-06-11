"""`rlat consolidate-insights <km.rlat>`

Close the confidence loop: fold the resolved-intent outcome records on the
unified `ClaimOutcomeLog` into the corpus insight layer's Beta confidence.

The criterion reducer (docs/internal/GROUNDING_MODEL.md §"Confidence &
attribution") maps each resolved intent's criterion verdicts to per-insight
corroboration / falsification weight — poison-guarded by verdict_confidence ×
source × provenance — and the weight is accumulated into each insight's tallies;
a sufficiently-falsified insight retires. This is the trigger side of
`store.insight_lifecycle.apply_weights_to_archive` — run it on your own cadence
(a session-end hook is the natural place).

Usage:

  rlat consolidate-insights project.rlat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_consolidate_insights(args: argparse.Namespace) -> int:
    from ..store.insight_attribution import criterion_weighted
    from ..store.insight_lifecycle import apply_weights_to_archive
    from . import _outcomes

    km_path = Path(args.knowledge_model)
    if not km_path.exists():
        print(f"error: {km_path} not found", file=sys.stderr)
        return 1

    intent_outcomes = _outcomes.read_intent_outcomes()
    if not intent_outcomes:
        print("[consolidate-insights] no outcomes recorded yet — "
              "nothing to consolidate", file=sys.stderr)
        return 0

    # Resolved-intent criterion verdicts → per-claim Beta weight, poison-guarded
    # by verdict_confidence × source × provenance. The apply re-derives each
    # seeded corpus claim's absolute tally from its born seed + this full-ledger
    # weight, so re-running consolidation is a no-op rather than re-folding the
    # ledger (§B BLOCKER idempotency). Reading the full ledger each run is what
    # makes the re-derivation correct.
    combined = criterion_weighted(intent_outcomes)

    try:
        n_updated, n_retired = apply_weights_to_archive(km_path, combined)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"[consolidate-insights] {len(intent_outcomes)} intent outcome(s) "
          f"via criterion reducer")
    print(f"[consolidate-insights] {n_updated} insight(s) updated, "
          f"{n_retired} retired")
    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "consolidate-insights",
        help="Fold resolved-intent outcomes into insight-layer confidence "
             "(attribution writeback)",
    )
    p.add_argument("knowledge_model", help="Path to a .rlat knowledge model")
    p.set_defaults(func=cmd_consolidate_insights)
