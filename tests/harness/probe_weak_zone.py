"""probe_weak_zone — weak-zone selection logic for `rlat probe`.

The probe substrate has two parts: (a) finding the gaps to re-attempt and
(b) re-running deep-search on them. Part (b) needs a live LLM and is tested
by manual usage; this suite exercises (a) — the pure-data selection logic.

Since the S4 keystone, a gap is an **intent whose evaluated success criteria
are unmet** (architecture §4), not the old dogfood-ledger heuristic.

Guarantees:

  1. No outcome log -> [].
  2. An intent whose latest outcome is `not_satisfied` is selected (its text).
  3. An intent whose latest outcome is `satisfied` is NOT selected.
  4. An intent unmet then later `satisfied` (resolved-since) drops out —
     the most-recent outcome wins.
  5. An `unknown` roll-up counts as unmet (undecided is a gap).
  6. --limit caps; results are ordered most-recently-unmet first.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def _add_intent(state_root: Path, text: str) -> str:
    from resonance_lattice.state import LiveIntentStore
    intent = LiveIntentStore(state_root).add_intent(
        level="task", text=text, stance="do", achievability="medium",
        success_criteria=[], constraints=[],
    )
    return intent.intent_id


def _resolve(state_root: Path, intent_id: str, verdict: str, when: str) -> None:
    from resonance_lattice.state import (
        ClaimOutcomeLog,
        ClaimOutcomeRecord,
        IntentOutcomeDetails,
    )
    ClaimOutcomeLog(state_root).write(ClaimOutcomeRecord(
        intent_id=intent_id,
        resolved_at=when,
        roll_up_verdict=verdict,
        attribution=[],
        details=IntentOutcomeDetails(intent_level="task"),
    ))


def run() -> int:
    from resonance_lattice.cli.probe import _weak_zone_queries

    failures = 0

    # ---- Guarantee 1: no outcome log -> [] ----
    with tempfile.TemporaryDirectory() as d:
        ledger_dir = Path(d) / ".rlat-state" / "ledger"
        out = _weak_zone_queries(ledger_dir, max_unique=5)
        if out != []:
            print(f"[probe_weak] FAIL g1: {out}", file=sys.stderr)
            failures += 1
        else:
            print("[probe_weak] g1 (no outcome log -> []) OK", file=sys.stderr)

    # ---- Guarantee 2 + 3: unmet selected, satisfied excluded ----
    with tempfile.TemporaryDirectory() as d:
        state_root = Path(d) / ".rlat-state"
        ledger_dir = state_root / "ledger"
        unmet_id = _add_intent(state_root, "ship the unmet thing")
        met_id = _add_intent(state_root, "ship the met thing")
        _resolve(state_root, unmet_id, "not_satisfied", "2026-06-01T00:00:00Z")
        _resolve(state_root, met_id, "satisfied", "2026-06-01T00:01:00Z")
        out = _weak_zone_queries(ledger_dir, max_unique=5)
        if out != ["ship the unmet thing"]:
            print(f"[probe_weak] FAIL g2/g3: {out}", file=sys.stderr)
            failures += 1
        else:
            print("[probe_weak] g2/g3 (unmet selected, satisfied excluded) OK",
                  file=sys.stderr)

    # ---- Guarantee 4: resolved-since drops out (latest wins) ----
    with tempfile.TemporaryDirectory() as d:
        state_root = Path(d) / ".rlat-state"
        ledger_dir = state_root / "ledger"
        iid = _add_intent(state_root, "later fixed")
        _resolve(state_root, iid, "not_satisfied", "2026-06-01T00:00:00Z")
        _resolve(state_root, iid, "satisfied", "2026-06-02T00:00:00Z")  # later
        out = _weak_zone_queries(ledger_dir, max_unique=5)
        if out != []:
            print(f"[probe_weak] FAIL g4: resolved-since still surfaced {out}",
                  file=sys.stderr)
            failures += 1
        else:
            print("[probe_weak] g4 (resolved-since drops out) OK",
                  file=sys.stderr)

    # ---- Guarantee 5: unknown roll-up counts as unmet ----
    with tempfile.TemporaryDirectory() as d:
        state_root = Path(d) / ".rlat-state"
        ledger_dir = state_root / "ledger"
        iid = _add_intent(state_root, "undecided thing")
        _resolve(state_root, iid, "unknown", "2026-06-01T00:00:00Z")
        out = _weak_zone_queries(ledger_dir, max_unique=5)
        if out != ["undecided thing"]:
            print(f"[probe_weak] FAIL g5: unknown not surfaced {out}",
                  file=sys.stderr)
            failures += 1
        else:
            print("[probe_weak] g5 (unknown counts as unmet) OK",
                  file=sys.stderr)

    # ---- Guarantee 6: --limit caps + most-recently-unmet ordering ----
    with tempfile.TemporaryDirectory() as d:
        state_root = Path(d) / ".rlat-state"
        ledger_dir = state_root / "ledger"
        # Five unmet intents resolved at increasing timestamps.
        for i in range(5):
            iid = _add_intent(state_root, f"gap {i}")
            _resolve(state_root, iid, "not_satisfied",
                     f"2026-06-0{i + 1}T00:00:00Z")
        out = _weak_zone_queries(ledger_dir, max_unique=3)
        # Most-recent (gap 4, 3, 2) first.
        if out != ["gap 4", "gap 3", "gap 2"]:
            print(f"[probe_weak] FAIL g6: cap/order wrong {out}",
                  file=sys.stderr)
            failures += 1
        else:
            print("[probe_weak] g6 (--limit caps + recency order) OK",
                  file=sys.stderr)

    # ---- Guarantee 7: latest is by resolved_at, not append order ----
    with tempfile.TemporaryDirectory() as d:
        state_root = Path(d) / ".rlat-state"
        ledger_dir = state_root / "ledger"
        iid = _add_intent(state_root, "skewed")
        # Append the SATISFIED (later-resolved) record FIRST, then a stale
        # not_satisfied with an EARLIER resolved_at — the satisfied must win.
        _resolve(state_root, iid, "satisfied", "2026-06-02T00:00:00Z")
        _resolve(state_root, iid, "not_satisfied", "2026-06-01T00:00:00Z")
        out = _weak_zone_queries(ledger_dir, max_unique=5)
        if out != []:
            print(f"[probe_weak] FAIL g7: stale not_satisfied won over later "
                  f"satisfied {out}", file=sys.stderr)
            failures += 1
        else:
            print("[probe_weak] g7 (latest by resolved_at, not append) OK",
                  file=sys.stderr)

    if failures:
        print(f"[probe_weak] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print("[probe_weak] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
