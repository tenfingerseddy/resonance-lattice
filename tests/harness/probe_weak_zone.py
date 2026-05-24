"""probe_weak_zone — weak-zone selection logic for `rlat probe`.

The probe substrate has two parts: (a) finding queries that didn't
ground well, and (b) re-running deep-search on them. Part (b) requires
a live LLM and is tested by manual usage; this suite exercises (a) —
the pure-data selection logic.

Guarantees:

  1. No event ledger -> [].
  2. Empty event ledger -> [].
  3. deep-search-failed events are selected.
  4. Repeated queries without 'accept' verdict are selected.
  5. Already-probed queries (events with intent_context starts 'probe-')
     are excluded from re-probing.
  6. --limit caps the unique queries returned.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def _write_events(ledger_path: Path, events: list[dict]) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, sort_keys=True) + "\n")


def run() -> int:
    from resonance_lattice.cli.probe import _weak_zone_queries

    failures = 0

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        ledger_dir = d / ".rlat-state" / "ledger"

        # ---- Guarantee 1: no ledger -> [] ----
        out = _weak_zone_queries(ledger_dir, max_unique=5)
        if out != []:
            print(f"[probe_weak] FAIL g1: {out}", file=sys.stderr)
            failures += 1
        else:
            print("[probe_weak] g1 (no ledger -> []) OK", file=sys.stderr)

        # ---- Guarantee 2: empty ledger -> [] ----
        ledger_dir.mkdir(parents=True)
        _write_events(ledger_dir / "dogfood_events.jsonl", [])
        out = _weak_zone_queries(ledger_dir, max_unique=5)
        if out != []:
            print(f"[probe_weak] FAIL g2: {out}", file=sys.stderr)
            failures += 1
        else:
            print("[probe_weak] g2 (empty ledger -> []) OK", file=sys.stderr)

        # ---- Guarantee 3: deep-search-failed events selected ----
        events = [
            {"query": "Q1", "intent_context": "deep-search-failed",
             "verdict": "pending"},
            {"query": "Q2", "intent_context": "deep-search",
             "verdict": "pending"},
            {"query": "Q3", "intent_context": "deep-search-failed",
             "verdict": "pending"},
        ]
        _write_events(ledger_dir / "dogfood_events.jsonl", events)
        out = _weak_zone_queries(ledger_dir, max_unique=5)
        if set(out) != {"Q1", "Q3"}:
            print(f"[probe_weak] FAIL g3: {out}", file=sys.stderr)
            failures += 1
        else:
            print("[probe_weak] g3 (deep-search-failed selected) OK",
                  file=sys.stderr)

        # ---- Guarantee 4: repeated query without accept selected ----
        events = [
            {"query": "Q-rep", "intent_context": "deep-search",
             "verdict": "pending"},
            {"query": "Q-rep", "intent_context": "deep-search",
             "verdict": "pending"},
            {"query": "Q-good", "intent_context": "deep-search",
             "verdict": "accept"},
            {"query": "Q-good", "intent_context": "deep-search",
             "verdict": "pending"},
        ]
        _write_events(ledger_dir / "dogfood_events.jsonl", events)
        out = _weak_zone_queries(ledger_dir, max_unique=5)
        if "Q-rep" not in out:
            print(f"[probe_weak] FAIL g4: Q-rep missing from {out}",
                  file=sys.stderr)
            failures += 1
        elif "Q-good" in out:
            print(f"[probe_weak] FAIL g4: Q-good (accepted) in {out}",
                  file=sys.stderr)
            failures += 1
        else:
            print("[probe_weak] g4 (repeated + no-accept selected) OK",
                  file=sys.stderr)

        # ---- Guarantee 5: already-probed queries excluded ----
        events = [
            {"query": "Q-old", "intent_context": "deep-search-failed",
             "verdict": "pending"},
            {"query": "Q-old", "intent_context": "probe-failed",
             "verdict": "pending"},  # marked probed
            {"query": "Q-new", "intent_context": "deep-search-failed",
             "verdict": "pending"},
        ]
        _write_events(ledger_dir / "dogfood_events.jsonl", events)
        out = _weak_zone_queries(ledger_dir, max_unique=5)
        if "Q-old" in out:
            print(f"[probe_weak] FAIL g5: Q-old re-probed in {out}",
                  file=sys.stderr)
            failures += 1
        elif "Q-new" not in out:
            print(f"[probe_weak] FAIL g5: Q-new missing from {out}",
                  file=sys.stderr)
            failures += 1
        else:
            print("[probe_weak] g5 (already-probed queries excluded) OK",
                  file=sys.stderr)

        # ---- Guarantee 6: --limit caps unique queries ----
        events = [
            {"query": f"Q{i}", "intent_context": "deep-search-failed",
             "verdict": "pending"}
            for i in range(10)
        ]
        _write_events(ledger_dir / "dogfood_events.jsonl", events)
        out = _weak_zone_queries(ledger_dir, max_unique=3)
        if len(out) != 3:
            print(f"[probe_weak] FAIL g6: {len(out)} results, expected 3",
                  file=sys.stderr)
            failures += 1
        else:
            print("[probe_weak] g6 (--limit caps unique queries) OK",
                  file=sys.stderr)

    if failures:
        print(f"[probe_weak] {failures} guarantee(s) failed", file=sys.stderr)
        return 1
    print("[probe_weak] all guarantees OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
