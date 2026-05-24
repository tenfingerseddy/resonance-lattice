"""`rlat probe <km.rlat>` — idle-cycle self-probe.

Architecture §14. The corpus-as-subject move: between active sessions,
identify weak zones (queries that failed to ground, repeated queries
without strong matches) and re-attempt deep-search against the current
state of the corpus. Successful re-attempts write synthesis_candidates
just like a normal deep-search; the next `bench_lensed_dogfood accept`
promotes them.

v1 surface: a CLI command the user invokes on demand. v2 may wire a
weekly schedule. The substrate is identical either way.

Weak-zone signals:

  - Past dogfood events with `intent_context` containing
    "deep-search-failed" — queries that previously failed to ground.
  - Repeated queries (same query in N+ events) without verdict-positive
    signal — the corpus didn't answer well.

For each weak-zone query, run `rlat deep-search` and capture the result.
Successful re-runs produce synthesis_candidate rows automatically (via
the existing writeback path). Failed re-runs append `probe_failed`
events so the loop knows not to retry them at the next probe.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _weak_zone_queries(state_dir: Path, max_unique: int) -> list[str]:
    """Scan the dogfood event ledger for queries that didn't ground well.

    Two signals:
      1. Events with intent_context startswith 'deep-search-failed'.
      2. Repeated queries (same query >= 2 events) that never received
         an 'accept' verdict.

    Returns up to `max_unique` unique queries in order of weakness
    (failures first, then repeated-no-accept).
    """
    ledger = state_dir / "dogfood_events.jsonl"
    if not ledger.exists():
        return []
    events: list[dict] = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    failed = [
        e["query"] for e in events
        if isinstance(e.get("intent_context"), str)
        and e["intent_context"].startswith("deep-search-failed")
    ]

    accepts_by_q: dict[str, int] = {}
    counts_by_q: Counter = Counter()
    for e in events:
        q = e.get("query")
        if not q:
            continue
        counts_by_q[q] += 1
        if e.get("verdict") == "accept":
            accepts_by_q[q] = accepts_by_q.get(q, 0) + 1

    repeated_no_accept = [
        q for q, c in counts_by_q.most_common()
        if c >= 2 and accepts_by_q.get(q, 0) == 0
    ]

    # Probe-already-attempted bookkeeping — events with
    # intent_context starting "probe-" should not be retried this cycle.
    probed_before = {
        e["query"] for e in events
        if isinstance(e.get("intent_context"), str)
        and e["intent_context"].startswith("probe-")
    }

    seen: set[str] = set()
    out: list[str] = []
    for q in list(failed) + list(repeated_no_accept):
        if q in seen or q in probed_before:
            continue
        seen.add(q)
        out.append(q)
        if len(out) >= max_unique:
            break
    return out


def _record_probe_event(
    state_dir: Path, km_path: Path, query: str, duration_ms: int,
    grounded: bool,
) -> None:
    """Mark this query as probed so a future probe pass doesn't retry."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_id": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "km_path": str(km_path),
        "query": query,
        "duration_ms": duration_ms,
        "insight_hits": 0,
        "source_hits": 0,
        "verdict": "pending",
        "intent_context": "probe-grounded" if grounded else "probe-failed",
        "lens_id": None,
    }
    try:
        (state_dir / "dogfood_events.jsonl").open("a", encoding="utf-8").write(
            json.dumps(event, sort_keys=True) + "\n"
        )
    except OSError:
        pass


def cmd_probe(args: argparse.Namespace) -> int:
    state_dir = Path.cwd() / ".rlat-state" / "ledger"
    if not state_dir.exists():
        print(
            "error: no .rlat-state/ledger/ in current directory — "
            "rlat probe needs accumulated event history to find weak "
            "zones. Run from your workspace root after some rlat usage.",
            file=sys.stderr,
        )
        return 1

    queries = _weak_zone_queries(state_dir, max_unique=args.limit)
    if not queries:
        print("[probe] no weak-zone queries found — nothing to probe")
        return 0

    if args.dry_run:
        print(f"[probe] would probe {len(queries)} query(ies):")
        for q in queries:
            print(f"  - {q}")
        return 0

    # Defer to deep-search via direct call so the synthesis_candidate
    # writeback path fires for free.
    from . import deep_search as _ds
    try:
        import anthropic
    except ImportError:
        print("error: rlat probe requires the `anthropic` package",
              file=sys.stderr)
        return 1

    from ..optimise.synth_queries import api_key_or_error
    try:
        api_key = api_key_or_error()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    client = anthropic.Anthropic(api_key=api_key)
    km_path = Path(args.knowledge_model)

    from ..deep_search.loop import deep_search

    n_grounded = 0
    n_failed = 0
    for i, query in enumerate(queries, start=1):
        print(f"[probe] {i}/{len(queries)}: {query[:60]}", file=sys.stderr)
        t0 = time.monotonic()
        try:
            result = deep_search(
                km_path, query, client=client,
                max_hops=args.max_hops, top_k=args.top_k,
                source_root=None, strict_names=False,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"[probe]   skipped: {e}", file=sys.stderr)
            continue
        duration_ms = int((time.monotonic() - t0) * 1000)

        grounded = bool(result.answer) and not result.answer.startswith(
            "I cannot produce an answer"
        )
        _record_probe_event(state_dir, km_path, query, duration_ms, grounded)

        if grounded:
            # Same autonomous faithfulness-gate-then-promote path the normal
            # deep-search CLI runs post-hoc.
            _ds._maybe_promote_faithful(km_path, result, client)
            n_grounded += 1
        else:
            n_failed += 1

    print(f"[probe] probed {len(queries)} weak-zone query(ies): "
          f"grounded={n_grounded} failed={n_failed}")
    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "probe",
        help="Idle-cycle self-probe: re-attempt failed / under-grounded queries",
    )
    p.add_argument("knowledge_model")
    p.add_argument("--limit", type=int, default=5,
                   help="Max number of weak-zone queries to probe this cycle "
                        "(default: 5; cost control)")
    p.add_argument("--max-hops", type=int, default=5)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--dry-run", action="store_true",
                   help="List weak-zone queries without running deep-search")
    p.set_defaults(func=cmd_probe)
