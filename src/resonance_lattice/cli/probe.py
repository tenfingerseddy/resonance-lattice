"""`rlat probe <km.rlat>` — idle-cycle self-probe.

Architecture §14. The corpus-as-subject move: between active sessions,
identify weak zones (intents whose success criteria went unmet — the
keystone gap) and re-attempt deep-search against the current state of the
corpus. Successful re-attempts write synthesis_candidates just like a
normal deep-search, promoted on the next consolidation pass.

Manual on-demand CLI today. A scheduled-cadence variant (cron, hook)
would reuse the same substrate; that's a Phase E candidate gated on a
measurement that on-demand misses queries the schedule would catch.

Weak-zone signal (the keystone gap — architecture §4):

  - An intent whose evaluated `success_criteria` are unmet: its most recent
    resolution rolled up `not_satisfied` or `unknown`. That intent's text is
    the query to re-attempt — the corpus did not let the agent meet what the
    user intended. (This replaced the older dogfood-ledger heuristic of
    failed/repeated queries; an unmet criterion is the principled gap.)

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
from datetime import datetime, timezone
from pathlib import Path


def _weak_zone_queries(state_dir: Path, max_unique: int) -> list[str]:
    """Surface intents whose evaluated success criteria are unmet — the
    keystone gap (architecture §4: a Gap is an Intent whose `criterion_checks`
    show unmet criteria), the principled replacement for the old dogfood-ledger
    heuristic.

    Reads the intent-kind outcome records and the live intent graph: an intent
    whose **most recent** resolution rolled up `not_satisfied` or `unknown`
    (its criteria were not all met) is a weak zone, and its text is the query
    to re-attempt. An intent later resolved `satisfied` drops out. Returns up
    to `max_unique` distinct intent texts, most-recently-unmet first.

    `state_dir` is the ledger directory `<root>/ledger`; the outcome log and
    the live intent store both hang off its parent state root.
    """
    from ..state import LiveIntentStore
    from ..state.claim_outcome import ClaimOutcomeLog

    state_root = state_dir.parent
    records = ClaimOutcomeLog(state_root).read(kind="intent")
    if not records:
        return []
    # Most-recent outcome per intent, by `resolved_at` (robust to a record
    # appended out of resolution order); ties keep the later-appended one.
    latest: dict = {}
    for r in records:
        prev = latest.get(r.intent_id)
        if prev is None or r.resolved_at >= prev.resolved_at:
            latest[r.intent_id] = r
    intents = {i.intent_id: i for i in LiveIntentStore(state_root).list_all()}

    unmet = [
        (r.resolved_at, intents[iid].text)
        for iid, r in latest.items()
        if r.roll_up_verdict in ("not_satisfied", "unknown")
        and iid in intents and intents[iid].text.strip()
    ]
    unmet.sort(key=lambda pair: pair[0], reverse=True)  # most recent first

    seen: set[str] = set()
    out: list[str] = []
    for _resolved_at, text in unmet:
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
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
            "rlat probe finds weak zones from resolved intents whose "
            "success criteria went unmet. Run from your workspace root "
            "after some intent activity.",
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

    from .._anthropic import api_key_or_error
    try:
        api_key = api_key_or_error()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.cost_cap_usd is not None and args.cost_cap_usd <= 0:
        print(
            f"error: --cost-cap-usd must be positive (got {args.cost_cap_usd})",
            file=sys.stderr,
        )
        return 1

    client = anthropic.Anthropic(api_key=api_key)
    km_path = Path(args.knowledge_model)

    from .._pricing import CostMeter
    from ..deep_search.loop import deep_search

    # One shared meter across every query in this probe pass. The cap is
    # session-wide: once observed spend crosses it, subsequent
    # `deep_search` calls short-circuit in their own pre-flight check.
    meter = CostMeter(cap_usd=args.cost_cap_usd)

    n_grounded = 0
    n_failed = 0
    for i, query in enumerate(queries, start=1):
        print(f"[probe] {i}/{len(queries)}: {query[:60]}", file=sys.stderr)
        if meter.has_exceeded_cap():
            print(
                f"[probe]   skipped: cost cap crossed "
                f"(${meter.cost_so_far():.4f} of ${meter.cap_usd:.4f})",
                file=sys.stderr,
            )
            n_failed += 1
            continue
        t0 = time.monotonic()
        try:
            result = deep_search(
                km_path, query, client=client,
                max_hops=args.max_hops, top_k=args.top_k,
                source_root=None, strict_names=False,
                meter=meter,
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
        help="Idle-cycle self-probe: re-attempt intents whose success "
             "criteria went unmet",
    )
    p.add_argument("knowledge_model")
    p.add_argument("--limit", type=int, default=5,
                   help="Max number of weak-zone queries to probe this cycle "
                        "(default: 5; cost control)")
    p.add_argument("--max-hops", type=int, default=5)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--dry-run", action="store_true",
                   help="List weak-zone queries without running deep-search")
    p.add_argument("--cost-cap-usd", type=float, default=None,
                   help="Cap cumulative LLM spend in USD across all queries "
                        "this pass. Once observed spend crosses the cap, "
                        "remaining queries are skipped. Mirrors `rlat "
                        "reverify` and `rlat memory verify`.")
    p.set_defaults(func=cmd_probe)
