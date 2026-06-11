"""`rlat grow <km.rlat>` — one opt-in pass of the self-improvement loop.

Wraps `curator.author.grow_from_telemetry` (decide → author → gate → land):
reads the archive's own persisted telemetry for confirmed-recurring,
relatively-undercovered intents, authors a grounded fill for the top
`--max-fills` of them, and lands each through the faithfulness gate +
compression test into the insight band.

This is the product surface the 2026-06 review found missing — the loop
was fully built and tested but reachable only from hand-written Python.
Opt-in by design: it spends LLM calls and writes to the archive, so it
never runs implicitly. `--dry-run` shows the decide-tier candidates
(no LLM, no writes) so you can see what a run WOULD fill.

Telemetry note: the loop feeds on persisted telemetry
(`insight/telemetry.jsonl`), which accrues when sessions run with
`RLAT_CAPTURE_PERSIST=1` (or a dogfood session). No telemetry → no
candidates → a clean no-op.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_grow(args: argparse.Namespace) -> int:
    from ..curator.decide import decide

    if args.max_fills <= 0:
        # Negative values invert the slice (-1 == "all but the last") and
        # turn the spend bound into near-unbounded LLM spend; 0 is a silent
        # no-op after key setup. Refuse both.
        print(f"error: --max-fills must be positive (got {args.max_fills})",
              file=sys.stderr)
        return 1

    km_path = Path(args.knowledge_model)
    if not km_path.is_file():
        print(f"error: {km_path} not found", file=sys.stderr)
        return 1

    candidates = decide(str(km_path), min_occurrences=2, min_sessions=1)
    if not candidates:
        print(f"[grow] {km_path}: no confirmed-recurring undercovered intents "
              "in the persisted telemetry — nothing to grow")
        return 0

    print(f"[grow] {km_path}: {len(candidates)} candidate intent(s) past the "
          f"decide gate (occurrences × sessions × relative undercoverage)")
    for c in candidates[: args.max_fills if not args.dry_run else None]:
        print(f"  - cluster {c.cluster_id}: occurrences={c.occurrences} "
              f"sessions={c.distinct_sessions} "
              f"mean_top_score={c.mean_top_score:.3f}")

    if args.dry_run:
        print("[grow] --dry-run: no fills authored, no writes")
        return 0

    try:
        import anthropic
    except ImportError:
        print(
            "error: rlat grow requires the `anthropic` package "
            "(pip install rlat[llm] or pip install anthropic)",
            file=sys.stderr,
        )
        return 1

    from .._anthropic import api_key_or_error
    try:
        api_key = api_key_or_error()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    from ..curator.author import grow_from_telemetry

    client = anthropic.Anthropic(api_key=api_key)
    # The previewed candidates ARE the run's candidates — one telemetry
    # snapshot, so what was printed above is exactly what gets filled.
    outcomes = grow_from_telemetry(
        km_path, client,
        max_fills=args.max_fills,
        source_root=args.source_root,
        model=args.model,
        candidates=candidates,
    )

    n_promoted = sum(1 for o in outcomes if o.promoted)
    n_authored = sum(1 for o in outcomes if o.pending is not None)
    print(f"[grow] attempted {len(outcomes)} fill(s): authored={n_authored} "
          f"landed={n_promoted} (gate-rejected={n_authored - n_promoted})")
    for line in _outcome_lines(outcomes):
        print(line)
    return 0


def _outcome_lines(outcomes) -> list[str]:
    """Per-fill report lines. `PendingFill.claim` is the claim TEXT (a str)."""
    lines = []
    for o in outcomes:
        if o.pending is not None:
            status = "landed" if o.promoted else "gate-rejected"
            preview = o.pending.claim[:80].replace("\n", " ")
            lines.append(f"  [{status}] {preview}…")
    return lines


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "grow",
        help="One opt-in self-improvement pass: fill the archive's own "
             "most-demanded, worst-covered gaps from its telemetry.",
    )
    p.add_argument("knowledge_model", help="Target .rlat archive.")
    p.add_argument("--max-fills", type=int, default=1,
                   help="Fills to author this run (bounds LLM spend; default 1).")
    p.add_argument("--dry-run", action="store_true",
                   help="Show the decide-tier candidates; no LLM, no writes.")
    p.add_argument("--source-root", default=None,
                   help="Source root for local-mode archives (defaults to cwd).")
    p.add_argument("--model", default=None,
                   help="Anthropic model override for the author pass.")
    p.set_defaults(func=cmd_grow)
