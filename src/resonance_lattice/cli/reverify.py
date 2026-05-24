"""`rlat reverify <km.rlat>` — LLM re-verification of stale insights.

After `rlat refresh` flags insights as `stale` (because their cited
source content_hash changed), this command asks an LLM whether the
updated source still supports the insight. Survivors flip back to
`accepted` with refreshed citations; failures retire.

LLM-driven; requires an Anthropic API key. Costs ~1 call per stale row.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_reverify(args: argparse.Namespace) -> int:
    try:
        import anthropic
    except ImportError:
        print(
            "error: rlat reverify requires the `anthropic` package "
            "(pip install rlat[optimise] or pip install anthropic)",
            file=sys.stderr,
        )
        return 1

    from ..optimise.synth_queries import api_key_or_error
    try:
        api_key = api_key_or_error()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    from ..store import archive
    from ..store.reverification import reverify_stale_insights

    km_path = Path(args.knowledge_model)
    contents = archive.read(km_path)
    stale_count = sum(
        1 for ins in contents.insights if ins.verdict_state == "stale"
    )
    if stale_count == 0:
        print(f"[reverify] {km_path}: no stale insights — nothing to do")
        return 0

    print(f"[reverify] {km_path}: re-verifying {stale_count} stale insight(s) "
          f"(limit={args.limit or 'all'})", file=sys.stderr)

    client = anthropic.Anthropic(api_key=api_key)
    outcomes = reverify_stale_insights(
        km_path, client, model=args.model, limit=args.limit,
    )

    n_accepted = sum(1 for o in outcomes if o.new_state == "accepted")
    n_retired = sum(1 for o in outcomes if o.new_state == "retired")
    n_skipped = sum(1 for o in outcomes if o.new_state == "skipped")

    print(f"[reverify] processed {len(outcomes)} stale insight(s): "
          f"accepted={n_accepted} retired={n_retired} skipped={n_skipped}")
    if args.verbose:
        for o in outcomes:
            print(f"  {o.insight_id}  -> {o.new_state}  ({o.reason})")
    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "reverify",
        help="LLM-judge stale insights against current source; revive or retire",
    )
    p.add_argument("knowledge_model", help="Path to a .rlat knowledge model")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of LLM calls (cost control)")
    p.add_argument("--model", default=None,
                   help="Anthropic model id (default: SONNET_MODEL)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print per-insight outcome")
    p.set_defaults(func=cmd_reverify)
