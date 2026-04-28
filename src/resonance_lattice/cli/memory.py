"""`rlat memory <subcommand>`

Subcommands:
  add          — append text to a tier (default: working)
  recall       — multi-tier weighted retrieval
  consolidate  — promote episodic → semantic (near-dup clustering)
  primer       — regenerate `.claude/memory-primer.md` from semantic+episodic
  gc           — apply retention decay + cap

Other LayeredMemory operations exist via the Python API; the CLI exposes
the four commands a user actually issues by hand.

Phase 5 deliverable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..memory.consolidation import consolidate
from ..memory.layered import DEFAULT_TIER_WEIGHTS, TIER_NAMES, LayeredMemory
from ..memory.primer import generate_memory_primer
from ..memory.retention import gc as _gc

DEFAULT_MEMORY_ROOT = "memory"
DEFAULT_PRIMER_OUT = ".claude/memory-primer.md"


def _resolve_root(args: argparse.Namespace) -> Path:
    return Path(args.memory_root or DEFAULT_MEMORY_ROOT)


def cmd_memory_add(args: argparse.Namespace) -> int:
    LayeredMemory.init(_resolve_root(args))  # ensure tier files exist
    memory = LayeredMemory(_resolve_root(args))
    text = args.text
    if text == "-":
        text = sys.stdin.read()
    if not text.strip():
        print("error: refusing to add empty text", file=sys.stderr)
        return 1
    entry = memory.add(
        text=text, tier=args.tier, salience=args.salience,
        source_id=args.source_id, session=args.session,
    )
    print(f"[memory] added to {args.tier}  source_id={entry.source_id or '·'}  "
          f"salience={entry.salience}", file=sys.stderr)
    return 0


def cmd_memory_recall(args: argparse.Namespace) -> int:
    memory = LayeredMemory(_resolve_root(args))
    weights = DEFAULT_TIER_WEIGHTS
    if args.tier_weights:
        try:
            override = json.loads(args.tier_weights)
            weights = {**DEFAULT_TIER_WEIGHTS, **override}
        except json.JSONDecodeError as exc:
            print(f"error: --tier-weights must be JSON: {exc}", file=sys.stderr)
            return 1
    hits = memory.recall(args.query, top_k=args.top_k, tier_weights=weights)
    if args.format == "json":
        print(json.dumps([
            {
                "score": score, "tier": e.tier, "text": e.text,
                "source_id": e.source_id, "session": e.session,
                "salience": e.salience, "recurrence_count": e.recurrence_count,
                "created_utc": e.created_utc,
            }
            for score, e in hits
        ], indent=2))
    else:
        for score, e in hits:
            preview = e.text.replace("\n", " ").strip()[:100]
            print(f"{score:+.3f}  [{e.tier:<8}]  {e.source_id or '·':<20}  {preview}")
    return 0


def cmd_memory_consolidate(args: argparse.Namespace) -> int:
    memory = LayeredMemory(_resolve_root(args))
    promoted = consolidate(
        memory,
        recurrence_threshold=args.recurrence_threshold,
        dup_threshold=args.dup_threshold,
        session=args.session,
    )
    print(f"[memory] consolidated {promoted} entries (episodic → semantic)",
          file=sys.stderr)
    return 0


def cmd_memory_primer(args: argparse.Namespace) -> int:
    n_chars = generate_memory_primer(
        _resolve_root(args),
        Path(args.output),
        novelty_threshold=args.novelty,
    )
    print(f"[memory] wrote {args.output} ({n_chars} chars, "
          f"~{n_chars // 4} tokens)", file=sys.stderr)
    return 0


def cmd_memory_gc(args: argparse.Namespace) -> int:
    memory = LayeredMemory(_resolve_root(args))
    total = 0
    for tier in (args.tier or TIER_NAMES):
        n = _gc(memory, tier)
        if n:
            print(f"[memory] gc({tier}): removed {n}", file=sys.stderr)
        total += n
    print(f"[memory] total removed: {total}", file=sys.stderr)
    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("memory", help="Layered memory subcommands")
    p.add_argument("--memory-root", default=None,
                   help=f"Memory directory (default: ./{DEFAULT_MEMORY_ROOT}/)")
    sub_mem = p.add_subparsers(dest="memory_subcommand", required=True)

    p_add = sub_mem.add_parser("add", help="Append text to a tier")
    p_add.add_argument("text", help="Text to add (or '-' for stdin)")
    p_add.add_argument("--tier", default="working", choices=list(TIER_NAMES))
    p_add.add_argument("--salience", type=float, default=1.0)
    p_add.add_argument("--source-id", default="")
    p_add.add_argument("--session", default=None)
    p_add.set_defaults(func=cmd_memory_add)

    p_rec = sub_mem.add_parser("recall", help="Multi-tier weighted retrieval")
    p_rec.add_argument("query", help="Query text")
    p_rec.add_argument("--top-k", type=int, default=10)
    p_rec.add_argument("--tier-weights", default=None,
                       help='JSON dict, e.g. \'{"working": 0.7}\' to override defaults')
    p_rec.add_argument("--format", default="text", choices=["text", "json"])
    p_rec.set_defaults(func=cmd_memory_recall)

    p_con = sub_mem.add_parser("consolidate",
                               help="Promote near-duplicate episodic → semantic")
    p_con.add_argument("--recurrence-threshold", type=int, default=3)
    p_con.add_argument("--dup-threshold", type=float, default=0.92)
    p_con.add_argument("--session", default=None,
                       help="Restrict to entries from this session")
    p_con.set_defaults(func=cmd_memory_consolidate)

    p_pri = sub_mem.add_parser("primer",
                               help="Regenerate .claude/memory-primer.md")
    p_pri.add_argument("-o", "--output", default=DEFAULT_PRIMER_OUT)
    p_pri.add_argument("--novelty", type=float, default=0.3,
                       help="Min cosine-to-centroid for entries to appear")
    p_pri.set_defaults(func=cmd_memory_primer)

    p_gc = sub_mem.add_parser("gc",
                              help="Apply retention decay + cap (drops expired)")
    p_gc.add_argument("--tier", action="append", default=None,
                      choices=list(TIER_NAMES),
                      help="Restrict gc to specific tier(s); default: all")
    p_gc.set_defaults(func=cmd_memory_gc)
