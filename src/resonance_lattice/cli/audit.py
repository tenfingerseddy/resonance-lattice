"""`rlat audit <km.rlat>` — knowledge-model trust-contract audit.

Read-only inspection of the source/insight provenance state. Surfaces:

  rlat audit km.rlat                  summary (layer sizes, states)
  rlat audit km.rlat --stale          list stale insights
  rlat audit km.rlat --orphans        list orphan insights
  rlat audit km.rlat --json           machine-readable form
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..store import archive
from ..store.audit import audit_orphans, audit_stale, audit_summary
from ._load import load_or_exit


def cmd_audit(args: argparse.Namespace) -> int:
    km_path = Path(args.knowledge_model)
    contents = load_or_exit(km_path)

    if args.stale:
        rows = audit_stale(contents)
        if args.format == "json":
            print(json.dumps([
                {"insight_id": r.insight_id, "kind": r.kind,
                 "content": r.content, "generated_at": r.generated_at,
                 "source_passage_hashes": list(r.source_passage_hashes)}
                for r in rows
            ], indent=2))
        else:
            if not rows:
                print("(no stale insights)")
            else:
                print(f"Stale insights ({len(rows)}):")
                for r in rows:
                    preview = r.content[:80].replace("\n", " ")
                    print(f"  {r.insight_id}  kind={r.kind}  {preview}...")
        return 0

    if args.orphans:
        rows = audit_orphans(contents)
        if args.format == "json":
            print(json.dumps([
                {"insight_id": r.insight_id, "kind": r.kind,
                 "missing_citations": [
                     c.passage_id for c in r.citations
                     if c.passage_id not in {p.passage_id for p in contents.registry}
                 ]}
                for r in rows
            ], indent=2))
        else:
            if not rows:
                print("(no orphan insights)")
            else:
                print(f"Orphan insights ({len(rows)}):")
                for r in rows:
                    print(f"  {r.insight_id}  kind={r.kind}")
        return 0

    summary = audit_summary(contents)
    if args.format == "json":
        from dataclasses import asdict
        print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    else:
        print(f"[audit] {km_path}")
        print(f"  source passages: {summary.source_passages}")
        print(f"  insight total:   {summary.insight_total}")
        if summary.insight_total:
            print(f"    accepted:  {summary.insight_accepted}")
            print(f"    candidate: {summary.insight_candidate}")
            print(f"    stale:     {summary.insight_stale}")
            print(f"    rejected:  {summary.insight_rejected}")
            print(f"    retired:   {summary.insight_retired}")
            print(f"    orphans:   {summary.insight_orphans}")
    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("audit", help="Trust-contract audit of a knowledge model")
    p.add_argument("knowledge_model", help="Path to a .rlat knowledge model")
    view = p.add_mutually_exclusive_group()
    view.add_argument("--stale", action="store_true",
                      help="List insights flagged stale by drift cascade")
    view.add_argument("--orphans", action="store_true",
                      help="List insights whose cited source has been removed")
    p.add_argument("--format", default="text", choices=["text", "json"],
                   help="Output format (default: text)")
    p.set_defaults(func=cmd_audit)
