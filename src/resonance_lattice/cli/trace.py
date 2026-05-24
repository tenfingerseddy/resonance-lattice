"""`rlat trace <km.rlat> <id>` — full provenance chain for an insight or
source passage.

For an `insight_id`: prints the cited source passages with hash + char
span, plus any lineage chain. For a `source_passage_id`: prints every
insight that cites this passage (reverse trace).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..store.audit import trace_insight, trace_source
from ._load import load_or_exit


def cmd_trace(args: argparse.Namespace) -> int:
    km_path = Path(args.knowledge_model)
    contents = load_or_exit(km_path)

    target_id: str = args.id

    # Try as insight_id first; fall back to source-passage reverse trace.
    try:
        trace = trace_insight(contents, target_id)
    except KeyError:
        reverse = trace_source(contents, target_id)
        if not reverse:
            print(f"error: id {target_id!r} matched no insight and no source "
                  f"passage cited by any insight", file=sys.stderr)
            return 1
        if args.format == "json":
            print(json.dumps([
                {"insight_id": r.insight_id, "kind": r.kind,
                 "verdict_state": r.verdict_state, "content": r.content}
                for r in reverse
            ], indent=2))
        else:
            print(f"[trace] {target_id} cited by {len(reverse)} insight(s):")
            for ins in reverse:
                preview = ins.content[:80].replace("\n", " ")
                print(f"  {ins.insight_id}  state={ins.verdict_state}  {preview}...")
        return 0

    if args.format == "json":
        print(json.dumps({
            "insight": {
                "id": trace.insight.insight_id,
                "kind": trace.insight.kind,
                "content": trace.insight.content,
                "verdict_state": trace.insight.verdict_state,
                "confidence": trace.insight.confidence,
                "generated_at": trace.insight.generated_at,
                "source_model_hash": trace.insight.source_model_hash,
            },
            "source_passages": trace.source_passages,
            "source_orphans": trace.source_orphans,
            "lineage": [
                {"insight_id": li.insight_id, "kind": li.kind}
                for li in trace.lineage_chain
            ],
        }, indent=2))
    else:
        ins = trace.insight
        print(f"[trace] insight {ins.insight_id}")
        print(f"  kind:        {ins.kind}")
        print(f"  verdict:     {ins.verdict_state}")
        print(f"  confidence:  {ins.confidence:.2f}")
        print(f"  generated:   {ins.generated_at}")
        print(f"  content:")
        print(f"    {ins.content}")
        if trace.source_passages:
            print(f"  cites {len(trace.source_passages)} source passage(s):")
            for sp in trace.source_passages:
                print(f"    {sp['source_file']}:{sp['char_offset']}+"
                      f"{sp['char_length']}  hash={sp['content_hash'][:12]}…  "
                      f"confidence={sp['citation_confidence']:.2f}")
        if trace.source_orphans:
            print(f"  ORPHANS ({len(trace.source_orphans)} cited passages no longer exist):")
            for pid in trace.source_orphans:
                print(f"    {pid}")
        if trace.lineage_chain:
            print(f"  lineage chain ({len(trace.lineage_chain)} parents):")
            for parent in trace.lineage_chain:
                print(f"    {parent.insight_id}  kind={parent.kind}")

    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("trace", help="Full provenance chain for an insight or source")
    p.add_argument("knowledge_model", help="Path to a .rlat knowledge model")
    p.add_argument("id", help="An insight_id or source passage_id")
    p.add_argument("--format", default="text", choices=["text", "json"],
                   help="Output format (default: text)")
    p.set_defaults(func=cmd_trace)
