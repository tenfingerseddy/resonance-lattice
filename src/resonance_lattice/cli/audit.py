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
                {"claim_id": r.claim_id, "kind": r.kind,
                 "content": r.content, "created_at": r.created_at,
                 "source_passage_hashes": list(r.facts.source_passage_hashes)}
                for r in rows
            ], indent=2))
        else:
            if not rows:
                print("(no stale insights)")
            else:
                print(f"Stale insights ({len(rows)}):")
                for r in rows:
                    preview = r.content[:80].replace("\n", " ")
                    print(f"  {r.claim_id}  kind={r.kind}  {preview}...")
        return 0

    if args.orphans:
        rows = audit_orphans(contents)
        if args.format == "json":
            print(json.dumps([
                {"claim_id": r.claim_id, "kind": r.kind,
                 "missing_citations": [
                     c.passage_id for c in r.facts.citations
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
                    print(f"  {r.claim_id}  kind={r.kind}")
        return 0

    if getattr(args, "external", False):
        # The external (web-fetched) claims + their source URLs — the FREE enumeration that feeds a world-freshness
        # re-check (re-fetch each URL, ask if it still supports the claim). Pure read; no network here.
        from ..store.external_freshness import external_claims
        refs = external_claims(contents)
        if args.format == "json":
            print(json.dumps([
                {"claim_id": r.claim_id, "state": r.state, "content": r.content,
                 "source_urls": list(r.source_urls)} for r in refs], indent=2))
        elif not refs:
            print("(no external claims — nothing has been fetched from outside the corpus)")
        else:
            print(f"External claims ({len(refs)}) — re-fetch their sources to check world-freshness:")
            for r in refs:
                preview = r.content[:80].replace("\n", " ")
                print(f"  {r.claim_id}  {preview}...")
                for u in r.source_urls:
                    print(f"      -> {u}")
        return 0

    if getattr(args, "shape", False):
        # The corpus SELF-AUDIT — the foundational, LLM-free shape-report stored at build/refresh: where the corpus
        # is under-served (demand gaps) and which cross-doc passages are same-topic candidates to judge for
        # contradiction. Default: read the stored (0.92-floor) report. `--min-cosine`/`--with-text` recompute the
        # contradiction candidates LIVE at a chosen floor and resolve each pair's text — judge-ready output for an
        # agent stance-judge (the `rlat-contradictions` skill) or a metered `curator.reconcile.judge_contradictions`.
        live = getattr(args, "min_cosine", None) is not None or getattr(args, "with_text", False)
        if live:
            from ..store.self_audit import (
                find_contradiction_candidates,
                rank_contradictions_by_demand,
            )
            floor = args.min_cosine if args.min_cosine is not None else 0.92
            cands = find_contradiction_candidates(
                km_path, min_cosine=floor, max_pairs=args.max_pairs, resolve_text=args.with_text)
            # GEOMETRY × DEMAND: order the candidates by query traffic (the stored telemetry) so the agent judges
            # the conflicts users actually hit first — a conflict nobody queries is academic. Best-effort.
            try:
                cands = rank_contradictions_by_demand(cands, contents, archive.read_telemetry(km_path))
            except Exception:
                pass
            rep = {
                "live": True,
                "min_cosine": floor,
                "demand_ranked": True,
                "high_cosine_pairs": [{"cosine": c.cosine, "a": c.a, "b": c.b} for c in cands],
                "counts": {"high_cosine_pairs": len(cands)},
            }
        else:
            rep = archive.read_self_audit(km_path)
        if args.format == "json":
            print(json.dumps(rep, indent=2, sort_keys=True))
        elif not rep:
            print("(no self-audit stored yet — run `rlat build` or `rlat refresh` to compute it)")
        else:
            c = rep.get("counts", {})
            print(f"[audit] corpus shape — {km_path}")
            if not rep.get("live"):
                print(f"  demand gaps (under-served intents): {c.get('gaps', 0)}")
            floor_note = f" (floor {rep['min_cosine']:.2f}, live)" if rep.get("live") else ""
            skip = "  — contradiction pass SKIPPED (corpus too large)" if rep.get("pairs_skipped") else ""
            print(f"  same-topic cross-doc pairs to review{floor_note}: {c.get('high_cosine_pairs', 0)}{skip}")
            for p in rep.get("high_cosine_pairs", [])[:10]:
                print(f"    {p.get('cosine', 0):.3f}  {p['a'].get('source_file')}  <->  {p['b'].get('source_file')}")
                if getattr(args, "with_text", False):
                    at = (p["a"].get("text", "") or "")[:100].replace("\n", " ")
                    bt = (p["b"].get("text", "") or "")[:100].replace("\n", " ")
                    print(f"        A: {at}")
                    print(f"        B: {bt}")
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
            print(f"    active:    {summary.insight_active}")
            print(f"    candidate: {summary.insight_candidate}")
            print(f"    stale:     {summary.insight_stale}")
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
    view.add_argument("--shape", action="store_true",
                      help="Show the corpus self-audit: demand gaps + same-topic cross-doc pairs to review")
    view.add_argument("--external", action="store_true",
                      help="List external (web-fetched) claims + their source URLs — the inputs to a "
                           "world-freshness re-check (the `rlat-refresh-facts` skill / recheck_external_freshness)")
    p.add_argument("--min-cosine", type=float, default=None,
                   help="With --shape: recompute the contradiction-candidate pairs LIVE at this cosine floor "
                        "instead of reading the stored 0.92-floor audit. A lower floor (~0.85) surfaces more "
                        "prose contradictions; a stance judge then filters precision.")
    p.add_argument("--with-text", action="store_true",
                   help="With --shape: resolve each candidate pair's passage text (judge-ready output). "
                        "Implies a live recompute.")
    p.add_argument("--max-pairs", type=int, default=200,
                   help="With --shape live recompute: cap on candidate pairs returned (default 200).")
    p.add_argument("--format", default="text", choices=["text", "json"],
                   help="Output format (default: text)")
    p.set_defaults(func=cmd_audit)
