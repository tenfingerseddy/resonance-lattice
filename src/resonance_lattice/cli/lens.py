"""`rlat lens <subcommand>` — manage lens artefacts.

Subcommands:

  rlat lens create --name N --scope user|role|team|project --id ID [--stance FILE]
  rlat lens show <lens.lens>
  rlat lens compose <lens1.lens> <lens2.lens> ... -o <out.lens> --id ID --name N
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from ..lens import schema as lens_mod


def _cmd_create(args: argparse.Namespace) -> int:
    stance = None
    if args.stance:
        stance = Path(args.stance).read_text(encoding="utf-8")
    lens = lens_mod.new_lens(
        lens_id=args.id, scope=args.scope, name=args.name,
        description=args.description, declared_stance=stance,
    )
    out = Path(args.output)
    lens_mod.save(lens, out)
    print(f"[lens] created {out} (lens_id={args.id}, scope={args.scope})")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    lens = lens_mod.load(Path(args.lens_path))
    if args.format == "json":
        print(json.dumps({
            "manifest": asdict(lens.manifest),
            "declared_stance_chars": len(lens.declared_stance) if lens.declared_stance else 0,
            "trust_weights": [asdict(tw) for tw in lens.trust_weights],
            "insight_preferences": len(lens.insight_preferences),
            "memory_rows": len(lens.memory),
            "intent_history": len(lens.intent_history),
            "verdict_log": len(lens.verdict_log),
            "private_insights": len(lens.private_insights),
        }, indent=2, sort_keys=True))
    else:
        m = lens.manifest
        print(f"[lens] {m.name} (id={m.lens_id})")
        print(f"  scope:           {m.scope}")
        if m.description:
            print(f"  description:     {m.description}")
        print(f"  created:         {m.created_at}")
        print(f"  last_active:     {m.last_active}")
        print(f"  schema_version:  {m.schema_version}")
        print(f"  encoder_version: {m.encoder_version or '(none)'}")
        if lens.declared_stance:
            print(f"  declared stance: {len(lens.declared_stance)} chars")
        if lens.trust_weights:
            print(f"  trust weights:   {len(lens.trust_weights)} pattern(s)")
            for tw in lens.trust_weights:
                print(f"    {tw.pattern}  -> {tw.weight}")
        if lens.insight_preferences:
            print(f"  insight prefs:   {len(lens.insight_preferences)} entr(ies)")
        if lens.memory:
            print(f"  memory rows:     {len(lens.memory)}")
        if lens.intent_history:
            print(f"  intent history:  {len(lens.intent_history)}")
        if lens.verdict_log:
            print(f"  verdict log:     {len(lens.verdict_log)} signal(s)")
        if lens.private_insights:
            print(f"  private insights: {len(lens.private_insights)}")
    return 0


def _cmd_compose(args: argparse.Namespace) -> int:
    lenses = [lens_mod.load(Path(p)) for p in args.inputs]
    out_lens = lens_mod.compose(
        lenses,
        composed_id=args.id,
        name=args.name,
        scope=args.scope,
    )
    lens_mod.save(out_lens, Path(args.output))
    print(f"[lens] composed {len(lenses)} lens(es) into {args.output} "
          f"(lens_id={args.id}, scope={args.scope})")
    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("lens", help="Manage lens artefacts (.lens files)")
    lens_sub = p.add_subparsers(dest="lens_command", required=True)

    create = lens_sub.add_parser("create", help="Create a new empty lens")
    create.add_argument("--id", required=True, help="Stable lens identifier")
    create.add_argument("--name", required=True, help="Human-readable name")
    create.add_argument("--scope", choices=["user", "role", "team", "project"],
                        default="user")
    create.add_argument("--description", default=None)
    create.add_argument("--stance", default=None,
                        help="Path to a markdown file with the declared editorial stance")
    create.add_argument("-o", "--output", required=True, help="Output .lens path")
    create.set_defaults(func=_cmd_create)

    show = lens_sub.add_parser("show", help="Inspect a .lens file")
    show.add_argument("lens_path", help="Path to a .lens file")
    show.add_argument("--format", default="text", choices=["text", "json"])
    show.set_defaults(func=_cmd_show)

    compose = lens_sub.add_parser("compose", help="Combine multiple lenses into one")
    compose.add_argument("inputs", nargs="+", help="Two or more .lens paths")
    compose.add_argument("--id", required=True, help="Composed lens identifier")
    compose.add_argument("--name", required=True, help="Composed lens name")
    compose.add_argument("--scope", choices=["user", "role", "team", "project"],
                        default="team")
    compose.add_argument("-o", "--output", required=True, help="Output .lens path")
    compose.set_defaults(func=_cmd_compose)
