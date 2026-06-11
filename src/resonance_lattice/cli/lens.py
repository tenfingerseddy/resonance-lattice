"""`rlat lens <subcommand>` — manage lens artefacts.

Subcommands:

  rlat lens create --name N --scope user|role|team|project --id ID [--stance FILE]
  rlat lens show <lens.lens>
  rlat lens compose <lens1.lens> <lens2.lens> ... -o <out.lens> --id ID --name N
  rlat lens set-trust <lens.lens> <pattern> <weight> [--remove]

`set-trust` is the write surface the 2026-06 review added — the lens layer
shipped read-complete but write-blind (no production path ever wrote a
TrustWeight), making the portable-perspective story a demo users could not
exercise. One command makes it real: boost (`>1`), suppress (`<1`), or
effectively exclude (`0`) sources by glob pattern.
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
        if lens.declared_stance:
            print(f"  declared stance: {len(lens.declared_stance)} chars")
        if lens.trust_weights:
            print(f"  trust weights:   {len(lens.trust_weights)} pattern(s)")
            for tw in lens.trust_weights:
                print(f"    {tw.pattern}  -> {tw.weight}")
        if lens.insight_preferences:
            print(f"  insight prefs:   {len(lens.insight_preferences)} entr(ies)")
    return 0


def _cmd_set_trust(args: argparse.Namespace) -> int:
    path = Path(args.lens_path)
    lens = lens_mod.load(path)
    if args.remove:
        kept = [tw for tw in lens.trust_weights if tw.pattern != args.pattern]
        if len(kept) == len(lens.trust_weights):
            print(f"error: no trust weight for pattern {args.pattern!r} — nothing to remove", file=sys.stderr)
            return 1
        lens.trust_weights[:] = kept
        lens_mod.save(lens, path)
        print(f"[lens] removed trust weight for {args.pattern!r} "
              f"({len(kept)} pattern(s) remain)")
        return 0
    if args.weight is None:
        print("error: a weight is required (or pass --remove)", file=sys.stderr)
        return 1
    if args.weight < 0:
        print(f"error: weight must be >= 0 (got {args.weight}); 0 excludes, "
              f"1 is identity, >1 boosts", file=sys.stderr)
        return 1
    new_tw = lens_mod.TrustWeight(pattern=args.pattern, weight=args.weight)
    for i, tw in enumerate(lens.trust_weights):
        if tw.pattern == args.pattern:
            lens.trust_weights[i] = new_tw
            verb = "updated"
            break
    else:
        lens.trust_weights.append(new_tw)
        verb = "added"
    lens_mod.save(lens, path)
    print(f"[lens] {verb} {args.pattern!r} -> {args.weight} "
          f"({len(lens.trust_weights)} pattern(s) total)")
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

    set_trust = lens_sub.add_parser(
        "set-trust",
        help="Add/update (or --remove) a source-pattern trust weight in a lens")
    set_trust.add_argument("lens_path", help="Path to a .lens file")
    set_trust.add_argument("pattern", help="Glob over source-file paths (e.g. 'docs/external/*')")
    set_trust.add_argument("weight", nargs="?", type=float, default=None,
                           help=">1 boosts, <1 suppresses, 0 excludes; identity 1.0")
    set_trust.add_argument("--remove", action="store_true",
                           help="Remove the pattern's trust weight instead")
    set_trust.set_defaults(func=_cmd_set_trust)
