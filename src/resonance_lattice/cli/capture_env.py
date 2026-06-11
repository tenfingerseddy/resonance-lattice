"""`rlat capture-env` / `rlat capture-attribute` — land user-environment ATTRIBUTES in the insight band.

The lowest-friction band-population path (capture-frontier charter): `rlat capture-env <km.rlat>` AUTO-reads the
machine-readable environment (`memory.env_probe`) and writes every observed attribute — one command, zero further
user effort, zero confabulation (a read value can't be invented). `rlat capture-attribute <km> "..."` is the
manual fallback for facts no probe can read. Both are single-shot, born-active (the user/machine asserts them),
zero-knob.
"""
from __future__ import annotations

import argparse
import sys


def cmd_capture_env(args: argparse.Namespace) -> int:
    from ..memory import env_probe
    claims = env_probe.probe_and_capture(args.km, include_corpus_size=not args.no_corpus_size)
    if not claims:
        print("[rlat capture-env] no readable environment attributes found", file=sys.stderr)
        return 0
    for c in claims:
        print(f"  + {c.content}", file=sys.stderr)
    print(f"[rlat capture-env] captured {len(claims)} attribute(s) into {args.km}", file=sys.stderr)
    return 0


# User-facing kind names → claim-schema kinds. Plain language at the CLI
# ("falsified"), schema vocabulary inside (`negation`).
_KIND_FOR_FLAG = {"attribute": "attribute", "constraint": "constraint",
                  "falsified": "negation"}


def cmd_capture_attribute(args: argparse.Namespace) -> int:
    from ..memory.attribute_capture import capture_attribute
    text = sys.stdin.read().strip() if args.text == "-" else args.text
    if not text.strip():
        print("[rlat capture-attribute] empty attribute", file=sys.stderr)
        return 1
    kind_flag = getattr(args, "kind", "attribute")
    c = capture_attribute(args.km, text, criticality=args.criticality,
                          attribute_key=getattr(args, "attribute_key", "") or "",
                          kind=_KIND_FOR_FLAG[kind_flag])
    print(f"[rlat capture-attribute] captured ({kind_flag}): {c.content}",
          file=sys.stderr)
    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    pe = sub.add_parser("capture-env",
                        help="Auto-probe the environment and capture attributes into a .rlat insight band.")
    pe.add_argument("km", help="Target .rlat archive.")
    pe.add_argument("--no-corpus-size", action="store_true", help="Skip the corpus-size attribute.")
    pe.set_defaults(func=cmd_capture_env)

    pa = sub.add_parser("capture-attribute",
                        help="Manually capture one user-environment attribute (the fallback for un-probeable facts).")
    pa.add_argument("km", help="Target .rlat archive.")
    pa.add_argument("text", help="Attribute text (or `-` for stdin).")
    pa.add_argument("--criticality", default="high", choices=["low", "normal", "high", "critical"])
    pa.add_argument("--attribute-key", default="", dest="attribute_key",
                    help="Normalized subject for serve-time newest-wins dedup "
                         "(e.g. 'workspace capacity'); dedup applies within the "
                         "same --kind only. Omit to leave unkeyed — never deduped.")
    pa.add_argument("--kind", default="attribute",
                    choices=sorted(_KIND_FOR_FLAG),
                    help="World content class: attribute = stable fact (default); "
                         "constraint = standing hard rule, always served; "
                         "falsified = tried-and-failed finding — include the "
                         "evidence in the text ('Tried X; falsified by Y').")
    pa.set_defaults(func=cmd_capture_attribute)
