"""`rlat expertise build` — write the expertise primer.

Architecture's North Star §"Structure": three primary context sources
(corpus + memory + intent) distil into a fourth derived layer —
*expertise* — the earned synthesis. This subcommand renders the
synthesis to disk (default `.claude/expertise.md`) so the agent reads
it on session start.

v0 ships memory + intent only. The corpus piece (top-N from the
workspace's primary `*.rlat`) is a follow-up.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..expertise import (
    DEFAULT_MAX_INTENTS,
    DEFAULT_MAX_MEMORY_ROWS,
    build_expertise_primer,
)
from ..state import resolve_workspace, state_root_for
from ._errors import EXIT_OK
from ._memory import _open_user_memory


def cmd_expertise_build(args: argparse.Namespace) -> int:
    cwd = Path(args.cwd) if args.cwd else Path.cwd()
    identity = resolve_workspace(cwd)
    state_root = state_root_for(identity.root)

    memory = _open_user_memory(args)
    output_path = (
        Path(args.out) if args.out
        else identity.root / ".claude" / "expertise.md"
    )
    written, char_count = build_expertise_primer(
        state_root=state_root,
        memory=memory,
        output_path=output_path,
        max_intents=args.max_intents,
        max_memory_rows=args.max_memory_rows,
    )
    print(f"[rlat expertise] wrote {written} ({char_count} chars)",
          file=sys.stderr)
    return EXIT_OK


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "expertise",
        help="Build the expertise primer — synthesis of memory + intent "
             "the agent reads on session start.",
    )
    expertise_sub = p.add_subparsers(dest="expertise_command", required=True)

    build_p = expertise_sub.add_parser(
        "build",
        help="Render the primer to .claude/expertise.md (or --out PATH).",
    )
    build_p.add_argument(
        "--out", default=None,
        help="output path (default: <workspace-root>/.claude/expertise.md)",
    )
    build_p.add_argument(
        "--max-intents", type=int, default=DEFAULT_MAX_INTENTS,
        help=f"cap on intent rows (default: {DEFAULT_MAX_INTENTS})",
    )
    build_p.add_argument(
        "--max-memory-rows", type=int, default=DEFAULT_MAX_MEMORY_ROWS,
        help=f"cap on memory rows (default: {DEFAULT_MAX_MEMORY_ROWS})",
    )
    build_p.add_argument(
        "--memory-root", default=None,
        help="override per-user memory root (default: ~/.rlat/memory/)",
    )
    build_p.add_argument(
        "--user", default=None,
        help="override user_id (default: $RLAT_MEMORY_USER / $USER)",
    )
    build_p.add_argument(
        "--cwd", help="override workspace cwd (defaults to $PWD)",
    )
    build_p.set_defaults(func=cmd_expertise_build)
