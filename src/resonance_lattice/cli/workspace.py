"""`rlat workspace <subcommand>` — Horizon 1 workspace identity CLI.

  rlat workspace status         — show resolved identity (id + source)
  rlat workspace declare [--name X] [--id ID]
                                — write `.rlat-state/workspace.json`

The slash command `/workspace declare <name>` shells out to this. Source
of identity (`declared` / `git` / `cwd`) is surfaced so the user can see
which arm of the resolver fired.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..state import declare_workspace, resolve_workspace, state_root_for

EXIT_OK = 0


def _cwd_path(args: argparse.Namespace) -> Path:
    return Path(args.cwd) if args.cwd else Path.cwd()


def _cmd_status(args: argparse.Namespace) -> int:
    identity = resolve_workspace(_cwd_path(args))
    print(f"workspace_id: {identity.workspace_id}")
    print(f"source:       {identity.source}")
    print(f"root:         {identity.root}")
    print(f"state_root:   {state_root_for(identity.root)}")
    return EXIT_OK


def _cmd_declare(args: argparse.Namespace) -> int:
    root = _cwd_path(args)
    identity = declare_workspace(
        root, name=args.name, workspace_id=args.id,
    )
    print(f"declared {identity.workspace_id} at {identity.root}")
    return EXIT_OK


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "workspace",
        help="manage workspace identity",
        description="Resolve / declare the per-project workspace identity "
                    "the harness uses for memory and intent tagging.",
    )
    p.add_argument("--cwd", help="override workspace cwd (defaults to $PWD)")
    ws_sub = p.add_subparsers(dest="workspace_command", required=True)

    status_p = ws_sub.add_parser("status", help="show resolved identity")
    status_p.set_defaults(func=_cmd_status)

    declare_p = ws_sub.add_parser(
        "declare", help="write a workspace declaration override",
    )
    declare_p.add_argument(
        "--name", default=None,
        help="display name (also seeds the id when --id is omitted)",
    )
    declare_p.add_argument(
        "--id", dest="id", default=None,
        help="explicit workspace_id; auto-derived from --name or path if omitted",
    )
    declare_p.set_defaults(func=_cmd_declare)
