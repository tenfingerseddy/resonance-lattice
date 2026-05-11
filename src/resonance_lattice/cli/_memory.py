"""CLI helpers: per-user Memory open + workspace polarity tag.

Lifted out of `cli/intent.py` so other CLI subcommands that touch the
durable per-user store (or need a workspace-scoped polarity tag for
manual rows) can reuse the same arg-shape contract:

    --memory-root  override base directory (default ~/.rlat/memory/)
    --user         override user_id (default $RLAT_MEMORY_USER / $USER /
                   $USERNAME)
    --cwd          override workspace cwd (default $PWD)

The leading underscore is preserved on the function names because
`workspace_polarity_tag` already exists at the `state` layer with a
different signature (`(workspace_id) -> tag` vs the args-shape wrapper
here); naming the wrapper distinctly avoids import shadowing in
callers that pull from both modules.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _open_user_memory(args: argparse.Namespace):
    """Open the per-user Memory store. Durable intents (goals + directions)
    live here alongside memory rows; live intents (steps + tasks) live in
    LiveIntentStore. Architecture §"Where intent lives — two homes"."""
    from ..memory.store import Memory, path_for_user
    base = Path(args.memory_root) if getattr(args, "memory_root", None) else None
    return Memory(root=path_for_user(user_id=getattr(args, "user", None), root=base))


def _workspace_polarity_tag(args: argparse.Namespace) -> str:
    from ..state import resolve_workspace, workspace_polarity_tag
    cwd = Path(args.cwd) if args.cwd else Path.cwd()
    identity = resolve_workspace(cwd)
    return workspace_polarity_tag(identity.workspace_id)
