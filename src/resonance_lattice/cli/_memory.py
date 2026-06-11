"""CLI helpers: per-user store opens.

Lifted out of `cli/intent.py` so other CLI subcommands that touch the
per-user stores can reuse the same arg-shape contract:

    --memory-root  override base directory (default ~/.rlat/memory/)
    --user         override user_id (default $RLAT_MEMORY_USER / $USER /
                   $USERNAME)
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _user_root(args: argparse.Namespace) -> Path:
    """Resolve the per-user store directory from `--memory-root` / `--user`."""
    from ..memory.store import path_for_user
    base = Path(args.memory_root) if getattr(args, "memory_root", None) else None
    return path_for_user(user_id=getattr(args, "user", None), root=base)


def _open_user_memory(args: argparse.Namespace):
    """Open the per-user experience-claim store — the earned-knowledge claims."""
    from ..memory.claim_store import ExperienceClaimStore
    return ExperienceClaimStore(root=_user_root(args))


def _open_durable_intents(args: argparse.Namespace):
    """Open the per-user DurableIntentStore — durable goals and directions.

    Live intents (steps + tasks) live in LiveIntentStore, per workspace;
    durable intents are per-user (claim-system-design §5)."""
    from ..state import DurableIntentStore
    return DurableIntentStore(_user_root(args))
