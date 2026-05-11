"""Workspace identity — the project-scoped unit of memory and intent isolation.

Architecture §"Workspace" specifies the resolution rule:

    workspace_id = hash(repo_initial_commit + repo_origin_url) if in a git repo
                 else hash(canonical_user_declared_root)        if user-declared
                 else hash(canonical_cwd)                       as fallback

The git case is path-independent — a repo cloned to `~/A` and `~/B` resolves
to the same workspace. User-declared overrides (`/workspace declare`) handle
monorepos, multi-repo projects, and any case where the git default is wrong.

Resolution walks *upward* from `cwd` looking for either a declaration file
(`.rlat-state/workspace.json`) or a `.git/` directory; the first hit wins,
declaration before git so an override can split a monorepo subtree.

The resolved identity stamps the `workspace:<hash>` polarity scope-tag on
every memory and intent row written from this workspace. Falls open: if every
resolution step fails, the cwd-hash fallback ensures recall and capture never
break — they just become per-cwd rather than per-project.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..memory._common import atomic_write_json, utcnow_iso, workspace_hash

# `<workspace-root>/.rlat-state/` — anchor directory for live state. Only
# created when the harness writes to it; resolution itself is read-only.
STATE_DIR = ".rlat-state"
WORKSPACE_DECLARATION_FILE = "workspace.json"

# How far up the directory tree to walk before giving up. 24 levels covers
# every realistic project depth (monorepo subpackages typically sit 3–6 levels
# deep) without scanning the entire filesystem on a misconfigured cwd.
_MAX_WALK_DEPTH = 24

# Default declaration-file schema version; bumped only on breaking shape change.
_DECLARATION_SCHEMA_VERSION = 1

WorkspaceSource = Literal["declared", "git", "cwd"]


@dataclass(frozen=True)
class WorkspaceIdentity:
    """Resolved workspace.

    `root` is the directory that owns the live `.rlat-state/` (always
    canonicalised). `workspace_id` is the 6-hex hash used as the
    `workspace:<hash>` polarity scope-tag. `source` records which arm of the
    resolution rule fired — useful for debug surfaces and for the
    `/workspace` slash commands' status output.
    """

    root: Path
    workspace_id: str
    source: WorkspaceSource


# `_common.workspace_hash` runs the same 6-hex sha256 over an
# `os.path.normcase`-folded cwd. The git + name arms feed it pre-folded
# strings (initial commit + URL, or a name) so the case-fold is a no-op for
# them — single helper, three call sites, one collision profile.


def state_root_for(workspace_root: Path) -> Path:
    """`<workspace-root>/.rlat-state/`."""
    return workspace_root / STATE_DIR


def workspace_polarity_tag(workspace_id: str) -> str:
    """Build the `workspace:<hash>` scope-tag for polarity."""
    return f"workspace:{workspace_id}"


def _load_declaration(state_dir: Path) -> dict | None:
    """Read `<state-dir>/workspace.json` if present.

    Returns `None` for missing file or malformed JSON; resolution then falls
    through to the git arm. Malformed declarations don't crash the resolver
    — the harness must always produce *some* identity.
    """
    path = state_dir / WORKSPACE_DECLARATION_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _git_identity(repo_root: Path) -> tuple[str, str] | None:
    """Compute (initial_commit_hex, origin_url) for `repo_root`.

    Shells out to `git` because pygit2/dulwich would add a heavyweight dep
    for a once-per-session lookup. `git rev-list --max-parents=0 HEAD`
    surfaces the initial commit (stable across clones); `git config --get
    remote.origin.url` surfaces the origin URL (empty string if no origin).

    Returns `None` if the repo has no commits yet (fresh `git init` with
    nothing committed) — caller falls back to cwd hashing.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-list", "--max-parents=0", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        ).stdout.strip().splitlines()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError, OSError):
        return None
    if not commit:
        return None
    initial_commit = commit[0]
    try:
        url_proc = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        origin_url = url_proc.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        origin_url = ""
    return initial_commit, origin_url


def _walk_up(start: Path) -> list[Path]:
    """Yield `start` then each parent up to filesystem root or depth cap.

    Bounded walk — `_MAX_WALK_DEPTH` is the upper bound on iterations so a
    pathological cwd (e.g. UNC root with a malformed structure) can't spin.
    """
    out: list[Path] = []
    current = start
    for _ in range(_MAX_WALK_DEPTH):
        out.append(current)
        if current.parent == current:
            break
        current = current.parent
    return out


def resolve_workspace(cwd: Path | str | None = None) -> WorkspaceIdentity:
    """Resolve workspace identity from `cwd`.

    Walks upward from `cwd` and returns the first hit:
      1. a directory containing `.rlat-state/workspace.json` → `declared`
      2. a directory containing `.git/` → `git`
      3. cwd itself → `cwd` fallback

    Always returns an identity — never raises. Callers use the resolution
    *source* for diagnostics; the identity hash itself is the same shape
    across all three arms so downstream code stays uniform.
    """
    start = Path(cwd) if cwd is not None else Path.cwd()
    start = start.resolve()

    declared_hit: tuple[Path, dict] | None = None
    git_hit: Path | None = None
    for candidate in _walk_up(start):
        if declared_hit is None:
            decl = _load_declaration(candidate / STATE_DIR)
            if decl is not None:
                declared_hit = (candidate, decl)
                break  # declaration is highest priority; stop walking
        if git_hit is None and (candidate / ".git").exists():
            git_hit = candidate
            # Don't break — a declaration further up still overrides this
            # git root (e.g. a monorepo split where a subdir was declared).

    if declared_hit is not None:
        root, decl = declared_hit
        wid = decl.get("workspace_id")
        if not isinstance(wid, str) or not wid:
            # Declaration without an explicit id hashes the resolved root
            # (canonical). Mirrors the cwd-fallback shape so polarity-tag
            # widths stay uniform regardless of source.
            wid = workspace_hash(os.path.normcase(str(root)))
        return WorkspaceIdentity(root=root, workspace_id=wid, source="declared")

    if git_hit is not None:
        identity = _git_identity(git_hit)
        if identity is not None:
            initial_commit, origin_url = identity
            wid = workspace_hash(f"{initial_commit}|{origin_url}")
            return WorkspaceIdentity(root=git_hit, workspace_id=wid, source="git")

    wid = workspace_hash(os.path.normcase(str(start)))
    return WorkspaceIdentity(root=start, workspace_id=wid, source="cwd")


def declare_workspace(
    root: Path | str,
    *,
    name: str | None = None,
    workspace_id: str | None = None,
) -> WorkspaceIdentity:
    """Write `<root>/.rlat-state/workspace.json` declaring this directory tree.

    Slash command `/workspace declare <name>` calls this. Existing declar-
    ation files are overwritten — the architecture treats declarations as
    durable but mutable (re-declare to rename). Returns the resulting
    identity so the caller can stamp it onto next-session memory writes.

    `workspace_id` may be passed explicitly (e.g. for `/workspace merge`,
    where the partner's id is the canonical one); otherwise it's derived
    from `name` if given, else from the canonical root path. Either way the
    declaration file holds the literal id so subsequent resolution returns
    a stable answer.
    """
    root_path = Path(root).resolve()
    state_dir = state_root_for(root_path)
    state_dir.mkdir(parents=True, exist_ok=True)
    if workspace_id is None:
        workspace_id = workspace_hash(name or str(root_path))
    payload = {
        "schema_version": _DECLARATION_SCHEMA_VERSION,
        "workspace_id": workspace_id,
        "name": name or root_path.name,
        "declared_at": utcnow_iso(),
    }
    atomic_write_json(state_dir / WORKSPACE_DECLARATION_FILE, payload)
    return WorkspaceIdentity(
        root=root_path, workspace_id=workspace_id, source="declared"
    )
