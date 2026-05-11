"""state_workspace — workspace-identity contracts.

Pins the architecture §"Workspace" resolution rule. Five contracts:

  (a) cwd fallback — no git, no declaration → identity hashes canonical cwd,
      source='cwd', never raises.

  (b) Declaration override — `.rlat-state/workspace.json` at any walked
      ancestor wins, regardless of git presence further up.

  (c) Declaration without explicit `workspace_id` derives a stable hash from
      the canonical root path (round-trip stable across re-resolves).

  (d) Sub-directory cwd resolves to the same workspace as the root — walking
      upward terminates at the declaration / .git boundary.

  (e) `declare_workspace` writes the declaration and `resolve_workspace`
      reads it back atomically; re-declaration overwrites cleanly.

Hermetic — no actual git invocation; the git arm is exercised by simply not
declaring and confirming the cwd fallback fires when no `.git/` exists.
The full git-identity probe is integration-tested by the dogfood resonance-
lattice.rlat refresh path, not here.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def _check_cwd_fallback() -> int:
    from resonance_lattice.state import resolve_workspace

    with tempfile.TemporaryDirectory() as td:
        cwd = Path(td) / "no-git-no-decl"
        cwd.mkdir()
        identity = resolve_workspace(cwd)
    if identity.source != "cwd":
        print(f"[state_workspace] FAIL (a): source={identity.source!r}", file=sys.stderr)
        return 1
    if not identity.workspace_id or len(identity.workspace_id) != 6:
        print(f"[state_workspace] FAIL (a): bad workspace_id={identity.workspace_id!r}", file=sys.stderr)
        return 1
    print("[state_workspace] (a) cwd fallback OK", file=sys.stderr)
    return 0


def _check_declaration_overrides() -> int:
    from resonance_lattice.state import declare_workspace, resolve_workspace

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "project"
        root.mkdir()
        (root / ".git").mkdir()  # would normally win — but declaration trumps
        identity = declare_workspace(root, name="rlat-private",
                                     workspace_id="abc123")
        seen = resolve_workspace(root)
    if seen.source != "declared":
        print(f"[state_workspace] FAIL (b): source={seen.source!r}", file=sys.stderr)
        return 1
    if seen.workspace_id != "abc123":
        print(f"[state_workspace] FAIL (b): id={seen.workspace_id!r}", file=sys.stderr)
        return 1
    if seen.workspace_id != identity.workspace_id:
        print("[state_workspace] FAIL (b): declare/resolve mismatch", file=sys.stderr)
        return 1
    print("[state_workspace] (b) declaration overrides .git OK", file=sys.stderr)
    return 0


def _check_declaration_no_explicit_id() -> int:
    from resonance_lattice.state import declare_workspace, resolve_workspace

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "no-id"
        root.mkdir()
        first = declare_workspace(root, name="some-project")
        second = resolve_workspace(root)
        third = resolve_workspace(root)
    if first.workspace_id != second.workspace_id != third.workspace_id:
        print("[state_workspace] FAIL (c): id not stable across re-resolves",
              file=sys.stderr)
        return 1
    if not first.workspace_id or len(first.workspace_id) != 6:
        print(f"[state_workspace] FAIL (c): bad id width: {first.workspace_id!r}",
              file=sys.stderr)
        return 1
    print("[state_workspace] (c) implicit id stable OK", file=sys.stderr)
    return 0


def _check_subdir_resolves_to_root() -> int:
    from resonance_lattice.state import declare_workspace, resolve_workspace

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "monorepo"
        root.mkdir()
        sub = root / "packages" / "deep" / "leaf"
        sub.mkdir(parents=True)
        declare_workspace(root, name="monorepo", workspace_id="aaaaaa")
        from_root = resolve_workspace(root)
        from_sub = resolve_workspace(sub)
    if from_root.workspace_id != from_sub.workspace_id:
        print(f"[state_workspace] FAIL (d): root={from_root.workspace_id!r} "
              f"sub={from_sub.workspace_id!r}", file=sys.stderr)
        return 1
    if from_sub.root.resolve() != root.resolve():
        print(f"[state_workspace] FAIL (d): sub.root={from_sub.root!r} "
              f"want {root!r}", file=sys.stderr)
        return 1
    print("[state_workspace] (d) subdir resolves to declared root OK",
          file=sys.stderr)
    return 0


def _check_redeclaration_overwrites() -> int:
    from resonance_lattice.state import declare_workspace, resolve_workspace

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "redecl"
        root.mkdir()
        declare_workspace(root, name="first", workspace_id="111111")
        declare_workspace(root, name="second", workspace_id="222222")
        seen = resolve_workspace(root)
    if seen.workspace_id != "222222":
        print(f"[state_workspace] FAIL (e): id={seen.workspace_id!r}", file=sys.stderr)
        return 1
    print("[state_workspace] (e) re-declare overwrites OK", file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_cwd_fallback,
        _check_declaration_overrides,
        _check_declaration_no_explicit_id,
        _check_subdir_resolves_to_root,
        _check_redeclaration_overwrites,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[state_workspace] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
