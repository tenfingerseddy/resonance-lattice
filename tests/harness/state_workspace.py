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


def _check_resolve_state_root_env_override() -> int:
    """(f) `resolve_state_root` honours the `RLAT_STATE_ROOT` env
    override — used by paired benches to isolate the intent graph +
    outcome ledger from the dogfood workspace. Unset → falls back to
    `<workspace-root>/.rlat-state/`.
    """
    import os

    from resonance_lattice.state import (
        STATE_ROOT_ENV,
        resolve_state_root,
    )

    with tempfile.TemporaryDirectory() as td:
        cwd = Path(td) / "proj"
        cwd.mkdir()
        override = Path(td) / "isolated-state"

        prior = os.environ.get(STATE_ROOT_ENV)
        try:
            os.environ[STATE_ROOT_ENV] = str(override)
            got = resolve_state_root(cwd)
            if got != override:
                print(f"[state_workspace] FAIL (f): override ignored — "
                      f"got {got!r} want {override!r}", file=sys.stderr)
                return 1
            del os.environ[STATE_ROOT_ENV]
            fallback = resolve_state_root(cwd)
            if fallback.name != ".rlat-state":
                print(f"[state_workspace] FAIL (f): fallback not "
                      f".rlat-state — got {fallback!r}", file=sys.stderr)
                return 1
        finally:
            if prior is not None:
                os.environ[STATE_ROOT_ENV] = prior
            else:
                os.environ.pop(STATE_ROOT_ENV, None)
    print("[state_workspace] (f) RLAT_STATE_ROOT env override OK",
          file=sys.stderr)
    return 0


def _check_resolve_primary_km() -> int:
    """(g) `resolve_primary_km` discovers the workspace's primary `.rlat`:
    none → None; exactly one → that one; several → the `<root-name>.rlat`
    tie-break, else None (ambiguous, never guess); and it scans the resolved
    workspace ROOT, so a subdirectory cwd still finds a root-level archive.
    """
    from resonance_lattice.state import declare_workspace, resolve_primary_km

    with tempfile.TemporaryDirectory() as td:
        # none → None
        empty = Path(td) / "empty"
        empty.mkdir()
        if resolve_primary_km(empty) is not None:
            print("[state_workspace] FAIL (g): empty workspace not None",
                  file=sys.stderr)
            return 1

        # exactly one → that one (any name)
        one = Path(td) / "one"
        one.mkdir()
        (one / "anything.rlat").write_bytes(b"")
        got = resolve_primary_km(one)
        if got is None or got.name != "anything.rlat":
            print(f"[state_workspace] FAIL (g): single archive → {got!r}",
                  file=sys.stderr)
            return 1

        # several + <root-name>.rlat present → the named one (tie-break)
        multi = Path(td) / "proj"
        multi.mkdir()
        for n in ("aaa.rlat", "proj.rlat", "zzz.rlat"):
            (multi / n).write_bytes(b"")
        got = resolve_primary_km(multi)
        if got is None or got.name != "proj.rlat":
            print(f"[state_workspace] FAIL (g): tie-break → {got!r} "
                  f"(want proj.rlat)", file=sys.stderr)
            return 1

        # several, none named <root-name> → None (ambiguous, never guess)
        amb = Path(td) / "ambiguous"
        amb.mkdir()
        (amb / "a.rlat").write_bytes(b"")
        (amb / "b.rlat").write_bytes(b"")
        if resolve_primary_km(amb) is not None:
            print("[state_workspace] FAIL (g): ambiguous not None",
                  file=sys.stderr)
            return 1

        # scans the resolved workspace ROOT, not raw cwd: a subdir cwd finds
        # the root-level archive (a declaration pins the root).
        decl_root = Path(td) / "declared"
        sub = decl_root / "packages" / "leaf"
        sub.mkdir(parents=True)
        declare_workspace(decl_root, name="declared", workspace_id="abcdef")
        (decl_root / "declared.rlat").write_bytes(b"")
        got = resolve_primary_km(sub)
        if got is None or got.name != "declared.rlat":
            print(f"[state_workspace] FAIL (g): subdir→root archive → {got!r}",
                  file=sys.stderr)
            return 1

    print("[state_workspace] (g) resolve_primary_km discovery + tie-break OK",
          file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_cwd_fallback,
        _check_declaration_overrides,
        _check_declaration_no_explicit_id,
        _check_subdir_resolves_to_root,
        _check_redeclaration_overwrites,
        _check_resolve_state_root_env_override,
        _check_resolve_primary_km,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[state_workspace] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
