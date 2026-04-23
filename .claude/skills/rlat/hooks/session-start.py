"""SessionStart hook — surface cartridge freshness + async primer refresh.

On Claude Code session open, this hook:

  1. Warns if any ``*.rlat`` knowledge model in the project root is stale
     (older than ``RLAT_STALE_HOURS``, default 72h). The warning goes back
     to Claude as ``hookSpecificOutput.additionalContext`` so the
     assistant knows to rebuild before trusting retrieval results.

  2. Fires a detached ``rlat primer refresh`` in the background when the
     code primer (``.claude/resonance-context.md``) is stale *or* its
     stamped git HEAD has diverged from the working tree. The current
     session is never blocked; the *next* session picks up the regen.

Failure policy: all work is guarded; exceptions hit stderr and the hook
exits 0 so session open is never blocked.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _hookutil as H  # noqa: E402

STALE_HOURS = float(os.environ.get("RLAT_STALE_HOURS", "72"))


def _cartridge_age_hours(path: Path) -> float | None:
    try:
        return (time.time() - path.stat().st_mtime) / 3600.0
    except Exception:
        return None


def main() -> None:
    with H.guard("session-start"):
        payload = H.read_hook_input()
        project_dir = H.project_dir(payload)

        stale_names: list[str] = []
        for c in sorted(p for p in project_dir.glob("*.rlat") if p.is_file()):
            age = _cartridge_age_hours(c)
            if age is not None and age > STALE_HOURS:
                stale_names.append(f"{c.name} ({age / 24:.1f}d old)")

        if stale_names:
            H.emit_hook_output({
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": (
                        "[rlat] Knowledge model freshness warning — rebuild "
                        "before trusting results:\n  "
                        + "\n  ".join(stale_names)
                        + "\n  Run: rlat build ./docs ./src -o project.rlat\n"
                    ),
                }
            })

        try:
            _maybe_async_primer_refresh(project_dir)
        except Exception:
            pass


def _maybe_async_primer_refresh(project_dir: Path) -> None:
    """Fire-and-forget primer refresh when stamp says it's out of date."""
    try:
        sys.path.insert(0, str(project_dir / "src"))
        from resonance_lattice.primer.refresh import is_stale  # type: ignore
    except Exception:
        return

    code_primer = project_dir / ".claude" / "resonance-context.md"
    stale, reason = is_stale(
        code_primer,
        repo_root=project_dir,
        stale_hours=STALE_HOURS,
    )
    if not stale:
        return

    cmd = [sys.executable, "-m", "resonance_lattice.cli", "primer", "refresh"]
    kwargs: dict = {
        "cwd": str(project_dir),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | 0x00000008
        )
    else:
        kwargs["start_new_session"] = True

    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONPATH", str(project_dir / "src"))
    kwargs["env"] = env

    try:
        subprocess.Popen(cmd, **kwargs)
        H.emit_hook_output({
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": (
                    f"[rlat] Primer refresh queued in background ({reason}). "
                    "Next session will pick up the regenerated primer.\n"
                ),
            }
        })
    except Exception:
        pass


if __name__ == "__main__":
    main()
