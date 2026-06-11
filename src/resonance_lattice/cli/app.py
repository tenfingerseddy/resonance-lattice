"""Command dispatch + entry point.

`rlat <command>` is wired here. Each subcommand lives in its own module
under `cli/` and registers its parser via `add_subparser(sub)` at dispatch
time — that keeps cold-start fast (an `rlat build` invocation never
reaches the lazy `import anthropic` inside the LLM-calling commands).

Phase 3 deliverable.
"""

from __future__ import annotations

import argparse
import sys

# Phase 3 wires `build`. Other subcommands stay scaffolded; they print a
# stub message rather than NotImplementedError so `rlat <subcmd> --help`
# is still useful while the rest of the surface lands.
from . import audit as audit_cmd
from . import build as build_cmd
from . import compare as compare_cmd
from . import consolidate_insights as consolidate_insights_cmd
from . import convert as convert_cmd
from . import deep_search as deep_search_cmd
from . import expertise as expertise_cmd
from . import fabric as fabric_cmd
from . import grow as grow_cmd
from . import init as init_cmd
from . import install_encoder as install_encoder_cmd
from . import intent as intent_cmd
from . import lens as lens_cmd
from . import maintain as maintain_cmd
from . import memory as memory_cmd
from . import profile as profile_cmd
from . import search as search_cmd
from . import skill_context as skill_context_cmd
from . import probe as probe_cmd
from . import capture_env as capture_env_cmd
from . import reverify as reverify_cmd
from . import summary as summary_cmd
from . import trace as trace_cmd
from . import watch as watch_cmd
from . import workspace as workspace_cmd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlat")
    sub = parser.add_subparsers(dest="command", required=True)

    build_cmd.add_subparser(sub)
    search_cmd.add_subparser(sub)
    profile_cmd.add_subparser(sub)
    compare_cmd.add_subparser(sub)
    summary_cmd.add_subparser(sub)
    init_cmd.add_subparser(sub)
    install_encoder_cmd.add_subparser(sub)
    maintain_cmd.add_subparser(sub)
    memory_cmd.add_subparser(sub)
    skill_context_cmd.add_subparser(sub)
    convert_cmd.add_subparser(sub)
    deep_search_cmd.add_subparser(sub)
    watch_cmd.add_subparser(sub)
    intent_cmd.add_subparser(sub)
    workspace_cmd.add_subparser(sub)
    fabric_cmd.add_subparser(sub)
    expertise_cmd.add_subparser(sub)
    audit_cmd.add_subparser(sub)
    trace_cmd.add_subparser(sub)
    lens_cmd.add_subparser(sub)
    reverify_cmd.add_subparser(sub)
    probe_cmd.add_subparser(sub)
    capture_env_cmd.add_subparser(sub)
    consolidate_insights_cmd.add_subparser(sub)
    grow_cmd.add_subparser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    # CLI output uses unicode (e.g. `→` in compare summaries, `≥` in
    # banners). On Windows the default console codec is cp1252, which
    # raises `UnicodeEncodeError` mid-print and aborts the command.
    # Reconfigure stdout/stderr to UTF-8 so every subcommand renders
    # cleanly regardless of host codec. No-op on terminals that already
    # use UTF-8 (most POSIX shells, Windows Terminal with `chcp 65001`).
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass  # Detached / non-tty streams; leave alone.

    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
