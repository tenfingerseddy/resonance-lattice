"""Command dispatch + entry point.

`rlat <command>` is wired here. Each subcommand lives in its own module
under `cli/` and registers its parser via `add_subparser(sub)` at dispatch
time — that keeps cold-start fast (an `rlat build` invocation doesn't
import `cli/optimise` and pull in anthropic).

Phase 3 deliverable.
"""

from __future__ import annotations

import argparse
import sys

# Phase 3 wires `build`. Other subcommands stay scaffolded; they print a
# stub message rather than NotImplementedError so `rlat <subcmd> --help`
# is still useful while the rest of the surface lands.
from . import build as build_cmd
from . import compare as compare_cmd
from . import convert as convert_cmd
from . import deep_search as deep_search_cmd
from . import init as init_cmd
from . import install_encoder as install_encoder_cmd
from . import maintain as maintain_cmd
from . import memory as memory_cmd
from . import profile as profile_cmd
from . import search as search_cmd
from . import skill_context as skill_context_cmd
from . import optimise as optimise_cmd
from . import summary as summary_cmd


def main(argv: list[str] | None = None) -> int:
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
    optimise_cmd.add_subparser(sub)
    memory_cmd.add_subparser(sub)
    skill_context_cmd.add_subparser(sub)
    convert_cmd.add_subparser(sub)
    deep_search_cmd.add_subparser(sub)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
