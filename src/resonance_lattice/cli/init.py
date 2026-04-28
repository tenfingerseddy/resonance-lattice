"""`rlat init-project [-o output.rlat] [--source DIR]... [--no-primer]`

One-command project setup. Detects source directories in the current working
directory, builds a knowledge model with sensible defaults, and (unless
`--no-primer` is passed) writes a context primer to
`.claude/resonance-context.md` for AI-assistant integration.

This is sugar over:

  rlat build <detected-sources> -o <output>.rlat
  rlat summary <output>.rlat -o .claude/resonance-context.md

Single recipe — no encoder/storage knobs surfaced. Users who need overrides
should call `rlat build` and `rlat summary` directly.

Phase 3 deliverable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .build import cmd_build
from .summary import cmd_summary

# Standard source-directory names to auto-detect at the project root.
# Conservative on purpose — surprising the user with what got ingested is
# worse than asking them to add `--source` for unconventional layouts.
# Top-level dirs only; `rlat build` walks recursively so subdirs are covered
# automatically. `_walk_sources` dedupes by relative posix path, so even if
# a future entry overlaps (e.g. `docs/` and `docs/internal/`) double-
# ingestion is structurally prevented.
_AUTO_DIRS = ("docs", "src", "lib", "notebooks", "examples")

# Top-level files of these extensions are also picked up so a flat repo
# (just README.md + a few scripts at the root) gets a useful primer
# without forcing the user to organise into directories first.
_AUTO_TOPLEVEL_EXTS = frozenset({".md", ".rst", ".txt"})


def _detect_sources(cwd: Path) -> list[Path]:
    """Auto-detect plausible source paths under `cwd`.

    Looks for known directory names (`docs/`, `src/`, …) and top-level
    text files (`README.md`, etc.). Returns absolute paths. Empty list if
    nothing recognisable is found — caller treats that as a usage error
    so the user is forced to be explicit rather than getting a surprise
    empty knowledge model.
    """
    found: list[Path] = []
    for name in _AUTO_DIRS:
        candidate = cwd / name
        if candidate.is_dir():
            found.append(candidate)
    for path in sorted(cwd.iterdir()):
        if path.is_file() and path.suffix.lower() in _AUTO_TOPLEVEL_EXTS:
            found.append(path)
    return found


def cmd_init_project(args: argparse.Namespace) -> int:
    cwd = Path.cwd().resolve()
    sources: list[Path] = [Path(s).resolve() for s in (args.source or [])]
    if not sources:
        sources = _detect_sources(cwd)
    if not sources:
        print(
            "error: no recognisable sources found. rlat init-project looks for "
            f"{', '.join(_AUTO_DIRS)} or top-level *.md/*.rst/*.txt files. Pass "
            "`--source <dir>` to be explicit, or run `rlat build <sources> "
            "-o ...` directly.",
            file=sys.stderr,
        )
        return 1

    output = Path(args.output) if args.output else cwd / f"{cwd.name}.rlat"
    print(f"[init] detected sources:")
    for s in sources:
        print(f"  - {s.relative_to(cwd) if s.is_relative_to(cwd) else s}")
    print(f"[init] output: {output}")

    rc = cmd_build(argparse.Namespace(
        sources=[str(s) for s in sources],
        output=str(output),
        store_mode="local",  # default — corpus stays on disk
        kind="corpus",
        source_root=str(cwd),
        min_chars=200,
        max_chars=3200,
        batch_size=32,
        ext=None,
        remote_url_base=None,
    ))
    if rc != 0:
        return rc

    if args.no_primer:
        print("[init] --no-primer set; skipping summary primer")
        return 0

    primer_path = cwd / ".claude" / "resonance-context.md"
    rc = cmd_summary(argparse.Namespace(
        knowledge_model=str(output),
        output=str(primer_path),
        queries=None,
        source_root=str(cwd),
    ))
    if rc != 0:
        return rc

    print(
        f"\n[init] done. Next steps:\n"
        f"  rlat search {output.name} \"<query>\"\n"
        f"  rlat profile {output.name}\n"
        f"  cat {primer_path.relative_to(cwd) if primer_path.is_relative_to(cwd) else primer_path}"
    )
    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("init-project", help="One-command project setup")
    p.add_argument(
        "-o", "--output", default=None,
        help="Output .rlat path (default: <cwd-name>.rlat)",
    )
    p.add_argument(
        "--source", action="append", default=None,
        help="Explicit source path (repeatable). Skips auto-detection.",
    )
    p.add_argument(
        "--no-primer", action="store_true",
        help="Skip writing .claude/resonance-context.md",
    )
    p.set_defaults(func=cmd_init_project)
