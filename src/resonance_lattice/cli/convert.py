"""`rlat convert <km.rlat> --to {bundled|local|remote} [flags]`

Switches a knowledge model between storage modes WITHOUT rebuilding
embeddings. The bands, registry, ANN index, and optimised W projection
are storage-mode-independent; conversion is just a metadata + payload
swap, atomically reshaping the on-disk layout.

Six pairwise transitions, all routed through `store.conversion.convert`:

  local    ↔ bundled
  local    ↔ remote
  bundled  ↔ remote

The "two-step optimise on remote" workflow lives here too:

    rlat convert upstream.rlat --to local --source-root ./local/ \\
                               -o working.rlat
    rlat optimise working.rlat --corpus-description "..."

Audit 08 commit 4/6.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..store.conversion import ConversionDriftError, convert
from ._load import load_or_exit


def cmd_convert(args: argparse.Namespace) -> int:
    km_path = Path(args.knowledge_model)
    output_path = Path(args.output) if args.output else km_path

    # Friendly upfront load so the no-op "already in mode X" path doesn't
    # crash inside the conversion pipeline. Uses the same friendly-error
    # wrapper every other CLI command uses.
    contents = load_or_exit(km_path)
    current_mode = contents.metadata.store_mode

    if args.to == current_mode:
        print(
            f"[convert] {km_path} is already in mode {current_mode!r}; "
            f"nothing to do.",
            file=sys.stderr,
        )
        return 0

    if args.dry_run:
        print(f"[convert] --dry-run: would convert {km_path} from "
              f"{current_mode!r} to {args.to!r} ({len(contents.registry)} "
              f"passages, {len({c.source_file for c in contents.registry})} "
              f"files)")
        return 0

    source_root = Path(args.source_root) if args.source_root else None

    try:
        result = convert(
            km_path,
            target_mode=args.to,
            source_root=source_root,
            remote_url_base=args.remote_url_base,
            output_path=output_path,
        )
    except ConversionDriftError as e:
        print(
            f"error: {e}",
            file=sys.stderr,
        )
        # Show the first few drifted paths so the user can diagnose.
        for path in e.drifted_paths[:10]:
            print(f"  drifted: {path}", file=sys.stderr)
        if len(e.drifted_paths) > 10:
            print(f"  … and {len(e.drifted_paths) - 10} more",
                  file=sys.stderr)
        return 2
    except ValueError as e:
        # Missing flag / unknown mode / idempotent no-op all surface here
        # as ValueError with a specific message.
        print(f"error: {e}", file=sys.stderr)
        return 1

    output_mb = result.output_bytes / (1024 * 1024)
    print(f"[convert] {result.from_mode} → {result.to_mode}: wrote "
          f"{result.archive_path} ({result.n_passages} passages, "
          f"{result.n_files} files, {output_mb:.2f} MB)")
    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "convert",
        help="Switch a knowledge model between storage modes (no rebuild)",
    )
    p.add_argument("knowledge_model", help="Path to a .rlat knowledge model")
    p.add_argument(
        "--to", required=True, choices=["bundled", "local", "remote"],
        help="Target storage mode.",
    )
    p.add_argument(
        "--source-root", default=None,
        help="Directory to materialise source files (required for --to local; "
             "also used as the read root when source mode is bundled and "
             "target needs disk-resident files).",
    )
    p.add_argument(
        "--remote-url-base", default=None,
        help="URL prefix joined with each source_file relative path to "
             "produce the upstream URL (required for --to remote).",
    )
    p.add_argument(
        "-o", "--output", default=None,
        help="Output path (default: in-place atomic replace).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Report the transition + counts; do not read source bytes or write.",
    )
    p.set_defaults(func=cmd_convert)
