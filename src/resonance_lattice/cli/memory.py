"""`rlat memory <subcommand>` — v2.1 flat-memory CLI.

The Sub-MVP slice of the §0.7 surface:

  add    — append a manual `["factual", ...]` row to the per-user store
  list   — tabular view of the sidecar with optional polarity / recurrence filters
  gc     — manual escape hatch (§0.5); never automatic

Subcommands shipping in MVP — `recall`, `distil`, `train`, `feedback`,
`doctor`, `migrate` — are stubbed here as banner-only entries so users
discover them via `rlat memory --help`. v2.0 names that have no v2.1
successor (`consolidate`, `primer`) print a deprecation banner pointing
at the migration path.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from ..memory._common import workspace_tag_for_cwd
from ..memory.store import (
    MANUAL_TRANSCRIPT_HASH,
    PRIMARY_POLARITY,
    Memory,
    path_for_user,
)

DEFAULT_PRIMARY_POLARITY = "factual"
PRIMARY_CHOICES: list[str] = sorted(PRIMARY_POLARITY)

# Exit codes:
#   0 — success
#   1 — user input error (bad polarity, empty text, unknown row id)
#   2 — deprecated subcommand: removed permanently in v2.1
#   3 — pending: subcommand ships in MVP, body not yet implemented
EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_DEPRECATED = 2
EXIT_PENDING_MVP = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_memory(args: argparse.Namespace) -> Memory:
    """Resolve the per-user memory root.

    `--memory-root` overrides the *base* directory (default
    `~/.rlat/memory/`); `--user` always picks the per-user subdirectory
    inside it. Passing both composes as `<base>/<user>/`. Memory's own
    constructor still accepts an exact root for tests + internal callers
    that need to bypass the user layer.
    """
    base = Path(args.memory_root) if args.memory_root else None
    return Memory(root=path_for_user(user_id=args.user, root=base))


def _print_banner(message: str, *, code: int) -> int:
    print(message, file=sys.stderr)
    return code


def _user_error(msg: str) -> int:
    """Print `error: <msg>` to stderr and return EXIT_USER_ERROR.

    Centralised so all `rlat memory` subcommands surface user-facing
    errors with the same prefix and exit code (memory_v21_hook (i)
    rc=1 contract).
    """
    print(f"error: {msg}", file=sys.stderr)
    return EXIT_USER_ERROR


def _deprecation_banner(old: str, replacement: str) -> int:
    return _print_banner(
        f"[rlat memory] `{old}` was removed in v2.1.\n"
        f"  → use `{replacement}` instead.\n"
        f"  See .claude/plans/fabric-agent-flat-memory.md §15 for the full deletion list.",
        code=EXIT_DEPRECATED,
    )


def _pending_banner(name: str) -> int:
    return _print_banner(
        f"[rlat memory] `{name}` ships in v2.1 MVP (not Sub-MVP).\n"
        f"  Tracking issue: https://github.com/tenfingerseddy/resonance-lattice/issues/88",
        code=EXIT_PENDING_MVP,
    )


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


def cmd_memory_add(args: argparse.Namespace) -> int:
    text = args.text
    if text == "-":
        text = sys.stdin.read()
    text = text.strip()
    if not text:
        return _user_error("refusing to add empty text")

    # §0.6 retrieval drops rows without a `workspace:<hash>` or
    # `cross-workspace` scope tag — manual rows must always carry one or
    # the other or they're unretrievable. Default to the cwd hash;
    # `--scope cross-workspace` adds the cross-workspace tag in addition.
    polarity = [args.polarity, workspace_tag_for_cwd()]
    if args.scope == "cross-workspace":
        polarity.append("cross-workspace")

    memory = _open_memory(args)
    try:
        row_id = memory.add_row(
            text=text,
            polarity=polarity,
            transcript_hash=MANUAL_TRANSCRIPT_HASH,
        )
    except ValueError as exc:
        return _user_error(str(exc))

    print(f"[rlat memory] added row {row_id} ({polarity[0]})", file=sys.stderr)
    return EXIT_OK


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def cmd_memory_list(args: argparse.Namespace) -> int:
    memory = _open_memory(args)
    rows, _ = memory.read_all()

    rows = [
        r
        for r in rows
        if (args.polarity is None or args.polarity in r.polarity)
        and (args.min_recurrence is None or r.recurrence_count >= args.min_recurrence)
        and (args.include_bad or not r.is_bad)
    ]
    rows.sort(key=lambda r: (r.recurrence_count, r.created_at), reverse=True)
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.format == "json":
        print(json.dumps([r.to_jsonl_dict() for r in rows], indent=2))
        return EXIT_OK

    if not rows:
        print("(no rows match)", file=sys.stderr)
        return EXIT_OK
    for row in rows:
        print(row.summary())
    print(f"\n[rlat memory] {len(rows)} row(s)", file=sys.stderr)
    return EXIT_OK


# ---------------------------------------------------------------------------
# gc — manual escape hatch
# ---------------------------------------------------------------------------


def cmd_memory_recall(args: argparse.Namespace) -> int:
    """`rlat memory recall <query> [--daemon]`.

    `--daemon` boots the long-lived recall server and blocks until
    idle exit. Without `--daemon`, runs the synchronous §0.6 pipeline
    against the on-disk store and prints hits in the requested format.
    """
    if args.daemon:
        return _run_recall_daemon(args)
    return _run_recall_oneshot(args)


def _run_recall_oneshot(args: argparse.Namespace) -> int:
    if not args.query:
        return _user_error(
            "`recall` requires a <query> argument (or --daemon to "
            "boot the long-lived server)."
        )

    from ..memory.recall import recall

    memory = _open_memory(args)
    hits = recall(args.query, store=memory, top_k=args.top_k)
    if args.polarity is not None:
        hits = [h for h in hits if args.polarity in h.row.polarity]

    if args.format == "json":
        print(json.dumps(
            [{"row": h.row.to_jsonl_dict(), "cosine": h.cosine} for h in hits],
            indent=2,
        ))
        return EXIT_OK

    if not hits:
        print("(no rows pass the §0.6 gates for this query)", file=sys.stderr)
        return EXIT_OK
    for hit in hits:
        if args.explain:
            print(f"{hit.row.summary()}  cos={hit.cosine:.3f}")
        else:
            print(hit.row.summary())
    print(f"\n[rlat memory] {len(hits)} hit(s)", file=sys.stderr)
    return EXIT_OK


def _run_recall_daemon(args: argparse.Namespace) -> int:
    from ..memory.daemon import (
        DaemonServer,
        daemon_socket_address,
        load_or_create_authkey,
    )

    memory = _open_memory(args)
    address = daemon_socket_address(memory.root)
    # POSIX: a stale socket file blocks `Listener` bind. Detect by
    # attempting a probe Client; if it succeeds, refuse to launch.
    if isinstance(address, str) and not address.startswith(r"\\."):
        if Path(address).exists():
            return _user_error(
                f"daemon socket already at {address}; another "
                f"daemon may be running. Remove the socket and retry "
                f"if you're certain no daemon is live."
            )
    try:
        encoder = memory._ensure_encoder()  # type: ignore[attr-defined]
        revision = getattr(encoder, "revision", "unknown")
        server = DaemonServer(
            store=memory,
            encoder=encoder,
            encoder_revision=revision,
            address=address,
            authkey=load_or_create_authkey(memory.root),
        )
        print(f"[rlat memory] daemon listening at {address}", file=sys.stderr)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[rlat memory] daemon stopped", file=sys.stderr)
    return EXIT_OK


def cmd_memory_doctor(args: argparse.Namespace) -> int:
    """`rlat memory doctor` — probe per-user store + daemon."""
    from ..field.encoder import MODEL_ID
    from ..memory.daemon import diagnose

    memory = _open_memory(args)
    report = diagnose(memory.root, encoder_revision=MODEL_ID)
    for check in report.checks:
        marker = "OK" if check["ok"] else "FAIL"
        print(f"[{marker}] {check['name']}: {check['message']}")
    return EXIT_OK


def cmd_memory_migrate(args: argparse.Namespace) -> int:
    """`rlat memory migrate <v2.0-root> --to <v2.1-root> --user <id>`.

    One-shot v2.0 LayeredMemory → v2.1 flat-memory migration per §14.4.
    Lossy by design (see §14.5 honest list); recommended first invocation
    is `--dry-run` to preview the polarity-heuristic classification.
    """
    from ..memory.migrate import migrate

    v20_root = Path(args.v20_root)
    v21_root = Path(args.to)
    if not v20_root.exists():
        return _user_error(f"v2.0 memory root not found: {v20_root}")
    if args.polarity_default not in PRIMARY_POLARITY:
        return _user_error(
            f"--polarity-default {args.polarity_default!r} not in "
            f"{sorted(PRIMARY_POLARITY)}"
        )
    try:
        result = migrate(
            v20_root,
            v21_root=v21_root,
            user_id=args.migrate_user,
            dry_run=args.dry_run,
            polarity_default=args.polarity_default,
        )
    except Exception as exc:
        return _user_error(f"migrate failed: {type(exc).__name__}: {exc}")
    print(result.summary(), file=sys.stderr)
    return EXIT_OK


def cmd_memory_hook(args: argparse.Namespace) -> int:
    """`rlat memory hook` — UserPromptSubmit hook entry point.

    Reads the Claude Code UserPromptSubmit envelope from stdin, runs
    §0.6 recall via the daemon (lazy-spawning on first fire per §5.2.1),
    and emits the §0.4 `<rlat-memory>` block to stdout as
    `hookSpecificOutput.additionalContext`. Always exits 0 (fail-open
    per §16.5 / §18.5).
    """
    from ..memory.user_prompt import _trace, run_hook

    _trace("cli:cmd_memory_hook entry")
    base = Path(args.memory_root) if args.memory_root else None
    return run_hook(user_id=args.user, memory_root_base=base)


def cmd_memory_capture(args: argparse.Namespace) -> int:
    """`rlat memory capture` — SessionEnd-hook entry point.

    Reads the Claude Code SessionEnd envelope from stdin
    (`{session_id, transcript_path, cwd, ...}`), parses the JSONL
    transcript best-effort, runs the §5.2 capture pipeline, and emits
    `{}` to stdout. Always exits 0 (fail-open per §16.5 / §18.5).

    Wire via `settings.json` `hooks.SessionEnd` — the plan §5.2 calls
    this the "Stop hook" but Claude Code's actual `Stop` event fires
    per-assistant-turn, not at session close. SessionEnd matches the
    spec's intent (capture-once-per-session).
    """
    from ..memory.user_prompt import _trace, run_capture_hook

    _trace("cli:cmd_memory_capture entry")
    base = Path(args.memory_root) if args.memory_root else None
    return run_capture_hook(user_id=args.user, memory_root_base=base)


def _format_train_status(result) -> str:
    """One status line per train operator action."""
    if result.field_changed == "recurrence_count":
        return (
            f"[rlat memory] {result.action} {result.row_id} "
            f"({result.field_changed}: {result.before} -> {result.after})"
        )
    return (
        f"[rlat memory] {result.action} {result.row_id} "
        f"({result.field_changed}: {str(result.before).lower()} -> "
        f"{str(result.after).lower()})"
    )


def cmd_memory_train(args: argparse.Namespace) -> int:
    """Train operator surface — `--bad-vote` / `--good-vote` /
    `--corroborate` mutate individual rows. The full §8 GRPO loop ships
    as the `/rlat-train` slash command (Day 9-10) — `train <task>`
    here just points at the slash command and exits 3 (pending-MVP).
    """
    from ..memory.train import bad_vote, corroborate, good_vote

    # Each entry: (cli flag attr, the operator function, extra kwargs).
    flag_table: tuple[tuple[str, callable, dict], ...] = (
        ("bad_vote", bad_vote, {"why": args.why}),
        ("good_vote", good_vote, {}),
        ("corroborate", corroborate, {}),
    )
    chosen = [(name, fn, kw) for name, fn, kw in flag_table
              if getattr(args, name) is not None]
    if len(chosen) > 1:
        return _user_error(
            "--bad-vote / --good-vote / --corroborate are mutually "
            "exclusive; pass at most one."
        )

    if not chosen:
        if args.task is None:
            return _user_error(
                "`rlat memory train` requires either a <task> "
                "argument (GRPO loop) or one of "
                "`--bad-vote` / `--good-vote` / `--corroborate <row_id>`."
            )
        return _print_banner(
            f"[rlat memory] `train <task>` runs the §8 GRPO loop, which "
            f"requires Claude Code's Task primitive — invoke "
            f"`/rlat-train {args.task}` from a Claude Code session "
            f"instead. The slash command ships in v2.1 MVP "
            f"(https://github.com/tenfingerseddy/resonance-lattice/issues/88).",
            code=EXIT_PENDING_MVP,
        )

    name, op, kwargs = chosen[0]
    memory = _open_memory(args)
    try:
        result = op(store=memory, row_id=getattr(args, name), **kwargs)
    except KeyError as exc:
        return _user_error(str(exc))
    print(_format_train_status(result), file=sys.stderr)
    return EXIT_OK


def cmd_memory_gc(args: argparse.Namespace) -> int:
    if not any([args.polarity, args.min_recurrence is not None,
                args.max_age_days is not None, args.is_bad]):
        return _user_error(
            "`gc` requires at least one filter "
            "(`--polarity`, `--min-recurrence`, `--max-age-days`, or `--is-bad`).\n"
            "Refusing to run with no filters — gc is a manual escape hatch, "
            "not a sweep (§0.5)."
        )

    cutoff_str: str | None = None
    if args.max_age_days is not None:
        cutoff = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=args.max_age_days)
        cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

    memory = _open_memory(args)
    rows, _ = memory.read_all()
    # Per §0.5 + Appendix D D.4 (c): bad-voted rows are kept for re-distil
    # suppression. Without `--is-bad`, gc skips them entirely; `--is-bad`
    # is the only way to delete a row tagged is_bad=True. Per §15.2 the
    # age clock uses `last_corroborated_at`, not `created_at` — a row that
    # corroborates again resets its eligibility window.
    targets = [
        r
        for r in rows
        if (r.is_bad if args.is_bad else not r.is_bad)
        and (args.polarity is None or args.polarity in r.polarity)
        and (args.min_recurrence is None or r.recurrence_count <= args.min_recurrence)
        and (cutoff_str is None or r.last_corroborated_at < cutoff_str)
    ]

    if not targets:
        print("(no rows match the filters)", file=sys.stderr)
        return EXIT_OK

    if args.dry_run:
        for r in targets:
            print(r.summary())
        print(
            f"\n[rlat memory] would delete {len(targets)} row(s) (--dry-run; nothing written)",
            file=sys.stderr,
        )
        return EXIT_OK

    n = memory.delete_rows([r.row_id for r in targets])
    print(f"[rlat memory] gc deleted {n} row(s)", file=sys.stderr)
    return EXIT_OK


# ---------------------------------------------------------------------------
# Stubs / deprecation banners
# ---------------------------------------------------------------------------


# Each tuple: (subcommand name, kind). `kind` drives both the help text
# rendered at parse time and the runtime banner.
_PENDING_MVP_SUBCOMMANDS: tuple[str, ...] = (
    "distil", "feedback",
)
# `(removed_name, v2.1 successor or guidance)`.
_DEPRECATED_SUBCOMMANDS: tuple[tuple[str, str], ...] = (
    ("consolidate", "rlat memory distil"),
    ("primer",
     "the per-prompt UserPromptSubmit hook (no static primer in v2.1; see §17.3)"),
)


def _make_pending_handler(name: str):
    def handler(_args: argparse.Namespace) -> int:
        return _pending_banner(name)
    return handler


def _make_deprecation_handler(name: str, replacement: str):
    def handler(_args: argparse.Namespace) -> int:
        return _deprecation_banner(name, replacement)
    return handler


# ---------------------------------------------------------------------------
# Parser wiring
# ---------------------------------------------------------------------------


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("memory", help="Per-user flat-memory operations (v2.1)")
    p.add_argument(
        "--memory-root",
        default=None,
        help="Override the memory root (default: ~/.rlat/memory/<user-id>/).",
    )
    p.add_argument(
        "--user",
        default=None,
        help="Override the user id (default: $RLAT_MEMORY_USER → $USER → $USERNAME).",
    )
    sub_mem = p.add_subparsers(dest="memory_subcommand", required=True)

    p_add = sub_mem.add_parser("add", help="Append a manual row.")
    p_add.add_argument("text", help="Row text (or `-` for stdin).")
    p_add.add_argument(
        "--polarity",
        default=DEFAULT_PRIMARY_POLARITY,
        choices=PRIMARY_CHOICES,
        help="Primary polarity tag (default: factual).",
    )
    p_add.add_argument(
        "--scope",
        default=None,
        choices=["cross-workspace"],
        help="Scope tag. Without this, the row is workspace-implicit.",
    )
    p_add.set_defaults(func=cmd_memory_add)

    p_list = sub_mem.add_parser("list", help="Tabular view of the sidecar.")
    p_list.add_argument("--polarity", default=None, help="Filter by polarity tag.")
    p_list.add_argument("--min-recurrence", type=int, default=None)
    p_list.add_argument("--limit", type=int, default=None)
    p_list.add_argument("--include-bad", action="store_true",
                        help="Show is_bad rows (default: hidden).")
    p_list.add_argument("--format", default="text", choices=["text", "json"])
    p_list.set_defaults(func=cmd_memory_list)

    p_recall = sub_mem.add_parser(
        "recall",
        help="Recall: §0.6 gated retrieval. Synchronous one-shot by "
             "default; --daemon boots the long-lived recall server.",
    )
    p_recall.add_argument("query", nargs="?", default=None,
                          help="Query text (required for one-shot mode; "
                               "ignored under --daemon).")
    p_recall.add_argument("--daemon", action="store_true",
                          help="Boot the long-lived recall daemon.")
    p_recall.add_argument("--top-k", type=int, default=5,
                          help="Maximum hits to return (default: 5).")
    p_recall.add_argument("--polarity", default=None, choices=PRIMARY_CHOICES,
                          help="Post-filter to hits with this primary tag.")
    p_recall.add_argument("--format", default="text", choices=["text", "json"],
                          help="Output format (default: text).")
    p_recall.add_argument("--explain", action="store_true",
                          help="Append per-hit cosine score to text output.")
    p_recall.set_defaults(func=cmd_memory_recall)

    p_doctor = sub_mem.add_parser(
        "doctor",
        help="Probe the per-user store + daemon. "
             "Prints one line per check.",
    )
    p_doctor.set_defaults(func=cmd_memory_doctor)

    p_hook = sub_mem.add_parser(
        "hook",
        help="UserPromptSubmit hook entry point. Reads JSON from stdin, "
             "writes JSON to stdout per Claude Code hook contract. "
             "Wire via settings.json `hooks.UserPromptSubmit`.",
    )
    p_hook.set_defaults(func=cmd_memory_hook)

    p_capture = sub_mem.add_parser(
        "capture",
        help="SessionEnd-hook entry point. Reads Claude Code SessionEnd "
             "envelope from stdin (`transcript_path` + `session_id` + "
             "`cwd`), runs the §5.2 capture pipeline. Wire via "
             "settings.json `hooks.SessionEnd`.",
    )
    p_capture.set_defaults(func=cmd_memory_capture)

    p_migrate = sub_mem.add_parser(
        "migrate",
        help="One-shot v2.0 LayeredMemory → v2.1 flat-memory migration "
             "(§14). Lossy by design; --dry-run to preview the polarity "
             "heuristic. Module deleted in v2.2.",
    )
    p_migrate.add_argument("v20_root", help="Path to the v2.0 memory root.")
    p_migrate.add_argument("--to", dest="to", required=True,
                            help="v2.1 base directory (per-user subdir created).")
    p_migrate.add_argument("--migrate-user", dest="migrate_user", required=True,
                            help="User id under <to>/<id>/ (distinct from --user).")
    p_migrate.add_argument("--dry-run", action="store_true",
                            help="Preview the migration without writing or "
                                 "archiving the v2.0 root.")
    p_migrate.add_argument("--polarity-default", default="factual",
                            choices=PRIMARY_CHOICES,
                            help="Polarity for rows the verb-scan heuristic "
                                 "doesn't classify (default: factual).")
    p_migrate.set_defaults(func=cmd_memory_migrate)

    p_train = sub_mem.add_parser(
        "train",
        help="Mutate a row (--bad-vote / --good-vote / --corroborate). "
             "GRPO loop runs via /rlat-train slash command.",
    )
    p_train.add_argument("task", nargs="?", default=None,
                         help="Task id (positional) — banner-only stub; use "
                              "/rlat-train slash command from Claude Code.")
    p_train.add_argument("--bad-vote", default=None, metavar="ROW_ID",
                         help="Mark row_id is_bad=True (drops from recall).")
    p_train.add_argument("--good-vote", default=None, metavar="ROW_ID",
                         help="Reverse a bad-vote: is_bad=False.")
    p_train.add_argument("--corroborate", default=None, metavar="ROW_ID",
                         help="Bump recurrence_count + last_corroborated_at.")
    p_train.add_argument("--why", default=None,
                         help="Optional rationale for --bad-vote (audit log).")
    p_train.set_defaults(func=cmd_memory_train)

    p_gc = sub_mem.add_parser(
        "gc",
        help="Manual escape-hatch deletion. Requires at least one filter.",
    )
    p_gc.add_argument("--polarity", default=None, help="Filter by polarity tag.")
    p_gc.add_argument(
        "--min-recurrence", type=int, default=None,
        help="Delete rows with recurrence_count <= this value.",
    )
    p_gc.add_argument(
        "--max-age-days", type=int, default=None,
        help="Delete rows whose created_at is older than this many days.",
    )
    p_gc.add_argument("--is-bad", action="store_true",
                      help="Delete rows tagged is_bad=true.")
    p_gc.add_argument("--dry-run", action="store_true",
                      help="Print what would go; don't write.")
    p_gc.set_defaults(func=cmd_memory_gc)

    # Pending MVP subcommands — banner stubs so `rlat memory --help`
    # documents the full §0.7 surface even though the bodies don't ship
    # until #88.
    for name in _PENDING_MVP_SUBCOMMANDS:
        sp = sub_mem.add_parser(name, help=f"(MVP) {name} — ships in v2.1 MVP.")
        sp.add_argument("args", nargs="*", help=argparse.SUPPRESS)
        sp.set_defaults(func=_make_pending_handler(name))

    # v2.0 names with no v2.1 successor — banner-only.
    for name, replacement in _DEPRECATED_SUBCOMMANDS:
        sp = sub_mem.add_parser(name, help=f"(removed) `{name}` — see banner.")
        sp.add_argument("args", nargs="*", help=argparse.SUPPRESS)
        sp.set_defaults(func=_make_deprecation_handler(name, replacement))
