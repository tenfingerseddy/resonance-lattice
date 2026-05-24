"""`rlat memory <subcommand>` — v2.1 flat-memory CLI.

The Sub-MVP slice of the §0.7 surface:

  add    — append a manual `["factual", ...]` row to the per-user store
  list   — tabular view of the sidecar with optional polarity / recurrence filters
  gc     — manual escape hatch (§0.5); never automatic

The full §0.7 surface ships: add / list / recall / distil / train /
feedback / verify / consolidate / gc / doctor / migrate, plus the
eval / rollup / dedup / session-mark / corroborate / capture / hook
entry points. v2.0 names with no v2.1 successor (`primer`) print a
deprecation banner.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from ..memory._common import workspace_tag_for_cwd
from ..memory.feedback import FEEDBACK_VERDICTS
from ..memory.store import (
    MANUAL_TRANSCRIPT_HASH,
    PRIMARY_POLARITY,
    Memory,
    path_for_user,
)
from ._errors import EXIT_OK, EXIT_USER_ERROR, user_error as _user_error

DEFAULT_PRIMARY_POLARITY = "factual"
PRIMARY_CHOICES: list[str] = sorted(PRIMARY_POLARITY)

# Exit codes:
#   0 — success (EXIT_OK, imported)
#   1 — user input error (EXIT_USER_ERROR, imported)
#   2 — deprecated subcommand: removed permanently in v2.1
#   3 — slash-command surface: the operation runs as a Claude Code slash
#       command, not a CLI body (currently `train <task>` → /rlat-train)
EXIT_DEPRECATED = 2
EXIT_PENDING_MVP = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_iso_strict(value: str) -> _dt.datetime:
    """Strict ISO-8601 parser for user-supplied flag values.

    Naïve timestamps are treated as UTC. Unparseable input raises
    `ValueError` so the caller can surface a usage error — the tolerant
    `memory._common.parse_iso_utc` falls back to "now", which is wrong
    for an explicit operator-supplied window.
    """
    cleaned = value.replace("Z", "+00:00") if value.endswith("Z") else value
    parsed = _dt.datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed


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


def _deprecation_banner(old: str, replacement: str) -> int:
    return _print_banner(
        f"[rlat memory] `{old}` was removed in v2.1.\n"
        f"  → use `{replacement}` instead.\n"
        f"  See .claude/plans/fabric-agent-flat-memory.md §15 for the full deletion list.",
        code=EXIT_DEPRECATED,
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
    # Match the production hook: it recalls via the daemon with
    # auto_tune_cold_start=True, so a sparse store surfaces hits under
    # the relaxed gates. Without this the inspection CLI would show a
    # blackout the live hook doesn't — the tool would disagree with
    # what it's meant to inspect.
    hits = recall(
        args.query, store=memory, top_k=args.top_k,
        auto_tune_cold_start=True,
    )
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


def cmd_memory_eval(args: argparse.Namespace) -> int:
    """`rlat memory eval` — print the longitudinal scorecard.

    Three modes:
      latest    — print the scorecard for the most recent window (default)
      compare   — split N day-windows into early + late and print the
                  pass/fail comparison
      json      — emit JSON for downstream tooling
    """
    import json as _json

    from ..state import (
        OutcomeLedger,
        RecallCache,
        WindowSpec,
        aggregate_windows,
        compute_session_scorecard,
        daily_windows,
        render_comparison,
        render_summary,
        resolve_state_root,
        scorecard_to_dict,
    )

    cwd = Path(args.cwd) if args.cwd else Path.cwd()
    state_root = resolve_state_root(cwd)
    memory = _open_memory(args)

    # Read ledger + cache + memory depth once and reuse across all
    # windows — without this, default `--sessions=20` would do 40 full
    # JSONL parses where 2 + filtering does the same work.
    outcomes = list(OutcomeLedger(state_root).iter_records())
    recalls = RecallCache(state_root).read_recent(limit=None)
    rows, _ = memory.read_all()
    memory_depth: dict[str, int] = {}
    for row in rows:
        memory_depth[row.level] = memory_depth.get(row.level, 0) + 1

    if args.since is not None or args.until is not None:
        # Explicit window overrides the daily-window default. Must be
        # paired; --compare needs ≥2 windows so it's incompatible here.
        if args.since is None or args.until is None:
            print(
                "error: --since and --until must be supplied together",
                file=sys.stderr,
            )
            return EXIT_USER_ERROR
        try:
            since_dt = _parse_iso_strict(args.since)
            until_dt = _parse_iso_strict(args.until)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_USER_ERROR
        if since_dt >= until_dt:
            print(
                "error: --since must be strictly earlier than --until",
                file=sys.stderr,
            )
            return EXIT_USER_ERROR
        if args.compare:
            print(
                "error: --compare requires multiple windows; "
                "drop --since/--until or remove --compare",
                file=sys.stderr,
            )
            return EXIT_USER_ERROR
        windows = [WindowSpec(
            since=since_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            until=until_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            label=since_dt.strftime("explicit-%Y-%m-%d"),
        )]
    else:
        windows = daily_windows(
            n_sessions=args.sessions, state_root=state_root,
        )
    scorecards = [
        compute_session_scorecard(
            state_root, memory=memory, window=w,
            outcomes=outcomes, recalls=recalls, memory_depth=memory_depth,
        )
        for w in windows
    ]

    if args.compare:
        early_n = max(1, args.sessions // 4)
        late_n = max(1, args.sessions // 4)
        comparison = aggregate_windows(
            scorecards,
            early=(0, early_n),
            late=(args.sessions - late_n, args.sessions),
        )
        if args.json:
            print(_json.dumps({
                "early": scorecard_to_dict(comparison.early),
                "late": scorecard_to_dict(comparison.late),
                "useful_passed": comparison.useful_passed,
                "effortless_passed": comparison.effortless_passed,
                "benchmark_passed": comparison.benchmark_passed,
            }, indent=2))
        else:
            print(render_comparison(comparison))
        return EXIT_OK

    latest = scorecards[-1]
    if args.json:
        print(_json.dumps(scorecard_to_dict(latest), indent=2))
    else:
        print(render_summary(latest))
    return EXIT_OK


def cmd_memory_rollup(args: argparse.Namespace) -> int:
    """`rlat memory rollup` — weekly digest: this week vs prior week.

    The operational measurement of the manifesto's falsifiable claim:
    "the harness gets measurably better at session N+1 than at N."
    Aggregates the trailing 7 days vs the 7 days before that, prints
    the same useful/effortless PASS/FAIL the cumulative-window
    `--compare` mode prints — but anchored on absolute time so
    regressions surface the week they land. Pair with `rlat freshness`
    so cartridge staleness can't quietly drag the numbers.

    `--weeks N` extends the comparison: with N=4, the early window
    aggregates the first 2 weeks and the late window aggregates the
    last 2 weeks. Default N=2 is the "this week vs last week" case.
    """
    from ..state import (
        OutcomeLedger,
        RecallCache,
        aggregate_windows,
        compute_session_scorecard,
        render_comparison,
        resolve_state_root,
        scorecard_to_dict,
        weekly_windows,
    )

    cwd = Path(args.cwd) if args.cwd else Path.cwd()
    state_root = resolve_state_root(cwd)
    memory = _open_memory(args)

    if args.weeks < 2:
        return _user_error(
            f"--weeks must be ≥2 to compare; got {args.weeks}"
        )

    outcomes = list(OutcomeLedger(state_root).iter_records())
    recalls = RecallCache(state_root).read_recent(limit=None)
    rows, _ = memory.read_all()
    memory_depth: dict[str, int] = {}
    for row in rows:
        memory_depth[row.level] = memory_depth.get(row.level, 0) + 1

    windows = weekly_windows(n_weeks=args.weeks)
    scorecards = [
        compute_session_scorecard(
            state_root, memory=memory, window=w,
            outcomes=outcomes, recalls=recalls, memory_depth=memory_depth,
        )
        for w in windows
    ]
    # For N=2, compare prev_week (early=[0:1]) vs current_week (late=[1:2]).
    # For N=4, compare first half (early=[0:2]) vs last half (late=[2:4]).
    half = args.weeks // 2
    comparison = aggregate_windows(
        scorecards, early=(0, half), late=(args.weeks - half, args.weeks),
    )
    if args.json:
        print(json.dumps({
            "early": scorecard_to_dict(comparison.early),
            "late": scorecard_to_dict(comparison.late),
            "useful_passed": comparison.useful_passed,
            "effortless_passed": comparison.effortless_passed,
            "benchmark_passed": comparison.benchmark_passed,
        }, indent=2))
    else:
        print(render_comparison(comparison))
    return EXIT_OK


def cmd_memory_dedup(args: argparse.Namespace) -> int:
    """`rlat memory dedup` — retroactive same-text-same-workspace collapse.

    Capture-time dedup (commit 46b74013) prevents new duplicates from
    accumulating. Memory written before that fix still carries N copies
    of every recurring captured event. This pass groups event rows by
    `(text, workspace_tag)`, keeps the oldest of each group with
    `recurrence_count` set to the cluster total, and deletes the rest.

    Idempotent. `--dry-run` reports what would change without touching disk.
    """
    from ..memory.dedup import dedup_event_rows

    memory = _open_memory(args)
    result = dedup_event_rows(memory, dry_run=args.dry_run)
    verb = "would collapse" if args.dry_run else "collapsed"
    print(
        f"[rlat memory] dedup: {verb} {result.rows_collapsed} row(s) into "
        f"{result.groups_collapsed} group(s); {result.rows_examined} event "
        f"rows examined"
    )
    return EXIT_OK


def cmd_memory_session_mark(args: argparse.Namespace) -> int:
    """`rlat memory session-mark` — record a session boundary timestamp.

    Appends one row to `<state-root>/ledger/sessions.jsonl`. When markers
    exist, `rlat memory eval` slices windows by session id instead of
    calendar day. Sessions that span midnight stay coherent; multiple
    sessions in one day each get their own scorecard.
    """
    from ..state import SessionMarkerLog, resolve_state_root

    cwd = Path(args.cwd) if args.cwd else Path.cwd()
    state_root = resolve_state_root(cwd)
    marker = SessionMarkerLog(state_root).write()
    print(
        f"[rlat memory] session-mark {marker.session_id} at {marker.timestamp}",
        file=sys.stderr,
    )
    return EXIT_OK


def cmd_memory_corroborate(args: argparse.Namespace) -> int:
    """`rlat memory corroborate <row_id>` — calibration mechanism 4.

    The user explicitly confirms a row is trustworthy; its confidence
    raises one step immediately (low→medium→high→verified). A no-op
    when the row is already `verified`.
    """
    from ..memory.confidence import corroborate_row

    memory = _open_memory(args)
    change = corroborate_row(memory, args.row_id)
    if change is None:
        rows, _ = memory.read_all()
        if not any(r.row_id == args.row_id for r in rows):
            return _user_error(f"no row with id {args.row_id!r}")
        print(f"[rlat memory] {args.row_id} already at verified — no change",
              file=sys.stderr)
        return EXIT_OK
    print(f"[rlat memory] corroborated {change.row_id}: "
          f"{change.from_confidence} → {change.to_confidence}",
          file=sys.stderr)
    return EXIT_OK


def cmd_memory_distil(args: argparse.Namespace) -> int:
    """`rlat memory distil` — §7 LLM lesson extraction over captured rows.

    Reads the recently-captured side of the per-user store, runs the
    distil prompt against an Anthropic model, and writes the resulting
    prefer/avoid/factual lessons back — corroborating near-duplicates
    instead of writing new rows. Needs an Anthropic API key; a store
    with no fresh capture rows is a no-op that returns before any LLM
    call. `--dry-run` reports candidates without writing.
    """
    from ..memory.distil import distil
    from ..memory.redaction import Redactor
    from .intent import _maybe_llm_client

    llm = _maybe_llm_client()
    if llm is None:
        return _user_error(
            "distil needs an Anthropic API key "
            "(set CLAUDE_API_2 / CLAUDE_API / ANTHROPIC_API_KEY)."
        )
    memory = _open_memory(args)
    redactor = Redactor(audit_log_path=memory.root / "redaction.log")
    result = distil(
        store=memory, redactor=redactor, client=llm,
        user_id=args.user or "user",
        since=args.since, all_rows=args.all, session=args.session,
        dry_run=args.dry_run, max_lessons=args.max_lessons,
    )
    verb = "would write" if args.dry_run else "wrote"
    print(
        f"[rlat memory] distil: processed {result.processed_count} capture "
        f"row(s) — {verb} {len(result.written_row_ids)}, corroborated "
        f"{len(result.corroborated_row_ids)}, skipped {result.skipped_count}",
        file=sys.stderr,
    )
    for note in result.notes:
        print(f"  note: {note}", file=sys.stderr)
    return EXIT_OK


def cmd_memory_feedback(args: argparse.Namespace) -> int:
    """`rlat memory feedback <good|bad>` — §9.5 vote on the most recent
    recall injection.

    Appends a `{verdict, timestamp}` line to `<memory-root>/feedback.log`.
    Logged, not acted on automatically — the weekly review reads it.
    """
    from ..memory.feedback import log_feedback

    memory = _open_memory(args)
    try:
        entry = log_feedback(memory.root, args.verdict)
    except ValueError as exc:
        return _user_error(str(exc))
    print(
        f"[rlat memory] feedback logged: {entry['verdict']} at "
        f"{entry['timestamp']}",
        file=sys.stderr,
    )
    return EXIT_OK


def _make_corpus_retriever(km_path: Path, source_root: str | None):
    """Build a `(query, top_k) -> [passage_text]` retriever over a
    knowledge model — the corpus seam for `corpus_verification_pass`.

    Loads the archive + encoder once; the returned closure runs one
    encode + dense retrieve per call. Drifted / missing passages carry
    empty text from `verify_hits` and are dropped."""
    from ..field import ann, retrieve
    from ..field.encoder import Encoder
    from ..store.verified import verify_hits
    from ._load import load_or_exit, open_store_or_exit

    contents = load_or_exit(km_path)
    handle = contents.select_band()
    ann_index = ann.deserialize(handle.ann_blob) if handle.ann_blob else None
    store = open_store_or_exit(km_path, contents, source_root)
    encoder = Encoder()

    def _retrieve(query: str, top_k: int) -> list[str]:
        emb = encoder.encode([query])[0]
        raw = retrieve(emb, handle, ann_index, contents.registry, top_k)
        return [
            h.text
            for h in verify_hits(raw, store, contents.registry)
            if h.text
        ]

    return _retrieve


def cmd_memory_verify(args: argparse.Namespace) -> int:
    """`rlat memory verify <corpus.rlat>` — calibration mechanism 2.

    The corpus-verification scan. Checks every high-criticality row at
    `low` or `verified` confidence against the supplied knowledge model:
    a confirmed row goes to `verified`, a contradicted one drops to
    `low` (the corpus-drift response for a row the source no longer
    supports), a silent one is unchanged.

    Needs an Anthropic API key for the judge calls.
    """
    from ..memory.confidence import corpus_verification_pass

    km_path = Path(args.corpus)
    if not km_path.is_file():
        return _user_error(f"corpus knowledge model not found: {km_path}")

    from .intent import _maybe_llm_client
    llm = _maybe_llm_client()
    if llm is None:
        return _user_error(
            "corpus verification needs an Anthropic API key "
            "(set CLAUDE_API_2 / CLAUDE_API / ANTHROPIC_API_KEY)."
        )

    memory = _open_memory(args)
    retriever = _make_corpus_retriever(km_path, args.source_root)
    results = corpus_verification_pass(
        memory, corpus=retriever, llm=llm,
        top_k=args.top_k, dry_run=args.dry_run,
    )
    if not results:
        print("(no high-criticality low/verified rows to verify)",
              file=sys.stderr)
        return EXIT_OK

    tally = {"confirmed": 0, "contradicted": 0, "unverifiable": 0}
    for r in results:
        tally[r.verdict] += 1
        print(f"  {r.verdict:13s} {r.row_id}  {r.reason}")
    suffix = " (dry-run — no writes)" if args.dry_run else ""
    print(
        f"\n[rlat memory] verify: {tally['confirmed']} confirmed, "
        f"{tally['contradicted']} contradicted, "
        f"{tally['unverifiable']} unverifiable{suffix}",
        file=sys.stderr,
    )
    return EXIT_OK


def cmd_memory_consolidate(args: argparse.Namespace) -> int:
    """`rlat memory consolidate` — run the session-end pass.

    Sequences distil Arrow 1 + Arrow 2 (LLM-driven; skipped when no API
    key OR `--no-llm`) → confidence raising → forget. Reports a one-line
    summary per stage so the user sees what happened.

    `--dry-run` runs every stage but skips every write; the per-stage
    counts then describe what *would* have changed.
    """
    from ..memory.session_end_pass import consolidation_pass
    from ..state import resolve_state_root

    cwd = Path(args.cwd) if args.cwd else Path.cwd()
    state_root = resolve_state_root(cwd)
    memory = _open_memory(args)
    llm = None
    if not args.no_llm:
        from .intent import _maybe_llm_client
        llm = _maybe_llm_client()
    result = consolidation_pass(
        memory, llm=llm, state_root=state_root, cwd=str(cwd),
        dry_run=args.dry_run,
    )
    promoted_verb = "would promote" if args.dry_run else "promoted"
    rejected_verb = "would reject" if args.dry_run else "rejected"
    def _arrow_line(name: str, result_obj) -> str:
        if result_obj is None:
            return f"{name}: skipped (no LLM)"
        return (
            f"{name}: {len(result_obj.promoted_row_ids)} {promoted_verb}, "
            f"{len(result_obj.rejections)} {rejected_verb}"
        )

    if args.dry_run:
        print("[rlat memory] --dry-run: pipeline ran, no writes",
              file=sys.stderr)
    print(_arrow_line("arrow1", result.arrow1))
    print(_arrow_line("arrow2", result.arrow2))
    print(_arrow_line("arrow3", result.arrow3))
    confidence_verb = "would change" if args.dry_run else "change(s)"
    forget_verb = "would drop" if args.dry_run else "dropped"
    print(f"confidence: {len(result.confidence_changes)} {confidence_verb}")
    print(f"forget: {result.forget_dropped} {forget_verb}")
    return EXIT_OK


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


# `(removed_name, v2.1 successor or guidance)`. v2.2 reclaims `consolidate`
# for the agent-harness session-end pass (distil → confidence → forget) —
# the v0.11 `consolidate` redirect to `distil` shipped through v2.0/v2.1
# and is removed here.
_DEPRECATED_SUBCOMMANDS: tuple[tuple[str, str], ...] = (
    ("primer",
     "the per-prompt UserPromptSubmit hook (no static primer in v2.1; see §17.3)"),
)


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

    p_eval = sub_mem.add_parser(
        "eval",
        help="Compute the longitudinal scorecard (useful + effortless "
             "axes from the agent-harness manifesto's benchmark).",
    )
    p_eval.add_argument(
        "--cwd", help="override workspace cwd (defaults to $PWD)",
    )
    p_eval.add_argument(
        "--sessions", type=int, default=20,
        help="number of day-windows to consider (default: 20)",
    )
    p_eval.add_argument(
        "--since",
        help="ISO-8601 start of an explicit window (UTC if naïve); "
             "must be paired with --until and overrides --sessions",
    )
    p_eval.add_argument(
        "--until",
        help="ISO-8601 end of an explicit window (UTC if naïve); "
             "must be paired with --since",
    )
    p_eval.add_argument(
        "--compare", action="store_true",
        help="aggregate first N/4 vs last N/4 windows; print pass/fail",
    )
    p_eval.add_argument(
        "--json", action="store_true",
        help="emit JSON instead of human-readable text",
    )
    p_eval.set_defaults(func=cmd_memory_eval)

    p_session_mark = sub_mem.add_parser(
        "session-mark",
        help="Record a session boundary in <state-root>/ledger/sessions.jsonl. "
             "When markers exist, `memory eval` slices by session_id instead "
             "of calendar day.",
    )
    p_session_mark.add_argument(
        "--cwd", help="override workspace cwd (defaults to $PWD)",
    )
    p_session_mark.set_defaults(func=cmd_memory_session_mark)

    p_corroborate = sub_mem.add_parser(
        "corroborate",
        help="Calibration mechanism 4 — user corroboration. Raise a row's "
             "confidence one step (low→medium→high→verified).",
    )
    p_corroborate.add_argument("row_id", help="The row id to corroborate.")
    p_corroborate.set_defaults(func=cmd_memory_corroborate)

    p_verify = sub_mem.add_parser(
        "verify",
        help="Calibration mechanism 2 — corpus verification. Check "
             "high-criticality low/verified rows against a knowledge "
             "model; confirm raises to verified, contradict drops to low.",
    )
    p_verify.add_argument(
        "corpus", help="Path to the .rlat knowledge model to check against.",
    )
    p_verify.add_argument(
        "--top-k", type=int, default=5,
        help="Corpus passages retrieved per row (default: 5).",
    )
    p_verify.add_argument(
        "--source-root", default=None,
        help="Override the knowledge model's recorded source_root "
             "(local mode).",
    )
    p_verify.add_argument(
        "--dry-run", action="store_true",
        help="Run the judge but skip the confidence write.",
    )
    p_verify.set_defaults(func=cmd_memory_verify)

    p_rollup = sub_mem.add_parser(
        "rollup",
        help="Weekly digest: this-week-vs-prior-week comparison on the "
             "manifesto's useful + effortless axes. Operational PASS/FAIL "
             "anchored on absolute time so regressions surface promptly.",
    )
    p_rollup.add_argument(
        "--weeks", type=int, default=2,
        help="number of trailing weeks; first half vs second half is the "
             "comparison (default: 2 → prior-week vs this-week; min: 2)",
    )
    p_rollup.add_argument(
        "--json", action="store_true",
        help="emit JSON {early, late, *_passed} instead of the text view",
    )
    p_rollup.add_argument(
        "--cwd", help="override workspace cwd (defaults to $PWD)",
    )
    p_rollup.set_defaults(func=cmd_memory_rollup)

    p_dedup = sub_mem.add_parser(
        "dedup",
        help="Retroactively collapse same-text-same-workspace event rows. "
             "Idempotent — safe to re-run.",
    )
    p_dedup.add_argument(
        "--dry-run", action="store_true",
        help="report what would collapse; do not touch disk",
    )
    p_dedup.set_defaults(func=cmd_memory_dedup)

    p_distil = sub_mem.add_parser(
        "distil",
        help="Run §7 LLM lesson extraction over recently-captured rows; "
             "writes prefer/avoid/factual lessons back to the store.",
    )
    p_distil.add_argument(
        "--since", default=None,
        help="Only distil capture rows created after this ISO-8601 "
             "timestamp (overrides the saved watermark).",
    )
    p_distil.add_argument(
        "--all", action="store_true",
        help="Distil every capture row, ignoring the watermark.",
    )
    p_distil.add_argument(
        "--session", default=None,
        help="Distil only the capture rows of one transcript hash.",
    )
    p_distil.add_argument(
        "--max-lessons", type=int, default=2,
        help="Cap on lessons written per invocation (default: 2).",
    )
    p_distil.add_argument(
        "--dry-run", action="store_true",
        help="Report the candidate lessons without writing.",
    )
    p_distil.set_defaults(func=cmd_memory_distil)

    p_feedback = sub_mem.add_parser(
        "feedback",
        help="Log a good/bad vote on the most recent recall injection "
             "(§9.5) to <memory-root>/feedback.log.",
    )
    p_feedback.add_argument(
        "verdict", choices=list(FEEDBACK_VERDICTS),
        help="`good` if the injection helped, `bad` if it was noise.",
    )
    p_feedback.set_defaults(func=cmd_memory_feedback)

    p_consolidate = sub_mem.add_parser(
        "consolidate",
        help="Run the per-session-end consolidation pass: distil arrows "
             "(events → patterns → learnings) → confidence raise → forget.",
    )
    p_consolidate.add_argument(
        "--cwd", help="override workspace cwd (defaults to $PWD)",
    )
    p_consolidate.add_argument(
        "--no-llm", action="store_true",
        help="skip distil arrows; run only confidence raise + forget",
    )
    p_consolidate.add_argument(
        "--dry-run", action="store_true",
        help="run the pipeline but skip every write; report what would change",
    )
    p_consolidate.set_defaults(func=cmd_memory_consolidate)

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

    # v2.0 names with no v2.1 successor — banner-only.
    for name, replacement in _DEPRECATED_SUBCOMMANDS:
        sp = sub_mem.add_parser(name, help=f"(removed) `{name}` — see banner.")
        sp.add_argument("args", nargs="*", help=argparse.SUPPRESS)
        sp.set_defaults(func=_make_deprecation_handler(name, replacement))
