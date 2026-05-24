"""`rlat intent <subcommand>` — Horizon 1 live-intent CLI.

The user-facing surface for the harness's two-question pattern:

  rlat intent add ...     — declare a live intent (`/want` slash-shape)
  rlat intent accept ID   — record user-source verdict satisfied (`/accept`)
  rlat intent reject ID   — record user-source verdict not_satisfied (`/reject`)
  rlat intent list        — show the active graph

Each subcommand resolves the workspace via `state.resolve_workspace(cwd)`
so the user never has to think about identity — the architecture's
auto-by-default rule applied at the CLI layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..state import (
    Attribution,
    CriterionCheck,
    LiveIntentStore,
    OutcomeLedger,
    OutcomeRecord,
    PendingSignalLog,
    RecallCache,
    Signal,
    attribution_from_entries,
    resolve_state_root,
)
from ..state.ledger import now_iso
from ._errors import EXIT_OK, EXIT_USER_ERROR, user_error as _user_error
from ._memory import _open_user_memory, _workspace_polarity_tag


def _state_root(args: argparse.Namespace) -> Path:
    cwd = Path(args.cwd) if args.cwd else Path.cwd()
    return resolve_state_root(cwd)


def _cmd_add(args: argparse.Namespace) -> int:
    store = LiveIntentStore(_state_root(args))
    success_criteria = []
    if args.criterion:
        for entry in args.criterion:
            if "=" not in entry:
                return _user_error(
                    f"--criterion expects 'measure=text'; got {entry!r}"
                )
            measure, text = entry.split("=", 1)
            success_criteria.append({"text": text, "measure": measure})
    intent = store.add_intent(
        level=args.level,
        text=args.text,
        stance=args.stance,
        achievability=args.achievability,
        success_criteria=success_criteria,
        constraints=list(args.constraint or []),
        created_under_intent_kind=args.kind,
        parent_ids=list(args.parent or []),
    )
    print(intent.intent_id)
    return EXIT_OK


def _cmd_list(args: argparse.Namespace) -> int:
    store = LiveIntentStore(_state_root(args))
    intents = store.list_active()
    if args.status:
        intents = [i for i in intents if i.status == args.status]
    if args.json:
        print(json.dumps([i.to_dict() for i in intents], indent=2))
        return EXIT_OK
    if not intents:
        print("(no live intents)", file=sys.stderr)
        return EXIT_OK
    for intent in intents:
        print(
            f"{intent.intent_id}  [{intent.level:<6} {intent.status:<10}] "
            f"{intent.text}"
        )
    return EXIT_OK


def _cmd_path(args: argparse.Namespace) -> int:
    """Walk parent_ids upward from `intent_id` and print the chain.

    Resolves IDs across both stores — live (steps + tasks) and durable
    (goals + directions) — so a step's task → goal → direction chain
    renders in one walk. Multi-parent rows are projected to a linear
    chain by following `parent_ids[0]` at each step; the skipped parent
    count is annotated on the line.
    """
    state_root = _state_root(args)
    live = LiveIntentStore(state_root)
    # node_id -> (level, text, source_tag, parent_ids)
    nodes: dict[str, tuple[str, str, str, list[str]]] = {}
    for i in live.list_active():
        nodes[i.intent_id] = (i.level, i.text, "live", list(i.parent_ids))
    # Durable lookup is best-effort: if no user_id can be derived (no
    # --user, no USER/USERNAME), traverse live-only rather than raise.
    try:
        memory = _open_user_memory(args)
        rows, _ = memory.read_all()
        for r in rows:
            if r.level in ("goal", "direction"):
                nodes[r.row_id] = (
                    r.level, r.text, "durable", list(r.parent_ids),
                )
    except RuntimeError:
        pass

    if args.intent_id not in nodes:
        return _user_error(
            f"intent_id {args.intent_id!r} not found in live or durable store"
        )

    chain: list[tuple[str, str, str, str, int]] = []  # leaf-first
    visited: set[str] = set()
    current: str | None = args.intent_id
    while current is not None:
        if current in visited:
            chain.append((current, "?", "(cycle detected)", "?", 0))
            break
        visited.add(current)
        node = nodes.get(current)
        if node is None:
            chain.append((current, "?", "(unknown id)", "?", 0))
            break
        level, text, source, parents = node
        chain.append((current, level, text, source, max(0, len(parents) - 1)))
        current = parents[0] if parents else None

    chain.reverse()  # root → leaf
    for depth, (node_id, level, text, source, extra) in enumerate(chain):
        indent = "  " * depth
        text_one_line = text.replace("\n", " ").strip()
        if len(text_one_line) > 80:
            text_one_line = text_one_line[:79] + "…"
        suffix = ""
        if extra == 1:
            suffix = "  (+1 more parent)"
        elif extra > 1:
            suffix = f"  (+{extra} more parents)"
        print(
            f"{indent}{node_id}  [{level:<9} {source:<7}] "
            f"{text_one_line}{suffix}"
        )
    return EXIT_OK


def _record_user_signal(
    args: argparse.Namespace,
    *,
    verdict: str,
    new_status: str,
) -> int:
    state_root = _state_root(args)
    store = LiveIntentStore(state_root)
    log = PendingSignalLog(state_root)
    ledger = OutcomeLedger(state_root)
    cache = RecallCache(state_root)
    intents = {i.intent_id: i for i in store.list_active()}
    intent = intents.get(args.intent_id)
    if intent is None:
        return _user_error(f"intent_id {args.intent_id!r} not in live graph")
    intent_started_at = intent.created_at
    try:
        store.set_status(args.intent_id, new_status, reason=args.reason or "")
    except KeyError as exc:
        return _user_error(str(exc))
    timestamp = now_iso()
    log.append(
        source="user",
        tool_name="cli:intent",
        tool_payload={"reason": args.reason or ""},
        value={"verdict": verdict},
        intent_id=args.intent_id,
    )
    # Read recall cache entries from the intent's lifetime + map them to
    # tier weights via state.attribution. This is the recall→action
    # attribution chain — the wire that lets confidence raising actually
    # credit specific rows for outcomes (architecture §"When attribution
    # is computed").
    #
    # Prefer entries stamped with this exact intent_id (Horizon 4
    # outcome-attributed retrieval); fall back to the timestamp window
    # only when no stamped entries exist (older recalls pre-Horizon-4
    # or recalls fired with no live intent active).
    recall_entries = cache.read_for_intent(
        args.intent_id, since_iso=intent_started_at,
    )
    if not recall_entries:
        recall_entries = cache.read_since(intent_started_at)
    attribution = attribution_from_entries(recall_entries)
    ledger.write(OutcomeRecord(
        intent_id=args.intent_id,
        intent_level=intent.level,
        criterion_checks=[CriterionCheck(
            criterion_text=f"user {verdict}: {args.reason or 'no reason'}",
            measure="user_confirms",
            verdict=verdict,
            signals_seen=[Signal(
                source="user",
                value={"verdict": verdict},
                timestamp=timestamp,
            )],
            verdict_confidence="high",
        )],
        roll_up_verdict=verdict,
        attribution=attribution,
        resolved_at=timestamp,
        intent_kind=intent.created_under_intent_kind,
    ))
    print(f"{verdict}: {args.intent_id} ({len(attribution)} attributed)")
    return EXIT_OK


def _cmd_accept(args: argparse.Namespace) -> int:
    return _record_user_signal(args, verdict="satisfied", new_status="satisfied")


def _cmd_reject(args: argparse.Namespace) -> int:
    return _record_user_signal(args, verdict="not_satisfied", new_status="abandoned")


def _cmd_activate(args: argparse.Namespace) -> int:
    """Flip a `proposed` intent (e.g. from `capture-plan`) to `active`.

    The manual half of the proposed→active transition — Claude Code
    publishes no plan-approval hook to automate it. A lifecycle
    transition, not an outcome: writes no signal, no ledger record.
    """
    store = LiveIntentStore(_state_root(args))
    try:
        intent = store.set_status(
            args.intent_id, "active", reason=args.reason or "",
        )
    except KeyError as exc:
        return _user_error(str(exc))
    print(f"active: {intent.intent_id}")
    return EXIT_OK


def _maybe_llm_client():
    """Resolve the Anthropic client, or None when no API key is set.

    Operations that LLM-call route through here so the missing-key path
    falls cleanly back to the cheap-path stub (what-next) or a structured
    refusal (decompose, distil arrows). API key whitespace is `.strip()`-ed
    because the env-var slot is known to ship with trailing newlines that
    crash `httpcore.LocalProtocolError` otherwise.
    """
    from ..optimise.synth_queries import default_client, discover_api_key
    key = discover_api_key()
    if not key:
        return None
    return default_client(key.strip())


def _cmd_what_next(args: argparse.Namespace) -> int:
    """Synthesise the next-move recommendation. LLM-optional — without a
    client (or with `--no-llm`) the cheap-path stub renders the top
    candidate verbatim."""
    from ..memory.what_next import pick_candidates, synthesise_recommendation

    state_root = _state_root(args)
    store = LiveIntentStore(state_root)
    candidates = pick_candidates(store.list_active(), top_k=args.top_k)
    llm = None if args.no_llm else _maybe_llm_client()
    print(synthesise_recommendation(candidates, llm=llm))
    return EXIT_OK


def _cmd_declare_durable(args: argparse.Namespace) -> int:
    """Write a durable goal or direction to the per-user memory store.

    The intent-interrogation skill orchestrates the conversation that
    produces the inputs; this CLI is the write seam. `confidence=verified`
    + `origin=manual` per architecture §"The intent-interrogation skill".
    """
    memory = _open_user_memory(args)
    success_criteria = []
    if args.criterion:
        for entry in args.criterion:
            if "=" not in entry:
                return _user_error(
                    f"--criterion expects 'measure=text'; got {entry!r}"
                )
            measure, text = entry.split("=", 1)
            success_criteria.append({"text": text, "measure": measure})
    polarity = ["factual", _workspace_polarity_tag(args)]
    row_id = memory.add_row(
        text=args.text,
        polarity=polarity,
        transcript_hash="manual",
        level=args.level,
        confidence="verified",
        origin="manual",
        stance=args.stance,
        achievability=args.achievability,
        status="active",
        success_criteria=success_criteria,
        constraints=list(args.constraint or []),
        created_under_intent_kind="none",
    )
    print(row_id)
    return EXIT_OK


def _cmd_list_durable(args: argparse.Namespace) -> int:
    """List durable goal and direction rows in the per-user memory store."""
    memory = _open_user_memory(args)
    rows, _ = memory.read_all()
    durable = [r for r in rows if r.level in ("goal", "direction")]
    if args.level:
        durable = [r for r in durable if r.level == args.level]
    if args.json:
        print(json.dumps([r.to_jsonl_dict() for r in durable], indent=2))
        return EXIT_OK
    if not durable:
        print("(no durable goals or directions)", file=sys.stderr)
        return EXIT_OK
    for row in durable:
        text = row.text.replace("\n", " ").strip()
        if len(text) > 80:
            text = text[:79] + "…"
        print(
            f"{row.row_id}  [{row.level:<10} {row.confidence:<8}] {text}"
        )
    return EXIT_OK


def _cmd_decompose(args: argparse.Namespace) -> int:
    """Decompose a task intent into step children via LLM."""
    from ..memory.decompose import DecompositionResult, decompose

    state_root = _state_root(args)
    store = LiveIntentStore(state_root)
    intents = {i.intent_id: i for i in store.list_active()}
    parent = intents.get(args.intent_id)
    if parent is None:
        return _user_error(f"intent_id {args.intent_id!r} not in live graph")
    llm = _maybe_llm_client()
    if llm is None:
        result = DecompositionResult(
            parent_intent_id=parent.intent_id,
            refused=True,
            rejection_reason=(
                "no Anthropic API key — set CLAUDE_API_2 (or CLAUDE_API / "
                "ANTHROPIC_API_KEY) and re-run"
            ),
        )
    else:
        result = decompose(parent, llm=llm, store=store)
    if result.refused:
        print(f"refused: {result.rejection_reason}", file=sys.stderr)
        return EXIT_USER_ERROR
    print(
        f"decomposed {parent.intent_id} → "
        f"{len(result.child_intent_ids)} step(s)",
    )
    return EXIT_OK


_PLAN_TITLE_MAX_CHARS = 200
_PLAN_STEP_MAX_CHARS = 200
_PLAN_MAX_STEPS = 10
_PLAN_NUMBERED_RE = None  # lazy-compile in _parse_plan_markdown
_PLAN_BULLET_RE = None


def _parse_plan_markdown(plan: str) -> tuple[str, list[str]]:
    """Extract `(task_title, [step_text, ...])` from an ExitPlanMode plan.

    Title: first non-empty line with markdown markers stripped, capped at
    `_PLAN_TITLE_MAX_CHARS`.

    Steps: numbered list items (`1. foo`) at top level. Falls back to
    top-level bullets (`- foo` / `* foo`) when the plan uses an unordered
    structure. Capped at `_PLAN_MAX_STEPS` × `_PLAN_STEP_MAX_CHARS`.
    """
    import re
    global _PLAN_NUMBERED_RE, _PLAN_BULLET_RE
    if _PLAN_NUMBERED_RE is None:
        _PLAN_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+(.+)$")
        _PLAN_BULLET_RE = re.compile(r"^\s*[-*]\s+(.+)$")

    lines = plan.splitlines()

    # Title: first non-empty line, strip leading `#`s and surrounding `**`.
    title = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        title = stripped.lstrip("#").strip()
        title = title.replace("**", "").strip()
        break
    if not title:
        title = "(plan with no title)"
    title = title[:_PLAN_TITLE_MAX_CHARS]

    numbered: list[str] = []
    bullets: list[str] = []
    for line in lines:
        m = _PLAN_NUMBERED_RE.match(line)
        if m:
            numbered.append(m.group(1).strip()[:_PLAN_STEP_MAX_CHARS])
            continue
        m = _PLAN_BULLET_RE.match(line)
        if m:
            bullets.append(m.group(1).strip()[:_PLAN_STEP_MAX_CHARS])
    steps = numbered if numbered else bullets
    return title, steps[:_PLAN_MAX_STEPS]


def _cmd_capture_plan(args: argparse.Namespace) -> int:
    """Capture ExitPlanMode plan-mode output as `proposed` task + steps.

    Wires the planning-mode-listening hook (PreToolUse matcher
    `ExitPlanMode`): Claude Code's hook payload arrives on stdin; this
    subcommand extracts `tool_input.plan` (markdown), parses a task title
    plus numbered/bulleted items, and writes them into the live store
    with `status="proposed"`.

    Captured intents enter `proposed`. The proposed→active flip cannot
    be automatic — Claude Code publishes no plan-approval hook event —
    so the operator runs `rlat intent activate <intent_id>` once the
    plan is adopted.
    """
    try:
        payload = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, OSError):
        return EXIT_OK  # fail-open — never block a tool call
    plan = (payload.get("tool_input") or {}).get("plan", "")
    if not isinstance(plan, str) or not plan.strip():
        return EXIT_OK  # silent no-op on empty plan

    cwd = payload.get("cwd") or (str(args.cwd) if args.cwd else None)
    if cwd:
        args.cwd = cwd  # let _state_root pick up the hook-supplied cwd
    store = LiveIntentStore(_state_root(args))

    title, steps = _parse_plan_markdown(plan)
    task = store.add_intent(
        level="task",
        text=title,
        stance="do",
        achievability="medium",
        success_criteria=[],
        constraints=[],
        status="proposed",
    )
    for step_text in steps:
        store.add_intent(
            level="step",
            text=step_text,
            stance="do",
            achievability="medium",
            success_criteria=[],
            constraints=[],
            parent_ids=[task.intent_id],
            status="proposed",
        )
    print(task.intent_id)
    return EXIT_OK


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "intent",
        help="declare and resolve live intent",
        description="Manage the live intent graph at "
                    "<workspace>/.rlat-state/intent/.",
    )
    p.add_argument("--cwd", help="override workspace cwd (defaults to $PWD)")
    intent_sub = p.add_subparsers(dest="intent_command", required=True)

    add_p = intent_sub.add_parser("add", help="declare a new live intent")
    add_p.add_argument("text", help="intent text")
    add_p.add_argument(
        "--level", default="task", choices=["step", "task"],
        help="intent level (default: task)",
    )
    add_p.add_argument(
        "--stance", default="do", choices=["do", "avoid", "know"],
        help="agent stance toward this intent (default: do)",
    )
    add_p.add_argument(
        "--achievability", default="medium",
        choices=["low", "medium", "high"],
        help="estimated achievability (default: medium)",
    )
    add_p.add_argument(
        "--kind", default="none",
        choices=["debug", "design", "implement", "review", "explain",
                 "refactor", "none"],
        help="intent kind for recall biasing (default: none)",
    )
    add_p.add_argument(
        "--criterion", action="append", metavar="MEASURE=TEXT",
        help="success criterion; repeat for multiple",
    )
    add_p.add_argument(
        "--constraint", action="append", metavar="TEXT",
        help="constraint string; repeat for multiple",
    )
    add_p.add_argument(
        "--parent", action="append", metavar="INTENT_ID",
        help="parent intent_id; repeat for multiple",
    )
    add_p.set_defaults(func=_cmd_add)

    list_p = intent_sub.add_parser("list", help="list live intents")
    list_p.add_argument("--status", help="filter by status")
    list_p.add_argument(
        "--json", action="store_true",
        help="emit JSON list of intent dicts (empty list when no matches)",
    )
    list_p.set_defaults(func=_cmd_list)

    path_p = intent_sub.add_parser(
        "path",
        help="print the parent chain (root → leaf) for an intent_id",
    )
    path_p.add_argument(
        "intent_id",
        help="live intent_id or durable row_id; cross-store traversal is "
             "supported (e.g. step → task → goal → direction)",
    )
    path_p.add_argument(
        "--memory-root", default=None,
        help="override per-user memory root (default: ~/.rlat/memory/)",
    )
    path_p.add_argument(
        "--user", default=None,
        help="override user_id (default: $RLAT_MEMORY_USER / $USER / $USERNAME)",
    )
    path_p.set_defaults(func=_cmd_path)

    accept_p = intent_sub.add_parser(
        "accept", help="record user-source satisfied verdict"
    )
    accept_p.add_argument("intent_id")
    accept_p.add_argument("--reason", default="")
    accept_p.set_defaults(func=_cmd_accept)

    reject_p = intent_sub.add_parser(
        "reject", help="record user-source not_satisfied verdict"
    )
    reject_p.add_argument("intent_id")
    reject_p.add_argument("--reason", default="")
    reject_p.set_defaults(func=_cmd_reject)

    activate_p = intent_sub.add_parser(
        "activate",
        help="flip a proposed intent (e.g. from capture-plan) to active",
    )
    activate_p.add_argument("intent_id")
    activate_p.add_argument("--reason", default="")
    activate_p.set_defaults(func=_cmd_activate)

    what_next_p = intent_sub.add_parser(
        "what-next",
        help="recommend the next move from the live intent graph",
    )
    what_next_p.add_argument(
        "--top-k", type=int, default=5,
        help="candidate pool size for the LLM (default: 5)",
    )
    what_next_p.add_argument(
        "--no-llm", action="store_true",
        help="skip LLM synthesis, render the cheap-path stub instead",
    )
    what_next_p.set_defaults(func=_cmd_what_next)

    decompose_p = intent_sub.add_parser(
        "decompose",
        help="decompose a task intent into step children via LLM",
    )
    decompose_p.add_argument("intent_id")
    decompose_p.set_defaults(func=_cmd_decompose)

    declare_p = intent_sub.add_parser(
        "declare-durable",
        help="declare a goal or direction (durable, written to per-user store)",
    )
    declare_p.add_argument("text", help="intent text")
    declare_p.add_argument(
        "--level", required=True, choices=["goal", "direction"],
        help="durable level (live store handles step/task; this writes "
             "goal or direction to the per-user memory store)",
    )
    declare_p.add_argument(
        "--stance", default="do", choices=["do", "avoid", "know"],
    )
    declare_p.add_argument(
        "--achievability", default="medium",
        choices=["low", "medium", "high"],
    )
    declare_p.add_argument(
        "--criterion", action="append", metavar="MEASURE=TEXT",
        help="success criterion; repeat for multiple",
    )
    declare_p.add_argument(
        "--constraint", action="append", metavar="TEXT",
        help="constraint string; repeat for multiple",
    )
    declare_p.add_argument("--memory-root", default=None)
    declare_p.add_argument("--user", default=None)
    declare_p.set_defaults(func=_cmd_declare_durable)

    durable_p = intent_sub.add_parser(
        "durable",
        help="list durable goals and directions from the per-user store",
    )
    durable_p.add_argument(
        "--level", default=None, choices=["goal", "direction"],
        help="filter by level",
    )
    durable_p.add_argument("--memory-root", default=None)
    durable_p.add_argument("--user", default=None)
    durable_p.add_argument(
        "--json", action="store_true",
        help="emit JSON list of row dicts (empty list when no matches)",
    )
    durable_p.set_defaults(func=_cmd_list_durable)

    capture_p = intent_sub.add_parser(
        "capture-plan",
        help="capture ExitPlanMode plan-mode output as proposed task + steps "
             "(reads PreToolUse hook payload from stdin)",
    )
    capture_p.set_defaults(func=_cmd_capture_plan)
