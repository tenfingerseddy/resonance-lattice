"""Trajectory primer — the bounded intent-path summary the SessionStart hook
loads via `additionalContext`.

Architecture §"LLM context is intent-path-scoped, not graph-scoped":

> When the LLM needs intent context, it gets:
> - The currently active intent path (this step → its parent task →
>   grandparent goal → great-grandparent direction) — at most 4 nodes
> - Sibling intents at the active level (other tasks under the same goal) —
>   typically 3–8 nodes
> - Recently-resolved intents in the same scope (last week or so)

> That's it. The full graph never goes to the LLM. Even a user with 10
> active goals and hundreds of historical tasks gets ~20 nodes of context
> per turn. **Bounded by design, not by hope.**

This module reads live + durable intent and emits a markdown block for the
hook layer. It never writes — read-only over `LiveIntentStore` plus a
short read of durable goals/directions from the memory store (when the
hook layer wants those).
"""

from __future__ import annotations

from pathlib import Path

from .intent import LiveIntent, LiveIntentStore

# Hard caps from the architecture's bounded-by-design rule.
_MAX_ACTIVE_PATH_NODES = 4
_MAX_SIBLINGS = 8
_MAX_RECENT_RESOLVED = 5


def _active_intents(store: LiveIntentStore) -> list[LiveIntent]:
    """Live rows in 'active' or 'blocked' status. Resolved/abandoned drop."""
    return [
        i for i in store.list_active()
        if i.status in ("active", "blocked")
    ]


def render_trajectory_primer(state_root: Path | str) -> str:
    """Build the markdown trajectory block for SessionStart.

    Returns an empty string when there's no active intent — the hook then
    skips the additionalContext injection (silent SessionStart). Otherwise
    a concise block:

        ## Active intents
        - **task** [in_progress] ship the harness
          - step [active] run the harness

    Bounded: at most 4 path nodes, 8 siblings, 5 recent resolutions per the
    architecture's intent-path-scoping rule.
    """
    intent_dir = Path(state_root)
    if not intent_dir.exists():
        return ""
    store = LiveIntentStore(intent_dir)
    active = _active_intents(store)
    if not active:
        return ""

    # Pick the *leaf of the deepest active chain* — that's the agent's
    # most fine-grained current intent ("this step → its parent task →
    # grandparent goal" reads from leaf to root). Tie-break by latest
    # updated_at so two equally-deep leaves pick the freshly-touched one.
    # Using parent depth (not just timestamp) makes the choice deterministic
    # under ms-tied timestamps that `utcnow_iso` produces in tight loops.
    by_id = {i.intent_id: i for i in store.list_active()}

    def _depth(intent: LiveIntent) -> int:
        depth = 0
        cursor: LiveIntent | None = intent
        seen: set[str] = set()
        while cursor is not None and cursor.parent_ids:
            if cursor.intent_id in seen:
                break
            seen.add(cursor.intent_id)
            cursor = by_id.get(cursor.parent_ids[0])
            depth += 1
        return depth

    most_recent = max(active, key=lambda i: (_depth(i), i.updated_at))

    path: list[LiveIntent] = []
    cursor: LiveIntent | None = most_recent
    seen: set[str] = set()
    while cursor is not None and len(path) < _MAX_ACTIVE_PATH_NODES:
        if cursor.intent_id in seen:
            break  # cycle defence — should never happen but cheap to guard
        seen.add(cursor.intent_id)
        path.append(cursor)
        if not cursor.parent_ids:
            break
        cursor = by_id.get(cursor.parent_ids[0])
    path.reverse()  # root → leaf

    # Siblings of the most-recent active row at its level.
    if most_recent.parent_ids:
        parent_id = most_recent.parent_ids[0]
        siblings = [
            i for i in active
            if i.intent_id != most_recent.intent_id
            and parent_id in i.parent_ids
        ][:_MAX_SIBLINGS]
    else:
        siblings = []

    # Recent resolutions for context (transitions log, last N).
    transitions = store.read_transitions()
    recent_resolved = [
        t for t in reversed(transitions)
        if t.get("to") in ("satisfied", "abandoned")
    ][:_MAX_RECENT_RESOLVED]

    lines: list[str] = ["## Active intents"]
    for depth, intent in enumerate(path):
        indent = "  " * depth
        lines.append(
            f"{indent}- **{intent.level}** [{intent.status}] {intent.text}"
        )
    if siblings:
        lines.append("")
        lines.append("### Ready siblings")
        for sib in siblings:
            lines.append(f"- [{sib.status}] {sib.text}")
    if recent_resolved:
        lines.append("")
        lines.append("### Recently resolved")
        for entry in recent_resolved:
            iid = entry.get("intent_id", "?")
            verdict = entry.get("to", "?")
            lines.append(f"- {iid}: → {verdict}")
    return "\n".join(lines) + "\n"
