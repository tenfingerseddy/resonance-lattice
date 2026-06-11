"""Decomposition — LLM-driven `task → steps` expansion.

Architecture §"Decomposition guidance for the LLM":

  Sizing.  A task is one session of work; if it's many sessions, it's
           probably a goal. A step is one tool call; if it needs multiple,
           it's probably a task.
  Boundary. If the criterion is "a person agrees," it's not a step —
           steps are mechanical.
  Fan-out. A direction has 2–5 goals typically; a goal has 3–8 tasks; a
           task has 2–10 steps. Outside these ranges, decomposition is
           probably wrong-shaped.

This module ships the live-store cut: **task → steps**. The wider vision
(direction → goal, goal → tasks) crosses into durable intent stored in
the memory store, which the user-interrogation skill path will populate
later. v1 demonstrates the operation works end-to-end on the level pair
the live store already supports.

Pipeline:

  1. Read parent task from live store
  2. Call LLM with parent text + sizing prompt → JSON list of step strings
  3. Post-validate — fan-out range (2–10) + step text shape
  4. Write each step as a child intent with parent_ids=[task_id]

Idempotency: re-running `decompose(task_id)` after children already exist
appends MORE children. Caller decides whether to clear existing children
first; the v1 CLI surfaces a `--replace` flag to make the destructive
intent explicit.

LLM seam: `(system, messages, max_tokens) -> LLMResponse` (the shared
`memory/_llm.py` type) so the harness suite injects a stub without
touching the network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..state import Intent, LiveIntentStore
from ._common import parse_llm_json
from ._llm import LLMClient

# Architecture's fan-out range for task → step decomposition.
DEFAULT_MIN_STEPS = 2
DEFAULT_MAX_STEPS = 10

# Step text length cap — keeps individual steps single-action-shaped.
DEFAULT_MAX_STEP_WORDS = 40


@dataclass
class DecompositionResult:
    """One decomposition pass outcome."""

    parent_intent_id: str
    child_intent_ids: list[str] = field(default_factory=list)
    refused: bool = False
    rejection_reason: str = ""


_PROMPT = """You decompose a task into the steps an engineer would run to satisfy it.

A STEP is ONE atomic action — a single tool call, a single edit, a single
command. If a step needs multiple tool calls or human deliberation, it's a
task, not a step. Don't generate "review" or "decide" steps; those aren't
mechanical.

OUTPUT FORMAT — read carefully:
  Output ONLY a JSON object. No prose, no markdown, no code fences, no
  explanation. The first character of your response is `{`. The last
  character is `}`. Nothing else.

  Exactly one of these two shapes:
    {"steps": ["<step text>", "<step text>", ...]}
    {"refuse": true, "reason": "<short reason>"}

Rules for `steps`:
  - 2 to 10 entries (otherwise refuse).
  - Each entry ≤40 words, single action.
  - Order matters — earlier steps land before later ones.
  - No "(optional)" or "(if needed)" — real steps that earn their place.
  - No success criteria, no acceptance commentary — just the action.

Refuse when the task is too vague to decompose mechanically, when it's clearly
a goal-sized chunk needing further breakdown first, or when it's already a
single-action step."""


def _build_messages(parent: Intent) -> list[dict]:
    user = (
        f"Task: {parent.text}\n\n"
        f"Constraints: {parent.constraints or '(none)'}\n"
        f"Success criteria (do not duplicate as steps):\n"
        + "\n".join(
            f"  - [{c['measure']}] {c['text']}"
            for c in (parent.success_criteria or [])
        )
        + "\n\nDecompose into 2–10 mechanical steps, or refuse."
    )
    return [{"role": "user", "content": user}]


def _validate_steps(
    steps: list[str],
    *,
    min_steps: int,
    max_steps: int,
    max_step_words: int,
) -> str | None:
    """Return None if valid, else a rejection reason."""
    if not isinstance(steps, list):
        return "steps must be a JSON array"
    if not (min_steps <= len(steps) <= max_steps):
        return (
            f"fan-out {len(steps)} outside [{min_steps}, {max_steps}]"
        )
    for i, step in enumerate(steps):
        if not isinstance(step, str) or not step.strip():
            return f"step {i} not a non-empty string"
        word_count = len(step.split())
        if word_count > max_step_words:
            return f"step {i} too long ({word_count} > {max_step_words} words)"
    return None


def decompose(
    parent: Intent,
    *,
    llm: LLMClient,
    store: LiveIntentStore,
    intent_kind: str | None = None,
    min_steps: int = DEFAULT_MIN_STEPS,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_step_words: int = DEFAULT_MAX_STEP_WORDS,
) -> DecompositionResult:
    """Decompose `parent` (a task) into mechanical steps.

    Returns a `DecompositionResult` with the new step intent_ids OR a
    rejection reason. Never raises on LLM error — the operation is
    user-facing and a transient API failure shouldn't blow up the CLI;
    callers see `refused=True` with the reason.
    """
    if parent.level != "task":
        return DecompositionResult(
            parent_intent_id=parent.intent_id,
            refused=True,
            rejection_reason=(
                f"v1 decomposes task→step only; parent.level={parent.level!r}"
            ),
        )
    try:
        response = llm(_PROMPT, _build_messages(parent), 512)
    except Exception as exc:
        return DecompositionResult(
            parent_intent_id=parent.intent_id,
            refused=True,
            rejection_reason=f"llm error: {type(exc).__name__}: {exc}",
        )
    try:
        payload = parse_llm_json(response.text)
    except json.JSONDecodeError as exc:
        return DecompositionResult(
            parent_intent_id=parent.intent_id,
            refused=True,
            rejection_reason=f"non-JSON response: {exc}",
        )
    if not isinstance(payload, dict):
        return DecompositionResult(
            parent_intent_id=parent.intent_id,
            refused=True,
            rejection_reason="response not a JSON object",
        )
    if payload.get("refuse"):
        return DecompositionResult(
            parent_intent_id=parent.intent_id,
            refused=True,
            rejection_reason=f"refused: {payload.get('reason', 'no reason')}",
        )
    rejection = _validate_steps(
        payload.get("steps", []),
        min_steps=min_steps,
        max_steps=max_steps,
        max_step_words=max_step_words,
    )
    if rejection:
        return DecompositionResult(
            parent_intent_id=parent.intent_id,
            refused=True,
            rejection_reason=rejection,
        )
    child_ids: list[str] = []
    kind = intent_kind or parent.created_under_intent_kind
    for step_text in payload["steps"]:
        child = store.add_intent(
            level="step",
            text=step_text.strip(),
            stance="do",
            achievability="medium",
            success_criteria=[],
            constraints=[],
            created_under_intent_kind=kind,
            parent_ids=[parent.intent_id],
        )
        child_ids.append(child.intent_id)
    store.record_decomposition(
        parent.intent_id, child_ids,
        rationale=f"task→step decomposition ({len(child_ids)} steps)",
    )
    return DecompositionResult(
        parent_intent_id=parent.intent_id,
        child_intent_ids=child_ids,
    )
