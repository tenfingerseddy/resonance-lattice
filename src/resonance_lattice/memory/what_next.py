"""What-next — the user-facing recommendation operation.

Architecture §"What-next — the user-facing recommendation operation":

> It's not a special case of recall — it's a synthesis over the intent
> graph + memory + outcome ledger that returns a single concrete
> suggestion.

Pipeline (this module):

  1. Filter — keep intents with status: active or status: blocked
  2. Score each candidate by:
     - Status (in_progress > ready > blocked)
     - Level coverage (prefer the deepest unsatisfied node)
     - Criticality of unsatisfied parents
     - Achievability of the candidate
     - Recency of activity on the parent
  3. Surface ready siblings of the active node + ready descendants
     of unblocked goals
  4. Synthesise — LLM produces 1–2 concrete suggestions in human terms

The scoring half (steps 1-3) is a pure function — testable without an
LLM key. The synthesis half (step 4) is a separable callable; without
an LLM the CLI falls back to printing the top candidate's text directly.

Output shape (architecture):
  > "You're partway through goal X. Task Y is ready and is the natural
  > next move — it unblocks Z. Want me to start, or is there something
  > else?"

Never a graph. Never enums. The user always sees a direct answer.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

from ..state.intent import Intent
from ..state.claim_outcome import ClaimOutcomeRecord
from ._common import parse_iso_utc
from ._llm import LLMClient

# Bounded by architecture §"LLM context is intent-path-scoped". At most
# ~20 nodes from the graph reach the LLM; what-next surfaces a tighter
# top-K so synthesis stays focused.
DEFAULT_TOP_K_CANDIDATES = 5
_MAX_LLM_CONTEXT_NODES = 20

# Status priority — architecture's "in_progress > ready > blocked" mapped
# onto Intent statuses. Live intents track {active, blocked, satisfied,
# abandoned, superseded}; we infer in_progress vs ready from updated_at
# recency on top of `active`.
_STATUS_PRIORITY = {"in_progress": 3, "ready": 2, "blocked": 1}
_LEVEL_DEPTH = {"step": 4, "task": 3, "goal": 2, "direction": 1}
_ACHIEVABILITY_PRIORITY = {"high": 3, "medium": 2, "low": 1}
_CRITICALITY_WEIGHT = {"low": 0, "normal": 1, "high": 2, "severe": 3}

# Recency window for "in_progress" classification — an active intent
# touched in the last 24h is treated as currently being worked on.
_IN_PROGRESS_WINDOW_HOURS = 24


@dataclass(frozen=True)
class Candidate:
    """One scored what-next candidate."""

    intent: Intent
    derived_status: str  # in_progress | ready | blocked
    score: tuple[int, int, int, str]  # for stable sort

    @property
    def text(self) -> str:
        return self.intent.text


def _hours_since(ts: str, now: _dt.datetime) -> float:
    return max(0.0, (now - parse_iso_utc(ts)).total_seconds() / 3600.0)


def _derive_status(
    intent: Intent,
    blocking_active_ids: set[str],
    now: _dt.datetime,
) -> str:
    """Map (status, blockers, recency) → in_progress | ready | blocked.

    `blocking_active_ids` is the set of intent_ids that some other active
    intent has named in its `blocks` list — those are still blocking
    progress and the candidate isn't ready to fire.
    """
    if intent.status == "blocked" or intent.intent_id in blocking_active_ids:
        return "blocked"
    if _hours_since(intent.updated_at, now) <= _IN_PROGRESS_WINDOW_HOURS:
        return "in_progress"
    return "ready"


def _candidate_score(
    intent: Intent,
    derived_status: str,
    now: _dt.datetime,
) -> tuple[int, int, int, str]:
    """Lexicographic-sort key — higher is better.

    Tuple shape: (status_priority, level_depth, achievability+criticality,
    -hours_since_update_as_string). Recency uses a string suffix so the
    tuple stays comparable; we negate by computing `(2**31 - hours)` and
    formatting fixed-width.
    """
    achievability = _ACHIEVABILITY_PRIORITY.get(intent.achievability or "medium", 2)
    criticality = _CRITICALITY_WEIGHT.get("normal", 1)  # default; live
    # intents don't carry criticality directly — Horizon 2 will lift it
    # from the parent durable goal. For now, default + achievability is
    # the dominant per-candidate signal.
    combined = achievability * 4 + criticality
    # Recency: more recent → higher key; fixed-width string for stable sort.
    hours_since = _hours_since(intent.updated_at, now)
    recency_key = f"{(10**9 - int(hours_since * 60)):011d}"
    return (
        _STATUS_PRIORITY[derived_status],
        _LEVEL_DEPTH.get(intent.level, 0),
        combined,
        recency_key,
    )


def pick_candidates(
    live_intents: list[Intent],
    *,
    recent_outcomes: list[ClaimOutcomeRecord] | None = None,
    top_k: int = DEFAULT_TOP_K_CANDIDATES,
    now: _dt.datetime | None = None,
) -> list[Candidate]:
    """Pure scoring — returns top-K active candidates ordered for synthesis.

    `recent_outcomes` is accepted for API symmetry with the synthesis
    layer; v1 doesn't fold it into the score, since the closed loop
    already routes outcomes back into recall via confidence raising.
    The architecture's "criticality of unsatisfied parents" signal will
    plug in here once durable-intent reads are wired.
    """
    if now is None:
        now = _dt.datetime.now(_dt.timezone.utc)
    active = [
        i for i in live_intents
        if i.status in ("active", "blocked")
    ]
    if not active:
        return []
    # Compute the set of intents another active intent is blocking on.
    blocking_active_ids: set[str] = set()
    for intent in active:
        if intent.status != "active":
            continue
        for blocked in intent.blocks:
            blocking_active_ids.add(blocked)
    candidates = [
        Candidate(
            intent=intent,
            derived_status=_derive_status(intent, blocking_active_ids, now),
            score=_candidate_score(
                intent,
                _derive_status(intent, blocking_active_ids, now),
                now,
            ),
        )
        for intent in active
    ]
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]


# ---------------------------------------------------------------------------
# LLM synthesis
# ---------------------------------------------------------------------------


_PROMPT = """You are recommending the next concrete move for an engineer mid-session.

You receive a small ranked list of live intents — tasks and steps the user
has explicitly declared, plus their derived status (in_progress / ready /
blocked). Output ONE short natural-language recommendation in plain prose:

  - Cite the most likely "next move" by name.
  - Explain in ≤2 sentences why it's next (what's in progress, what it
    unblocks, what's at risk if it slips).
  - End with a single open offer: "Want me to start, or is there something
    else?"
  - Never include a graph, an enum, a status table, or markdown headers.

If only one candidate is ready, just recommend it. If two are equally
strong, pick one and mention the second as an alternative."""


def _build_messages(candidates: list[Candidate]) -> list[dict]:
    """Build the LLM messages — bounded to ≤20 nodes per architecture."""
    body_lines: list[str] = []
    for c in candidates[:_MAX_LLM_CONTEXT_NODES]:
        body_lines.append(
            f"  - [{c.derived_status} / {c.intent.level} / "
            f"achievability={c.intent.achievability}] {c.intent.text}"
        )
    user = (
        "Active intents (highest-ranked first):\n\n"
        + "\n".join(body_lines)
        + "\n\nRecommend the next move."
    )
    return [{"role": "user", "content": user}]


def synthesise_recommendation(
    candidates: list[Candidate],
    *,
    llm: LLMClient | None = None,
) -> str:
    """Render a one-shot recommendation.

    With `llm`: synthesises a natural-language line via the LLM call.
    Without `llm`: falls back to the top candidate's text plus a stub
    framing — useful in environments without an API key, and as the
    cheap path for the harness suite.
    """
    if not candidates:
        return ("No active intents. Declare what you're working on with "
                "/want, or use /workspace status to confirm where you are.")
    top = candidates[0]
    if llm is None:
        return (
            f"Top candidate: {top.intent.text} "
            f"({top.derived_status}, {top.intent.level}, "
            f"achievability={top.intent.achievability}). "
            "Want me to start, or is there something else?"
        )
    try:
        response = llm(_PROMPT, _build_messages(candidates), 256)
    except Exception:
        # Fail-open to the cheap path — never block the user surface.
        return synthesise_recommendation(candidates, llm=None)
    text = response.text.strip()
    return text or synthesise_recommendation(candidates, llm=None)
