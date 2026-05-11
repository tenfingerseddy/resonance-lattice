"""Cheap-path intent classifier for UserPromptSubmit.

Architecture §"Intent extraction inside UserPromptSubmit":

> Cheap path (default for v1) — fast classifier (regex / small model) for
> the intent's level + stance; deeper extraction deferred to async
> background. Stays well within the 200ms hot-path budget.

This module ships the regex variant. Maps a prompt to one of the seven
`IntentKind` values (debug / design / implement / review / explain /
refactor / none); the recall daemon then conditions the manifesto re-rank
on this kind. Wrong classifications are recoverable — the user surfaces
them via session-start summaries or `/correct` (architecture §"The
three-line defence").

No allocations beyond the prompt-text scan; no model load; no network.
Sub-millisecond on a 1KB prompt.
"""

from __future__ import annotations

import re

from .store import IntentKind

# Each pattern fires on word-boundary matches in the prompt. Order matters
# only for tie-breaks: when two kinds match, the one with more matches
# wins; on a perfect tie the earlier list entry wins. The patterns are
# intentionally short — high precision in common cases, "none" fallback
# when the prompt is ambiguous (which the manifesto handles safely with
# the "neutral" valence/level profile).
_PATTERNS: list[tuple[IntentKind, re.Pattern[str]]] = [
    ("debug", re.compile(
        r"\b(?:fix|bug|broken|crash(?:ing|ed)?|stack\s*trace|fail(?:ing|ed|ure)?|"
        r"error(?:s)?|investigate|diagnose|why\s+(?:is|does|isn'?t|doesn'?t)|"
        r"doesn'?t\s+work|not\s+working|reproduce|stale)\b",
        re.IGNORECASE,
    )),
    ("design", re.compile(
        r"\b(?:design|architecture|architect|approach|plan|propose|"
        r"should\s+(?:we|i|it)|how\s+should|consider|trade.?off|tradeoffs?|"
        r"options?|alternatives?|brainstorm)\b",
        re.IGNORECASE,
    )),
    ("implement", re.compile(
        r"\b(?:add|build|implement|write|create|ship|wire|land|hook\s*up|"
        r"introduce|spec\s+out|stand\s*up)\b",
        re.IGNORECASE,
    )),
    ("review", re.compile(
        r"\b(?:review|check|audit|validate|verify|inspect|look\s+at|"
        r"sanity.?check|simplify\s+pass|second\s+opinion|grill)\b",
        re.IGNORECASE,
    )),
    ("explain", re.compile(
        r"\b(?:what\s+(?:is|are|does)|explain|how\s+does|show\s+me|"
        r"tell\s+me\s+about|walk\s+me\s+through|describe|summari[sz]e|"
        r"clarify)\b",
        re.IGNORECASE,
    )),
    ("refactor", re.compile(
        r"\b(?:refactor|clean(?:\s*up)?|simplify|rename|extract|consolidate|"
        r"deduplicate|de.?dup|tidy|reorganise|reorganize)\b",
        re.IGNORECASE,
    )),
]


def classify_intent_kind(prompt: str) -> IntentKind:
    """Map `prompt` to an IntentKind via keyword cues.

    Returns `"none"` when nothing matches or when the prompt is empty —
    the manifesto's neutral valence/level profile then applies, which
    falls back to cosine-only ordering.

    Tie-break: most matches wins; on equal counts the earlier entry in
    `_PATTERNS` wins (debug > design > implement > review > explain >
    refactor). Empirically debug-shaped prompts dominate Kane's actual
    sessions, so prioritising debug matches the dogfood distribution.
    """
    if not prompt or not prompt.strip():
        return "none"
    best_kind: IntentKind = "none"
    best_score = 0
    for kind, pattern in _PATTERNS:
        score = len(pattern.findall(prompt))
        if score > best_score:
            best_kind = kind
            best_score = score
    return best_kind
