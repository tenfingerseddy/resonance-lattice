"""Extract atomic events from a captured session transcript.

One LLM call per capture decomposes the session text into *specific
facts and decisions* — short, self-contained items likely to repeat
across sessions about the same work. The substrate's existing recall
gates (cosine, workspace, confidence-gap, recurrence) only function
on a grain that *can* repeat; session-dump text almost never does
(see `_probe/closed_loop_audit/HIT_RATE.md`).

Phase E/3a; the integration into `capture.py` is the next step.
Prompt lives inline, matching `decompose.py`.
"""
from __future__ import annotations

from ._common import parse_llm_json
from ._llm import LLMClient


_PROMPT = """You read assistant turns from coding sessions and extract specific facts and decisions worth remembering across sessions.

A FACT is a self-contained statement that would make sense to a future agent reading it cold, with no knowledge of the session it came from. Examples of GOOD facts:

  - The project's Bronze layer uses Spark Structured Streaming for Event Hubs JSON ingest.
  - The user prefers compact tables over verbose prose for tabular data.
  - The events_fact_hour grain handles the current 500 GB/day volume; events_fact_day did not.

Examples of BAD facts (do NOT emit these):

  - We discussed the architecture (vague, no content).
  - This approach should work (deictic — what approach?).
  - The test is still failing (transient state — if the failure has a known root cause, state the cause as the durable fact instead).
  - Earlier we decided X (references the session we're capturing, not durable).

Each fact also carries a POLARITY — how a future agent should treat it:

  - "factual"  — how things are (architecture, grain, config, behaviour).
  - "prefer"   — the user/project chose or wants this (style, tooling,
                 approach the user asked for).
  - "avoid"    — warned against, tried-and-failed, or explicitly ruled
                 out ("don't use X", "Y was falsified", "Z regressed").

When unsure, use "factual".

OUTPUT FORMAT — read carefully:
  Output ONLY a JSON object. No prose, no markdown, no code fences, no
  explanation. The first character of your response is `{`. The last
  character is `}`. Nothing else.

  Exactly this shape:
    {"facts": [{"text": "<fact text>", "polarity": "factual"}, ...]}

  - Use an empty list `[]` when the session contained no durable facts
    (planning, conversation, navigation — no actual decisions or facts
    that would help a future session).
  - Each fact ≤40 words. Concrete, specific, self-contained.
  - "polarity" is exactly one of "factual" / "prefer" / "avoid".
  - No "(maybe)" or "(if applicable)" — durable facts only.
"""


_MAX_TOKENS = 1024


def _build_messages(text: str) -> list[dict]:
    return [{
        "role": "user",
        "content": (
            "ASSISTANT TURN (capped at 24K chars):\n\n"
            + text.strip()
            + "\n\nExtract the durable facts. Empty list if none."
        ),
    }]


_FACT_POLARITIES = frozenset({"factual", "prefer", "avoid"})


def extract_events(
    text: str, *, client: LLMClient | None,
) -> list[tuple[str, str]] | None:
    """One LLM pass over `text` → list of `(fact_text, polarity)`, or None.

    Returns:
      - `list[(str, str)]` on success (possibly empty for
        no-extractable-content sessions). Each fact is self-contained and
        durable; polarity is one of "factual" / "prefer" / "avoid" —
        the valence the recall rerank weighs (2026-06 review: capture
        hardcoded everything "factual", making the valence term inert).
      - `None` on any failure (LLM exception, malformed JSON, wrong
        payload shape, entries that are neither strings nor fact dicts).
        The caller falls back to the pre-E/3a single-row capture so the
        baseline is preserved.

    Entry tolerance: a bare string entry is accepted as a "factual" fact
    (the pre-polarity output shape — some models will keep emitting it);
    an unknown polarity value coerces to "factual" rather than failing
    the whole extraction.

    `client is None` is a non-failure no-op: returns `None` so the
    caller takes the fallback. This matches the existing
    capture-path convention where the LLM seam is optional.
    """
    if client is None:
        return None
    if not text or not text.strip():
        return []
    try:
        response = client(_PROMPT, _build_messages(text), _MAX_TOKENS)
    except Exception:  # noqa: BLE001 — LLM failure must not raise
        return None
    try:
        payload = parse_llm_json(response.text)
    except Exception:  # noqa: BLE001 — parse failure must not raise
        return None
    if not isinstance(payload, dict):
        return None
    facts = payload.get("facts")
    if not isinstance(facts, list):
        return None
    out: list[tuple[str, str]] = []
    for item in facts:
        if isinstance(item, str):
            fact_text, polarity = item, "factual"
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            fact_text = item["text"]
            polarity = item.get("polarity", "factual")
            if polarity not in _FACT_POLARITIES:
                polarity = "factual"
        else:
            return None
        cleaned = fact_text.strip()
        if cleaned:
            out.append((cleaned, polarity))
    return out
