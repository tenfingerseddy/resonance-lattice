"""Expertise primer — the fourth context layer.

Architecture's North Star §"Structure": three primary context sources
(corpus + memory + intent) distil into a fourth derived layer —
*expertise* — the earned synthesis. The agent reads this on session
start as "what this project knows that I should be using right now."
Not another search surface; a synthesis of the other three.

v0 ships memory + intent only. The corpus piece (top-N passages from
the workspace's primary `*.rlat`) is a follow-up — it depends on the
KM existing and adds an encoder dependency to the path; the
memory + intent slice is enough to demonstrate the layer earns its keep.

Earn its keep: replace 2-3 things the agent currently rediscovers each
session — active intents the user has open, recent learnings the
agent already proved, project-shape feedback that's been corroborated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..memory._common import utcnow_iso
from ..memory.store import Memory, Row
from ..state.intent import LiveIntent, LiveIntentStore

# Default caps — engineering-spec parameters; the layer should be terse
# enough to read on session start without hijacking attention.
DEFAULT_MAX_INTENTS = 8
DEFAULT_MAX_MEMORY_ROWS = 10
DEFAULT_MEMORY_ROW_CHARS = 200


# Confidence sort priority — higher index → higher confidence.
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2, "verified": 3}

# Intent display priority — active before blocked before proposed.
_STATUS_RANK = {"active": 3, "blocked": 2, "proposed": 1}


@dataclass(frozen=True)
class ExpertiseSection:
    """One rendered section of the primer."""

    heading: str
    body: str

    def render(self) -> str:
        if not self.body.strip():
            return f"## {self.heading}\n\n_(none)_\n"
        return f"## {self.heading}\n\n{self.body.rstrip()}\n"


def _intent_line(intent: LiveIntent, *, max_chars: int) -> str:
    text = intent.text.strip().replace("\n", " ")
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return (
        f"- **{intent.level}** [{intent.status}] — {text} "
        f"(`{intent.intent_id}`)"
    )


def _memory_line(row: Row, *, max_chars: int) -> str:
    polarity = row.primary_polarity()
    text = row.text.strip().replace("\n", " ")
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return (
        f"- *{polarity}* — {text} "
        f"({row.confidence}, recur={row.recurrence_count}, level={row.level})"
    )


def _rank_intents(intents: Iterable[LiveIntent]) -> list[LiveIntent]:
    """Order intents for display: status > level > created_at-desc.

    `satisfied` and `abandoned` are excluded — they're terminal and
    surface in eval / rollup outputs, not in the live primer the agent
    reads first thing.
    """
    live = [
        i for i in intents if i.status in ("active", "blocked", "proposed")
    ]
    level_rank = {"direction": 4, "goal": 3, "task": 2, "step": 1}
    return sorted(
        live,
        key=lambda i: (
            -_STATUS_RANK.get(i.status, 0),
            -level_rank.get(i.level, 0),
            i.created_at,  # tie-break: oldest first so long-standing
                           # intents stay visible
        ),
    )


def _rank_memory_rows(rows: Iterable[Row]) -> list[Row]:
    """Order durable memory rows for display: confidence > level > recur.

    Filters out intent rows (they live in the live store; the primer
    surfaces them in the Active intents section). Falsified / is_bad
    rows are dropped — they exist for the forget pipeline, not for the
    agent's working surface.
    """
    out = []
    for r in rows:
        if r.is_bad:
            continue
        if r.level in ("direction", "goal", "task", "step"):
            continue  # intent shape — surfaced separately
        out.append(r)
    level_rank = {
        "principle": 4, "learning": 3, "pattern": 2, "event": 1,
    }
    return sorted(
        out,
        key=lambda r: (
            -_CONFIDENCE_RANK.get(r.confidence, 0),
            -level_rank.get(r.level, 0),
            -r.recurrence_count,
        ),
    )


def render_expertise_primer(
    *,
    intents: Iterable[LiveIntent],
    memory_rows: Iterable[Row],
    max_intents: int = DEFAULT_MAX_INTENTS,
    max_memory_rows: int = DEFAULT_MAX_MEMORY_ROWS,
    memory_row_chars: int = DEFAULT_MEMORY_ROW_CHARS,
    now: str | None = None,
) -> str:
    """Pure renderer — takes the inputs already loaded, returns markdown.

    No I/O, no encoder, no LLM. The CLI wrapper does the workspace
    resolution + store opens; this function is the contract harness
    suites pin against.
    """
    when = now or utcnow_iso()
    ranked_intents = _rank_intents(intents)[:max_intents]
    ranked_rows = _rank_memory_rows(memory_rows)[:max_memory_rows]

    intent_body = "\n".join(
        _intent_line(i, max_chars=memory_row_chars) for i in ranked_intents
    )
    memory_body = "\n".join(
        _memory_line(r, max_chars=memory_row_chars) for r in ranked_rows
    )

    sections = [
        ExpertiseSection(
            heading=f"Active intents ({len(ranked_intents)})",
            body=intent_body,
        ),
        ExpertiseSection(
            heading=f"Project memory (top {len(ranked_rows)})",
            body=memory_body,
        ),
    ]
    header = (
        "# Expertise primer\n\n"
        f"_Generated: {when}_\n\n"
        "Synthesis of corpus + memory + intent — the agent reads this on "
        "session start. Active intents come from the live store; project "
        "memory comes from the durable store, ordered by confidence × "
        "level × recurrence.\n\n"
    )
    return header + "\n".join(s.render() for s in sections)


def build_expertise_primer(
    *,
    state_root: Path,
    memory: Memory,
    output_path: Path,
    max_intents: int = DEFAULT_MAX_INTENTS,
    max_memory_rows: int = DEFAULT_MAX_MEMORY_ROWS,
) -> tuple[Path, int]:
    """Read state + memory, render, write atomically. Returns
    `(output_path, char_count)` so the CLI can report what landed.
    """
    store = LiveIntentStore(state_root)
    intents = store.list_active()
    rows, _band = memory.read_all()

    body = render_expertise_primer(
        intents=intents,
        memory_rows=rows,
        max_intents=max_intents,
        max_memory_rows=max_memory_rows,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    import os
    os.replace(tmp, output_path)
    return output_path, len(body)
