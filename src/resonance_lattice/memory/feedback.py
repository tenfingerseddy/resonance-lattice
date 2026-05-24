"""Feedback log — §9.5 `/rlat-feedback` thumbs-up/down on recall injections.

Append-only `<memory-root>/feedback.log` of good/bad votes the user casts
on the most recent UserPromptSubmit injection. Logged, not acted on
automatically (architecture §9.5) — the weekly review reads it; a future
version may threshold on the aggregate once signal volume warrants it.
"""

from __future__ import annotations

import json
from pathlib import Path

from ._common import utcnow_iso

FEEDBACK_FILENAME = "feedback.log"
FEEDBACK_VERDICTS: tuple[str, ...] = ("good", "bad")


def feedback_log_path(memory_root: Path | str) -> Path:
    return Path(memory_root) / FEEDBACK_FILENAME


def log_feedback(
    memory_root: Path | str, verdict: str, *, timestamp: str | None = None,
) -> dict:
    """Append one `{verdict, timestamp}` vote to the feedback log.

    `verdict` must be `good` or `bad`. Returns the written entry.
    """
    if verdict not in FEEDBACK_VERDICTS:
        raise ValueError(
            f"verdict must be one of {FEEDBACK_VERDICTS}; got {verdict!r}"
        )
    root = Path(memory_root)
    root.mkdir(parents=True, exist_ok=True)
    entry = {"verdict": verdict, "timestamp": timestamp or utcnow_iso()}
    with feedback_log_path(root).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def read_feedback(memory_root: Path | str) -> list[dict]:
    """Read every logged vote in write order; skip truncated lines."""
    p = feedback_log_path(memory_root)
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
