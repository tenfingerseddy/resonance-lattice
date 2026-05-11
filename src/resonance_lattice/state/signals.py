"""Pending-signal log — `<workspace-root>/.rlat-state/ledger/pending_signals.jsonl`.

Architecture §"Outcomes" splits the closed loop into:

  PostToolUse hook → captures signals (mechanical / user / LLM)
                    → writes them to this pending-signal log
  Intent resolution → reads signals → synthesises a CriterionCheck
                    → writes the OutcomeRecord

The split lets the hook layer fire fast (no LLM calls, no heavy I/O on the
hot path) while the slower intent-resolution path is free to LLM-judge or
ask the user. Pending signals decay — anything older than the
configured window is ignored at resolution time (engineering-spec
parameter).

This module owns *only* the pending-signal record + log; the consumer-side
synthesis (signals → criterion check) is a Horizon 2 deliverable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import portalocker

from ..memory._common import utcnow_iso, validate_enum as _validate_enum
from .ledger import LEDGER_DIR, SIGNAL_SOURCES, SignalSource

PENDING_SIGNALS_FILE = "pending_signals.jsonl"


@dataclass
class PendingSignal:
    """One observation that hasn't been bound to a criterion yet.

    `tool_name` and `tool_payload` come from PostToolUse — they describe
    the action the signal observed. `intent_id` is the *current* live-
    intent root if known (the agent's understanding of what it's working
    on), and is `None` when the agent has no active intent. Resolution
    later filters signals by intent_id when synthesising outcomes.

    The signal `value` opaquely carries the verdict-shaped payload — for
    PostToolUse it's typically `{"verdict": "satisfied"}` if exit_code
    equals 0, `{"verdict": "not_satisfied"}` otherwise.
    """

    source: SignalSource
    tool_name: str
    tool_payload: dict[str, Any]
    value: Any
    intent_id: str | None
    captured_at: str


def _validate(signal: PendingSignal) -> None:
    _validate_enum("signal source", signal.source, SIGNAL_SOURCES)
    if not isinstance(signal.tool_name, str):
        raise ValueError("tool_name must be a string")
    if not isinstance(signal.tool_payload, dict):
        raise ValueError("tool_payload must be a dict")


class PendingSignalLog:
    """Append-only pending-signals log under `<state-root>/ledger/`.

    Lock-protected appends so concurrent PostToolUse + Stop don't
    interleave their lines. Reads are unlocked — JSONL parsing tolerates
    a truncated trailing line.
    """

    def __init__(self, state_root: Path | str):
        self._root = Path(state_root) / LEDGER_DIR
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._root / ".signals.lock"
        self._lock_path.touch(exist_ok=True)

    def _lock(self) -> portalocker.Lock:
        return portalocker.Lock(
            str(self._lock_path), mode="r+b", flags=portalocker.LOCK_EX,
        )

    def append(
        self,
        *,
        source: SignalSource,
        tool_name: str,
        tool_payload: dict[str, Any] | None,
        value: Any,
        intent_id: str | None = None,
    ) -> PendingSignal:
        """Append one pending signal. Returns the persisted record."""
        signal = PendingSignal(
            source=source,
            tool_name=tool_name,
            tool_payload=dict(tool_payload) if tool_payload is not None else {},
            value=value,
            intent_id=intent_id,
            captured_at=utcnow_iso(),
        )
        _validate(signal)
        path = self._root / PENDING_SIGNALS_FILE
        line = json.dumps(asdict(signal), sort_keys=True) + "\n"
        with self._lock():
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        return signal

    def read(
        self,
        *,
        intent_id: str | None = None,
        since: str | None = None,
    ) -> list[PendingSignal]:
        """Read pending signals, optionally filtered.

        `intent_id` filter keeps only signals whose `intent_id` matches OR
        is `None` (un-bound signals are ambient and may be relevant to any
        intent). `since` is an ISO timestamp lower bound on `captured_at`.
        """
        path = self._root / PENDING_SIGNALS_FILE
        if not path.exists():
            return []
        out: list[PendingSignal] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            sig = PendingSignal(**payload)
            if intent_id is not None and sig.intent_id is not None and sig.intent_id != intent_id:
                continue
            if since is not None and sig.captured_at < since:
                continue
            out.append(sig)
        return out
