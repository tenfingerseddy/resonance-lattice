"""Session boundary markers — `<state-root>/ledger/sessions.jsonl`.

A marker records the start of a logical session. Consecutive markers (or the
last marker → now) define session-bounded windows for the longitudinal
scorecard. Sessions that span midnight stay coherent; multiple sessions in
one calendar day each get their own scorecard.

Calendar-day windows remain the fallback when no markers exist — so the
scorecard surface keeps working out-of-the-box and only switches to
session-aware slicing once the operator opts in by writing markers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import portalocker

from ..memory._common import make_ulid, utcnow_iso
from ._jsonl_log import LEDGER_DIR

SESSIONS_FILE = "sessions.jsonl"


@dataclass(frozen=True)
class SessionMarker:
    """One session boundary."""

    session_id: str
    timestamp: str


def sessions_path(state_root: Path | str) -> Path:
    return Path(state_root) / LEDGER_DIR / SESSIONS_FILE


class SessionMarkerLog:
    """Append-only session boundary log under `<state-root>/ledger/`.

    Lock-protected so concurrent writers (CLI invocation racing a hook)
    don't interleave their lines.
    """

    def __init__(self, state_root: Path | str):
        self._path = sessions_path(state_root)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._path.parent / ".sessions.lock"
        self._lock_path.touch(exist_ok=True)

    def _lock(self) -> portalocker.Lock:
        return portalocker.Lock(
            str(self._lock_path), mode="r+b", flags=portalocker.LOCK_EX,
        )

    def write(
        self,
        *,
        session_id: str | None = None,
        timestamp: str | None = None,
    ) -> SessionMarker:
        """Append one marker. `session_id` and `timestamp` default to a
        fresh ULID + utcnow — pass explicit values for hook plumbing that
        wants to mirror the harness's own session id."""
        marker = SessionMarker(
            session_id=session_id or make_ulid(),
            timestamp=timestamp or utcnow_iso(),
        )
        line = json.dumps(asdict(marker), sort_keys=True) + "\n"
        with self._lock():
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)
        return marker

    def read_all(self) -> list[SessionMarker]:
        """Read every marker in chronological order (skip truncated lines)."""
        if not self._path.exists():
            return []
        out: list[SessionMarker] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append(SessionMarker(
                session_id=payload["session_id"],
                timestamp=payload["timestamp"],
            ))
        out.sort(key=lambda m: m.timestamp)
        return out
