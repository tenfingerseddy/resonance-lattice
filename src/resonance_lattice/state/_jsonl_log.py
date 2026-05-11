"""Lock-protected size-capped JSONL log — base class.

`RecallCache` and `RecallDiagnosticLog` shared 80% of their implementation
(lock file + lock acquisition + append + ring-buffer trim + read_recent).
The simplify review flagged the duplication; this module factors it out.

`SessionMarkerLog` deliberately stays separate: it doesn't cap, and the
unbounded shape would awkwardly subclass.

Subclasses declare three class-level constants (`LOCK_FILENAME`,
`FILE_NAME`, `DEFAULT_CACHE_SIZE`) and provide a `_decode(dict) -> Entry`
classmethod. The base handles all I/O.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

import portalocker

from .ledger import LEDGER_DIR

EntryT = TypeVar("EntryT")


class JsonlRingBufferLog(Generic[EntryT]):
    """Append-only JSONL log under `<state-root>/ledger/`, capped at
    `DEFAULT_CACHE_SIZE` entries. Trim triggers on append once the file
    overflows; rewrite is atomic via .tmp + os.replace.
    """

    LOCK_FILENAME: ClassVar[str]
    FILE_NAME: ClassVar[str]
    DEFAULT_CACHE_SIZE: ClassVar[int]

    def __init__(
        self,
        state_root: Path | str,
        *,
        cache_size: int | None = None,
    ):
        self._root = Path(state_root) / LEDGER_DIR
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._root / self.LOCK_FILENAME
        self._lock_path.touch(exist_ok=True)
        self._path = self._root / self.FILE_NAME
        self._cache_size = (
            cache_size if cache_size is not None else self.DEFAULT_CACHE_SIZE
        )

    def _lock(self) -> portalocker.Lock:
        return portalocker.Lock(
            str(self._lock_path), mode="r+b", flags=portalocker.LOCK_EX,
        )

    def _append_dict(self, payload: dict) -> None:
        """Append one serialised entry; trim when over cap. Subclasses
        call this after serialising their typed entry to a dict.
        """
        line = json.dumps(payload, sort_keys=True) + "\n"
        with self._lock():
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)
            self._maybe_trim_unlocked()

    def _maybe_trim_unlocked(self) -> None:
        """Rewrite the file to the most-recent `cache_size` entries.

        Caller holds the lock. No-op when below cap; only triggers the
        full read+rewrite on overflow.
        """
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return
        if len(lines) <= self._cache_size:
            return
        keep = lines[-self._cache_size:]
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(keep)
        os.replace(tmp, self._path)

    def _read_dicts(self, *, limit: int | None = None) -> list[dict]:
        """Parse JSONL into a list of payload dicts. Subclasses decode
        these into typed entries. Truncated or invalid lines are skipped.
        """
        if not self._path.exists():
            return []
        out: list[dict] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if limit is not None:
            out = out[-limit:]
        return out
