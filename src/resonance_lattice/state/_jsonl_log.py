"""Lock-protected JSONL logs under `<state-root>/ledger/` — base classes.

`JsonlLog` is the append-only base: lock file + lock acquisition +
append + JSONL read. `JsonlRingBufferLog` extends it with a ring-buffer
trim for the size-capped logs.

`RecallCache` and `RecallDiagnosticLog` shared 80% of their
implementation; the simplify review flagged the duplication and this
module factors it out. `ClaimOutcomeLog` — an uncapped outcome log —
uses `JsonlLog` directly.

Subclasses declare `LOCK_FILENAME` + `FILE_NAME` (and, for the ring
buffer, `DEFAULT_CACHE_SIZE`); the base owns all I/O.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

import portalocker

# The `ledger/` subdirectory under a `.rlat-state` root holds every
# append-only log (outcomes, recall cache + diagnostics, session markers,
# pending signals). `_jsonl_log` is the base class for all of them, so it
# owns the layout constant.
LEDGER_DIR = "ledger"

EntryT = TypeVar("EntryT")


def ledger_dir(state_root: Path) -> Path:
    """`<state-root>/ledger/`."""
    return state_root / LEDGER_DIR


class JsonlLog(Generic[EntryT]):
    """Append-only JSONL log under `<state-root>/ledger/`, lock-protected.

    Uncapped — every appended entry is kept. Writes serialise under a
    portalocker advisory lock; reads are unlocked and silently drop a
    partial trailing line.
    """

    LOCK_FILENAME: ClassVar[str]
    FILE_NAME: ClassVar[str]

    def __init__(self, state_root: Path | str):
        self._root = Path(state_root) / LEDGER_DIR
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._root / self.LOCK_FILENAME
        self._lock_path.touch(exist_ok=True)
        self._path = self._root / self.FILE_NAME

    def _lock(self) -> portalocker.Lock:
        return portalocker.Lock(
            str(self._lock_path), mode="r+b", flags=portalocker.LOCK_EX,
        )

    def _append_dict(self, payload: dict) -> None:
        """Append one serialised entry; subclasses call this after
        serialising their typed entry to a dict."""
        line = json.dumps(payload, sort_keys=True) + "\n"
        with self._lock():
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)
            self._post_append_locked()

    def _post_append_locked(self) -> None:
        """Hook run under the append lock — a no-op for an uncapped log;
        `JsonlRingBufferLog` overrides it to trim."""

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


class JsonlRingBufferLog(JsonlLog[EntryT]):
    """A `JsonlLog` capped at `DEFAULT_CACHE_SIZE` entries. Trim triggers
    on append once the file overflows; rewrite is atomic via .tmp +
    os.replace.
    """

    DEFAULT_CACHE_SIZE: ClassVar[int]

    def __init__(
        self,
        state_root: Path | str,
        *,
        cache_size: int | None = None,
    ):
        super().__init__(state_root)
        self._cache_size = (
            cache_size if cache_size is not None else self.DEFAULT_CACHE_SIZE
        )

    def _post_append_locked(self) -> None:
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
