"""Recall cache — `<workspace-root>/.rlat-state/ledger/recall_cache.jsonl`.

Architecture §"When attribution is computed":

> Recall happens (per turn, hot path) — records rows surfaced
> Action happens — records what the agent did
> PostToolUse fires — captures action result
> Attribution computed — pairs action result with recall result, computes
>   tier weights per row
> Outcome record written to ledger with attribution baked in
>
> One requirement on upstream: the recall result for each turn must be
> persisted at least until the matching outcomes resolve. A short-lived
> in-memory cache (last N turns) is sufficient.

In-memory wouldn't survive across processes — the UserPromptSubmit hook,
PostToolUse hook, and `rlat intent accept|reject` CLI all run as separate
subprocesses. Disk-backed cache it is. Append-only JSONL with ring-buffer
trim every N entries so the file stays bounded without hot-path I/O on
every write.

One entry per recall:
  turn_id        — derived from prompt hash + timestamp; stable across hooks
  timestamp      — when the recall fired
  prompt_hash    — sha256[:16] of the prompt for cross-hook correlation
  intent_kind    — the classifier's verdict on this turn's prompt
  intent_id      — the live intent_id the recall fired against, when one is
                   active at recall time. None when no active intent exists.
                   This is the *outcome-attributed retrieval* enabler — when
                   present, accept/reject attribution skips the timestamp
                   window heuristic and matches exactly. (Architecture
                   §"Trace as corpus", Horizon 4 substrate primitive.)
  row_metadata   — list of {row_id, rank, cosine} for each surfaced hit
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..memory._common import utcnow_iso
from ._jsonl_log import JsonlRingBufferLog

RECALL_CACHE_FILE = "recall_cache.jsonl"


@dataclass(frozen=True)
class RecallHitMetadata:
    """One surfaced row's attribution-relevant metadata."""

    row_id: str
    rank: int
    cosine: float


@dataclass
class RecallEntry:
    """One cached recall result."""

    turn_id: str
    timestamp: str
    prompt_hash: str
    intent_kind: str
    row_metadata: list[RecallHitMetadata] = field(default_factory=list)
    intent_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
            "prompt_hash": self.prompt_hash,
            "intent_kind": self.intent_kind,
            "intent_id": self.intent_id,
            "row_metadata": [asdict(m) for m in self.row_metadata],
        }


def make_turn_id(prompt: str, timestamp: str | None = None) -> str:
    """Deterministic turn id from prompt + timestamp.

    The hook can derive the same id later (PostToolUse, intent resolution)
    without round-tripping through the cache file — useful when correlating
    a tool result back to the recall that preceded it.
    """
    when = timestamp or utcnow_iso()
    h = hashlib.sha256(f"{when}|{prompt}".encode("utf-8")).hexdigest()[:16]
    return h


def hash_prompt(prompt: str) -> str:
    """sha256[:16] of the prompt — same shape `_hookutil.hash_query` produces
    so cross-hook correlation aligns."""
    return hashlib.sha256(prompt.encode("utf-8", "ignore")).hexdigest()[:16]


class RecallCache(JsonlRingBufferLog[RecallEntry]):
    """Append-only recall result cache under `<state-root>/ledger/`.

    Cap (50) is sized for one intent's lifetime: even a chatty session
    rarely fires UserPromptSubmit more than 30 times, and the architecture
    only attributes outcomes to recalls fired during the active intent's
    lifetime.
    """

    LOCK_FILENAME = ".recall_cache.lock"
    FILE_NAME = RECALL_CACHE_FILE
    DEFAULT_CACHE_SIZE = 50

    def append(self, entry: RecallEntry) -> None:
        self._append_dict(entry.to_dict())

    def read_recent(self, *, limit: int | None = None) -> list[RecallEntry]:
        return [self._decode(p) for p in self._read_dicts(limit=limit)]

    @staticmethod
    def _decode(payload: dict) -> RecallEntry:
        return RecallEntry(
            turn_id=payload["turn_id"],
            timestamp=payload["timestamp"],
            prompt_hash=payload["prompt_hash"],
            intent_kind=payload.get("intent_kind", "none"),
            intent_id=payload.get("intent_id"),
            row_metadata=[
                RecallHitMetadata(**m)
                for m in payload.get("row_metadata", [])
            ],
        )

    def read_since(self, since_iso: str) -> list[RecallEntry]:
        """Read entries with timestamp >= `since_iso`."""
        return [e for e in self.read_recent() if e.timestamp >= since_iso]

    def read_for_intent(
        self, intent_id: str, *, since_iso: str | None = None,
    ) -> list[RecallEntry]:
        """Read entries stamped with this exact `intent_id`.

        When the recall path stamps `intent_id` at recall time (the
        outcome-attributed retrieval primitive), this returns the precise
        set of recalls fired against the intent — independent of the
        timestamp window the older path uses. `since_iso` filters within
        that match for callers that also want a lifetime bound.

        Returns an empty list when no entry carries the intent_id (older
        recall_cache.jsonl pre-Horizon-4, or recalls fired with no live
        intent active). Caller falls back to `read_since` in that case.
        """
        out = [e for e in self.read_recent() if e.intent_id == intent_id]
        if since_iso is not None:
            out = [e for e in out if e.timestamp >= since_iso]
        return out
