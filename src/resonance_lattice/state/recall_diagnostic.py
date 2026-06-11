"""Recall diagnostic log — `<state-root>/ledger/recall_diagnostic.jsonl`.

One entry per UserPromptSubmit recall attempt, hit or miss. The
longitudinal-v3 bench surfaced "16/20 sessions had no recall" without
any way to attribute the misses — were they daemon timeouts, sparse
corpus, gate rejections, or workspace mismatches? Bench-time-only
investigation found a mix, but only post-hoc and only because we had
the raw prompts. This log puts that attribution in-band so any future
bench (or live debugging) can answer "why did this recall return
nothing" by reading the file.

Separate from `recall_cache.jsonl` so:
  - misses don't crowd out hits under the 50-entry attribution cap
  - the attribution path stays untouched (it only reads hits)
  - log size can grow larger (200 entries) to capture full bench arcs
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from ._jsonl_log import JsonlRingBufferLog

RECALL_DIAGNOSTIC_FILE = "recall_diagnostic.jsonl"

# Status codes — captured at the hook level, orthogonal to RankDiagnostic's
# in-recall `dropped_at`. Together they answer "did the call reach the
# daemon, and if so, what did the gates decide?"
STATUS_OK = "ok"                              # daemon returned hits
STATUS_NO_HIT = "no_hit"                      # daemon returned 0 hits
STATUS_DAEMON_UNREACHABLE = "daemon_unreachable"  # connect failed both tries
STATUS_DAEMON_ERROR = "daemon_error"          # daemon returned error field
STATUS_NO_STORE = "no_store"                  # per-user root missing


@dataclass(frozen=True)
class RecallDiagnosticEntry:
    """One recall attempt's full diagnostic record."""

    turn_id: str
    timestamp: str
    prompt_hash: str
    intent_kind: str
    intent_id: str | None
    status: str
    n_hits: int
    # Serialised `memory.recall.RankDiagnostic` when the daemon replied;
    # None when the call never reached the daemon (unreachable / no store).
    diagnostic: dict | None = field(default=None)

    def to_dict(self) -> dict:
        return {**asdict(self), "diagnostic": _sanitise_diagnostic(self.diagnostic)}


def _sanitise_diagnostic(diag: dict | None) -> dict | None:
    """Coerce non-JSON-finite floats (-inf / NaN sentinels) to None.

    `RankDiagnostic` uses -inf to mark "no top1 cosine at this stage"
    (e.g. n_rows == 0). Stdlib json writes -inf / NaN as `-Infinity` /
    `NaN`, which most readers reject. Replacing with None preserves the
    semantic intent and stays inside RFC 8259.
    """
    if diag is None:
        return None
    return {
        k: (None if isinstance(v, float) and not math.isfinite(v) else v)
        for k, v in diag.items()
    }


class RecallDiagnosticLog(JsonlRingBufferLog[RecallDiagnosticEntry]):
    """Append-only diagnostic log under `<state-root>/ledger/`.

    Cap (200) is 4× RecallCache's cap so a single bench arc that fires
    recall 50 times AND misses 150 times still fits.
    """

    LOCK_FILENAME = ".recall_diagnostic.lock"
    FILE_NAME = RECALL_DIAGNOSTIC_FILE
    DEFAULT_CACHE_SIZE = 200

    def append(self, entry: RecallDiagnosticEntry) -> None:
        self._append_dict(entry.to_dict())

    def read_recent(self, *, limit: int | None = None) -> list[RecallDiagnosticEntry]:
        return [self._decode(p) for p in self._read_dicts(limit=limit)]

    @staticmethod
    def _decode(payload: dict) -> RecallDiagnosticEntry:
        return RecallDiagnosticEntry(
            turn_id=payload["turn_id"],
            timestamp=payload["timestamp"],
            prompt_hash=payload["prompt_hash"],
            intent_kind=payload.get("intent_kind", "none"),
            intent_id=payload.get("intent_id"),
            status=payload.get("status", STATUS_OK),
            n_hits=int(payload.get("n_hits", 0)),
            diagnostic=payload.get("diagnostic"),
        )
