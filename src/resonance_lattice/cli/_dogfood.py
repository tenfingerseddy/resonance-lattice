"""Shared dogfood-ledger helper for the `rlat search` / `rlat deep-search`
auto-recorders.

The Claim 1 scorecard groups events by `session_id`. Calendar-day
granularity is the default. A controlled run (e.g. 20 ordered batches
in one sitting) sets `RLAT_DOGFOOD_SESSION` so each batch lands in its
own orderable bucket instead of collapsing into a single day.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

_SESSION_ENV = "RLAT_DOGFOOD_SESSION"


def session_id() -> str:
    """Resolve the dogfood session id.

    `RLAT_DOGFOOD_SESSION` wins when set (controlled multi-batch runs);
    otherwise the UTC calendar day.
    """
    return (
        os.environ.get(_SESSION_ENV)
        or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )


def record_event(
    km_path, query: str, *, duration_ms: int, n_source: int,
    insight_ids: list[str], faithfulness: float | None,
    intent_context: str | None, lens_id: str | None,
) -> None:
    """Append one dogfood event to `.rlat-state/ledger/dogfood_events.jsonl`.

    Opt-in by ledger presence; silent on absence or write failure —
    telemetry must never break retrieval. `insight_ids` is rank-ordered
    (list position = retrieval rank), the raw substrate for attribution.
    """
    state_dir = Path.cwd() / ".rlat-state" / "ledger"
    if not state_dir.exists():
        return
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_id": session_id(),
        "km_path": str(km_path),
        "query": query,
        "duration_ms": duration_ms,
        "insight_hits": len(insight_ids),
        "insight_ids": insight_ids,
        "source_hits": n_source,
        "faithfulness": faithfulness,
        "intent_context": intent_context,
        "lens_id": lens_id,
    }
    try:
        (state_dir / "dogfood_events.jsonl").open("a", encoding="utf-8").write(
            json.dumps(event, sort_keys=True) + "\n"
        )
    except OSError:
        pass
