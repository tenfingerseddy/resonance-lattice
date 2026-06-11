"""Session-end consolidation pass — confidence → forget.

Per session-end pass:
  1. Run confidence raising (re-derive confidence from cumulative ledger)
  2. Run forget (apply the drop conditions)
  3. Persist

Confidence raising sits before forget so condition 3 of forget
(falsified by outcomes) sees the freshly-derived confidence — a
low-confidence row that just got bumped to medium is no longer eligible
for the falsification drop, which is the right behaviour.

Both stages are mechanical — neither needs an LLM client.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

from ..state.claim_outcome import ClaimOutcomeLog
from .claim_store import ExperienceClaimStore
from .confidence import ConfidenceChange, raise_confidence_pass
from .forget import ForgetVerdict, apply_forget


@dataclass
class ConsolidationResult:
    """One session-end pass outcome."""

    confidence_changes: list[ConfidenceChange] = field(default_factory=list)
    forget_dropped: int = 0
    forget_verdicts: list[ForgetVerdict] = field(default_factory=list)


def consolidation_pass(
    memory: ExperienceClaimStore,
    *,
    state_root: Path | None = None,
    now: _dt.datetime | None = None,
    drifted_claim_ids: list[str] | None = None,
    dry_run: bool = False,
) -> ConsolidationResult:
    """Run the per-session-end pass: confidence → forget.

    `state_root=None` skips confidence raising (no outcome ledger) and
    forget condition 3.

    `drifted_claim_ids` drives forget condition 4 (stale-due-to-corpus-
    drift) — the set of claims whose cited passages a corpus-aware caller
    (`rlat watch`) found drifted. Omitted → condition 4 never fires.

    `dry_run=True` runs every stage to completion but suppresses every
    write. The returned `ConsolidationResult` reports what *would* have
    changed.
    """
    outcomes: list = []
    if state_root is not None:
        outcomes = list(
            ClaimOutcomeLog(state_root).iter_records(kind="intent")
        )

    confidence_changes: list[ConfidenceChange] = []
    if state_root is not None:
        # state_root must flow through: raise_confidence_pass gates its
        # recall-cache read on it, and without it implicit corroboration
        # (M3) silently never fires — 2026-06 review found every
        # production consolidate had run with M3 off.
        confidence_changes = raise_confidence_pass(
            memory, outcomes=outcomes, state_root=state_root, dry_run=dry_run,
        )
    n_dropped, verdicts = apply_forget(
        memory, outcomes=outcomes, now=now,
        drifted_claim_ids=drifted_claim_ids, dry_run=dry_run,
    )
    return ConsolidationResult(
        confidence_changes=confidence_changes,
        forget_dropped=n_dropped,
        forget_verdicts=verdicts,
    )
