"""Session-end consolidation pass — distil → confidence → forget.

Architecture §"Forget runs after distil completes":

  Per session-end pass:
    1. Run distil (cluster events into patterns where triggers met,
       extract learnings from patterns with outcome evidence)
    2. Run confidence raising (re-derive confidence from cumulative ledger)
    3. Run forget (apply the five drop conditions)
    4. Persist

Distil runs both arrows in order — Arrow 1 (events → pattern) writes
fresh patterns at low confidence; Arrow 2 (pattern → learning) reads
those plus earlier patterns and extracts prescriptive learnings where
the outcome ledger has accumulated evidence.

Confidence raising sits between distil and forget so condition 3 of
forget (falsified by outcomes) sees the freshly-derived confidence —
a low-confidence row that just got bumped to medium is no longer
eligible for the falsification drop, which is the right behaviour.

Distil arrows need an LLM client; missing client skips both. Confidence
raising and forget are mechanical and run regardless.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

from ..field.encoder import Encoder
from ..state.ledger import OutcomeLedger
from .confidence import ConfidenceChange, raise_confidence_pass
from .distil_arrow1 import Arrow1Result, LLMClient, arrow1_pass
from .distil_arrow2 import Arrow2Result, arrow2_pass
from .distil_arrow3 import Arrow3Result, arrow3_pass
from .forget import ForgetVerdict, apply_forget
from .store import Memory


@dataclass
class ConsolidationResult:
    """One session-end pass outcome."""

    arrow1: Arrow1Result | None = None
    arrow2: Arrow2Result | None = None
    arrow3: Arrow3Result | None = None
    confidence_changes: list[ConfidenceChange] = field(default_factory=list)
    forget_dropped: int = 0
    forget_verdicts: list[ForgetVerdict] = field(default_factory=list)


def consolidation_pass(
    memory: Memory,
    *,
    llm: LLMClient | None = None,
    encoder: Encoder | None = None,
    state_root: Path | None = None,
    cwd: str | None = None,
    now: _dt.datetime | None = None,
    dry_run: bool = False,
) -> ConsolidationResult:
    """Run the per-session-end pass: distil arrows → confidence → forget.

    Sequencing rationale:
      Arrow 1 first → patterns from events.
      Arrow 2 next  → learnings from patterns + outcome ledger.
      Arrow 3 next  → principles from learnings + cross-domain evidence.
      Confidence    → re-derive from cumulative ledger before forget.
      Forget        → applies the five drop conditions on fresh state.

    `llm=None` skips all arrows. `state_root=None` skips Arrow 2 + 3
    (no outcome ledger), confidence raising, and forget condition 3.

    `dry_run=True` runs every stage to completion but suppresses every
    write (`add_row` / `update_row` / `delete_rows`). The returned
    `ConsolidationResult` reports what *would* have changed. Note the
    dry-run does not simulate cross-arrow read-after-write — Arrow 2/3
    see the on-disk row set, not Arrow 1's would-be promotions.
    """
    # Read the outcome ledger once and share across the four readers
    # (arrow2, arrow3, confidence raising, forget condition 3). Each
    # would otherwise re-parse the full JSONL.
    outcomes: list = []
    if state_root is not None:
        outcomes = list(OutcomeLedger(state_root).iter_records())

    arrow1 = None
    arrow2 = None
    arrow3 = None
    if llm is not None:
        arrow1 = arrow1_pass(
            memory, llm=llm, encoder=encoder, cwd=cwd, dry_run=dry_run,
        )
        if state_root is not None:
            arrow2 = arrow2_pass(
                memory, outcomes=outcomes, llm=llm,
                encoder=encoder, cwd=cwd, dry_run=dry_run,
            )
            arrow3 = arrow3_pass(
                memory, outcomes=outcomes, llm=llm,
                encoder=encoder, cwd=cwd, dry_run=dry_run,
            )
    confidence_changes: list[ConfidenceChange] = []
    if state_root is not None:
        confidence_changes = raise_confidence_pass(
            memory, outcomes=outcomes, dry_run=dry_run,
        )
    n_dropped, verdicts = apply_forget(
        memory, outcomes=outcomes, now=now, dry_run=dry_run,
    )
    return ConsolidationResult(
        arrow1=arrow1,
        arrow2=arrow2,
        arrow3=arrow3,
        confidence_changes=confidence_changes,
        forget_dropped=n_dropped,
        forget_verdicts=verdicts,
    )
