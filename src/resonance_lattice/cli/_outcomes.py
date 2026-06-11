"""Resolved-intent outcome reader for the corpus-trust loop.

`rlat intent accept|reject` writes one `intent`-kind record to the unified
`ClaimOutcomeLog` per resolved intent (`cli/intent.py`). This module projects
those records into the `CriterionOutcome` records the criterion reducer folds
into corpus-insight confidence (`rlat consolidate-insights`).
"""

from __future__ import annotations

from pathlib import Path

from ..store.insight_attribution import CriterionOutcome, decisive_verdict


def read_intent_outcomes() -> list[CriterionOutcome]:
    """Project the `intent`-kind outcome records into criterion-level
    attribution records for the criterion reducer (S4 d3).

    Reads the `ClaimOutcomeLog`, keeps `intent`-kind records, and yields one
    `CriterionOutcome` per record: the record's attribution as `(claim_id,
    tier)` pairs, its roll-up verdict, the decisive `(verdict_confidence,
    verdict_source)` across its criterion checks (the poison-guard inputs),
    and the intent provenance (`"user"` until S5 harvests). Records with no
    attributed claim, or an unknown/pending roll-up (no trust signal), are
    dropped. Returns `[]` when the log is absent.
    """
    from ..state import resolve_state_root
    from ..state.claim_outcome import ClaimOutcomeLog

    # Read the intent-kind records from the SAME root `rlat intent accept/reject`
    # wrote them to. The writer (cli/intent.py `_state_root` → resolve_state_root)
    # honours `$RLAT_STATE_ROOT` and walks up to the workspace/git root, so the
    # other intent-kind consumers (cli/memory, cli/probe) resolve the same way.
    out: list[CriterionOutcome] = []
    log = ClaimOutcomeLog(resolve_state_root(Path.cwd()))
    for record in log.iter_records(kind="intent"):
        attributed = tuple((a.claim_id, a.tier) for a in record.attribution)
        if not attributed:
            continue
        if record.roll_up_verdict not in ("satisfied", "not_satisfied"):
            continue
        confidence, source = decisive_verdict(
            record.details.criterion_checks, record.roll_up_verdict)
        out.append(CriterionOutcome(
            attributed=attributed,
            roll_up=record.roll_up_verdict,
            verdict_confidence=confidence,
            verdict_source=source,
            provenance="user",  # S5 sets "harvested" for auto-captured intents
        ))
    return out
