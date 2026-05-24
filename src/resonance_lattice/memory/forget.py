"""Forget — Horizon 2 memory pruning.

Architecture §"Forget" specifies five drop conditions and five protections:

  Drop conditions:
    1. Decay below floor — strength(row) < strength_floor AND not protected
    2. Redundant after promotion — event whose role is now carried by a
       medium+ confidence pattern parent
    3. Falsified by outcomes — low-confidence row with ≥3 failed primary/
       secondary attributions and ≤1 success
    4. Stale due to corpus drift — high/verified row whose cited passages
       have drifted; confidence drops to low (stage 1). Not a row drop —
       the drop-to-low enrols it in mechanism 2's re-verification scan
       (`confidence.corpus_verification_pass`, stage 2).
    5. Trivial from start — age >7d + recurrence==1 + criticality
       low/normal + never recalled / corroborated / attributed

  Protections (override drops):
    1. Active provenance — referenced in another row's parent_ids
    2. Severe avoid — avoid + severe criticality (don't-touch-the-flame)
    3. User-declared — origin: manual
    4. Recently active — corroborated in last N days (proxy for
       "recalled / corroborated / attributed in last N sessions")
    5. Breadcrumbs — falsification records (origin: outcome_derived
       AND criticality: high)

Mechanical — no LLM judgement. Sequenced *after* distil so condition 2
sees freshly-promoted patterns.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from ..state.ledger import OutcomeLedger, Verdict
from ._common import parse_iso_utc
from .rerank import strength
from .store import Confidence, Memory, Row

ForgetCondition = Literal[
    "decay", "redundant", "falsified", "trivial", "stale_drift", "kept",
]

# Thresholds — engineering-spec parameters per architecture §"Operations
# are bounded by depth, not breadth"; tunable without rewriting the spec.
DEFAULT_STRENGTH_FLOOR = 0.05
DEFAULT_TRIVIAL_AGE_DAYS = 7
DEFAULT_RECENT_ACTIVITY_DAYS = 14
DEFAULT_FALSIFICATION_FAIL_COUNT = 3
DEFAULT_FALSIFICATION_SUCCESS_COUNT = 1


@dataclass(frozen=True)
class ForgetVerdict:
    """One row's forget decision with reason.

    `downgrade_to` is set only by condition 4 (stale_drift): the row is
    not dropped, its confidence is lowered to that level. `drop` and
    `downgrade_to` are mutually exclusive — a verdict either removes the
    row or recalibrates it, never both.
    """

    row_id: str
    drop: bool
    condition: ForgetCondition
    protection: str | None  # name of overriding protection, if any
    downgrade_to: Confidence | None = None  # condition 4 — new level


def _age_days(ts: str, now: _dt.datetime) -> float:
    return max(0.0, (now - parse_iso_utc(ts)).total_seconds() / 86400.0)


def _is_protected(
    row: Row,
    *,
    referenced_ids: set[str],
    now: _dt.datetime,
    recent_activity_days: int,
) -> str | None:
    """Return the name of the firing protection, or None if unprotected.

    Protection 1 (active provenance) is *not absolute*: condition 2
    (redundant after promotion) bypasses it for events whose role is
    captured by a confident parent. The check happens at the call site,
    not here — this helper reports protection presence; the caller
    decides whether to honour it.
    """
    if row.row_id in referenced_ids:
        return "active_provenance"
    if row.primary_polarity() == "avoid" and row.criticality == "severe":
        return "severe_avoid"
    if row.origin == "manual":
        return "user_declared"
    if _age_days(row.last_corroborated_at, now) <= recent_activity_days:
        return "recently_active"
    if row.origin == "outcome_derived" and row.criticality == "high":
        return "breadcrumb"
    return None


def _falsification_counts(
    row_id: str,
    outcomes: list,
) -> tuple[int, int]:
    """Count failed and successful primary/secondary attributions for a row.

    Architecture §"How attribution flows downstream — falsification (forget
    condition 3)": failed outcomes drop confidence symmetrically;
    incidental rows take none. So we count only primary + secondary tiers.
    """
    fail = success = 0
    for record in outcomes:
        for att in record.attribution:
            if att.row_id != row_id or att.tier == "incidental":
                continue
            if record.roll_up_verdict == "satisfied":
                success += 1
            elif record.roll_up_verdict == "not_satisfied":
                fail += 1
    return fail, success


def _has_independent_strength_signal(row: Row) -> bool:
    """Architecture: 'events with high recurrence, severe criticality, or
    heavy outcome attribution stay regardless'."""
    if row.recurrence_count >= 10:
        return True
    if row.criticality in ("high", "severe"):
        return True
    return False


def evaluate_row(
    row: Row,
    *,
    referenced_ids: set[str],
    confident_parent_lookup: dict[str, list[Row]],
    outcomes: list,
    now: _dt.datetime,
    drifted_row_ids: frozenset[str] = frozenset(),
    strength_floor: float = DEFAULT_STRENGTH_FLOOR,
    trivial_age_days: int = DEFAULT_TRIVIAL_AGE_DAYS,
    recent_activity_days: int = DEFAULT_RECENT_ACTIVITY_DAYS,
    falsification_fail_count: int = DEFAULT_FALSIFICATION_FAIL_COUNT,
    falsification_success_count: int = DEFAULT_FALSIFICATION_SUCCESS_COUNT,
) -> ForgetVerdict:
    """Decide whether `row` should be dropped this pass.

    `confident_parent_lookup` maps an event row_id to the list of
    patterns/learnings that named it as a parent and have confidence
    >= medium — pre-computed by the caller so this function stays O(1)
    per row.

    `drifted_row_ids` is the set of rows whose cited passages have
    drifted against the corpus (computed by the caller — `rlat watch`
    or a corpus-aware pass). It drives condition 4.
    """
    # Condition 4: stale due to corpus drift. Checked first — decay
    # would otherwise rank on a stale confidence. Not a row drop:
    # lowering to `low` enrols the row in mechanism 2's re-verification.
    # Fires regardless of protections (recalibration, not pruning).
    if (row.confidence in ("high", "verified")
            and row.row_id in drifted_row_ids):
        return ForgetVerdict(row.row_id, drop=False, condition="stale_drift",
                             protection=None, downgrade_to="low")

    protection = _is_protected(
        row,
        referenced_ids=referenced_ids,
        now=now,
        recent_activity_days=recent_activity_days,
    )

    # Condition 2: redundant after promotion. Bypasses active_provenance
    # only — the event's role is now captured by a confident parent, so
    # the parent chain is preserved while the redundant event drops.
    if (
        row.level == "event"
        and confident_parent_lookup.get(row.row_id)
        and not _has_independent_strength_signal(row)
    ):
        if protection is None or protection == "active_provenance":
            return ForgetVerdict(row.row_id, drop=True,
                                 condition="redundant", protection=None)
        return ForgetVerdict(row.row_id, drop=False,
                             condition="redundant", protection=protection)

    if protection is not None:
        return ForgetVerdict(row.row_id, drop=False,
                             condition="kept", protection=protection)

    # Condition 3: falsified by outcomes.
    if row.confidence == "low":
        fail, success = _falsification_counts(row.row_id, outcomes)
        if (fail >= falsification_fail_count
                and success <= falsification_success_count):
            return ForgetVerdict(row.row_id, drop=True,
                                 condition="falsified", protection=None)

    # Condition 5: trivial from start. Tighter than the others (every
    # sub-condition must hold) so it doesn't punish slow-burn events.
    age = _age_days(row.created_at, now)
    if (age > trivial_age_days
            and row.recurrence_count == 1
            and row.criticality in ("low", "normal")
            and row.created_at == row.last_corroborated_at):
        return ForgetVerdict(row.row_id, drop=True,
                             condition="trivial", protection=None)

    # Condition 1: decay below floor.
    if strength(row, now=now) < strength_floor:
        return ForgetVerdict(row.row_id, drop=True,
                             condition="decay", protection=None)

    return ForgetVerdict(row.row_id, drop=False,
                         condition="kept", protection=None)


def forget_pass(
    rows: list[Row],
    *,
    outcomes: Iterable | None = None,
    now: _dt.datetime | None = None,
    drifted_row_ids: Iterable[str] | None = None,
    **thresholds,
) -> list[ForgetVerdict]:
    """Evaluate every row in `rows`. Returns one verdict per row.

    Caller deletes rows where `verdict.drop` is True and lowers
    confidence on rows where `verdict.downgrade_to` is set (condition 4);
    rows where `verdict.protection` is set are kept and the protection
    name is reported for diagnostics. Pure function — no I/O. The
    session-end runner (`consolidation_pass`) handles persistence,
    outcome-ledger plumbing, and the downgrade writes.
    """
    if now is None:
        now = _dt.datetime.now(_dt.timezone.utc)
    outcomes_list = list(outcomes) if outcomes is not None else []
    drifted = frozenset(drifted_row_ids) if drifted_row_ids else frozenset()
    # Pre-compute the parent index once: which row_ids are referenced as
    # parents by other rows, and for each event, which medium+ confidence
    # parents claim it.
    referenced_ids: set[str] = set()
    by_id: dict[str, Row] = {r.row_id: r for r in rows}
    confident_parent_lookup: dict[str, list[Row]] = {}
    for parent in rows:
        for child_id in parent.parent_ids:
            referenced_ids.add(child_id)
            if (parent.level in ("pattern", "learning")
                    and parent.confidence in ("medium", "high", "verified")):
                confident_parent_lookup.setdefault(child_id, []).append(parent)
    return [
        evaluate_row(
            row,
            referenced_ids=referenced_ids,
            confident_parent_lookup=confident_parent_lookup,
            outcomes=outcomes_list,
            now=now,
            drifted_row_ids=drifted,
            **thresholds,
        )
        for row in rows
    ]


def apply_forget(
    memory: Memory,
    *,
    state_root: Path | None = None,
    outcomes: Iterable | None = None,
    now: _dt.datetime | None = None,
    drifted_row_ids: Iterable[str] | None = None,
    dry_run: bool = False,
    **thresholds,
) -> tuple[int, list[ForgetVerdict]]:
    """End-to-end: read rows + outcomes, evaluate, drop + recalibrate.
    Returns (n_dropped, all_verdicts) — verdicts include kept rows and
    condition-4 downgrades for audit.

    Caller can pass `outcomes` directly (used by `consolidation_pass` to
    share one ledger read across distil + confidence + forget); falls
    back to reading from `state_root` when not provided. `drifted_row_ids`
    drives condition 4 — the set of rows whose cited passages have
    drifted; when omitted, condition 4 never fires.

    `dry_run=True` skips both the delete and the condition-4 confidence
    writes; `n_dropped` then reflects the count that *would* have been
    dropped."""
    rows, _ = memory.read_all()
    if outcomes is None:
        outcomes = (
            list(OutcomeLedger(state_root).iter_records())
            if state_root is not None
            else []
        )
    verdicts = forget_pass(
        rows, outcomes=outcomes, now=now,
        drifted_row_ids=drifted_row_ids, **thresholds,
    )
    drop_ids = [v.row_id for v in verdicts if v.drop]
    if dry_run:
        return len(drop_ids), verdicts
    for v in verdicts:
        if v.downgrade_to is not None:
            memory.update_row(v.row_id, confidence=v.downgrade_to)
    n_dropped = memory.delete_rows(drop_ids) if drop_ids else 0
    return n_dropped, verdicts
