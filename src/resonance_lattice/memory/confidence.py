"""Confidence raising — mechanisms 1 + 5 from architecture §"Calibration".

Distillation dilutes confidence on every promotion. Without a recovery path,
learnings and principles would be permanently downranked despite being the
most valuable rows. The architecture specifies five mechanisms; this module
implements the two that the outcome ledger fully drives:

  Mechanism 1 — Outcome corroboration
    Threshold-based: 2 successes raise low → medium; 3 raise medium → high;
    5 (across ≥2 intent_kinds) raise high → verified.

  Mechanism 5 — Cross-domain accumulation (principle-only)
    Principle attributed to a successful outcome in a NEW intent_kind →
    one-step raise after the cross-domain count threshold.

Mechanisms 2 (corpus verification, requires `rlat watch`), 3 (implicit
corroboration, requires the recall→action attribution cache), and 4 (user
corroborate CLI) ship in later horizons.

Stateless re-derivation: each pass scans the cumulative outcome ledger and
maps `(net_score, distinct_intent_kinds_with_wins)` to the target
confidence level. That avoids the per-pass checkpoint dance — one less
state file to keep consistent.

Symmetric: failures attributed to a row count as -1 to net_score
(architecture §"Calibration mechanisms — failed outcomes drop confidence
by one step"). Forget condition 3 still handles the *drop-row* extreme;
this module handles the confidence-drift gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..state.ledger import OutcomeLedger
from .store import CONFIDENCE_VALUES, Confidence, Memory, Row

# Threshold cuts. Values come from architecture §"Mechanism 1 — Outcome
# corroboration" — engineering-spec parameters; tunable without rewriting
# the manifesto.
_NET_TO_CONFIDENCE: list[tuple[int, Confidence]] = [
    # net_score >= 5 → verified (cross-domain check applied separately)
    (5, "verified"),
    (3, "high"),
    (2, "medium"),
]
_DROP_FLOOR_NET = -2
_VERIFIED_REQUIRES_INTENT_KINDS = 2
_PRINCIPLE_LEVEL = "principle"


@dataclass(frozen=True)
class ConfidenceChange:
    """One row's confidence transition with the evidence that drove it."""

    row_id: str
    from_confidence: Confidence
    to_confidence: Confidence
    net_score: int
    distinct_intent_kinds: int


def _bucket_evidence_by_row(
    outcomes: Iterable,
) -> dict[str, tuple[int, set[str]]]:
    """Single pass over the ledger → `{row_id: (net_score, intent_kinds_with_wins)}`.

    Replaces the prior O(M×N) inner-loop scan (one full ledger walk per
    row) with O(M+N): bucket outcomes by attributed row_id once, look
    up per row. `incidental` tier excluded per architecture §"How
    attribution flows downstream".
    """
    by_row: dict[str, tuple[int, set[str]]] = {}
    for record in outcomes:
        for att in record.attribution:
            if att.tier == "incidental":
                continue
            net, kinds = by_row.get(att.row_id, (0, set()))
            if record.roll_up_verdict == "satisfied":
                net += 1
                if record.intent_kind:
                    kinds = kinds | {record.intent_kind}
            elif record.roll_up_verdict == "not_satisfied":
                net -= 1
            by_row[att.row_id] = (net, kinds)
    return by_row


def _row_evidence(
    row_id: str, outcomes: Iterable,
) -> tuple[int, set[str]]:
    """Count primary+secondary attributions for `row_id`.

    Single-row convenience wrapper used by tests + `target_confidence`'s
    public surface; the pass-level path uses `_bucket_evidence_by_row`
    once and indexes it.
    """
    return _bucket_evidence_by_row(outcomes).get(row_id, (0, set()))


def _target_from_evidence(
    row: Row, net: int, intent_kinds: set[str],
) -> Confidence | None:
    """Map (net_score, intent_kinds) → target confidence for `row`."""
    target: Confidence | None = None
    for threshold, level in _NET_TO_CONFIDENCE:
        if net >= threshold:
            if level == "verified":
                # Cross-domain requirement (mechanism 5). Principles can
                # reach verified via cross-domain accumulation; non-
                # principle rows cap at high until corpus verification
                # (mechanism 2, deferred) lands.
                if (row.level == _PRINCIPLE_LEVEL
                        and len(intent_kinds) >= _VERIFIED_REQUIRES_INTENT_KINDS):
                    target = "verified"
                else:
                    target = "high"
            else:
                target = level
            break
    if target is None and net <= _DROP_FLOOR_NET:
        target = "low"
    if target is None or target == row.confidence:
        return None
    return target


def target_confidence(
    row: Row, outcomes: Iterable,
) -> Confidence | None:
    """Map cumulative evidence to a confidence level for `row`.

    Returns None when the evidence is too thin to suggest a change OR
    when the suggested level matches the current one. Caller skips
    rows where this returns None.
    """
    net, intent_kinds = _row_evidence(row.row_id, outcomes)
    return _target_from_evidence(row, net, intent_kinds)


def raise_confidence_pass(
    memory: Memory,
    *,
    state_root: Path | None = None,
    outcomes: Iterable | None = None,
    dry_run: bool = False,
) -> list[ConfidenceChange]:
    """Re-derive every row's confidence from the cumulative ledger.

    `outcomes` overrides the on-disk ledger (used by tests). When neither
    `outcomes` nor `state_root` is supplied, the pass returns immediately
    — there's no evidence to fold in.

    `dry_run=True` skips the per-row update; the returned changes list
    still describes what *would* have been written.

    Architecture's "step at a time" framing is preserved by the threshold
    bands: a row with 2 wins lands at medium (not verified) regardless of
    its starting confidence; a row with 5 wins + cross-domain lands at
    verified. Walking through every level isn't necessary because we re-
    derive from cumulative state every pass.
    """
    if outcomes is None:
        if state_root is None:
            return []
        outcomes = list(OutcomeLedger(state_root).iter_records())
    else:
        outcomes = list(outcomes)
    rows, _ = memory.read_all()
    # One pass over the ledger to bucket evidence per row_id; per-row
    # lookups below are O(1). Avoids the per-row full-ledger scan that
    # the simpler `target_confidence` path does.
    evidence = _bucket_evidence_by_row(outcomes)
    changes: list[ConfidenceChange] = []
    for row in rows:
        net, intent_kinds = evidence.get(row.row_id, (0, set()))
        target = _target_from_evidence(row, net, intent_kinds)
        if target is None:
            continue
        changes.append(ConfidenceChange(
            row_id=row.row_id,
            from_confidence=row.confidence,
            to_confidence=target,
            net_score=net,
            distinct_intent_kinds=len(intent_kinds),
        ))
        if not dry_run:
            memory.update_row(row.row_id, confidence=target)
    return changes


# Sanity-check at import time so a future Confidence enum drift can't
# silently break the threshold map.
assert all(level in CONFIDENCE_VALUES for _, level in _NET_TO_CONFIDENCE), (
    "_NET_TO_CONFIDENCE references unknown Confidence values"
)
