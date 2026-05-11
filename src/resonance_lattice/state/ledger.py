"""Outcome ledger — `<workspace-root>/.rlat-state/ledger/outcomes.jsonl`.

Architecture §"Outcomes" specifies an append-only ledger; one record per
intent resolution at any level. The ledger outlives the rows it references —
when a memory row is dropped (forget), its outcome records stay as the audit
trail. Readers handle "row no longer exists" gracefully.

Three signal sources (mechanical / user / LLM) are combined under the
authority rule (architecture §"The combination rule — authority, not
averaging") — *highest-trust signal wins, lower-trust signals are recorded
but don't override*. User > mechanical > LLM. User-mechanical disagreement
flags conflict and resolves conservatively to `not_satisfied`.

This module ships the data shape + write API + combination rule. The
downstream consumers (confidence raising, distil arrows 2/3, forget
condition 3) are wired in Horizon 2 — they read this ledger.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import portalocker

from ..memory._common import (
    utcnow_iso,
    validate_criterion as _validate_criterion,
    validate_enum as _validate_enum,
)
from ..memory.store import Criterion

LEDGER_DIR = "ledger"
OUTCOMES_FILE = "outcomes.jsonl"

# Verdict states per criterion. `pending` is temporary (re-fires); `unknown`
# is terminal (the check ran but couldn't decide).
Verdict = Literal["satisfied", "not_satisfied", "unknown", "pending"]
VERDICT_VALUES: frozenset[str] = frozenset(
    {"satisfied", "not_satisfied", "unknown", "pending"}
)

# Signal source — the three input channels feeding combination.
SignalSource = Literal["mechanical", "user", "llm"]
SIGNAL_SOURCES: frozenset[str] = frozenset({"mechanical", "user", "llm"})

# Verdict confidence — separate from the verdict itself.
VerdictConfidence = Literal["high", "medium", "low"]
VERDICT_CONFIDENCE_VALUES: frozenset[str] = frozenset({"high", "medium", "low"})

# Verdict source after combination — `signal` (combined from observed signals)
# vs `user_override` (user explicitly accepted incompletion).
VerdictSource = Literal["signal", "user_override"]
VERDICT_SOURCE_VALUES: frozenset[str] = frozenset({"signal", "user_override"})

# Attribution tier weights from architecture §"Attribution".
AttributionTier = Literal["primary", "secondary", "incidental"]
ATTRIBUTION_TIERS: frozenset[str] = frozenset(
    {"primary", "secondary", "incidental"}
)
TIER_WEIGHTS: dict[str, float] = {
    "primary": 1.0,
    "secondary": 0.5,
    "incidental": 0.1,
}


@dataclass
class Signal:
    """One observation feeding a criterion check.

    `value` is the observed payload — for `mechanical`, this is typically
    `{"exit_code": 0}` or `{"file_exists": true}`; for `user`, the boolean
    they signalled; for `llm`, the rubric verdict + reasoning. Stored
    opaque so the ledger doesn't constrain measurement specifics.
    """

    source: SignalSource
    value: Any
    timestamp: str


@dataclass
class CriterionCheck:
    """One success criterion's resolution."""

    criterion_text: str
    measure: str
    verdict: Verdict
    signals_seen: list[Signal] = field(default_factory=list)
    verdict_confidence: VerdictConfidence = "low"
    conflict_flag: bool = False
    verdict_source: VerdictSource = "signal"
    mechanical_check_bypassed: bool = False


@dataclass
class Attribution:
    """One contributing-row → tier link."""

    row_id: str
    tier: AttributionTier
    # `recall_rank` is the row's position in the recall result for the turn
    # whose action produced this outcome. `cosine` is the row's score at
    # recall time. `alignment` is the row↔action cosine — the load-bearing
    # signal that distinguishes "context that shaped the action" from
    # "context that was retrieved but ignored". Optional because v1
    # mechanical-only outcomes may not compute alignment.
    recall_rank: int | None = None
    cosine: float | None = None
    alignment: float | None = None


@dataclass
class OutcomeRecord:
    """One outcome record. Append-only on the ledger.

    `intent_id` references a live or durable intent (the ledger doesn't
    care which). `roll_up_verdict` is the AND across `criterion_checks`
    plus child intent verdicts — computed by the writer, stored explicitly
    so readers don't need the rule embedded.

    `intent_kind` is the agent's intent-kind context at outcome time
    (debug / design / implement / …). Confidence raising's cross-domain
    mechanism (architecture §"Calibration mechanisms" #5) reads this to
    decide whether a principle is earning evidence in a *new* intent_kind.
    Optional + additive — older records load with intent_kind=None.
    """

    intent_id: str
    intent_level: str
    criterion_checks: list[CriterionCheck]
    roll_up_verdict: Verdict
    attribution: list[Attribution]
    resolved_at: str
    intent_was_corrected: bool = False
    notes: str = ""
    intent_kind: str | None = None


# ---------------------------------------------------------------------------
# Combination rule — authority, not averaging
# ---------------------------------------------------------------------------


def combine_signals(
    signals: Iterable[Signal],
    *,
    user_override: bool = False,
) -> tuple[Verdict, VerdictConfidence, bool, VerdictSource]:
    """Combine raw signals into (verdict, confidence, conflict_flag, source).

    Authority rule (architecture §"The combination rule"):

      User=not_satisfied + mechanical=satisfied → not_satisfied (user wins)
      Mechanical=satisfied + LLM=not_satisfied → satisfied (mechanical wins)
      User=satisfied + mechanical=not_satisfied → not_satisfied + conflict flag
      LLM-only → use it, mark `low_confidence`

    `user_override=True` means the user explicitly accepted incompletion;
    the verdict is forced to `satisfied` regardless of other signals, but
    the source records that fact for downstream weighting.
    """
    by_source: dict[str, Verdict] = {}
    for sig in signals:
        # Map raw payload → coarse verdict. Most signals carry a payload
        # the harness translates to satisfied/not_satisfied at observation
        # time; here we accept whatever the caller pre-distilled. If
        # `value` is a dict with a `verdict` key, prefer that; else treat
        # truthiness as satisfied.
        verdict: Verdict
        v = sig.value
        if isinstance(v, dict) and v.get("verdict") in VERDICT_VALUES:
            verdict = v["verdict"]
        elif v is None:
            verdict = "unknown"
        else:
            verdict = "satisfied" if bool(v) else "not_satisfied"
        # Last-wins per source — multiple mechanical signals collapse to
        # the most recent. Callers that need historical signals read the
        # raw `signals_seen` list directly.
        by_source[sig.source] = verdict

    if user_override:
        return "satisfied", "high", False, "user_override"

    user = by_source.get("user")
    mech = by_source.get("mechanical")
    llm = by_source.get("llm")

    conflict = False
    verdict_confidence: VerdictConfidence = "low"

    if user is not None and mech is not None:
        if user == mech:
            final = user
            verdict_confidence = "high"
        else:
            # Loud disagreement — fail safe to not_satisfied + flag.
            final = "not_satisfied"
            verdict_confidence = "high"
            conflict = True
    elif user is not None:
        final = user
        verdict_confidence = "high"
    elif mech is not None:
        final = mech
        verdict_confidence = "high"
    elif llm is not None:
        final = llm
        verdict_confidence = "low"
    else:
        final = "unknown"
        verdict_confidence = "low"

    return final, verdict_confidence, conflict, "signal"


def roll_up(criterion_checks: Iterable[CriterionCheck]) -> Verdict:
    """AND across criterion checks. Architecture §"Roll-up — the aggregation
    rule": every criterion must be `satisfied` for the row-up to be
    `satisfied`; any `not_satisfied` makes the roll-up `not_satisfied`;
    otherwise (some unknown/pending) → `unknown`.
    """
    checks = list(criterion_checks)
    if not checks:
        return "unknown"
    verdicts = [c.verdict for c in checks]
    if any(v == "not_satisfied" for v in verdicts):
        return "not_satisfied"
    if all(v == "satisfied" for v in verdicts):
        return "satisfied"
    return "unknown"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_signal(sig: Signal) -> None:
    _validate_enum("signal source", sig.source, SIGNAL_SOURCES)
    if not isinstance(sig.timestamp, str) or not sig.timestamp:
        raise ValueError("signal.timestamp must be a non-empty string")


def _validate_criterion_check(cc: CriterionCheck) -> None:
    _validate_criterion({"text": cc.criterion_text, "measure": cc.measure})
    _validate_enum("verdict", cc.verdict, VERDICT_VALUES)
    _validate_enum(
        "verdict_confidence", cc.verdict_confidence, VERDICT_CONFIDENCE_VALUES
    )
    _validate_enum(
        "verdict_source", cc.verdict_source, VERDICT_SOURCE_VALUES
    )
    if not isinstance(cc.signals_seen, list):
        raise ValueError("signals_seen must be a list")
    for sig in cc.signals_seen:
        _validate_signal(sig)


def _validate_attribution(att: Attribution) -> None:
    if not isinstance(att.row_id, str) or not att.row_id:
        raise ValueError("attribution.row_id must be a non-empty string")
    _validate_enum("attribution.tier", att.tier, ATTRIBUTION_TIERS)


def _validate_record(record: OutcomeRecord) -> None:
    if not isinstance(record.intent_id, str) or not record.intent_id:
        raise ValueError("intent_id must be a non-empty string")
    _validate_enum(
        "roll_up_verdict", record.roll_up_verdict, VERDICT_VALUES
    )
    if not isinstance(record.criterion_checks, list):
        raise ValueError("criterion_checks must be a list")
    for cc in record.criterion_checks:
        _validate_criterion_check(cc)
    for att in record.attribution:
        _validate_attribution(att)


# ---------------------------------------------------------------------------
# Ledger I/O
# ---------------------------------------------------------------------------


def ledger_dir(state_root: Path) -> Path:
    """`<state-root>/ledger/`."""
    return state_root / LEDGER_DIR


class OutcomeLedger:
    """Append-only outcome ledger.

    Writes serialised under a portalocker advisory lock so concurrent hooks
    (PostToolUse + Stop) don't interleave their lines. Reads are unlocked —
    JSONL is line-oriented and a partial trailing line is silently dropped.
    """

    def __init__(self, state_root: Path | str):
        self._root = ledger_dir(Path(state_root))
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._root / ".lock"
        self._lock_path.touch(exist_ok=True)

    def _lock(self) -> portalocker.Lock:
        return portalocker.Lock(
            str(self._lock_path), mode="r+b", flags=portalocker.LOCK_EX,
        )

    def write(self, record: OutcomeRecord) -> None:
        """Append one outcome record."""
        _validate_record(record)
        path = self._root / OUTCOMES_FILE
        line = json.dumps(_record_to_dict(record), sort_keys=True) + "\n"
        with self._lock():
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)

    def read(
        self,
        *,
        intent_id: str | None = None,
        since: str | None = None,
    ) -> list[OutcomeRecord]:
        """Read records, optionally filtered by intent_id or by ISO timestamp.

        The architecture's downstream consumers (confidence raising, distil
        arrows 2 + 3, forget condition 3) read this — keep the surface
        simple and let consumers do their own additional filtering.
        """
        out: list[OutcomeRecord] = []
        for record in self.iter_records():
            if intent_id is not None and record.intent_id != intent_id:
                continue
            if since is not None and record.resolved_at < since:
                continue
            out.append(record)
        return out

    def iter_records(self) -> Iterator[OutcomeRecord]:
        """Yield every record in append order (skip truncated trailing line)."""
        path = self._root / OUTCOMES_FILE
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield _record_from_dict(payload)


def _record_to_dict(record: OutcomeRecord) -> dict[str, Any]:
    return asdict(record)


def _record_from_dict(payload: dict[str, Any]) -> OutcomeRecord:
    """Re-hydrate a record from its JSONL payload.

    Tolerates extra unknown keys (additive forward compat) and re-builds
    nested dataclasses by name. The shape is stable across Horizon 1 — only
    additions are forward-compat through this helper.
    """
    cc = [
        CriterionCheck(
            criterion_text=item["criterion_text"],
            measure=item["measure"],
            verdict=item["verdict"],
            signals_seen=[
                Signal(**s) for s in item.get("signals_seen", [])
            ],
            verdict_confidence=item.get("verdict_confidence", "low"),
            conflict_flag=item.get("conflict_flag", False),
            verdict_source=item.get("verdict_source", "signal"),
            mechanical_check_bypassed=item.get(
                "mechanical_check_bypassed", False
            ),
        )
        for item in payload.get("criterion_checks", [])
    ]
    attribution = [
        Attribution(
            row_id=a["row_id"],
            tier=a["tier"],
            recall_rank=a.get("recall_rank"),
            cosine=a.get("cosine"),
            alignment=a.get("alignment"),
        )
        for a in payload.get("attribution", [])
    ]
    return OutcomeRecord(
        intent_id=payload["intent_id"],
        intent_level=payload["intent_level"],
        criterion_checks=cc,
        roll_up_verdict=payload["roll_up_verdict"],
        attribution=attribution,
        resolved_at=payload["resolved_at"],
        intent_was_corrected=payload.get("intent_was_corrected", False),
        notes=payload.get("notes", ""),
        intent_kind=payload.get("intent_kind"),
    )


def now_iso() -> str:
    """Re-export `utcnow_iso` so callers don't need to thread the import."""
    return utcnow_iso()
