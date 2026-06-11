"""The claim-outcome log — `<state-root>/ledger/claim_outcomes.jsonl`.

One append-only log, one record per resolved intent: the retrieved
`claim_id`s, the verdict, and the attribution. It is the single log both
halves of the closed learning loop read — the experience calibration
pass and the corpus attribution reducers.

Three signal sources (mechanical / user / LLM) combine under the
authority rule (architecture §"The combination rule — authority, not
averaging") — *highest-trust signal wins, lower-trust signals are
recorded but don't override*. User > mechanical > LLM. A user-mechanical
disagreement flags conflict and resolves conservatively to
`not_satisfied`.

The log outlives the claims it references — when a claim is dropped
(forget / retirement), its outcome records stay as the audit trail.
Readers handle "claim no longer exists" gracefully.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Iterator, Literal

from ..memory._common import (
    utcnow_iso,
    validate_criterion as _validate_criterion,
    validate_enum as _validate_enum,
)
from ._jsonl_log import JsonlLog
from .claim import DEFAULT_WRITER

CLAIM_OUTCOMES_FILE = "claim_outcomes.jsonl"

# Verdict states per criterion. `pending` is temporary (re-fires); `unknown`
# is terminal (the check ran but couldn't decide).
Verdict = Literal["satisfied", "not_satisfied", "unknown", "pending"]
VERDICT_VALUES: frozenset[str] = frozenset(
    {"satisfied", "not_satisfied", "unknown", "pending"}
)

# Record origin. `intent` records are per-intent resolutions (the
# `/accept` `/reject` path). The `session` kind (rlat assess-session) was
# removed in 8ab08e13; the Literal is the single source of truth.
OutcomeRecordKind = Literal["intent"]

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
    opaque so the log doesn't constrain measurement specifics.
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
    """One contributing-claim → tier link."""

    claim_id: str
    tier: AttributionTier
    # `recall_rank` is the claim's position in the recall result for the
    # turn whose action produced this outcome. `cosine` is the claim's
    # score at recall time. `alignment` is the claim↔action cosine — the
    # load-bearing signal that distinguishes "context that shaped the
    # action" from "context that was retrieved but ignored". Optional
    # because v1 mechanical-only outcomes may not compute alignment.
    recall_rank: int | None = None
    cosine: float | None = None
    alignment: float | None = None


@dataclass
class IntentOutcomeDetails:
    """The fields a per-intent outcome record carries — the criteria the
    intent was checked against, the manifesto level it sat at, and the
    `intent_kind` context the calibration cross-domain mechanism reads."""

    intent_level: str
    criterion_checks: list[CriterionCheck] = field(default_factory=list)
    intent_kind: str | None = None
    intent_was_corrected: bool = False


OutcomeDetails = IntentOutcomeDetails


@dataclass
class ClaimOutcomeRecord:
    """One outcome record. Append-only on the claim-outcome log.

    `intent_id` references a live or durable intent (the log doesn't care
    which). `roll_up_verdict` is the AND across criterion checks plus child
    intent verdicts — computed by the writer, stored explicitly so readers
    don't need the rule embedded.

    `details` is `IntentOutcomeDetails` — the one record kind today.
    `record.kind` is derived from `type(details)`; the discriminator stays so
    a future record kind slots in without a format change.
    """

    intent_id: str
    resolved_at: str
    roll_up_verdict: Verdict
    attribution: list[Attribution]
    details: OutcomeDetails
    session_id: str = ""
    notes: str = ""
    # H3 identity (defaulted single-writer in v1) — who recorded this
    # outcome (invariant 7). Defaulted so pre-S1.5 rows still load.
    writer: str = DEFAULT_WRITER

    @property
    def kind(self) -> OutcomeRecordKind:
        # One record kind today; the discriminator stays for the next.
        return "intent"


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
    rule": every criterion must be `satisfied` for the roll-up to be
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
    if not isinstance(att.claim_id, str) or not att.claim_id:
        raise ValueError("attribution.claim_id must be a non-empty string")
    _validate_enum("attribution.tier", att.tier, ATTRIBUTION_TIERS)


def _validate_record(record: ClaimOutcomeRecord) -> None:
    if not isinstance(record.intent_id, str) or not record.intent_id:
        raise ValueError("intent_id must be a non-empty string")
    _validate_enum(
        "roll_up_verdict", record.roll_up_verdict, VERDICT_VALUES
    )
    for att in record.attribution:
        _validate_attribution(att)
    details = record.details
    if isinstance(details, IntentOutcomeDetails):
        if not isinstance(details.criterion_checks, list):
            raise ValueError("criterion_checks must be a list")
        for cc in details.criterion_checks:
            _validate_criterion_check(cc)
    else:
        raise ValueError(
            f"details must be IntentOutcomeDetails; "
            f"got {type(details).__name__}"
        )


# ---------------------------------------------------------------------------
# Log I/O
# ---------------------------------------------------------------------------


class ClaimOutcomeLog(JsonlLog[ClaimOutcomeRecord]):
    """Append-only claim-outcome log.

    Built on `JsonlLog` — lock-protected append, unlocked JSONL read,
    partial trailing line dropped. Shares the `ledger/` directory's
    `.lock`, so concurrent writers across the ledger's logs never
    interleave a line.
    """

    LOCK_FILENAME = ".lock"
    FILE_NAME = CLAIM_OUTCOMES_FILE

    def write(self, record: ClaimOutcomeRecord) -> None:
        """Append one outcome record."""
        _validate_record(record)
        self._append_dict(_record_to_dict(record))

    def read(
        self,
        *,
        intent_id: str | None = None,
        since: str | None = None,
        kind: OutcomeRecordKind | None = None,
    ) -> list[ClaimOutcomeRecord]:
        """Read records, optionally filtered by `kind`, intent_id, or ISO
        timestamp.

        `kind` filters by record origin. One kind exists today (`"intent"`),
        read by the experience consumers (confidence raising, the distil
        arrows, forget) and the corpus attribution reducer alike; the filter
        stays for the next record kind.
        """
        out: list[ClaimOutcomeRecord] = []
        for record in self.iter_records(kind=kind):
            if intent_id is not None and record.intent_id != intent_id:
                continue
            if since is not None and record.resolved_at < since:
                continue
            out.append(record)
        return out

    def iter_records(
        self, *, kind: OutcomeRecordKind | None = None,
    ) -> Iterator[ClaimOutcomeRecord]:
        """Yield records in append order (skip a truncated trailing line),
        optionally filtered to one `kind` of record origin.

        Filter on the raw payload's discriminator before rehydrating —
        the typical caller (corpus reducers on an intent-heavy log) only
        wants one kind, and skipping the dataclass construction for the
        other is the difference between O(N) and O(N×nested_fields).
        """
        for payload in self._read_dicts():
            if kind is not None and payload.get("kind", "intent") != kind:
                continue
            yield _record_from_dict(payload)


def _record_to_dict(record: ClaimOutcomeRecord) -> dict[str, Any]:
    """Serialise a record to JSON-ready dict. The wire format is flat —
    `details` is unwrapped into top-level keys, with `kind` as the
    discriminator. Flat lets the reader skip rehydration of off-kind
    records on the `iter_records(kind=)` filter path; nesting under a
    `details` object would force a full decode per row."""
    d: dict[str, Any] = {
        "intent_id": record.intent_id,
        "resolved_at": record.resolved_at,
        "roll_up_verdict": record.roll_up_verdict,
        "attribution": [asdict(a) for a in record.attribution],
        "session_id": record.session_id,
        "notes": record.notes,
        "writer": record.writer,
        "kind": record.kind,
    }
    d.update({
        "intent_level": record.details.intent_level,
        "criterion_checks": [
            asdict(cc) for cc in record.details.criterion_checks
        ],
        "intent_kind": record.details.intent_kind,
        "intent_was_corrected": record.details.intent_was_corrected,
    })
    return d


def _record_from_dict(payload: dict[str, Any]) -> ClaimOutcomeRecord:
    """Re-hydrate a record from its JSONL payload.

    The `kind` discriminator picks which `details` type to build.
    Tolerates extra unknown keys (additive forward compat) and re-builds
    nested dataclasses by name.
    """
    # Forward-compat: select known fields only — an additive key on a
    # persisted row must not break a reader.
    attribution = [
        Attribution(
            claim_id=a["claim_id"],
            tier=a["tier"],
            recall_rank=a.get("recall_rank"),
            cosine=a.get("cosine"),
            alignment=a.get("alignment"),
        )
        for a in payload.get("attribution", [])
    ]
    details = IntentOutcomeDetails(
        intent_level=payload.get("intent_level", ""),
        criterion_checks=[
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
        ],
        intent_kind=payload.get("intent_kind"),
        intent_was_corrected=payload.get("intent_was_corrected", False),
    )
    return ClaimOutcomeRecord(
        intent_id=payload["intent_id"],
        resolved_at=payload["resolved_at"],
        roll_up_verdict=payload["roll_up_verdict"],
        attribution=attribution,
        details=details,
        session_id=payload.get("session_id", ""),
        notes=payload.get("notes", ""),
        writer=payload.get("writer", DEFAULT_WRITER),
    )


def now_iso() -> str:
    """Re-export `utcnow_iso` so callers don't need to thread the import."""
    return utcnow_iso()
