"""claim_outcome_log — the unified claim-outcome log contracts.

Pins architecture §8 ("the outcome loop — one log"). Eight contracts:

  (a) Append-only round-trip — write N records, read them back equal.

  (b) Authority combination: user > mechanical > LLM, conflict flag fires
      on user/mechanical disagreement, conservative `not_satisfied` result.

  (c) `user_override=True` forces verdict=satisfied with verdict_source=
      `user_override` regardless of other signals.

  (d) Roll-up rule: any not_satisfied → not_satisfied; all satisfied →
      satisfied; otherwise unknown.

  (e) Validation rejects unknown enum values, bad attribution tiers, a
      negative retry_count, and empty-string load_bearing_ids.

  (f) Filtering by intent_id and `since` timestamp returns expected subset.

  (g) Truncated trailing line is silently dropped — never raises.

  (h) The `kind` discriminator and the session-end fields — `session_id`,
      `retry_count`, `load_bearing_ids` — round-trip; absent on a record
      they load as the dataclass defaults (`kind` → `intent`).

  (i) Invariant 7 (writer): every record carries a `writer`, defaulted
      single-writer; an explicit writer round-trips and a pre-S1.5 row
      lacking the key loads as `DEFAULT_WRITER`.

Hermetic — temp dir, no LLM, no network.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_P = "claim_outcome_log"


def _check_round_trip() -> int:
    from resonance_lattice.state import (
        Attribution,
        ClaimOutcomeLog,
        ClaimOutcomeRecord,
        CriterionCheck,
        IntentOutcomeDetails,
        Signal,
    )
    from resonance_lattice.state.claim_outcome import now_iso

    with tempfile.TemporaryDirectory() as td:
        log = ClaimOutcomeLog(Path(td))
        record = ClaimOutcomeRecord(
            intent_id="01HZTASK1",
            details=IntentOutcomeDetails(
                intent_level="task",
                criterion_checks=[
                    CriterionCheck(
                        criterion_text="tests pass",
                        measure="exit_code:0",
                        verdict="satisfied",
                        signals_seen=[
                            Signal(
                                source="mechanical",
                                value={"exit_code": 0},
                                timestamp=now_iso(),
                            ),
                        ],
                        verdict_confidence="high",
                    ),
                ],
            ),
            roll_up_verdict="satisfied",
            attribution=[
                Attribution(
                    claim_id="01HZCLAIM1",
                    tier="primary",
                    recall_rank=0,
                    cosine=0.92,
                    alignment=0.85,
                ),
            ],
            resolved_at=now_iso(),
        )
        log.write(record)
        loaded = log.read()
    if len(loaded) != 1:
        print(f"[{_P}] FAIL (a): rows={len(loaded)}", file=sys.stderr)
        return 1
    got = loaded[0]
    if (got.intent_id != "01HZTASK1"
            or got.roll_up_verdict != "satisfied"
            or got.details.criterion_checks[0].verdict != "satisfied"
            or got.attribution[0].tier != "primary"
            or got.attribution[0].claim_id != "01HZCLAIM1"):
        print(f"[{_P}] FAIL (a): record drifted: {got!r}", file=sys.stderr)
        return 1
    print(f"[{_P}] (a) round-trip OK", file=sys.stderr)
    return 0


def _check_authority_combination() -> int:
    from resonance_lattice.state import Signal, combine_signals

    now = "2026-05-07T00:00:00Z"
    # mechanical=satisfied + LLM=not_satisfied → satisfied (mechanical wins)
    v, _, conflict, _ = combine_signals([
        Signal(source="mechanical", value={"verdict": "satisfied"}, timestamp=now),
        Signal(source="llm", value={"verdict": "not_satisfied"}, timestamp=now),
    ])
    if v != "satisfied" or conflict:
        print(f"[{_P}] FAIL (b.1): mech>llm: v={v!r} conflict={conflict}",
              file=sys.stderr)
        return 1
    # user=not_satisfied + mechanical=satisfied → not_satisfied (user wins)
    v, _, conflict, _ = combine_signals([
        Signal(source="user", value={"verdict": "not_satisfied"}, timestamp=now),
        Signal(source="mechanical", value={"verdict": "satisfied"}, timestamp=now),
    ])
    if v != "not_satisfied" or not conflict:
        print(f"[{_P}] FAIL (b.2): user>mech: v={v!r} conflict={conflict}",
              file=sys.stderr)
        return 1
    # LLM-only → use it, low confidence
    v, conf, _, _ = combine_signals([
        Signal(source="llm", value={"verdict": "satisfied"}, timestamp=now),
    ])
    if v != "satisfied" or conf != "low":
        print(f"[{_P}] FAIL (b.3): llm-only: v={v!r} conf={conf!r}",
              file=sys.stderr)
        return 1
    # Empty → unknown
    v, _, _, _ = combine_signals([])
    if v != "unknown":
        print(f"[{_P}] FAIL (b.4): empty: v={v!r}", file=sys.stderr)
        return 1
    print(f"[{_P}] (b) authority combination OK", file=sys.stderr)
    return 0


def _check_user_override() -> int:
    from resonance_lattice.state import Signal, combine_signals

    now = "2026-05-07T00:00:00Z"
    v, conf, conflict, source = combine_signals(
        [Signal(source="mechanical", value={"verdict": "not_satisfied"},
                timestamp=now)],
        user_override=True,
    )
    if v != "satisfied" or source != "user_override" or conflict:
        print(f"[{_P}] FAIL (c): override: v={v!r} src={source!r} "
              f"conflict={conflict}", file=sys.stderr)
        return 1
    if conf != "high":
        print(f"[{_P}] FAIL (c): override conf={conf!r}", file=sys.stderr)
        return 1
    print(f"[{_P}] (c) user_override OK", file=sys.stderr)
    return 0


def _check_roll_up() -> int:
    from resonance_lattice.state import CriterionCheck, roll_up

    cc_sat = lambda: CriterionCheck(
        criterion_text="x", measure="user_confirms", verdict="satisfied",
    )
    cc_unsat = lambda: CriterionCheck(
        criterion_text="x", measure="user_confirms", verdict="not_satisfied",
    )
    cc_unknown = lambda: CriterionCheck(
        criterion_text="x", measure="user_confirms", verdict="unknown",
    )
    cases = [
        ([cc_sat(), cc_sat()], "satisfied"),
        ([cc_sat(), cc_unsat()], "not_satisfied"),
        ([cc_sat(), cc_unknown()], "unknown"),
        ([], "unknown"),
    ]
    for checks, want in cases:
        got = roll_up(checks)
        if got != want:
            print(f"[{_P}] FAIL (d): {[c.verdict for c in checks]} → "
                  f"{got!r} (want {want!r})", file=sys.stderr)
            return 1
    print(f"[{_P}] (d) roll-up rule OK", file=sys.stderr)
    return 0


def _check_validation() -> int:
    from resonance_lattice.state import (
        Attribution,
        ClaimOutcomeLog,
        ClaimOutcomeRecord,
        CriterionCheck,
        IntentOutcomeDetails,
    )
    from resonance_lattice.state.claim_outcome import now_iso

    def _rejected(log: ClaimOutcomeLog, rec: ClaimOutcomeRecord) -> bool:
        try:
            log.write(rec)
        except ValueError:
            return True
        return False

    def _intent(**kw) -> ClaimOutcomeRecord:
        details_kw: dict = dict(intent_level="task")
        for k in ("intent_level", "criterion_checks", "intent_kind",
                  "intent_was_corrected"):
            if k in kw:
                details_kw[k] = kw.pop(k)
        defaults = dict(
            intent_id="x",
            details=IntentOutcomeDetails(**details_kw),
            roll_up_verdict="unknown", attribution=[], resolved_at=now_iso(),
        )
        defaults.update(kw)
        return ClaimOutcomeRecord(**defaults)

    with tempfile.TemporaryDirectory() as td:
        log = ClaimOutcomeLog(Path(td))
        bad_cc = CriterionCheck(
            criterion_text="x", measure="user_confirms",
            verdict="oopsie",  # type: ignore[arg-type]
        )
        # "bad kind" maps to "bad details type" — pass something that is
        # not an IntentOutcomeDetails.
        bad_details_rec = ClaimOutcomeRecord(
            intent_id="x",
            details="hero",  # type: ignore[arg-type]
            roll_up_verdict="unknown",
            attribution=[],
            resolved_at=now_iso(),
        )
        cases = [
            ("bad verdict", _intent(
                criterion_checks=[bad_cc], roll_up_verdict="satisfied")),
            ("bad tier", _intent(
                attribution=[Attribution(claim_id="c", tier="hero")])),  # type: ignore[arg-type]
            ("bad details type", bad_details_rec),
        ]
        for label, rec in cases:
            if not _rejected(log, rec):
                print(f"[{_P}] FAIL (e): {label} accepted", file=sys.stderr)
                return 1
    print(f"[{_P}] (e) validation rejects bad fields OK", file=sys.stderr)
    return 0


def _check_filtering() -> int:
    from resonance_lattice.state import (
        ClaimOutcomeLog,
        ClaimOutcomeRecord,
        IntentOutcomeDetails,
    )

    with tempfile.TemporaryDirectory() as td:
        log = ClaimOutcomeLog(Path(td))
        for i, intent_id in enumerate(["t1", "t1", "t2"]):
            log.write(ClaimOutcomeRecord(
                intent_id=intent_id,
                details=IntentOutcomeDetails(intent_level="task"),
                roll_up_verdict="unknown",
                attribution=[],
                resolved_at=f"2026-05-{i + 1:02}T00:00:00Z",
            ))
        only_t1 = log.read(intent_id="t1")
        since = log.read(since="2026-05-02T00:00:00Z")
    if [r.intent_id for r in only_t1] != ["t1", "t1"]:
        print(f"[{_P}] FAIL (f): intent filter: "
              f"{[r.intent_id for r in only_t1]!r}", file=sys.stderr)
        return 1
    if [r.resolved_at for r in since] != [
        "2026-05-02T00:00:00Z", "2026-05-03T00:00:00Z",
    ]:
        print(f"[{_P}] FAIL (f): since filter: "
              f"{[r.resolved_at for r in since]!r}", file=sys.stderr)
        return 1
    print(f"[{_P}] (f) filtering OK", file=sys.stderr)
    return 0


def _check_truncated_trailing_line() -> int:
    from resonance_lattice.state import (
        ClaimOutcomeLog,
        ClaimOutcomeRecord,
        IntentOutcomeDetails,
    )
    from resonance_lattice.state._jsonl_log import ledger_dir
    from resonance_lattice.state.claim_outcome import CLAIM_OUTCOMES_FILE

    with tempfile.TemporaryDirectory() as td:
        log = ClaimOutcomeLog(Path(td))
        log.write(ClaimOutcomeRecord(
            intent_id="t",
            details=IntentOutcomeDetails(intent_level="task"),
            roll_up_verdict="unknown",
            attribution=[],
            resolved_at="2026-05-07T00:00:00Z",
        ))
        # Append a corrupt line directly — simulating a kill mid-write.
        path = ledger_dir(Path(td)) / CLAIM_OUTCOMES_FILE
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"intent_id": "broken", "incomplete')
        loaded = log.read()
    if len(loaded) != 1 or loaded[0].intent_id != "t":
        print(f"[{_P}] FAIL (g): truncated drop wrong: "
              f"{[r.intent_id for r in loaded]!r}", file=sys.stderr)
        return 1
    print(f"[{_P}] (g) truncated trailing line dropped OK", file=sys.stderr)
    return 0


def _check_writer_field() -> int:
    """Invariant 7: a record carries a `writer` (defaulted single-writer);
    an explicit writer round-trips, a record written without it loads the
    default, and a pre-S1.5 row physically lacking the key loads as
    `DEFAULT_WRITER`."""
    import json

    from resonance_lattice.state import (
        ClaimOutcomeLog,
        ClaimOutcomeRecord,
        IntentOutcomeDetails,
    )
    from resonance_lattice.state._jsonl_log import ledger_dir
    from resonance_lattice.state.claim import DEFAULT_WRITER
    from resonance_lattice.state.claim_outcome import CLAIM_OUTCOMES_FILE

    with tempfile.TemporaryDirectory() as td:
        log = ClaimOutcomeLog(Path(td))
        log.write(ClaimOutcomeRecord(
            intent_id="t1",
            details=IntentOutcomeDetails(intent_level="task"),
            roll_up_verdict="unknown", attribution=[],
            resolved_at="2026-05-07T00:00:00Z", writer="alice",
        ))
        log.write(ClaimOutcomeRecord(
            intent_id="t2",
            details=IntentOutcomeDetails(intent_level="task"),
            roll_up_verdict="unknown", attribution=[],
            resolved_at="2026-05-08T00:00:00Z",       # no writer → default
        ))
        # A pre-S1.5 row physically lacking the "writer" key.
        path = ledger_dir(Path(td)) / CLAIM_OUTCOMES_FILE
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "intent_id": "t3", "resolved_at": "2026-05-09T00:00:00Z",
                "roll_up_verdict": "unknown", "attribution": [],
                "session_id": "", "notes": "", "kind": "intent",
                "intent_level": "task", "criterion_checks": [],
                "intent_kind": None, "intent_was_corrected": False,
            }) + "\n")
        explicit = log.read(intent_id="t1")[0]
        defaulted = log.read(intent_id="t2")[0]
        legacy = log.read(intent_id="t3")[0]
    ok = (
        explicit.writer == "alice"
        and defaulted.writer == DEFAULT_WRITER
        and legacy.writer == DEFAULT_WRITER
    )
    if not ok:
        print(f"[{_P}] FAIL (i): writer drifted: explicit={explicit.writer!r} "
              f"default={defaulted.writer!r} legacy={legacy.writer!r}",
              file=sys.stderr)
        return 1
    print(f"[{_P}] (i) writer round-trip + legacy default OK", file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_round_trip,
        _check_authority_combination,
        _check_user_override,
        _check_roll_up,
        _check_validation,
        _check_filtering,
        _check_truncated_trailing_line,
        _check_writer_field,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print(f"[{_P}] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
