"""state_ledger — outcome ledger contracts.

Pins architecture §"Outcomes". Seven contracts:

  (a) Append-only round-trip — write N records, read them back equal.

  (b) Authority combination: user > mechanical > LLM, conflict flag fires
      on user/mechanical disagreement, conservative `not_satisfied` result.

  (c) `user_override=True` forces verdict=satisfied with verdict_source=
      `user_override` regardless of other signals.

  (d) Roll-up rule: any not_satisfied → not_satisfied; all satisfied →
      satisfied; otherwise unknown.

  (e) Validation rejects unknown enum values + bad attribution tiers.

  (f) Filtering by intent_id and `since` timestamp returns expected subset.

  (g) Truncated trailing line is silently dropped — never raises.

Hermetic — temp dir, no LLM, no network.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def _check_round_trip() -> int:
    from resonance_lattice.state import (
        Attribution,
        CriterionCheck,
        OutcomeLedger,
        OutcomeRecord,
        Signal,
    )
    from resonance_lattice.state.ledger import now_iso

    with tempfile.TemporaryDirectory() as td:
        ledger = OutcomeLedger(Path(td))
        record = OutcomeRecord(
            intent_id="01HZTASK1",
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
            roll_up_verdict="satisfied",
            attribution=[
                Attribution(
                    row_id="01HZROW1",
                    tier="primary",
                    recall_rank=0,
                    cosine=0.92,
                    alignment=0.85,
                ),
            ],
            resolved_at=now_iso(),
        )
        ledger.write(record)
        loaded = ledger.read()
    if len(loaded) != 1:
        print(f"[state_ledger] FAIL (a): rows={len(loaded)}", file=sys.stderr)
        return 1
    got = loaded[0]
    if (got.intent_id != "01HZTASK1"
            or got.roll_up_verdict != "satisfied"
            or got.criterion_checks[0].verdict != "satisfied"
            or got.attribution[0].tier != "primary"):
        print(f"[state_ledger] FAIL (a): record drifted: {got!r}",
              file=sys.stderr)
        return 1
    print("[state_ledger] (a) round-trip OK", file=sys.stderr)
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
        print(f"[state_ledger] FAIL (b.1): mech>llm: v={v!r} conflict={conflict}",
              file=sys.stderr)
        return 1
    # user=not_satisfied + mechanical=satisfied → not_satisfied (user wins)
    v, _, conflict, _ = combine_signals([
        Signal(source="user", value={"verdict": "not_satisfied"}, timestamp=now),
        Signal(source="mechanical", value={"verdict": "satisfied"}, timestamp=now),
    ])
    if v != "not_satisfied" or not conflict:
        print(f"[state_ledger] FAIL (b.2): user>mech: v={v!r} conflict={conflict}",
              file=sys.stderr)
        return 1
    # LLM-only → use it, low confidence
    v, conf, _, _ = combine_signals([
        Signal(source="llm", value={"verdict": "satisfied"}, timestamp=now),
    ])
    if v != "satisfied" or conf != "low":
        print(f"[state_ledger] FAIL (b.3): llm-only: v={v!r} conf={conf!r}",
              file=sys.stderr)
        return 1
    # Empty → unknown
    v, _, _, _ = combine_signals([])
    if v != "unknown":
        print(f"[state_ledger] FAIL (b.4): empty: v={v!r}", file=sys.stderr)
        return 1
    print("[state_ledger] (b) authority combination OK", file=sys.stderr)
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
        print(f"[state_ledger] FAIL (c): override: v={v!r} src={source!r} "
              f"conflict={conflict}", file=sys.stderr)
        return 1
    if conf != "high":
        print(f"[state_ledger] FAIL (c): override conf={conf!r}",
              file=sys.stderr)
        return 1
    print("[state_ledger] (c) user_override OK", file=sys.stderr)
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
            print(f"[state_ledger] FAIL (d): {[c.verdict for c in checks]} → "
                  f"{got!r} (want {want!r})", file=sys.stderr)
            return 1
    print("[state_ledger] (d) roll-up rule OK", file=sys.stderr)
    return 0


def _check_validation() -> int:
    from resonance_lattice.state import (
        Attribution,
        CriterionCheck,
        OutcomeLedger,
        OutcomeRecord,
    )
    from resonance_lattice.state.ledger import now_iso

    with tempfile.TemporaryDirectory() as td:
        ledger = OutcomeLedger(Path(td))
        # Bad verdict.
        bad_cc = CriterionCheck(
            criterion_text="x", measure="user_confirms",
            verdict="oopsie",  # type: ignore[arg-type]
        )
        rec = OutcomeRecord(
            intent_id="x", intent_level="task",
            criterion_checks=[bad_cc],
            roll_up_verdict="satisfied",
            attribution=[],
            resolved_at=now_iso(),
        )
        try:
            ledger.write(rec)
        except ValueError:
            pass
        else:
            print("[state_ledger] FAIL (e): bad verdict accepted",
                  file=sys.stderr)
            return 1
        # Bad attribution tier.
        rec2 = OutcomeRecord(
            intent_id="x", intent_level="task",
            criterion_checks=[],
            roll_up_verdict="unknown",
            attribution=[Attribution(row_id="r", tier="hero")],  # type: ignore[arg-type]
            resolved_at=now_iso(),
        )
        try:
            ledger.write(rec2)
        except ValueError:
            pass
        else:
            print("[state_ledger] FAIL (e): bad tier accepted", file=sys.stderr)
            return 1
    print("[state_ledger] (e) validation rejects bad enums OK", file=sys.stderr)
    return 0


def _check_filtering() -> int:
    from resonance_lattice.state import OutcomeLedger, OutcomeRecord

    with tempfile.TemporaryDirectory() as td:
        ledger = OutcomeLedger(Path(td))
        for i, intent_id in enumerate(["t1", "t1", "t2"]):
            ledger.write(OutcomeRecord(
                intent_id=intent_id,
                intent_level="task",
                criterion_checks=[],
                roll_up_verdict="unknown",
                attribution=[],
                resolved_at=f"2026-05-{i + 1:02}T00:00:00Z",
            ))
        only_t1 = ledger.read(intent_id="t1")
        since = ledger.read(since="2026-05-02T00:00:00Z")
    if [r.intent_id for r in only_t1] != ["t1", "t1"]:
        print(f"[state_ledger] FAIL (f): intent filter: "
              f"{[r.intent_id for r in only_t1]!r}", file=sys.stderr)
        return 1
    if [r.resolved_at for r in since] != [
        "2026-05-02T00:00:00Z", "2026-05-03T00:00:00Z",
    ]:
        print(f"[state_ledger] FAIL (f): since filter: "
              f"{[r.resolved_at for r in since]!r}", file=sys.stderr)
        return 1
    print("[state_ledger] (f) filtering OK", file=sys.stderr)
    return 0


def _check_truncated_trailing_line() -> int:
    from resonance_lattice.state import OutcomeLedger, OutcomeRecord
    from resonance_lattice.state.ledger import OUTCOMES_FILE, ledger_dir

    with tempfile.TemporaryDirectory() as td:
        ledger = OutcomeLedger(Path(td))
        ledger.write(OutcomeRecord(
            intent_id="t",
            intent_level="task",
            criterion_checks=[],
            roll_up_verdict="unknown",
            attribution=[],
            resolved_at="2026-05-07T00:00:00Z",
        ))
        # Append a corrupt line directly — simulating a kill mid-write.
        path = ledger_dir(Path(td)) / OUTCOMES_FILE
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"intent_id": "broken", "incomplete')
        loaded = ledger.read()
    if len(loaded) != 1 or loaded[0].intent_id != "t":
        print(f"[state_ledger] FAIL (g): truncated drop wrong: "
              f"{[r.intent_id for r in loaded]!r}", file=sys.stderr)
        return 1
    print("[state_ledger] (g) truncated trailing line dropped OK",
          file=sys.stderr)
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
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[state_ledger] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
