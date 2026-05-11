"""memory_v22_confidence — confidence raising contracts.

Pins architecture §"Calibration mechanisms — how rows earn back trust",
mechanisms 1 (outcome corroboration) and 5 (cross-domain accumulation).
Seven contracts:

  (a) 2 wins raise low → medium.

  (b) 3 wins raise medium → high.

  (c) Non-principle row caps at high — verified requires principle level
      AND cross-domain (mechanism 5).

  (d) Principle with 5 wins across ≥2 intent_kinds reaches verified.

  (e) Principle with 5 wins in only 1 intent_kind caps at high.

  (f) Symmetric drop — 2 net losses pull confidence down to low.

  (g) Incidental tier doesn't count — primary/secondary only.

Hermetic — synthetic outcomes + temp memory store.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from resonance_lattice.memory.confidence import (
    raise_confidence_pass,
    target_confidence,
)
from resonance_lattice.memory.store import Memory, Row
from resonance_lattice.state import (
    Attribution,
    CriterionCheck,
    OutcomeRecord,
)


def _row(
    *,
    row_id: str = "01HZ_R1",
    level: str = "pattern",
    confidence: str = "low",
) -> Row:
    return Row(
        row_id=row_id,
        text="row",
        polarity=["factual", "workspace:abc123"],
        recurrence_count=2,
        created_at="2026-05-01T00:00:00Z",
        last_corroborated_at="2026-05-01T00:00:00Z",
        transcript_hash="manual",
        is_bad=False,
        level=level,
        confidence=confidence,
        origin="distilled",
    )


def _outcome(
    *,
    row_id: str,
    verdict: str,
    intent_kind: str = "implement",
    tier: str = "primary",
) -> OutcomeRecord:
    return OutcomeRecord(
        intent_id=f"01HZ_INTENT_{verdict}",
        intent_level="task",
        criterion_checks=[CriterionCheck(
            criterion_text="x", measure="user_confirms", verdict=verdict,
        )],
        roll_up_verdict=verdict,
        attribution=[Attribution(row_id=row_id, tier=tier)],
        resolved_at="2026-05-07T00:00:00Z",
        intent_kind=intent_kind,
    )


def _check_2_wins_low_to_medium() -> int:
    row = _row(level="pattern", confidence="low")
    outcomes = [_outcome(row_id="01HZ_R1", verdict="satisfied") for _ in range(2)]
    target = target_confidence(row, outcomes)
    if target != "medium":
        print(f"[memory_v22_confidence] FAIL (a): target={target!r}",
              file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (a) 2 wins → medium OK", file=sys.stderr)
    return 0


def _check_3_wins_medium_to_high() -> int:
    row = _row(level="pattern", confidence="medium")
    outcomes = [_outcome(row_id="01HZ_R1", verdict="satisfied") for _ in range(3)]
    target = target_confidence(row, outcomes)
    if target != "high":
        print(f"[memory_v22_confidence] FAIL (b): target={target!r}",
              file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (b) 3 wins → high OK", file=sys.stderr)
    return 0


def _check_non_principle_caps_at_high() -> int:
    row = _row(level="pattern", confidence="high")
    outcomes = [
        _outcome(row_id="01HZ_R1", verdict="satisfied", intent_kind=k)
        for k in ["debug", "design", "implement", "review", "explain"]
    ]
    target = target_confidence(row, outcomes)
    # Pattern-level row should NOT be promoted to verified, cross-domain
    # or otherwise — only principles can hit verified.
    if target == "verified":
        print(f"[memory_v22_confidence] FAIL (c): pattern hit verified",
              file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (c) non-principle caps at high OK",
          file=sys.stderr)
    return 0


def _check_principle_cross_domain_to_verified() -> int:
    row = _row(level="principle", confidence="high")
    outcomes = (
        [_outcome(row_id="01HZ_R1", verdict="satisfied", intent_kind="debug")
         for _ in range(3)]
        + [_outcome(row_id="01HZ_R1", verdict="satisfied", intent_kind="design")
           for _ in range(2)]
    )
    target = target_confidence(row, outcomes)
    if target != "verified":
        print(f"[memory_v22_confidence] FAIL (d): target={target!r}",
              file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (d) principle cross-domain → verified OK",
          file=sys.stderr)
    return 0


def _check_principle_single_domain_caps_at_high() -> int:
    row = _row(level="principle", confidence="high")
    outcomes = [
        _outcome(row_id="01HZ_R1", verdict="satisfied", intent_kind="debug")
        for _ in range(5)
    ]
    target = target_confidence(row, outcomes)
    if target == "verified":
        print(f"[memory_v22_confidence] FAIL (e): single-domain principle "
              f"hit verified", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (e) single-domain principle caps OK",
          file=sys.stderr)
    return 0


def _check_symmetric_drop_to_low() -> int:
    row = _row(level="pattern", confidence="medium")
    # 2 net losses (3 losses, 1 win) → low
    outcomes = (
        [_outcome(row_id="01HZ_R1", verdict="not_satisfied") for _ in range(3)]
        + [_outcome(row_id="01HZ_R1", verdict="satisfied")]
    )
    target = target_confidence(row, outcomes)
    if target != "low":
        print(f"[memory_v22_confidence] FAIL (f): target={target!r}",
              file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (f) symmetric drop to low OK",
          file=sys.stderr)
    return 0


def _check_incidental_tier_excluded() -> int:
    row = _row(level="pattern", confidence="low")
    outcomes = [
        _outcome(row_id="01HZ_R1", verdict="satisfied", tier="incidental")
        for _ in range(10)
    ]
    target = target_confidence(row, outcomes)
    # Incidental shouldn't credit anything — row stays at low (target=None).
    if target is not None:
        print(f"[memory_v22_confidence] FAIL (g): incidental credited; "
              f"target={target!r}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (g) incidental tier excluded OK",
          file=sys.stderr)
    return 0


def _check_end_to_end_pass() -> int:
    """raise_confidence_pass actually mutates the store."""
    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u")
        row_id = memory.add_row(
            text="distilled pattern",
            polarity=["factual", "workspace:abc123"],
            transcript_hash="distilled:x",
            embedding=np.zeros(768, dtype=np.float32),
            level="pattern",
            confidence="low",
            origin="distilled",
        )
        outcomes = [
            _outcome(row_id=row_id, verdict="satisfied") for _ in range(2)
        ]
        changes = raise_confidence_pass(memory, outcomes=outcomes)
        rows, _ = memory.read_all()
    if len(changes) != 1:
        print(f"[memory_v22_confidence] FAIL (h): changes={len(changes)}",
              file=sys.stderr)
        return 1
    if rows[0].confidence != "medium":
        print(f"[memory_v22_confidence] FAIL (h): confidence="
              f"{rows[0].confidence!r}", file=sys.stderr)
        return 1
    print("[memory_v22_confidence] (h) end-to-end pass OK", file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_2_wins_low_to_medium,
        _check_3_wins_medium_to_high,
        _check_non_principle_caps_at_high,
        _check_principle_cross_domain_to_verified,
        _check_principle_single_domain_caps_at_high,
        _check_symmetric_drop_to_low,
        _check_incidental_tier_excluded,
        _check_end_to_end_pass,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v22_confidence] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
