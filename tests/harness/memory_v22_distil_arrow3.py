"""memory_v22_distil_arrow3 — learning→principle generalisation contracts.

Pins architecture §"Distil Arrow Generalise" + §"New-principle protection
window". Eight contracts:

  (a) Cross-domain discovery — a learning with successful attributions
      across ≥2 distinct intent_kinds surfaces; single-domain doesn't.

  (b) Already-promoted learnings skipped — learnings that have a
      confident principle child don't re-promote.

  (c) Promote() refuses on `promote: false`; no principle written.

  (d) Promote() rejects on cosine misalignment with parent learning.

  (e) Promote() rejects when principle text is NOT shorter than the
      learning (architecture's "shorter than the learning" rule).

  (f) Promote() emits a row payload with level=principle, parent_ids
      pointing at the learning, confidence diluted one step.

  (g) New-principle protection window — a low-confidence principle
      within 30 days of created_at gets confidence_floor as if medium
      (architecture §"New-principle protection window").

  (h) Outside the protection window the principle floors normally
      (low → 0.9 for principle level; not lifted to medium).

Hermetic — synthetic rows + fake LLM client; no encoder, no network.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from collections import namedtuple

import numpy as np

from resonance_lattice.memory.distil_arrow3 import (
    PrincipleCandidate,
    find_promotion_candidates,
    promote,
)
from resonance_lattice.memory.rerank import (
    _within_new_principle_protection,
    confidence_floor,
)
from resonance_lattice.memory.store import Row
from resonance_lattice.state import (
    Attribution,
    CriterionCheck,
    OutcomeRecord,
)

LLMResponse = namedtuple("LLMResponse", "text input_tokens output_tokens")


def _row(
    *,
    row_id: str = "01HZ_L1",
    text: str = "always log reasoning before commits at the boundary layer",
    level: str = "learning",
    confidence: str = "medium",
    parent_ids: list[str] | None = None,
    created_at: str = "2026-05-01T00:00:00Z",
) -> Row:
    return Row(
        row_id=row_id,
        text=text,
        polarity=["factual", "workspace:abc123"],
        recurrence_count=3,
        created_at=created_at,
        last_corroborated_at=created_at,
        transcript_hash="distilled:x",
        is_bad=False,
        level=level,
        confidence=confidence,
        parent_ids=parent_ids or [],
        origin="distilled",
    )


def _outcome(
    *, row_id: str, intent_kind: str, verdict: str = "satisfied",
    tier: str = "primary",
) -> OutcomeRecord:
    return OutcomeRecord(
        intent_id="t",
        intent_level="task",
        criterion_checks=[CriterionCheck(
            criterion_text="x", measure="user_confirms", verdict=verdict,
        )],
        roll_up_verdict=verdict,
        attribution=[Attribution(row_id=row_id, tier=tier)],
        resolved_at="2026-05-07T00:00:00Z",
        intent_kind=intent_kind,
    )


def _unit(*coords: float) -> np.ndarray:
    v = np.zeros(768, dtype=np.float32)
    for i, c in enumerate(coords):
        v[i] = c
    n = np.linalg.norm(v)
    return v / (n if n else 1.0)


def _check_cross_domain_discovery() -> int:
    learning = _row(row_id="01HZ_L_OK")
    single_domain = _row(row_id="01HZ_L_SINGLE")
    rows = [learning, single_domain]
    band = np.stack([_unit(1.0), _unit(0.0, 1.0)]).astype(np.float32)
    outcomes = [
        _outcome(row_id="01HZ_L_OK", intent_kind="debug"),
        _outcome(row_id="01HZ_L_OK", intent_kind="design"),
        _outcome(row_id="01HZ_L_SINGLE", intent_kind="debug"),
        _outcome(row_id="01HZ_L_SINGLE", intent_kind="debug"),
    ]
    candidates = find_promotion_candidates(rows, band, outcomes=outcomes)
    if len(candidates) != 1 or candidates[0].learning_row.row_id != "01HZ_L_OK":
        print(f"[memory_v22_distil_arrow3] FAIL (a): {candidates!r}",
              file=sys.stderr)
        return 1
    if candidates[0].intent_kinds_with_success != {"debug", "design"}:
        print(f"[memory_v22_distil_arrow3] FAIL (a): kinds="
              f"{candidates[0].intent_kinds_with_success!r}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow3] (a) cross-domain discovery OK",
          file=sys.stderr)
    return 0


def _check_already_promoted_skipped() -> int:
    learning = _row(row_id="01HZ_L_DONE")
    principle = _row(
        row_id="01HZ_P", level="principle", confidence="high",
        parent_ids=["01HZ_L_DONE"],
    )
    rows = [learning, principle]
    band = np.stack([_unit(1.0), _unit(0.5, 0.5)]).astype(np.float32)
    outcomes = [
        _outcome(row_id="01HZ_L_DONE", intent_kind="debug"),
        _outcome(row_id="01HZ_L_DONE", intent_kind="design"),
    ]
    candidates = find_promotion_candidates(rows, band, outcomes=outcomes)
    if candidates:
        print(f"[memory_v22_distil_arrow3] FAIL (b): {len(candidates)}",
              file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow3] (b) already-promoted skipped OK",
          file=sys.stderr)
    return 0


class _ZeroEncoder:
    revision = "test"

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 768), dtype=np.float32)


def _check_refusal() -> int:
    candidate = PrincipleCandidate(
        learning_row=_row(),
        learning_embedding=_unit(1.0),
        intent_kinds_with_success={"debug", "design"},
        total_success_count=4,
    )
    refuse_llm = lambda system, msgs, tokens: LLMResponse(
        json.dumps({"promote": False, "reason": "too narrow to generalise"}),
        20, 8,
    )
    payload, rejection = promote(
        candidate, llm=refuse_llm, encoder=_ZeroEncoder(),
    )
    if payload is not None or "refused" not in (rejection or ""):
        print(f"[memory_v22_distil_arrow3] FAIL (c): payload={payload!r} "
              f"rejection={rejection!r}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow3] (c) refusal honoured OK",
          file=sys.stderr)
    return 0


class _OrthogonalEncoder:
    revision = "test"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 768), dtype=np.float32)
        out[:, 100] = 1.0  # orthogonal to candidate centroid (dim 0)
        return out


def _check_misalignment_rejected() -> int:
    candidate = PrincipleCandidate(
        learning_row=_row(),
        learning_embedding=_unit(1.0),
        intent_kinds_with_success={"debug", "design"},
        total_success_count=4,
    )
    promote_llm = lambda system, msgs, tokens: LLMResponse(
        json.dumps({
            "promote": True,
            "text": "make actions reversible by default",
            "polarity": "prefer",
        }),
        25, 12,
    )
    payload, rejection = promote(
        candidate, llm=promote_llm, encoder=_OrthogonalEncoder(),
    )
    if payload is not None or "alignment" not in (rejection or ""):
        print(f"[memory_v22_distil_arrow3] FAIL (d): payload={payload!r} "
              f"rejection={rejection!r}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow3] (d) misalignment rejected OK",
          file=sys.stderr)
    return 0


class _AlignedEncoder:
    revision = "test"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 768), dtype=np.float32)
        out[:, 0] = 1.0
        return out


def _check_principle_must_be_shorter() -> int:
    # Parent text has 8 words. A principle that's also 8 words must
    # be rejected — architecture says "shorter than the learning".
    parent = _row(text="one two three four five six seven eight")
    candidate = PrincipleCandidate(
        learning_row=parent,
        learning_embedding=_unit(1.0),
        intent_kinds_with_success={"debug", "design"},
        total_success_count=4,
    )
    promote_llm = lambda system, msgs, tokens: LLMResponse(
        json.dumps({
            "promote": True,
            "text": "alpha beta gamma delta epsilon zeta eta theta",
            "polarity": "prefer",
        }),
        20, 10,
    )
    payload, rejection = promote(
        candidate, llm=promote_llm, encoder=_AlignedEncoder(),
    )
    if payload is not None or "shorter" not in (rejection or ""):
        print(f"[memory_v22_distil_arrow3] FAIL (e): payload={payload!r} "
              f"rejection={rejection!r}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow3] (e) principle must be shorter OK",
          file=sys.stderr)
    return 0


def _check_promotion_payload_shape() -> int:
    parent = _row(
        text="prefer pytest -xvs when debugging single boundary failures",
        confidence="medium",
    )
    candidate = PrincipleCandidate(
        learning_row=parent,
        learning_embedding=_unit(1.0),
        intent_kinds_with_success={"debug", "design"},
        total_success_count=4,
    )
    promote_llm = lambda system, msgs, tokens: LLMResponse(
        json.dumps({
            "promote": True,
            "text": "match test mode to failure narrowness",
            "polarity": "prefer",
        }),
        50, 20,
    )
    payload, rejection = promote(
        candidate, llm=promote_llm, encoder=_AlignedEncoder(),
    )
    if payload is None:
        print(f"[memory_v22_distil_arrow3] FAIL (f): rejected: {rejection!r}",
              file=sys.stderr)
        return 1
    if (payload["level"] != "principle"
            or payload["parent_ids"] != [parent.row_id]
            or payload["confidence"] != "low"  # medium → low (one-step dilute)
            or payload["origin"] != "distilled"):
        print(f"[memory_v22_distil_arrow3] FAIL (f): bad payload: "
              f"{payload!r}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow3] (f) promotion payload shape OK",
          file=sys.stderr)
    return 0


_NOW = _dt.datetime(2026, 5, 8, tzinfo=_dt.timezone.utc)


def _check_protection_window_lifts_floor() -> int:
    """Within 30 days of created_at, a low-confidence principle's
    confidence_floor multiplier matches what `medium` would give."""
    fresh = _row(
        row_id="01HZ_FRESH", level="principle", confidence="low",
        # 5 days old — well inside the 30-day window.
        created_at=(_NOW - _dt.timedelta(days=5)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
    )
    if not _within_new_principle_protection(fresh, now=_NOW):
        print("[memory_v22_distil_arrow3] FAIL (g): not detected as fresh",
              file=sys.stderr)
        return 1
    floor = confidence_floor(fresh, now=_NOW)
    medium_principle_floor = 0.95  # _CONFIDENCE_FLOOR["medium"]["principle"]
    if abs(floor - medium_principle_floor) > 1e-9:
        print(f"[memory_v22_distil_arrow3] FAIL (g): floor={floor!r} "
              f"(want {medium_principle_floor!r})", file=sys.stderr)
        return 1
    print(f"[memory_v22_distil_arrow3] (g) protection window lifts floor "
          f"({floor:.3f}) OK", file=sys.stderr)
    return 0


def _check_outside_protection_window() -> int:
    """Outside the window the principle floors at its actual confidence."""
    aged = _row(
        row_id="01HZ_AGED", level="principle", confidence="low",
        created_at=(_NOW - _dt.timedelta(days=60)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
    )
    if _within_new_principle_protection(aged, now=_NOW):
        print("[memory_v22_distil_arrow3] FAIL (h): aged still protected",
              file=sys.stderr)
        return 1
    floor = confidence_floor(aged, now=_NOW)
    low_principle_floor = 0.9  # _CONFIDENCE_FLOOR["low"]["principle"]
    if abs(floor - low_principle_floor) > 1e-9:
        print(f"[memory_v22_distil_arrow3] FAIL (h): floor={floor!r} "
              f"(want {low_principle_floor!r})", file=sys.stderr)
        return 1
    print(f"[memory_v22_distil_arrow3] (h) outside-window floor "
          f"({floor:.3f}) OK", file=sys.stderr)
    return 0


def _check_sessions_arm_expires_protection() -> int:
    """Architecture §"New-principle protection window": grace period is
    "5 sessions or 30 days, whichever first." A principle that's 5 days
    old (well inside the 30-day clock) but has lived through 6 sessions
    must drop to its actual confidence floor.
    """
    fresh = _row(
        row_id="01HZ_BURN", level="principle", confidence="low",
        created_at=(_NOW - _dt.timedelta(days=5)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
    )
    # Within both arms — protected.
    if not _within_new_principle_protection(
        fresh, now=_NOW, sessions_since_created=3,
    ):
        print("[memory_v22_distil_arrow3] FAIL (k): 3-session principle "
              "should still be protected", file=sys.stderr)
        return 1
    # Past the session arm — protection drops even though clock hasn't.
    if _within_new_principle_protection(
        fresh, now=_NOW, sessions_since_created=6,
    ):
        print("[memory_v22_distil_arrow3] FAIL (k): 6-session principle "
              "should have lost protection (5-session arm)", file=sys.stderr)
        return 1
    # Floor reflects the loss.
    floor = confidence_floor(fresh, now=_NOW, sessions_since_created=6)
    low_principle_floor = 0.9
    if abs(floor - low_principle_floor) > 1e-9:
        print(f"[memory_v22_distil_arrow3] FAIL (k): floor={floor!r} "
              f"(want {low_principle_floor!r})", file=sys.stderr)
        return 1
    print(f"[memory_v22_distil_arrow3] (k) sessions arm expires protection "
          f"({floor:.3f}) OK", file=sys.stderr)
    return 0


def _check_only_satisfied_outcomes_count() -> int:
    """Cross-domain evidence means successful evidence — a learning that
    contributed to not_satisfied verdicts across two kinds isn't
    promotion-worthy. Pins distil_arrow3:_success_attributions_by_intent_kind.
    """
    learning = _row(row_id="01HZ_L_FAIL")
    rows = [learning]
    band = np.stack([_unit(1.0)]).astype(np.float32)
    outcomes = [
        _outcome(row_id="01HZ_L_FAIL", intent_kind="debug",
                 verdict="not_satisfied"),
        _outcome(row_id="01HZ_L_FAIL", intent_kind="design",
                 verdict="not_satisfied"),
    ]
    candidates = find_promotion_candidates(rows, band, outcomes=outcomes)
    if candidates:
        print(f"[memory_v22_distil_arrow3] FAIL (j): not_satisfied counted "
              f"as cross-domain evidence: {candidates!r}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow3] (j) only satisfied outcomes count OK",
          file=sys.stderr)
    return 0


def _check_hedge_phrase_rejected() -> int:
    """Architecture §"Distil — Arrow Generalise": post-LLM hedges must reject."""
    parent = _row(text="prefer composition over inheritance for testability gains")
    candidate = PrincipleCandidate(
        learning_row=parent,
        learning_embedding=_unit(1.0),
        intent_kinds_with_success={"debug", "design"},
        total_success_count=4,
    )
    promote_llm = lambda system, msgs, tokens: LLMResponse(
        json.dumps({
            "promote": True,
            "text": "abstractions might leak",
            "polarity": "prefer",
        }),
        20, 10,
    )
    payload, rejection = promote(
        candidate, llm=promote_llm, encoder=_AlignedEncoder(),
    )
    if payload is not None or "hedge" not in (rejection or "").lower():
        print(f"[memory_v22_distil_arrow3] FAIL (i): payload={payload!r} "
              f"rejection={rejection!r}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow3] (i) hedge phrase rejected OK",
          file=sys.stderr)
    return 0


def _check_cold_start_arrow3_gates() -> int:
    """(l) `cold_start_arrow3_gates(n_rows)` returns
    `(min_distinct_intent_kinds,)` below threshold; None at/above.
    Mirrors W2/W4 cold-start gate functions."""
    from resonance_lattice.memory.distil_arrow3 import (
        cold_start_arrow3_gates,
    )
    from resonance_lattice.memory.recall import COLD_START_ROW_THRESHOLD

    relaxed = cold_start_arrow3_gates(0)
    if relaxed != (1,):
        print(f"[memory_v22_distil_arrow3] FAIL (l): empty store gates="
              f"{relaxed!r}", file=sys.stderr)
        return 1
    if cold_start_arrow3_gates(COLD_START_ROW_THRESHOLD - 1) != (1,):
        print("[memory_v22_distil_arrow3] FAIL (l): just-below threshold "
              "should still relax", file=sys.stderr)
        return 1
    if cold_start_arrow3_gates(COLD_START_ROW_THRESHOLD) is not None:
        print("[memory_v22_distil_arrow3] FAIL (l): at-threshold should "
              "NOT relax (returns None)", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow3] (l) cold-start arrow3 gates OK",
          file=sys.stderr)
    return 0


def _check_cold_start_promotes_single_domain() -> int:
    """(m) When memory is sparse AND `auto_tune_cold_start=True`,
    `arrow3_pass` accepts a learning whose successful attributions span
    exactly 1 intent_kind that the default threshold (2) would reject.
    Single-domain workloads physically cannot meet the 2-kind bar.

    Inverse check: `auto_tune_cold_start=False` leaves the default
    trigger in place. Explicit caller override wins.
    """
    import tempfile
    from pathlib import Path

    from resonance_lattice.memory.distil_arrow3 import arrow3_pass
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u", encoder=_AlignedEncoder())
        # Plant a learning row with embedding on dim-0 (AlignedEncoder
        # outputs unit-on-dim-0 vectors so post-LLM cosine alignment
        # check passes).
        memory.add_row(
            text="prefer composition over inheritance for testability",
            polarity=["prefer"],
            transcript_hash="distilled:planted",
            embedding=_unit(1.0),
            level="learning",
            origin="distilled",
        )
        rows, _ = memory.read_all()
        learning_row_id = rows[0].row_id
        # Single-domain success attribution — only 'design'.
        outcomes = [_outcome(row_id=learning_row_id, intent_kind="design")]

        promote_llm = lambda system, msgs, tokens: LLMResponse(
            json.dumps({
                "promote": True,
                "text": "favour structure that supports change",
                "polarity": "prefer",
            }),
            25, 12,
        )
        result = arrow3_pass(
            memory, outcomes=outcomes, llm=promote_llm,
            encoder=_AlignedEncoder(), dry_run=True,
        )
        if result.candidates_found != 1:
            print(f"[memory_v22_distil_arrow3] FAIL (m.1): cold-start "
                  f"auto-tune should find 1 candidate from a single-domain "
                  f"learning; got candidates_found="
                  f"{result.candidates_found}", file=sys.stderr)
            return 1
        if len(result.promoted_row_ids) != 1:
            print(f"[memory_v22_distil_arrow3] FAIL (m.2): expected 1 "
                  f"promoted row; got {result.promoted_row_ids!r} "
                  f"rejections={result.rejections!r}", file=sys.stderr)
            return 1

        # Auto-tune disabled → default min_distinct_intent_kinds=2 → no candidate.
        result_off = arrow3_pass(
            memory, outcomes=outcomes, llm=promote_llm,
            encoder=_AlignedEncoder(), dry_run=True,
            auto_tune_cold_start=False,
        )
        if result_off.candidates_found != 0:
            print(f"[memory_v22_distil_arrow3] FAIL (m.3): default-gates "
                  f"pass should find 0 candidates with 1 kind; got "
                  f"{result_off.candidates_found}", file=sys.stderr)
            return 1

        # Explicit caller override wins.
        result_override = arrow3_pass(
            memory, outcomes=outcomes, llm=promote_llm,
            encoder=_AlignedEncoder(), dry_run=True,
            min_distinct_intent_kinds=2,
        )
        if result_override.candidates_found != 0:
            print(f"[memory_v22_distil_arrow3] FAIL (m.4): explicit "
                  f"min_distinct_intent_kinds=2 should override "
                  f"cold-start relax", file=sys.stderr)
            return 1

    print("[memory_v22_distil_arrow3] (m) cold-start auto-tune promotes "
          "single-domain learnings + override wins OK", file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_cross_domain_discovery,
        _check_already_promoted_skipped,
        _check_refusal,
        _check_misalignment_rejected,
        _check_principle_must_be_shorter,
        _check_promotion_payload_shape,
        _check_protection_window_lifts_floor,
        _check_outside_protection_window,
        _check_hedge_phrase_rejected,
        _check_only_satisfied_outcomes_count,
        _check_sessions_arm_expires_protection,
        _check_cold_start_arrow3_gates,
        _check_cold_start_promotes_single_domain,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v22_distil_arrow3] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
