"""memory_v22_distil_arrow2 — pattern→learning extraction contracts.

Pins architecture §"Distil Arrow Extract (pattern → learning)". Six
contracts:

  (a) Candidate discovery — a pattern with ≥2 attributed primary/
      secondary outcomes surfaces; a pattern with no attributions
      doesn't.

  (b) Already-promoted patterns skipped — patterns that have a confident
      learning child don't re-promote.

  (c) Promote() refuses on `promote: false`; no learning row written.

  (d) Promote() rejects on cosine misalignment with parent pattern.

  (e) Promote() emits a row payload with level=learning, parent_ids
      pointing at the pattern, and confidence diluted one step.

  (f) `incidental` tier attributions don't count toward the trigger.

Hermetic — synthetic rows + fake LLM client; no encoder, no network.
"""

from __future__ import annotations

import json
import sys
from collections import namedtuple

import numpy as np

from resonance_lattice.memory.distil_arrow2 import (
    LearningCandidate,
    _build_messages,
    find_promotion_candidates,
    promote,
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
    row_id: str = "01HZ_P1",
    level: str = "pattern",
    confidence: str = "medium",
    parent_ids: list[str] | None = None,
) -> Row:
    return Row(
        row_id=row_id,
        text=f"pattern {row_id}",
        polarity=["factual", "workspace:abc123"],
        recurrence_count=3,
        created_at="2026-05-01T00:00:00Z",
        last_corroborated_at="2026-05-01T00:00:00Z",
        transcript_hash="distilled:x",
        is_bad=False,
        level=level,
        confidence=confidence,
        parent_ids=parent_ids or [],
        origin="distilled",
    )


def _outcome(
    *, row_id: str, verdict: str, tier: str = "primary",
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
        intent_kind="implement",
    )


def _unit(*coords: float) -> np.ndarray:
    v = np.zeros(768, dtype=np.float32)
    for i, c in enumerate(coords):
        v[i] = c
    n = np.linalg.norm(v)
    return v / (n if n else 1.0)


def _check_candidate_discovery() -> int:
    pattern_a = _row(row_id="01HZ_P_A")
    pattern_b = _row(row_id="01HZ_P_B")
    rows = [pattern_a, pattern_b]
    band = np.stack([_unit(1.0), _unit(0.0, 1.0)]).astype(np.float32)
    outcomes = [
        _outcome(row_id="01HZ_P_A", verdict="satisfied"),
        _outcome(row_id="01HZ_P_A", verdict="not_satisfied"),
    ]
    candidates = find_promotion_candidates(rows, band, outcomes=outcomes)
    if len(candidates) != 1 or candidates[0].pattern_row.row_id != "01HZ_P_A":
        print(f"[memory_v22_distil_arrow2] FAIL (a): {candidates!r}",
              file=sys.stderr)
        return 1
    if (candidates[0].success_count != 1
            or candidates[0].failure_count != 1):
        print(f"[memory_v22_distil_arrow2] FAIL (a): bad counts: "
              f"{candidates[0]!r}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow2] (a) candidate discovery OK",
          file=sys.stderr)
    return 0


def _check_already_promoted_skipped() -> int:
    pattern = _row(row_id="01HZ_P", level="pattern")
    learning = _row(
        row_id="01HZ_L", level="learning", confidence="high",
        parent_ids=["01HZ_P"],
    )
    rows = [pattern, learning]
    band = np.stack([_unit(1.0), _unit(0.5, 0.5)]).astype(np.float32)
    outcomes = [
        _outcome(row_id="01HZ_P", verdict="satisfied"),
        _outcome(row_id="01HZ_P", verdict="satisfied"),
    ]
    candidates = find_promotion_candidates(rows, band, outcomes=outcomes)
    if candidates:
        print(f"[memory_v22_distil_arrow2] FAIL (b): {len(candidates)} "
              f"candidates surfaced", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow2] (b) already-promoted skipped OK",
          file=sys.stderr)
    return 0


class _ZeroEncoder:
    revision = "test"

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 768), dtype=np.float32)


def _check_refusal() -> int:
    candidate = LearningCandidate(
        pattern_row=_row(),
        pattern_embedding=_unit(1.0),
        success_count=2,
        failure_count=0,
    )
    refuse_llm = lambda system, msgs, tokens: LLMResponse(
        json.dumps({"promote": False, "reason": "evidence too thin"}),
        20, 8,
    )
    payload, rejection = promote(
        candidate, llm=refuse_llm, encoder=_ZeroEncoder(),
    )
    if payload is not None or "refused" not in (rejection or ""):
        print(f"[memory_v22_distil_arrow2] FAIL (c): payload={payload!r} "
              f"rejection={rejection!r}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow2] (c) refusal honoured OK",
          file=sys.stderr)
    return 0


class _OrthogonalEncoder:
    revision = "test"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 768), dtype=np.float32)
        out[:, 100] = 1.0  # orthogonal to candidate centroid (dim 0)
        return out


def _check_misalignment_rejected() -> int:
    candidate = LearningCandidate(
        pattern_row=_row(),
        pattern_embedding=_unit(1.0),
        success_count=2,
        failure_count=0,
    )
    promote_llm = lambda system, msgs, tokens: LLMResponse(
        json.dumps({
            "promote": True,
            "text": "always log reasoning before commits",
            "polarity": "prefer",
        }),
        30, 15,
    )
    payload, rejection = promote(
        candidate, llm=promote_llm, encoder=_OrthogonalEncoder(),
    )
    if payload is not None or "alignment" not in (rejection or ""):
        print(f"[memory_v22_distil_arrow2] FAIL (d): payload={payload!r} "
              f"rejection={rejection!r}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow2] (d) misalignment rejected OK",
          file=sys.stderr)
    return 0


class _AlignedEncoder:
    """Returns the same dim-0 vector the candidate centroid lives on."""
    revision = "test"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 768), dtype=np.float32)
        out[:, 0] = 1.0
        return out


def _check_promotion_payload_shape() -> int:
    parent = _row(confidence="medium")
    candidate = LearningCandidate(
        pattern_row=parent,
        pattern_embedding=_unit(1.0),
        success_count=2,
        failure_count=1,
    )
    promote_llm = lambda system, msgs, tokens: LLMResponse(
        json.dumps({
            "promote": True,
            "text": "prefer pytest -xvs when debugging single failures",
            "polarity": "prefer",
        }),
        50, 20,
    )
    payload, rejection = promote(
        candidate, llm=promote_llm, encoder=_AlignedEncoder(),
    )
    if payload is None:
        print(f"[memory_v22_distil_arrow2] FAIL (e): rejected: {rejection!r}",
              file=sys.stderr)
        return 1
    if (payload["level"] != "learning"
            or payload["parent_ids"] != [parent.row_id]
            or payload["confidence"] != "low"  # medium → low (one-step dilute)
            or payload["origin"] != "distilled"):
        print(f"[memory_v22_distil_arrow2] FAIL (e): bad payload: "
              f"{payload!r}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow2] (e) promotion payload shape OK",
          file=sys.stderr)
    return 0


def _check_incidental_excluded() -> int:
    pattern = _row(row_id="01HZ_P_INC")
    band = np.stack([_unit(1.0)]).astype(np.float32)
    outcomes = [
        _outcome(row_id="01HZ_P_INC", verdict="satisfied", tier="incidental")
        for _ in range(5)
    ]
    candidates = find_promotion_candidates(
        [pattern], band, outcomes=outcomes,
    )
    if candidates:
        print(f"[memory_v22_distil_arrow2] FAIL (f): incidental credited: "
              f"{len(candidates)}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow2] (f) incidental tier excluded OK",
          file=sys.stderr)
    return 0


def _check_outcome_citations_collected_and_built() -> int:
    """Architecture §"Distil — Arrow Extract": the LLM extracts a rule
    "with named conditions and cited outcomes". Pins that
    `find_promotion_candidates` collects per-attribution citations and
    that `_build_messages` surfaces them in the prompt body.
    """
    pattern = _row(row_id="01HZ_P_CITE")
    rows = [pattern]
    band = np.stack([_unit(1.0)]).astype(np.float32)
    outcomes = [
        OutcomeRecord(
            intent_id="t1", intent_level="task",
            criterion_checks=[CriterionCheck(
                criterion_text="tests pass after refactor",
                measure="user_confirms", verdict="satisfied",
            )],
            roll_up_verdict="satisfied",
            attribution=[Attribution(row_id="01HZ_P_CITE", tier="primary")],
            resolved_at="2026-05-07T00:00:00Z",
            intent_kind="refactor",
        ),
        OutcomeRecord(
            intent_id="t2", intent_level="task",
            criterion_checks=[CriterionCheck(
                criterion_text="user accepted explanation",
                measure="user_confirms", verdict="not_satisfied",
            )],
            roll_up_verdict="not_satisfied",
            attribution=[Attribution(row_id="01HZ_P_CITE", tier="primary")],
            resolved_at="2026-05-07T00:00:00Z",
            intent_kind="explain",
        ),
    ]
    candidates = find_promotion_candidates(rows, band, outcomes=outcomes)
    if len(candidates) != 1:
        print(f"[memory_v22_distil_arrow2] FAIL (h): {candidates!r}",
              file=sys.stderr)
        return 1
    cand = candidates[0]
    if (len(cand.success_citations) != 1
            or "tests pass after refactor" not in cand.success_citations[0]
            or "[refactor]" not in cand.success_citations[0]):
        print(f"[memory_v22_distil_arrow2] FAIL (h): success citations "
              f"missing: {cand.success_citations!r}", file=sys.stderr)
        return 1
    if (len(cand.failure_citations) != 1
            or "user accepted explanation" not in cand.failure_citations[0]
            or "[explain]" not in cand.failure_citations[0]):
        print(f"[memory_v22_distil_arrow2] FAIL (h): failure citations "
              f"missing: {cand.failure_citations!r}", file=sys.stderr)
        return 1
    body = _build_messages(cand)[0]["content"]
    if ("Success citations:" not in body
            or "Failure citations:" not in body
            or "tests pass after refactor" not in body
            or "user accepted explanation" not in body):
        print(f"[memory_v22_distil_arrow2] FAIL (h): prompt body missing "
              f"citations: {body!r}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow2] (h) outcome citations collected + "
          "surfaced in prompt OK", file=sys.stderr)
    return 0


def _check_hedge_phrase_rejected() -> int:
    """Architecture §"Distil — Arrow Extract": post-LLM hedges must reject."""
    candidate = LearningCandidate(
        pattern_row=_row(),
        pattern_embedding=_unit(1.0),
        success_count=2,
        failure_count=0,
    )
    promote_llm = lambda system, msgs, tokens: LLMResponse(
        json.dumps({
            "promote": True,
            "text": "in some cases prefer pytest over unittest",
            "polarity": "prefer",
        }),
        30, 15,
    )
    payload, rejection = promote(
        candidate, llm=promote_llm, encoder=_AlignedEncoder(),
    )
    if payload is not None or "hedge" not in (rejection or "").lower():
        print(f"[memory_v22_distil_arrow2] FAIL (g): payload={payload!r} "
              f"rejection={rejection!r}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow2] (g) hedge phrase rejected OK",
          file=sys.stderr)
    return 0


def _check_cold_start_arrow2_gates() -> int:
    """(h) `cold_start_arrow2_gates(n_rows)` returns relaxed
    `(min_attributions=1,)` when memory is sparse, else None. Mirrors
    `recall.cold_start_gates` and `distil_arrow1.cold_start_arrow1_gates`,
    using the same `COLD_START_ROW_THRESHOLD` so the "sparse memory"
    definition stays consistent across recall, arrow1 promotion, and
    arrow2 promotion.

    Motivation: v5_paired* benches consistently produced 1-2 patterns
    but 0 learnings because no pattern hit ≥2 outcome attributions.
    With 5 arc-shaped prompt clusters and 6 sessions per arc, the
    typical pattern gets surfaced ~1 time when later same-arc prompts
    fire — borderline against the default min_attributions=2. Cold-start
    relax lets a single attribution attempt promotion.
    """
    from resonance_lattice.memory.distil_arrow2 import cold_start_arrow2_gates
    from resonance_lattice.memory.recall import COLD_START_ROW_THRESHOLD

    relaxed = cold_start_arrow2_gates(0)
    if relaxed != (1,):
        print(f"[memory_v22_distil_arrow2] FAIL (h.1): empty store should "
              f"return (1,); got {relaxed!r}", file=sys.stderr)
        return 1
    if cold_start_arrow2_gates(COLD_START_ROW_THRESHOLD - 1) != (1,):
        print(f"[memory_v22_distil_arrow2] FAIL (h.2): below threshold "
              f"should relax", file=sys.stderr)
        return 1
    if cold_start_arrow2_gates(COLD_START_ROW_THRESHOLD) is not None:
        print(f"[memory_v22_distil_arrow2] FAIL (h.3): at threshold should "
              f"NOT relax (strict <)", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow2] (h) cold-start arrow2 gates OK",
          file=sys.stderr)
    return 0


def _check_cold_start_promotes_single_attribution() -> int:
    """(i) When memory is sparse AND `auto_tune_cold_start=True`,
    `arrow2_pass` accepts a pattern with exactly 1 outcome attribution
    that the default threshold (2) would reject.

    This is the v5_paired blocker: patterns formed but no learnings
    because attribution count was 1 in arcs where patterns existed.

    Inverse check: `auto_tune_cold_start=False` correctly leaves the
    default trigger in place.

    Explicit caller override check: passing `min_attributions=2` even
    with auto_tune=True correctly enforces the explicit threshold.
    """
    import tempfile
    from pathlib import Path

    from resonance_lattice.memory.distil_arrow2 import arrow2_pass
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u", encoder=_AlignedEncoder())
        # Plant a pattern row with embedding on dim-0 (aligned encoder
        # outputs unit-on-dim-0 vectors so post-LLM cosine alignment
        # check passes).
        memory.add_row(
            text="pattern about logging tool calls",
            polarity=["factual"],
            transcript_hash="distilled:planted",
            embedding=_unit(1.0),
            level="pattern",
            origin="distilled",
        )
        rows, _ = memory.read_all()
        pattern_row_id = rows[0].row_id
        outcomes = [_outcome(row_id=pattern_row_id, verdict="satisfied")]

        promote_llm = lambda system, msgs, tokens: LLMResponse(
            json.dumps({
                "promote": True,
                "text": "log tool calls when debugging side effects",
                "polarity": "prefer",
            }),
            30, 15,
        )
        result = arrow2_pass(
            memory, outcomes=outcomes, llm=promote_llm,
            encoder=_AlignedEncoder(), dry_run=True,
        )
        if result.candidates_found != 1:
            print(f"[memory_v22_distil_arrow2] FAIL (i.1): cold-start auto-tune "
                  f"should find 1 candidate from a single-attribution pattern; "
                  f"got candidates_found={result.candidates_found}",
                  file=sys.stderr)
            return 1
        if len(result.promoted_row_ids) != 1:
            print(f"[memory_v22_distil_arrow2] FAIL (i.2): expected 1 promoted "
                  f"row; got {result.promoted_row_ids!r} rejections="
                  f"{result.rejections!r}", file=sys.stderr)
            return 1

        # Auto-tune disabled → default min_attributions=2 → no candidate.
        result_off = arrow2_pass(
            memory, outcomes=outcomes, llm=promote_llm,
            encoder=_AlignedEncoder(), dry_run=True,
            auto_tune_cold_start=False,
        )
        if result_off.candidates_found != 0:
            print(f"[memory_v22_distil_arrow2] FAIL (i.3): default-gates pass "
                  f"should find 0 candidates with 1 attribution; got "
                  f"{result_off.candidates_found}", file=sys.stderr)
            return 1

        # Explicit caller override wins.
        result_override = arrow2_pass(
            memory, outcomes=outcomes, llm=promote_llm,
            encoder=_AlignedEncoder(), dry_run=True,
            min_attributions=2,
        )
        if result_override.candidates_found != 0:
            print(f"[memory_v22_distil_arrow2] FAIL (i.4): explicit "
                  f"min_attributions=2 should override cold-start relax",
                  file=sys.stderr)
            return 1

    print("[memory_v22_distil_arrow2] (i) cold-start auto-tune promotes "
          "single-attribution patterns + override wins OK", file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_candidate_discovery,
        _check_already_promoted_skipped,
        _check_refusal,
        _check_misalignment_rejected,
        _check_promotion_payload_shape,
        _check_incidental_excluded,
        _check_hedge_phrase_rejected,
        _check_outcome_citations_collected_and_built,
        _check_cold_start_arrow2_gates,
        _check_cold_start_promotes_single_attribution,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v22_distil_arrow2] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
