"""memory_v22_full_chain — event→pattern→learning→principle end-to-end.

Hermetic smoke for the full distil chain. Plants synthetic rows +
outcomes + fake LLM, runs arrow1→arrow2→arrow3 in sequence on a fresh
per-user store, asserts each stage produces the next level and that
recurrence_count is inherited through every promotion.

Three checks:

  (a) arrow1 cold-start promotes a size-2 cluster to a pattern with
      recurrence_count = cluster total.

  (b) arrow2 cold-start promotes that pattern to a learning when a
      single successful outcome attributes to it; learning inherits
      the pattern's recurrence.

  (c) arrow3 cold-start promotes that learning to a principle when a
      single-domain successful outcome attributes to it; principle
      inherits the learning's recurrence.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import namedtuple
from pathlib import Path

import numpy as np

from resonance_lattice.memory.distil_arrow1 import arrow1_pass
from resonance_lattice.memory.distil_arrow2 import arrow2_pass
from resonance_lattice.memory.distil_arrow3 import arrow3_pass
from resonance_lattice.memory.store import Memory
from resonance_lattice.state import (
    Attribution,
    CriterionCheck,
    OutcomeRecord,
)

LLMResponse = namedtuple("LLMResponse", "text input_tokens output_tokens")


def _unit(*coords: float) -> np.ndarray:
    v = np.zeros(768, dtype=np.float32)
    for i, c in enumerate(coords):
        v[i] = c
    n = np.linalg.norm(v)
    return v / (n if n else 1.0)


class _AlignedEncoder:
    """Returns unit-on-dim-0 vectors so cosine alignment with planted
    dim-0 candidates always passes post-validation."""

    revision = "test"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 768), dtype=np.float32)
        out[:, 0] = 1.0
        return out


def _arrow1_llm(system: str, msgs: list[dict], tokens: int) -> LLMResponse:
    return LLMResponse(
        json.dumps({
            "promote": True,
            "text": "prefer logging tool calls when debugging boundary issues",
            "polarity": "prefer",
        }),
        30, 12,
    )


def _arrow2_llm(system: str, msgs: list[dict], tokens: int) -> LLMResponse:
    return LLMResponse(
        json.dumps({
            "promote": True,
            "text": "log tool calls before commits when debugging side effects",
            "polarity": "prefer",
        }),
        30, 15,
    )


def _arrow3_llm(system: str, msgs: list[dict], tokens: int) -> LLMResponse:
    return LLMResponse(
        json.dumps({
            "promote": True,
            "text": "log decisions before committing to them",
            "polarity": "prefer",
        }),
        25, 10,
    )


def _outcome_for(row_id: str, intent_kind: str = "design") -> OutcomeRecord:
    return OutcomeRecord(
        intent_id="t",
        intent_level="task",
        criterion_checks=[CriterionCheck(
            criterion_text="x", measure="user_confirms", verdict="satisfied",
        )],
        roll_up_verdict="satisfied",
        attribution=[Attribution(row_id=row_id, tier="primary")],
        resolved_at="2026-05-14T00:00:00Z",
        intent_kind=intent_kind,
    )


def _plant_cluster(memory: Memory) -> None:
    """Two events on dim-0 with recurrence_count=2 each — clears
    cold-start arrow1 (cluster_min_size=2, min_total_recurrence=2)."""
    for i in range(2):
        memory.add_row(
            text=f"observed tool call before commit at boundary #{i}",
            polarity=["factual"],
            transcript_hash=f"event:{i}",
            embedding=_unit(1.0),
            level="event",
            origin="manual",
            recurrence_count=2,
        )


def _find_row(memory: Memory, level: str):
    rows, _ = memory.read_all()
    for r in rows:
        if r.level == level:
            return r
    return None


def _check_full_chain() -> int:
    encoder = _AlignedEncoder()
    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u", encoder=encoder)
        _plant_cluster(memory)

        # arrow1: event cluster → pattern.
        r1 = arrow1_pass(memory, llm=_arrow1_llm, encoder=encoder)
        if len(r1.promoted_row_ids) != 1:
            print(f"[memory_v22_full_chain] FAIL (a): arrow1 promoted="
                  f"{r1.promoted_row_ids!r} rejections={r1.rejections!r}",
                  file=sys.stderr)
            return 1
        pattern = _find_row(memory, "pattern")
        if pattern is None:
            print("[memory_v22_full_chain] FAIL (a): no pattern row after arrow1",
                  file=sys.stderr)
            return 1
        # W5: pattern inherits cluster total_recurrence (2 events × 2 = 4).
        if pattern.recurrence_count != 4:
            print(f"[memory_v22_full_chain] FAIL (a): pattern recurrence_count="
                  f"{pattern.recurrence_count} (want 4)", file=sys.stderr)
            return 1
        print("[memory_v22_full_chain] (a) event-cluster → pattern OK "
              f"(recurrence={pattern.recurrence_count})", file=sys.stderr)

        # arrow2: pattern + attribution → learning. Cold-start auto-tune
        # accepts the single attribution (W4).
        pattern_outcomes = [_outcome_for(pattern.row_id)]
        r2 = arrow2_pass(
            memory, outcomes=pattern_outcomes,
            llm=_arrow2_llm, encoder=encoder,
        )
        if len(r2.promoted_row_ids) != 1:
            print(f"[memory_v22_full_chain] FAIL (b): arrow2 promoted="
                  f"{r2.promoted_row_ids!r} rejections={r2.rejections!r}",
                  file=sys.stderr)
            return 1
        learning = _find_row(memory, "learning")
        if learning is None:
            print("[memory_v22_full_chain] FAIL (b): no learning row after arrow2",
                  file=sys.stderr)
            return 1
        # Learning inherits parent pattern's recurrence so it clears
        # recall's cold-start min_recurrence=2 gate.
        if learning.recurrence_count != pattern.recurrence_count:
            print(f"[memory_v22_full_chain] FAIL (b): learning recurrence="
                  f"{learning.recurrence_count} should inherit pattern's "
                  f"{pattern.recurrence_count}", file=sys.stderr)
            return 1
        print(f"[memory_v22_full_chain] (b) pattern + attribution → learning OK "
              f"(recurrence={learning.recurrence_count} inherited)",
              file=sys.stderr)

        # arrow3: learning + single-domain success → principle. Cold-start
        # auto-tune accepts the single-domain attribution (W6).
        learning_outcomes = pattern_outcomes + [
            _outcome_for(learning.row_id),
        ]
        r3 = arrow3_pass(
            memory, outcomes=learning_outcomes,
            llm=_arrow3_llm, encoder=encoder,
        )
        if len(r3.promoted_row_ids) != 1:
            print(f"[memory_v22_full_chain] FAIL (c): arrow3 promoted="
                  f"{r3.promoted_row_ids!r} rejections={r3.rejections!r}",
                  file=sys.stderr)
            return 1
        principle = _find_row(memory, "principle")
        if principle is None:
            print("[memory_v22_full_chain] FAIL (c): no principle row after arrow3",
                  file=sys.stderr)
            return 1
        if principle.parent_ids != [learning.row_id]:
            print(f"[memory_v22_full_chain] FAIL (c): principle parent_ids="
                  f"{principle.parent_ids!r} (want [{learning.row_id!r}])",
                  file=sys.stderr)
            return 1
        # Principle inherits parent learning's recurrence so it clears
        # recall's cold-start gate.
        if principle.recurrence_count != learning.recurrence_count:
            print(f"[memory_v22_full_chain] FAIL (c): principle recurrence="
                  f"{principle.recurrence_count} should inherit learning's "
                  f"{learning.recurrence_count}", file=sys.stderr)
            return 1
        print(f"[memory_v22_full_chain] (c) learning + attribution → principle OK "
              f"(recurrence={principle.recurrence_count} inherited)",
              file=sys.stderr)
    return 0


def run() -> int:
    rc = _check_full_chain()
    if rc != 0:
        return rc
    print("[memory_v22_full_chain] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
