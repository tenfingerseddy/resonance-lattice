"""memory_v22_distil_arrow1 — Distil Arrow 1 contracts.

Pins architecture §"Distil — Arrow Cluster (events → pattern)". Six
contracts:

  (a) Cluster discovery — a tight cluster of ≥3 events with cosine ≥0.85
      and total recurrence ≥5 surfaces; loose pairs don't.

  (b) Criticality precondition — events at `low` criticality are excluded
      regardless of cluster cosine.

  (c) Total recurrence threshold — a cluster of 3 events with total
      recurrence < 5 doesn't qualify.

  (d) Already-promoted events skipped — events that appear in another
      row's parent_ids (where the parent is a confident pattern) don't
      re-promote.

  (e) Promote() refuses on `promote: false` — refusal is honoured;
      no row is written.

  (f) Promote() rejects on cosine misalignment — a stub LLM that emits
      text orthogonal to the cluster centroid is caught by post-validation.

Hermetic — synthetic rows + a fake LLM client + ZeroEncoder; no network.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import namedtuple
from pathlib import Path

import numpy as np

from resonance_lattice.memory.distil_arrow1 import (
    PromotionCandidate,
    find_promotion_candidates,
    promote,
)
from resonance_lattice.memory.store import Memory, Row

LLMResponse = namedtuple("LLMResponse", "text input_tokens output_tokens")


def _row(row_id: str, *, level: str = "event", criticality: str = "normal",
         recurrence_count: int = 2, parent_ids: list[str] | None = None,
         confidence: str = "medium") -> Row:
    return Row(
        row_id=row_id,
        text=f"event {row_id}",
        polarity=["factual", "workspace:abc123"],
        recurrence_count=recurrence_count,
        created_at="2026-05-01T00:00:00Z",
        last_corroborated_at="2026-05-01T00:00:00Z",
        transcript_hash="manual",
        is_bad=False,
        level=level,
        criticality=criticality,
        confidence=confidence,
        parent_ids=parent_ids or [],
        origin="manual",
    )


def _band(vectors: list[np.ndarray]) -> np.ndarray:
    return np.stack(vectors).astype(np.float32)


def _unit(*coords: float) -> np.ndarray:
    v = np.zeros(768, dtype=np.float32)
    for i, c in enumerate(coords):
        v[i] = c
    n = np.linalg.norm(v)
    return v / (n if n else 1.0)


def _cluster_band(seed: float, n: int) -> list[np.ndarray]:
    """Build n nearly-identical unit vectors so cosine pairs ≥0.99."""
    base = _unit(1.0, seed)
    out = [base]
    for k in range(1, n):
        # Tiny perturbation that keeps cosine well above 0.85.
        v = base.copy()
        v[10 + k] = 0.05
        v /= np.linalg.norm(v)
        out.append(v.astype(np.float32))
    return out


def _check_cluster_discovery() -> int:
    rows = [_row(f"01HZ_EV{i}", recurrence_count=2) for i in range(4)]
    band = _band(_cluster_band(1.0, 4))
    candidates = find_promotion_candidates(rows, band)
    if len(candidates) != 1:
        print(f"[memory_v22_distil_arrow1] FAIL (a): expected 1 cluster, "
              f"got {len(candidates)}", file=sys.stderr)
        return 1
    if len(candidates[0].parent_rows) != 4:
        print(f"[memory_v22_distil_arrow1] FAIL (a): cluster size "
              f"{len(candidates[0].parent_rows)}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow1] (a) cluster discovery OK",
          file=sys.stderr)
    return 0


def _check_criticality_filter() -> int:
    rows = [
        _row(f"01HZ_LOW{i}", criticality="low", recurrence_count=5)
        for i in range(4)
    ]
    band = _band(_cluster_band(2.0, 4))
    candidates = find_promotion_candidates(rows, band)
    if candidates:
        print(f"[memory_v22_distil_arrow1] FAIL (b): low-criticality cluster "
              f"surfaced: {len(candidates)}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow1] (b) criticality precondition OK",
          file=sys.stderr)
    return 0


def _check_total_recurrence_threshold() -> int:
    # 3 events × recurrence 1 = total 3 < threshold 5 → no candidate.
    rows = [_row(f"01HZ_R{i}", recurrence_count=1) for i in range(3)]
    band = _band(_cluster_band(3.0, 3))
    candidates = find_promotion_candidates(rows, band)
    if candidates:
        print(f"[memory_v22_distil_arrow1] FAIL (c): low-recurrence cluster "
              f"surfaced", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow1] (c) total recurrence threshold OK",
          file=sys.stderr)
    return 0


def _check_already_promoted_skipped() -> int:
    """Two sub-checks. The 'high' case is the v2.1 contract — events
    that became a high-confidence pattern don't re-cluster. The 'low'
    case is the v4.2 fix: previously, low-confidence patterns didn't
    lock their parents, so the same events kept re-promoting on every
    consolidate. v4.2 bench: 144 events → 873 patterns (6× density)
    until the lock-at-any-confidence change shipped.
    """
    # (d.1) High-confidence parent locks its events.
    events = [_row(f"01HZ_E{i}", recurrence_count=2) for i in range(3)]
    pattern = _row("01HZ_P1", level="pattern", confidence="high",
                   parent_ids=[e.row_id for e in events])
    rows = events + [pattern]
    # Encoder vectors include the pattern centroid; we feed cluster vectors
    # that match the events but the pattern's vector is orthogonal so it
    # doesn't cluster with them anyway.
    vecs = _cluster_band(4.0, 3)
    pattern_vec = _unit(0.0, 0.0, 1.0)
    band = _band(vecs + [pattern_vec])
    candidates = find_promotion_candidates(rows, band)
    if candidates:
        print(f"[memory_v22_distil_arrow1] FAIL (d.1): high-confidence "
              f"parent didn't lock events: {len(candidates)}", file=sys.stderr)
        return 1

    # (d.2) Low-confidence parent ALSO locks its events.
    events2 = [_row(f"01HZ_F{i}", recurrence_count=2) for i in range(3)]
    pattern2 = _row("01HZ_P2", level="pattern", confidence="low",
                    parent_ids=[e.row_id for e in events2])
    rows2 = events2 + [pattern2]
    band2 = _band(_cluster_band(5.0, 3) + [_unit(0.0, 0.0, 1.0)])
    candidates2 = find_promotion_candidates(rows2, band2)
    if candidates2:
        print(f"[memory_v22_distil_arrow1] FAIL (d.2): low-confidence "
              f"parent didn't lock events — same v4.2 over-promotion "
              f"pattern: {len(candidates2)}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow1] (d) already-promoted events skipped at "
          "any pattern confidence OK", file=sys.stderr)
    return 0


class _ZeroEncoder:
    revision = "test"

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), 768), dtype=np.float32)


def _check_promote_refusal() -> int:
    candidate = PromotionCandidate(
        parent_rows=[_row(f"01HZ_E{i}", recurrence_count=2) for i in range(3)],
        centroid=_unit(1.0, 0.0),
        total_recurrence=6,
    )
    refuse_llm = lambda system, msgs, tokens: LLMResponse(
        json.dumps({"promote": False, "reason": "no coherent regularity"}),
        10, 5,
    )
    payload, rejection = promote(
        candidate, llm=refuse_llm, encoder=_ZeroEncoder(),
    )
    if payload is not None or "refused" not in (rejection or ""):
        print(f"[memory_v22_distil_arrow1] FAIL (e): refusal not honoured: "
              f"payload={payload!r} rejection={rejection!r}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow1] (e) promote refusal OK",
          file=sys.stderr)
    return 0


class _OrthogonalEncoder:
    """Returns a vector orthogonal to the candidate centroid so the
    post-LLM cosine alignment check rejects the promotion."""

    revision = "test"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 768), dtype=np.float32)
        # Place mass on dim 100 — orthogonal to the centroid which lives
        # on dims 0..10 in the cluster fixtures.
        out[:, 100] = 1.0
        return out


def _check_promote_misalignment_rejected() -> int:
    candidate = PromotionCandidate(
        parent_rows=[_row(f"01HZ_E{i}", recurrence_count=2) for i in range(3)],
        centroid=_unit(1.0, 0.0),
        total_recurrence=6,
    )
    promote_llm = lambda system, msgs, tokens: LLMResponse(
        json.dumps({
            "promote": True,
            "text": "agents should always log their reasoning",
            "polarity": "prefer",
        }),
        20, 12,
    )
    payload, rejection = promote(
        candidate, llm=promote_llm, encoder=_OrthogonalEncoder(),
    )
    if payload is not None or "alignment" not in (rejection or ""):
        print(f"[memory_v22_distil_arrow1] FAIL (f): misalignment not caught: "
              f"payload={payload!r} rejection={rejection!r}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow1] (f) post-LLM misalignment rejected OK",
          file=sys.stderr)
    return 0


class _AlignedEncoder:
    """Returns a vector aligned with the candidate centroid (placed on
    dims 0..10) so the post-LLM cosine alignment check passes."""

    revision = "test"

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.tile(_unit(1.0, 0.0), (len(texts), 1)).astype(np.float32)


def _check_dilute_from_weakest_parent() -> int:
    """Architecture §"Field interactions worth knowing": "a distilled memory
    inherits the **minimum** confidence of its parents minus one step."
    A mixed `{verified, low, low}` cluster must ship as `pattern@low`
    (dilute(low)=low, the floor), not `pattern@high` (dilute(verified)=high).

    The earlier shape used `min(indices)` over the
    `("verified","high","medium","low")` ordering — which gave the
    *highest* parent index → lowest end of the tuple → strongest parent.
    Variable name lied, math was inverted. This contract pins the fix.
    """
    parents = [
        _row("01HZ_VER", confidence="verified", recurrence_count=2),
        _row("01HZ_LO1", confidence="low", recurrence_count=2),
        _row("01HZ_LO2", confidence="low", recurrence_count=2),
    ]
    candidate = PromotionCandidate(
        parent_rows=parents,
        centroid=_unit(1.0, 0.0),
        total_recurrence=6,
    )
    promote_llm = lambda system, msgs, tokens: LLMResponse(
        json.dumps({
            "promote": True,
            "text": "agents log their tool calls",
            "polarity": "factual",
        }),
        20, 12,
    )
    payload, rejection = promote(
        candidate, llm=promote_llm, encoder=_AlignedEncoder(),
    )
    if payload is None:
        print(f"[memory_v22_distil_arrow1] FAIL (i): expected promotion, "
              f"got rejection={rejection!r}", file=sys.stderr)
        return 1
    # min(parent confidences) = "low"; dilute(low) = "low" (floor).
    if payload["confidence"] != "low":
        print(f"[memory_v22_distil_arrow1] FAIL (i): expected confidence='low' "
              f"(dilute of weakest parent), got {payload['confidence']!r}",
              file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow1] (i) dilution from weakest parent "
          "confidence OK", file=sys.stderr)
    return 0


def _check_cold_start_arrow1_gates() -> int:
    """(k) `cold_start_arrow1_gates(n_rows)` returns relaxed
    `(min_size=2, min_total_recurrence=2)` when memory is sparse, else
    None. Mirrors `recall.cold_start_gates` and uses the same
    `COLD_START_ROW_THRESHOLD` so the "sparse memory" definition stays
    consistent across recall and promotion.

    Motivation: v5_paired bench (FINDINGS_v5_paired) showed a
    30-session fresh store accumulated 27 events but 0 patterns
    because the default (min_size=3, min_total_recurrence=5) gates
    aren't met when each topic has 2-3 events at recurrence=1. Without
    patterns, recall surfaces only event rows, and the manifesto's
    value proposition (pattern + learning injection) can't be tested.
    """
    from resonance_lattice.memory.distil_arrow1 import cold_start_arrow1_gates
    from resonance_lattice.memory.recall import COLD_START_ROW_THRESHOLD

    relaxed = cold_start_arrow1_gates(0)
    if relaxed != (2, 2):
        print(f"[memory_v22_distil_arrow1] FAIL (k.1): empty store should "
              f"return (2, 2); got {relaxed!r}", file=sys.stderr)
        return 1

    just_below = cold_start_arrow1_gates(COLD_START_ROW_THRESHOLD - 1)
    if just_below != (2, 2):
        print(f"[memory_v22_distil_arrow1] FAIL (k.2): below-threshold should "
              f"relax; got {just_below!r}", file=sys.stderr)
        return 1

    at_threshold = cold_start_arrow1_gates(COLD_START_ROW_THRESHOLD)
    if at_threshold is not None:
        print(f"[memory_v22_distil_arrow1] FAIL (k.3): at threshold should "
              f"NOT relax (strict <); got {at_threshold!r}", file=sys.stderr)
        return 1

    print("[memory_v22_distil_arrow1] (k) cold-start arrow1 gates OK",
          file=sys.stderr)
    return 0


def _check_cold_start_promotes_size_2_cluster() -> int:
    """(l) When memory is sparse AND `auto_tune_cold_start=True`,
    `arrow1_pass` accepts a 2-event cluster with total recurrence 2
    that the default thresholds (size=3, recurrence=5) would reject.

    This is the v5_paired blocker — fresh bench accumulates 2-3 events
    per topic at recurrence=1 each; without cold-start relaxation
    nothing ever promotes and recall has no patterns to surface.

    Inverse check: same store with `auto_tune_cold_start=False`
    finds no candidates (default thresholds still apply).
    """
    from resonance_lattice.memory.distil_arrow1 import arrow1_pass

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u", encoder=_AlignedEncoder())
        # seed=0.0 so the cluster centroid lives on dim 0 — same axis as
        # _AlignedEncoder's output. Non-zero seeds rotate the centroid
        # off dim 0 and the post-LLM cosine alignment check rejects.
        cluster_vecs = _cluster_band(0.0, 2)
        for i, vec in enumerate(cluster_vecs):
            memory.add_row(
                text=f"agent should log {i}", polarity=["factual"],
                transcript_hash=f"manual{i}", embedding=vec,
            )

        promote_llm = lambda system, msgs, tokens: LLMResponse(
            json.dumps({
                "promote": True,
                "text": "agents log their tool calls",
                "polarity": "factual",
            }),
            20, 12,
        )
        result = arrow1_pass(
            memory, llm=promote_llm, encoder=_AlignedEncoder(),
            dry_run=True,
        )
        if result.candidates_found != 1:
            print(f"[memory_v22_distil_arrow1] FAIL (l.1): cold-start "
                  f"auto-tune should find 1 candidate from a size-2 "
                  f"cluster; got candidates_found={result.candidates_found}",
                  file=sys.stderr)
            return 1
        if len(result.promoted_row_ids) != 1:
            print(f"[memory_v22_distil_arrow1] FAIL (l.2): expected 1 promoted "
                  f"row; got {result.promoted_row_ids!r} rejections="
                  f"{result.rejections!r}", file=sys.stderr)
            return 1

        # Auto-tune disabled → default (size=3, recurrence=5) gates → no candidate.
        result_off = arrow1_pass(
            memory, llm=promote_llm, encoder=_AlignedEncoder(),
            dry_run=True, auto_tune_cold_start=False,
        )
        if result_off.candidates_found != 0:
            print(f"[memory_v22_distil_arrow1] FAIL (l.3): default-gates "
                  f"pass should find 0 candidates; got "
                  f"{result_off.candidates_found}", file=sys.stderr)
            return 1

        # Explicit caller override wins over cold-start auto-tune: passing
        # min_size=3 even with auto_tune_cold_start=True must reject the
        # size-2 cluster.
        result_override = arrow1_pass(
            memory, llm=promote_llm, encoder=_AlignedEncoder(),
            dry_run=True, min_size=3,
        )
        if result_override.candidates_found != 0:
            print(f"[memory_v22_distil_arrow1] FAIL (l.4): explicit "
                  f"min_size=3 should override cold-start relax; got "
                  f"{result_override.candidates_found}", file=sys.stderr)
            return 1

    print("[memory_v22_distil_arrow1] (l) cold-start auto-tune promotes "
          "size-2 clusters in sparse store + override wins OK",
          file=sys.stderr)
    return 0


def _check_pattern_inherits_cluster_recurrence() -> int:
    """(m) Promoted patterns inherit `candidate.total_recurrence` so they
    clear recall's cold-start `min_recurrence=2` gate.

    Without this, patterns enter at `recurrence_count=1` (Memory.add_row
    default), get filtered by recall's cold-start recurrence gate, are
    never surfaced, never attributed, and arrow2 never has candidates.
    v5_paired5 exposed this end-to-end: patterns formed but arrow2
    found zero candidates because no pattern was ever in the recall
    cache.

    Fixture: promote a 2-event cluster with recurrence 2 each (total=4)
    via arrow1_pass; verify the persisted pattern has
    `recurrence_count=4` (cluster total), not 1.
    """
    import tempfile

    from resonance_lattice.memory.distil_arrow1 import arrow1_pass
    from resonance_lattice.memory.store import Memory

    with tempfile.TemporaryDirectory() as td:
        memory = Memory(root=Path(td) / "u", encoder=_AlignedEncoder())
        # Plant 2 events at recurrence=2 each → cluster total_recurrence=4.
        for i, vec in enumerate(_cluster_band(0.0, 2)):
            memory.add_row(
                text=f"agent should log {i}", polarity=["factual"],
                transcript_hash=f"manual{i}", embedding=vec,
                recurrence_count=2,
            )

        promote_llm = lambda system, msgs, tokens: LLMResponse(
            json.dumps({
                "promote": True,
                "text": "agents log their tool calls",
                "polarity": "factual",
            }),
            20, 12,
        )
        result = arrow1_pass(
            memory, llm=promote_llm, encoder=_AlignedEncoder(),
            dry_run=False,
        )
        if len(result.promoted_row_ids) != 1:
            print(f"[memory_v22_distil_arrow1] FAIL (m.1): expected 1 "
                  f"promoted row; got {result.promoted_row_ids!r}",
                  file=sys.stderr)
            return 1
        pattern_id = result.promoted_row_ids[0]

        rows, _ = memory.read_all()
        pattern = next((r for r in rows if r.row_id == pattern_id), None)
        if pattern is None:
            print(f"[memory_v22_distil_arrow1] FAIL (m.2): promoted pattern "
                  f"{pattern_id} not in store", file=sys.stderr)
            return 1
        if pattern.level != "pattern":
            print(f"[memory_v22_distil_arrow1] FAIL (m.3): expected level="
                  f"pattern; got {pattern.level!r}", file=sys.stderr)
            return 1
        if pattern.recurrence_count != 4:
            print(f"[memory_v22_distil_arrow1] FAIL (m.4): pattern should "
                  f"inherit cluster total_recurrence=4 (sum of parent "
                  f"recurrences); got {pattern.recurrence_count}",
                  file=sys.stderr)
            return 1
    print("[memory_v22_distil_arrow1] (m) pattern recurrence_count inherits "
          "cluster total_recurrence OK", file=sys.stderr)
    return 0


def _check_hedge_phrase_rejected() -> int:
    """Architecture §"Distil — Arrow Cluster": post-LLM hedges must reject."""
    candidate = PromotionCandidate(
        parent_rows=[_row(f"01HZ_H{i}", recurrence_count=2) for i in range(3)],
        centroid=_unit(1.0, 0.0),
        total_recurrence=6,
    )
    promote_llm = lambda system, msgs, tokens: LLMResponse(
        json.dumps({
            "promote": True,
            "text": "agents sometimes log their reasoning",
            "polarity": "prefer",
        }),
        20, 12,
    )
    payload, rejection = promote(
        candidate, llm=promote_llm, encoder=_AlignedEncoder(),
    )
    if payload is not None or "hedge" not in (rejection or "").lower():
        print(f"[memory_v22_distil_arrow1] FAIL (j): hedge not caught: "
              f"payload={payload!r} rejection={rejection!r}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow1] (j) post-LLM hedge phrase rejected OK",
          file=sys.stderr)
    return 0


def run() -> int:
    for check in [
        _check_cluster_discovery,
        _check_criticality_filter,
        _check_total_recurrence_threshold,
        _check_already_promoted_skipped,
        _check_promote_refusal,
        _check_promote_misalignment_rejected,
        _check_dilute_from_weakest_parent,
        _check_hedge_phrase_rejected,
        _check_cold_start_arrow1_gates,
        _check_cold_start_promotes_size_2_cluster,
        _check_pattern_inherits_cluster_recurrence,
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v22_distil_arrow1] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
