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
        print(f"[memory_v22_distil_arrow1] FAIL (d): already-promoted events "
              f"re-clustered: {len(candidates)}", file=sys.stderr)
        return 1
    print("[memory_v22_distil_arrow1] (d) already-promoted events skipped OK",
          file=sys.stderr)
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
    ]:
        rc = check()
        if rc != 0:
            return rc
    print("[memory_v22_distil_arrow1] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
