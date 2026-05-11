"""Distil Arrow 1 — cluster events into patterns.

Architecture §"Distil — Arrow Cluster (events → pattern)":

  Trigger:   cluster of ≥3 events with cosine ≥0.85, criticality ≥normal,
             total recurrence ≥5
  Mechanism: LLM extracts a *regularity* statement from the cluster

  Failure modes Distil defends against:
    - Hallucinated promotions — post-LLM cosine check against parent
      cluster; reject misalignments
    - Premature generalisation — criticality precondition
    - Stale promotions — corpus drift drops parent confidence

  Distil never deletes parents — promotion preserves the chain. Forget
  handles cleanup later (sequencing rule: distil first, then forget).

This module ships:

  find_promotion_candidates(rows, band)   pure cluster discovery — no LLM
  promote(candidate, llm)                 LLM call + post-validation
  arrow1_pass(memory, llm)                end-to-end runner

The LLM seam mirrors `optimise/synth_queries.py` and `memory/distil.py`:
a callable `(system, messages, max_tokens) -> LLMResponse`. The harness
suite injects a stub; production wires the Anthropic client.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..field._runtime_common import l2_normalize
from ..field.encoder import Encoder
from ._common import (
    CONFIDENCE_DILUTION,
    parse_llm_json,
    reject_text_quality,
    utcnow_iso,
    workspace_tag_for_cwd,
)
from .distil import LLMResponse  # reuse the same shape across distillers
from .store import DISTILLED_PREFIX, Memory, Row

LLMClient = Callable[[str, list[dict], int], LLMResponse]

# Thresholds from architecture §"Distil — Arrow Cluster" — engineering
# spec parameters; tunable without rewriting the manifesto.
DEFAULT_CLUSTER_COSINE = 0.85
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_MIN_TOTAL_RECURRENCE = 5
DEFAULT_MIN_CRITICALITY_RANK = 1  # normal=1, high=2, severe=3 (low=0)
_CRITICALITY_RANK = {"low": 0, "normal": 1, "high": 2, "severe": 3}

# Post-LLM cosine alignment floor. The pattern's embedding must align
# with the parent cluster's centroid to at least this cosine — otherwise
# the LLM hallucinated. Looser than DEFAULT_CLUSTER_COSINE because the
# regularity statement is by construction more abstract than its parents.
DEFAULT_POST_VALIDATION_COSINE = 0.55

# Confidence dilutes one step per promotion unless separately verified
# (architecture §"Field interactions worth knowing"). Lifted to
# `memory/_common.CONFIDENCE_DILUTION` so arrows 2 + 3 share the map
# without cross-module private import; re-bound here as a deprecated
# alias for any external callers still hitting the underscore name.
_CONFIDENCE_DILUTION = CONFIDENCE_DILUTION


@dataclass(frozen=True)
class PromotionCandidate:
    """One cluster of events that meets all Arrow 1 triggers.

    `parent_rows` are the events that will become `parent_ids` on the
    promoted pattern; `centroid` is their L2-normalised mean (used both
    for post-LLM cosine validation and for the promoted pattern's own
    band entry on initial write before the LLM regularity is encoded).
    """

    parent_rows: list[Row]
    centroid: np.ndarray
    total_recurrence: int


@dataclass
class Arrow1Result:
    """End-to-end Arrow 1 pass outcome."""

    candidates_found: int = 0
    promoted_row_ids: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)


def _criticality_meets(row: Row, min_rank: int) -> bool:
    return _CRITICALITY_RANK.get(row.criticality, 0) >= min_rank


def _connected_components(
    band: np.ndarray, threshold: float,
) -> list[list[int]]:
    """Union-find connected components on the cosine-≥-threshold graph.

    Same shape as `field.algebra.greedy_cluster` but operates on a band
    indices subset rather than full RQL hits. Splits a transitive chain
    into one cluster (A↔B↔C all transitively similar → one cluster).
    """
    n = band.shape[0]
    if n == 0:
        return []
    parent = list(range(n))

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    sims = band @ band.T
    rows, cols = np.where(sims >= threshold)
    for a, b in zip(rows, cols):
        if a == b:
            continue
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, list[int]] = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def find_promotion_candidates(
    rows: list[Row],
    band: np.ndarray,
    *,
    cluster_cosine: float = DEFAULT_CLUSTER_COSINE,
    min_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_total_recurrence: int = DEFAULT_MIN_TOTAL_RECURRENCE,
    min_criticality_rank: int = DEFAULT_MIN_CRITICALITY_RANK,
) -> list[PromotionCandidate]:
    """Pure cluster discovery — no LLM, no I/O.

    Returns clusters of events that meet ALL three triggers (size,
    cosine, total recurrence) and the criticality precondition. Skips
    rows already promoted (i.e. that appear in another row's parent_ids
    where the parent is a confident pattern) — we don't double-promote.
    """
    if not rows:
        return []
    # Filter to events meeting the criticality precondition.
    eligible_idx = [
        i for i, row in enumerate(rows)
        if row.level == "event" and _criticality_meets(row, min_criticality_rank)
    ]
    if len(eligible_idx) < min_size:
        return []

    # Skip events already promoted (parent of a confident pattern).
    already_promoted: set[str] = set()
    for parent in rows:
        if (parent.level == "pattern"
                and parent.confidence in ("medium", "high", "verified")):
            already_promoted.update(parent.parent_ids)
    eligible_idx = [
        i for i in eligible_idx if rows[i].row_id not in already_promoted
    ]
    if len(eligible_idx) < min_size:
        return []

    sub_band = band[eligible_idx].astype(np.float32, copy=False)
    components = _connected_components(sub_band, cluster_cosine)
    candidates: list[PromotionCandidate] = []
    for component in components:
        if len(component) < min_size:
            continue
        member_idx = [eligible_idx[c] for c in component]
        member_rows = [rows[i] for i in member_idx]
        total_recur = sum(r.recurrence_count for r in member_rows)
        if total_recur < min_total_recurrence:
            continue
        centroid = sub_band[component].mean(axis=0)
        l2_normalize(centroid)
        candidates.append(PromotionCandidate(
            parent_rows=member_rows,
            centroid=centroid,
            total_recurrence=total_recur,
        ))
    return candidates


# ---------------------------------------------------------------------------
# LLM step
# ---------------------------------------------------------------------------


_PROMPT = """You extract a regularity from a cluster of similar events captured
during agent sessions. The cluster has been validated as cohesive (cosine ≥0.85,
criticality ≥normal). Produce ONE concise prescriptive statement that names the
regularity these events share, OR refuse if there is no coherent regularity.

OUTPUT FORMAT — read carefully:
  Output ONLY a JSON object. No prose, no markdown, no code fences, no
  explanation. The first character of your response is `{`. The last
  character is `}`. Nothing else.

  Exactly one of these two shapes:
    {"promote": true, "text": "<one-sentence regularity>", "polarity": "prefer|avoid|factual"}
    {"promote": false, "reason": "<short reason>"}

Choose `promote: false` when the events don't share a coherent regularity, when
the statement would be too vague to be actionable, or when the cluster is noise
(e.g. identical copies of the same event). Principled refusal beats hallucination.

When `promote: true`, the statement must:
  - Be ONE sentence, ≤25 words
  - Be prescriptive (what the agent should do, avoid, or know)
  - Not name specific files, line numbers, or session-specific identifiers
  - Be falsifiable — a future event could contradict it"""


def _build_messages(candidate: PromotionCandidate) -> list[dict]:
    body = "\n".join(
        f"  - [{r.primary_polarity()}] {r.text.strip()[:200]}"
        for r in candidate.parent_rows
    )
    user = (
        f"Cluster of {len(candidate.parent_rows)} events "
        f"(total recurrence {candidate.total_recurrence}):\n\n{body}\n\n"
        "Extract the regularity, or refuse with `promote: false`."
    )
    return [{"role": "user", "content": user}]


def _validate_promotion(
    candidate: PromotionCandidate,
    encoded_text: np.ndarray,
    *,
    post_validation_cosine: float,
    max_words: int,
    text: str,
) -> str | None:
    """Return None if valid, else a rejection reason."""
    return reject_text_quality(
        text, encoded_text, candidate.centroid,
        max_words=max_words,
        post_validation_cosine=post_validation_cosine,
    )


def promote(
    candidate: PromotionCandidate,
    *,
    llm: LLMClient,
    encoder: Encoder,
    cwd: str | None = None,
    post_validation_cosine: float = DEFAULT_POST_VALIDATION_COSINE,
    max_words: int = 25,
) -> tuple[dict | None, str | None]:
    """Run the LLM step + post-validation.

    Returns `(row_payload, rejection_reason)` — exactly one is None.
    `row_payload` is the kwargs dict for `Memory.add_row`; caller writes
    it. Refusal (`promote: false`), validation failure, or LLM error all
    return `(None, reason)` so the caller can log + continue.
    """
    try:
        response = llm(_PROMPT, _build_messages(candidate), 256)
    except Exception as exc:
        return None, f"llm error: {type(exc).__name__}: {exc}"
    try:
        payload = parse_llm_json(response.text)
    except json.JSONDecodeError as exc:
        return None, f"non-JSON response: {exc}"

    if not isinstance(payload, dict) or not payload.get("promote"):
        return None, f"refused: {payload.get('reason', 'no reason given')}"

    text = payload.get("text", "")
    polarity_primary = payload.get("polarity", "factual")
    if not isinstance(text, str) or not text.strip():
        return None, "empty text"
    if polarity_primary not in ("prefer", "avoid", "factual"):
        return None, f"bad polarity: {polarity_primary!r}"

    embedding = encoder.encode([text])[0]
    l2_normalize(embedding)
    rejection = _validate_promotion(
        candidate, embedding,
        post_validation_cosine=post_validation_cosine,
        max_words=max_words,
        text=text,
    )
    if rejection:
        return None, rejection

    # Architecture §"Field interactions worth knowing": "A distilled memory
    # inherits the minimum confidence of its parents minus one step." In the
    # `("verified", "high", "medium", "low")` ordering, weaker confidence has
    # a *higher* index — so "minimum confidence" is `max(indices)`, not
    # `min`. The earlier shape used `min` and the variable name lied: a
    # mixed `{verified, low, low}` cluster shipped as `pattern@high`
    # instead of `pattern@low`. Latent because real clusters at v0 are
    # uniformly `medium`, but a real correctness gap.
    weakest_parent_idx = max(
        ("verified", "high", "medium", "low").index(p.confidence)
        for p in candidate.parent_rows
    )
    parent_confidence = ("verified", "high", "medium", "low")[weakest_parent_idx]
    diluted = _CONFIDENCE_DILUTION[parent_confidence]

    transcript_hash = (
        f"{DISTILLED_PREFIX}arrow1:{utcnow_iso()}:"
        f"{len(candidate.parent_rows)}"
    )
    workspace_tag = workspace_tag_for_cwd(cwd)
    polarity = [polarity_primary, workspace_tag]
    return ({
        "text": text,
        "polarity": polarity,
        "transcript_hash": transcript_hash,
        "embedding": embedding,
        "level": "pattern",
        "criticality": _highest_criticality(candidate.parent_rows),
        "confidence": diluted,
        "parent_ids": [r.row_id for r in candidate.parent_rows],
        "origin": "distilled",
    }, None)


def _highest_criticality(rows: list[Row]) -> str:
    """Inherit the strongest criticality among the parent cluster — the
    architecture's `severe avoid` floor needs criticality to propagate
    upward through the ladder."""
    return max(
        (r.criticality for r in rows),
        key=lambda c: _CRITICALITY_RANK.get(c, 0),
    )


def arrow1_pass(
    memory: Memory,
    *,
    llm: LLMClient,
    encoder: Encoder | None = None,
    cwd: str | None = None,
    dry_run: bool = False,
    **thresholds,
) -> Arrow1Result:
    """End-to-end pass: discover → promote → write. Returns a result
    summary. Caller (the session-end runner) sequences this BEFORE Forget
    so condition 2 (redundant after promotion) can fire on freshly-
    promoted patterns.

    `dry_run=True` skips `memory.add_row` and appends the synthetic
    placeholder `<dry-run>` to `promoted_row_ids` so `len(...)` reflects
    the would-promote count.
    """
    rows, band = memory.read_all()
    candidate_thresholds = {
        k: v for k, v in thresholds.items()
        if k in {"cluster_cosine", "min_size", "min_total_recurrence",
                 "min_criticality_rank"}
    }
    promote_thresholds = {
        k: v for k, v in thresholds.items()
        if k in {"post_validation_cosine", "max_words"}
    }
    candidates = find_promotion_candidates(rows, band, **candidate_thresholds)
    result = Arrow1Result(candidates_found=len(candidates))
    if not candidates:
        return result
    if encoder is None:
        encoder = memory._ensure_encoder()  # type: ignore[attr-defined]
    for candidate in candidates:
        payload, rejection = promote(
            candidate, llm=llm, encoder=encoder, cwd=cwd,
            **promote_thresholds,
        )
        if payload is None:
            result.rejections.append(rejection or "unknown")
            continue
        new_id = "<dry-run>" if dry_run else memory.add_row(**payload)
        result.promoted_row_ids.append(new_id)
    return result
