"""Distil Arrow 3 — learning → principle.

Architecture §"Distil — Arrow Generalise (learning → principle)":

  Trigger:   learning + cross-context evidence (≥2 distinct intent_kinds)
  Mechanism: LLM removes domain-specific details to surface the
             *underlying truth*; must be falsifiable, shorter than the
             learning

Architecture §"New-principle protection window":

  Newly promoted principles get a bounded grace period (5 sessions or 30
  days, whichever first):
    - `confidence_floor` is treated as `medium` regardless of actual value
    - One boosted recall opportunity to prove themselves
    - After the window, they compete on the regular formula

This module ships the *promote* path; the protection window itself is
applied by `memory.rerank.confidence_floor` reading the row's
`created_at` against the configured window.

Where Arrow 2 abstracts a single pattern's outcome evidence into a
prescriptive rule, Arrow 3 *generalises* across domains: a learning that
has earned attribution under multiple intent_kinds (debug + design;
implement + review; etc.) earns extraction into a domain-free principle.

Same LLM seam, dilution rule, and writer shape as Arrows 1 + 2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..field._runtime_common import l2_normalize
from ..field.encoder import Encoder
from ..state.ledger import OutcomeRecord
from ._common import (
    CONFIDENCE_DILUTION,
    parse_llm_json,
    reject_text_quality,
    utcnow_iso,
    workspace_tag_for_cwd,
)
from .distil import LLMResponse
from .store import DISTILLED_PREFIX, Memory, Row

LLMClient = Callable[[str, list[dict], int], LLMResponse]

# Cross-context threshold per architecture §"Generalise" — at least two
# distinct intent_kinds attributed to the learning, with at least
# `min_attributions_per_kind` successful primary/secondary outcomes per
# kind. Engineering-spec parameters; tunable.
DEFAULT_MIN_DISTINCT_INTENT_KINDS = 2
DEFAULT_MIN_ATTRIBUTIONS_PER_KIND = 1
DEFAULT_POST_VALIDATION_COSINE = 0.45  # looser than Arrow 2 — principles
                                       # are by construction more abstract
                                       # than their parent learning
DEFAULT_MAX_WORDS = 20  # shorter than learning (≤30); architecture's
                        # "shorter than the learning" rule

# Cold-start: a single-domain workload (e.g. all `intent_kind=design`)
# physically cannot meet `min_distinct_intent_kinds=2` and arrow3 never
# fires. Relax the cross-domain bar to 1 distinct kind below
# COLD_START_ROW_THRESHOLD and let LLM-promote's domain-free constraint
# + post-validation cosine ≥0.45 refuse non-generalisable candidates.
COLD_START_MIN_DISTINCT_INTENT_KINDS = 1


def cold_start_arrow3_gates(n_rows: int) -> tuple[int] | None:
    """Return `(min_distinct_intent_kinds,)` relaxed gate when memory
    is sparse, else None. Mirrors `cold_start_arrow{1,2}_gates`."""
    from .recall import COLD_START_ROW_THRESHOLD

    if n_rows < COLD_START_ROW_THRESHOLD:
        return (COLD_START_MIN_DISTINCT_INTENT_KINDS,)
    return None


@dataclass(frozen=True)
class PrincipleCandidate:
    """A learning row with cross-domain evidence supporting promotion."""

    learning_row: Row
    learning_embedding: np.ndarray
    intent_kinds_with_success: set[str]
    total_success_count: int


@dataclass
class Arrow3Result:
    """End-to-end Arrow 3 pass outcome."""

    candidates_found: int = 0
    promoted_row_ids: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)


def _success_attributions_by_intent_kind(
    row_id: str,
    outcomes: list[OutcomeRecord],
    *,
    min_attributions_per_kind: int,
) -> tuple[set[str], int]:
    """Count successful primary+secondary attributions for `row_id`,
    grouped by `intent_kind`. Returns (intent_kinds_meeting_threshold,
    total_success_count) — only kinds whose count clears the per-kind
    threshold contribute to cross-domain qualification."""
    counts_by_kind: dict[str, int] = {}
    for record in outcomes:
        # Cross-domain *evidence* in arrow3's promotion gate means
        # successful evidence — a learning that contributed to failed
        # verdicts across multiple kinds isn't promotion-worthy. The
        # architecture says "cross-context evidence (≥2 distinct
        # intent_kinds)"; we read that as worked-across-domains, not
        # was-tried-across-domains.
        if record.roll_up_verdict != "satisfied":
            continue
        if not record.intent_kind:
            continue
        for att in record.attribution:
            if att.row_id != row_id or att.tier == "incidental":
                continue
            counts_by_kind[record.intent_kind] = (
                counts_by_kind.get(record.intent_kind, 0) + 1
            )
    qualifying = {
        kind for kind, count in counts_by_kind.items()
        if count >= min_attributions_per_kind
    }
    total = sum(counts_by_kind.values())
    return qualifying, total


def find_promotion_candidates(
    rows: list[Row],
    band: np.ndarray,
    *,
    outcomes: list[OutcomeRecord],
    min_distinct_intent_kinds: int = DEFAULT_MIN_DISTINCT_INTENT_KINDS,
    min_attributions_per_kind: int = DEFAULT_MIN_ATTRIBUTIONS_PER_KIND,
) -> list[PrincipleCandidate]:
    """Pure cross-domain discovery — no LLM, no I/O.

    Yields each `learning`-level row whose successful attributions span
    ≥`min_distinct_intent_kinds` distinct intent_kinds (each with at
    least `min_attributions_per_kind` successes). Already-promoted
    learnings (those with a confident principle child) skip.
    """
    if not rows:
        return []
    by_id = {r.row_id: i for i, r in enumerate(rows)}
    already_promoted: set[str] = set()
    for parent in rows:
        if (parent.level == "principle"
                and parent.confidence in ("medium", "high", "verified")):
            already_promoted.update(parent.parent_ids)
    candidates: list[PrincipleCandidate] = []
    for row in rows:
        if row.level != "learning" or row.row_id in already_promoted:
            continue
        kinds, total = _success_attributions_by_intent_kind(
            row.row_id, outcomes,
            min_attributions_per_kind=min_attributions_per_kind,
        )
        if len(kinds) < min_distinct_intent_kinds:
            continue
        idx = by_id[row.row_id]
        embedding = band[idx].astype(np.float32, copy=False)
        candidates.append(PrincipleCandidate(
            learning_row=row,
            learning_embedding=embedding,
            intent_kinds_with_success=kinds,
            total_success_count=total,
        ))
    return candidates


_PROMPT = """You extract a PRINCIPLE from a learning that has earned cross-domain
evidence — successful outcomes attributed across multiple intent kinds (debug,
design, implement, etc.). Remove the domain-specific details and surface the
underlying truth.

OUTPUT FORMAT — read carefully:
  Output ONLY a JSON object. No prose, no markdown, no code fences, no
  explanation. The first character of your response is `{`. The last
  character is `}`. Nothing else.

  Exactly one of these two shapes:
    {"promote": true, "text": "<one-sentence principle>", "polarity": "prefer|avoid|factual"}
    {"promote": false, "reason": "<short reason>"}

Choose `promote: false` when the learning is too narrow to generalise, when the
cross-domain evidence is incidental (the learning happened to fire in different
kinds but isn't actually domain-free), or when a generalisation would require
fabricating coverage the evidence doesn't support.

When `promote: true`, the principle must:
  - Be ONE sentence, ≤20 words
  - Be SHORTER than the parent learning
  - Be falsifiable — a future outcome could contradict it
  - Be domain-free — name no specific tools, files, languages, or project terms
  - Stay coherent with the parent learning (refinement up the ladder)"""


def _build_messages(candidate: PrincipleCandidate) -> list[dict]:
    intent_kinds = ", ".join(sorted(candidate.intent_kinds_with_success))
    body = (
        f"Parent learning: {candidate.learning_row.text.strip()}\n"
        f"Cross-domain attributions: {candidate.total_success_count} "
        f"successes across {{{intent_kinds}}}"
    )
    return [{"role": "user", "content": body
             + "\n\nGeneralise into a domain-free principle, or refuse."}]


def _validate_promotion(
    candidate: PrincipleCandidate,
    encoded_text: np.ndarray,
    *,
    post_validation_cosine: float,
    max_words: int,
    text: str,
) -> str | None:
    rejection = reject_text_quality(
        text, encoded_text, candidate.learning_embedding,
        max_words=max_words,
        post_validation_cosine=post_validation_cosine,
    )
    if rejection is not None:
        return rejection
    word_count = len(text.split())
    parent_word_count = len(candidate.learning_row.text.split())
    if word_count >= parent_word_count:
        return (
            f"principle not shorter than learning "
            f"({word_count} >= {parent_word_count} words)"
        )
    return None


def promote(
    candidate: PrincipleCandidate,
    *,
    llm: LLMClient,
    encoder: Encoder,
    cwd: str | None = None,
    post_validation_cosine: float = DEFAULT_POST_VALIDATION_COSINE,
    max_words: int = DEFAULT_MAX_WORDS,
) -> tuple[dict | None, str | None]:
    """Run the LLM step + post-validation. Returns `(row_payload, reason)`
    — exactly one is None. The caller writes the row when payload is set."""
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

    parent_confidence = candidate.learning_row.confidence
    diluted = CONFIDENCE_DILUTION[parent_confidence]
    transcript_hash = (
        f"{DISTILLED_PREFIX}arrow3:{utcnow_iso()}:"
        f"{candidate.learning_row.row_id}"
    )
    workspace_tag = workspace_tag_for_cwd(cwd)
    polarity = [polarity_primary, workspace_tag]
    return ({
        "text": text,
        "polarity": polarity,
        "transcript_hash": transcript_hash,
        "embedding": embedding,
        "level": "principle",
        "criticality": candidate.learning_row.criticality,
        "confidence": diluted,
        "parent_ids": [candidate.learning_row.row_id],
        "origin": "distilled",
    }, None)


def arrow3_pass(
    memory: Memory,
    *,
    outcomes: list[OutcomeRecord],
    llm: LLMClient,
    encoder: Encoder | None = None,
    cwd: str | None = None,
    dry_run: bool = False,
    auto_tune_cold_start: bool = True,
    **thresholds,
) -> Arrow3Result:
    """End-to-end pass: discover → promote → write. Sequence after Arrow 2
    so freshly-extracted learnings can themselves participate in the next
    pass once they've accumulated cross-domain evidence.

    `dry_run=True` skips `memory.add_row`; see `arrow1_pass` for the
    placeholder convention.

    `auto_tune_cold_start=True` (default) relaxes
    `min_distinct_intent_kinds` to 1 when the per-user store is below
    `recall.COLD_START_ROW_THRESHOLD` rows AND the caller didn't pass
    `min_distinct_intent_kinds` explicitly. Lets single-domain
    learnings attempt promotion at fresh-bench scale; the LLM
    domain-free constraint + post-validation cosine stay the safety
    net against same-domain duplicates being promoted as principles.
    """
    rows, band = memory.read_all()
    candidate_thresholds = {
        k: v for k, v in thresholds.items()
        if k in {"min_distinct_intent_kinds", "min_attributions_per_kind"}
    }
    if auto_tune_cold_start:
        relaxed = cold_start_arrow3_gates(len(rows))
        if relaxed is not None:
            (cold_min_kinds,) = relaxed
            candidate_thresholds.setdefault(
                "min_distinct_intent_kinds", cold_min_kinds,
            )
    promote_thresholds = {
        k: v for k, v in thresholds.items()
        if k in {"post_validation_cosine", "max_words"}
    }
    candidates = find_promotion_candidates(
        rows, band, outcomes=outcomes, **candidate_thresholds,
    )
    result = Arrow3Result(candidates_found=len(candidates))
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
        if dry_run:
            new_id = "<dry-run>"
        else:
            new_id = memory.add_row(
                **payload,
                recurrence_count=candidate.learning_row.recurrence_count,
            )
        result.promoted_row_ids.append(new_id)
    return result
