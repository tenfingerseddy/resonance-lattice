"""Distil Arrow 2 — pattern → learning.

Architecture §"Distil — Arrow Extract (pattern → learning)":

  Trigger:   pattern + outcome ledger evidence (success or failure outcomes
             attributed)
  Mechanism: LLM extracts a *prescriptive rule* with named conditions and
             cited outcomes

  Failure modes Distil defends against:
    - Hallucinated promotions  — post-LLM cosine alignment with parent
    - Premature generalisation — outcome-evidence threshold blocks learnings
                                 with insufficient ground truth
    - Stale promotions         — corpus drift drops parent confidence

Where Arrow 1 promotes by *aggregating* (a cluster of events becomes one
pattern), Arrow 2 promotes by *abstracting* — a single pattern with enough
outcome attribution becomes a prescriptive learning that names the
conditions under which the pattern should fire.

Reuses the LLM seam + dilution rule + writer shape from Arrow 1; the
trigger and prompt differ.
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

# Trigger thresholds — engineering-spec parameters per architecture
# §"Operations are bounded by depth, not breadth".
DEFAULT_MIN_OUTCOME_ATTRIBUTIONS = 2
DEFAULT_MIN_DISTINCT_VERDICTS = 1  # 1 = same-verdict is fine; 2 = needs both
DEFAULT_POST_VALIDATION_COSINE = 0.50
DEFAULT_MAX_WORDS = 30


_MAX_CITATIONS_PER_VERDICT = 3
"""How many success / failure outcomes to cite in the LLM prompt body.
The architecture asks for cited outcomes; the cap keeps message size
bounded once a row accumulates dozens of attributions."""


@dataclass(frozen=True)
class LearningCandidate:
    """A pattern row with enough outcome evidence to extract a learning.

    `success_citations` / `failure_citations` carry up to
    `_MAX_CITATIONS_PER_VERDICT` short strings of the form
    ``"[<intent_kind>] <criterion_text>"`` for the LLM to ground its
    extracted rule against. Counts stay alongside the citations so
    callers don't need to re-derive them.
    """

    pattern_row: Row
    pattern_embedding: np.ndarray
    success_count: int
    failure_count: int
    success_citations: tuple[str, ...] = ()
    failure_citations: tuple[str, ...] = ()


@dataclass
class Arrow2Result:
    """End-to-end Arrow 2 pass outcome."""

    candidates_found: int = 0
    promoted_row_ids: list[str] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)


def _outcome_counts_for_row(
    row_id: str,
    outcomes: list[OutcomeRecord],
) -> tuple[int, int, tuple[str, ...], tuple[str, ...]]:
    """Count primary+secondary attributions for a row split by verdict,
    plus a short citation per attribution for the LLM prompt.

    Mirrors `confidence._row_evidence` but keeps the counts split so
    Arrow 2's trigger can require both directions when configured.

    Citations are ``"[<intent_kind>] <criterion_text>"`` strings,
    capped at `_MAX_CITATIONS_PER_VERDICT` per verdict so the prompt
    body stays bounded.
    """
    success = failure = 0
    success_cites: list[str] = []
    failure_cites: list[str] = []
    for record in outcomes:
        for att in record.attribution:
            if att.row_id != row_id or att.tier == "incidental":
                continue
            if record.roll_up_verdict == "satisfied":
                success += 1
                if len(success_cites) < _MAX_CITATIONS_PER_VERDICT:
                    success_cites.append(_format_citation(record))
            elif record.roll_up_verdict == "not_satisfied":
                failure += 1
                if len(failure_cites) < _MAX_CITATIONS_PER_VERDICT:
                    failure_cites.append(_format_citation(record))
    return success, failure, tuple(success_cites), tuple(failure_cites)


def _format_citation(record: OutcomeRecord) -> str:
    """One-line citation: ``[<intent_kind>] <criterion_text>``. Falls
    back to verdict + notes when criterion checks are missing so the
    LLM still has something to ground against."""
    kind = record.intent_kind or "unknown"
    if record.criterion_checks:
        text = record.criterion_checks[0].criterion_text.strip()
    else:
        text = (record.notes or record.roll_up_verdict).strip()
    snippet = text[:120]
    return f"[{kind}] {snippet}"


def find_promotion_candidates(
    rows: list[Row],
    band: np.ndarray,
    *,
    outcomes: list[OutcomeRecord],
    min_attributions: int = DEFAULT_MIN_OUTCOME_ATTRIBUTIONS,
    min_distinct_verdicts: int = DEFAULT_MIN_DISTINCT_VERDICTS,
) -> list[LearningCandidate]:
    """Pure candidate discovery — no LLM, no I/O.

    Yields each `pattern`-level row that meets the outcome-attribution
    trigger. Patterns already promoted to a confident learning are
    skipped (no double-promotion).
    """
    if not rows:
        return []
    by_id = {r.row_id: i for i, r in enumerate(rows)}
    already_promoted: set[str] = set()
    for parent in rows:
        if (parent.level == "learning"
                and parent.confidence in ("medium", "high", "verified")):
            already_promoted.update(parent.parent_ids)
    candidates: list[LearningCandidate] = []
    for row in rows:
        if row.level != "pattern" or row.row_id in already_promoted:
            continue
        success, failure, success_cites, failure_cites = (
            _outcome_counts_for_row(row.row_id, outcomes)
        )
        total = success + failure
        if total < min_attributions:
            continue
        distinct = (1 if success > 0 else 0) + (1 if failure > 0 else 0)
        if distinct < min_distinct_verdicts:
            continue
        idx = by_id[row.row_id]
        embedding = band[idx].astype(np.float32, copy=False)
        candidates.append(LearningCandidate(
            pattern_row=row,
            pattern_embedding=embedding,
            success_count=success,
            failure_count=failure,
            success_citations=success_cites,
            failure_citations=failure_cites,
        ))
    return candidates


_PROMPT = """You extract a prescriptive RULE from a pattern that has accumulated
outcome evidence. The pattern is a regularity observed across events; the outcome
attributions name when the pattern's prediction was borne out (success) or
violated (failure).

OUTPUT FORMAT — read carefully:
  Output ONLY a JSON object. No prose, no markdown, no code fences, no
  explanation. The first character of your response is `{`. The last
  character is `}`. Nothing else.

  Exactly one of these two shapes:
    {"promote": true, "text": "<one-sentence prescriptive rule>", "polarity": "prefer|avoid|factual"}
    {"promote": false, "reason": "<short reason>"}

Choose `promote: false` when the evidence is too thin to support a rule, when
the pattern's outcomes contradict its framing, or when extracting a rule would
require fabricating conditions the evidence doesn't support.

When `promote: true`, the rule must:
  - Be ONE sentence, ≤30 words
  - Be prescriptive — name what the agent should do, avoid, or know
  - Cite at least one named condition (not "in some cases")
  - Be falsifiable — a future outcome could contradict it
  - Stay coherent with the parent pattern (refinement, not contradiction)"""


def _build_messages(candidate: LearningCandidate) -> list[dict]:
    lines = [
        f"Parent pattern: {candidate.pattern_row.text.strip()}",
        f"Outcome attributions: {candidate.success_count} success, "
        f"{candidate.failure_count} failure",
    ]
    if candidate.success_citations:
        lines.append("Success citations:")
        lines.extend(f"  - {c}" for c in candidate.success_citations)
    if candidate.failure_citations:
        lines.append("Failure citations:")
        lines.extend(f"  - {c}" for c in candidate.failure_citations)
    body = "\n".join(lines)
    return [{"role": "user", "content": body
             + "\n\nExtract the prescriptive rule, or refuse."}]


def _validate_promotion(
    candidate: LearningCandidate,
    encoded_text: np.ndarray,
    *,
    post_validation_cosine: float,
    max_words: int,
    text: str,
) -> str | None:
    return reject_text_quality(
        text, encoded_text, candidate.pattern_embedding,
        max_words=max_words,
        post_validation_cosine=post_validation_cosine,
    )


def promote(
    candidate: LearningCandidate,
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

    parent_confidence = candidate.pattern_row.confidence
    diluted = CONFIDENCE_DILUTION[parent_confidence]
    transcript_hash = (
        f"{DISTILLED_PREFIX}arrow2:{utcnow_iso()}:"
        f"{candidate.pattern_row.row_id}"
    )
    workspace_tag = workspace_tag_for_cwd(cwd)
    polarity = [polarity_primary, workspace_tag]
    return ({
        "text": text,
        "polarity": polarity,
        "transcript_hash": transcript_hash,
        "embedding": embedding,
        "level": "learning",
        "criticality": candidate.pattern_row.criticality,
        "confidence": diluted,
        "parent_ids": [candidate.pattern_row.row_id],
        "origin": "distilled",
    }, None)


def arrow2_pass(
    memory: Memory,
    *,
    outcomes: list[OutcomeRecord],
    llm: LLMClient,
    encoder: Encoder | None = None,
    cwd: str | None = None,
    dry_run: bool = False,
    **thresholds,
) -> Arrow2Result:
    """End-to-end pass: discover → promote → write. Sequence between Arrow 1
    and forget so condition 2 of forget (redundant after promotion) sees
    learnings that have been freshly extracted from confident patterns.

    `dry_run=True` skips `memory.add_row`; see `arrow1_pass` for the
    placeholder convention.
    """
    rows, band = memory.read_all()
    candidate_thresholds = {
        k: v for k, v in thresholds.items()
        if k in {"min_attributions", "min_distinct_verdicts"}
    }
    promote_thresholds = {
        k: v for k, v in thresholds.items()
        if k in {"post_validation_cosine", "max_words"}
    }
    candidates = find_promotion_candidates(
        rows, band, outcomes=outcomes, **candidate_thresholds,
    )
    result = Arrow2Result(candidates_found=len(candidates))
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
