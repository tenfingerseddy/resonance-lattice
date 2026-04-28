"""Typed return values for the deep-search loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


HopKind = Literal[
    "plan",                  # Planner generated the initial query.
    "search",                # Retrieved top-k passages for current_query.
    "decide_answer",         # Refiner decided to answer.
    "decide_search",         # Refiner decided to issue another query.
    "decide_give_up",        # Refiner declared the corpus doesn't cover the question.
    "synth_after_max_hops",  # Hops budget exhausted; synth from accumulated evidence.
    "search_failed",         # Retrieval errored out (unrecoverable).
    "parse_failed",          # Refiner output didn't match the JSON contract.
]


@dataclass(frozen=True)
class DeepSearchHop:
    """One step in the deep-search loop.

    `query` is set on plan/search/decide_search hops; `action` on
    decide_*; `error` on search_failed/parse_failed. Other fields stay
    None so the same dataclass shape covers every hop kind without an
    inheritance hierarchy.
    """
    n: int
    kind: HopKind
    query: str | None = None
    action: str | None = None
    n_passages: int | None = None
    error: str | None = None


@dataclass
class DeepSearchResult:
    """Final result of a deep-search invocation.

    `answer` is the synthesized natural-language answer the consumer
    LLM (or end user) reads. `hops` is the chronological log — useful
    for debugging "did the loop give up too early?" or "which query
    surfaced the load-bearing passage?".

    `evidence_passages` is the union of distinct retrieved passages
    across all hops, deduped on `(source_file, char_offset)`. The
    name-check operates over this union because a fact-extraction loop
    may surface the missing-token-bearing passage on hop 3 even if hop
    1 missed it.

    `name_check_missing` is the list of distinctive question tokens
    that did NOT appear in any hop's evidence. Empty list = check
    passed; non-empty = the answer was prefixed with a refusal
    directive (or, under strict_names, replaced entirely).
    """
    question: str
    answer: str
    hops: list[DeepSearchHop] = field(default_factory=list)
    evidence_passages: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    name_check_missing: list[str] = field(default_factory=list)
    strict_names_aborted: bool = False
