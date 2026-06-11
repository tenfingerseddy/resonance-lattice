"""Unified, relevance-gated context assembler — the keystone.

STATUS (2026-06 review, roadmap 4.6): chartered build-ahead, NOT on a
production path yet. Consumed today by the value-proof benchmarks
(`benchmarks/user_bench/value_proof/`) and its unit test; the intended
production consumer is the recall/serve path once the harness injects
assembled context per query. Not dead code — and not to be wired
casually: the value-proof's relevance-gating constraints are load-bearing.

CLAUDE.md promises "corpus + memory + intent → expertise"; until now those were
four separate streams. This is the single per-query assembler that combines the
personal-context sources into ONE block — but **relevance-gated**: a source's
content is included only when it clears a relevance floor for the query, so
off-topic memory or corpus is dropped rather than injected.

The empirical mandate (value-proof, 2026-06-03): a *naive* concat of all recalled
memory + all retrieved passages mildly HURT answers on questions where a source
was irrelevant (memory −0.05..−0.09 on off-domain questions). Gating removes that
drag while keeping the load-bearing hits.

Design: pure + dependency-injected. The caller supplies `memory_recall(query) ->
list[MemoryHit]` and `corpus_retrieve(query) -> list[CorpusHit]` (each carrying a
relevance score); the assembler gates, ranks, and renders. This keeps it testable
and lets the same core serve the benchmark and the product (which inject the real
recall path and store retrieval respectively).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# Relevance floors. Calibrated (value-proof, 2026-06-03) on real recall cosines:
# load-bearing lessons top out at ~0.585-0.79, clearly off-domain domain-questions
# at ~0.56. The boundary is FUZZY (cosine similarity is not a perfect "does this
# change the answer" signal — one load-bearing case sits at 0.585, one off-domain
# control at 0.609), so 0.58 is the best single cut: keeps observed load-bearing,
# drops clear off-domain, tolerates a rare weak leak. Override per call; auto by
# default, explicit override wins.
DEFAULT_MEM_FLOOR = 0.58
DEFAULT_CORPUS_FLOOR = 0.62
DEFAULT_TOP_K = 5

_SOURCES = ("memory", "corpus")


@dataclass
class MemoryHit:
    content: str
    polarity: str          # prefer | avoid | factual
    relevance: float       # cosine of query↔claim (the gate signal)


@dataclass
class CorpusHit:
    text: str
    source: str
    score: float           # retrieval score (trust/lens-weighted, the gate signal)


@dataclass
class AssembledContext:
    text: str = ""                              # the rendered block ("" = nothing relevant)
    sources_included: list[str] = field(default_factory=list)
    memory_hits: list[MemoryHit] = field(default_factory=list)
    corpus_hits: list[CorpusHit] = field(default_factory=list)


def _render_memory(hits: list[MemoryHit]) -> str:
    lines = [f"- *{h.polarity}* — {h.content}" for h in hits]
    return ("--- WHAT I'VE LEARNED WORKING ON THIS PROJECT ---\n"
            "(earned lessons from past sessions; weigh them when they apply)\n\n"
            + "\n".join(lines))


def _render_corpus(hits: list[CorpusHit]) -> str:
    lines = [f"- ({h.source}) {h.text}" for h in hits]
    return ("--- RETRIEVED FROM THE PROJECT KNOWLEDGE MODEL ---\n"
            "(domain passages relevant to the question; cite when used)\n\n"
            + "\n".join(lines))


def assemble(
    query: str,
    *,
    memory_recall: Callable[[str], list[MemoryHit]] | None = None,
    corpus_retrieve: Callable[[str], list[CorpusHit]] | None = None,
    enable: tuple[str, ...] = _SOURCES,
    mem_floor: float = DEFAULT_MEM_FLOOR,
    corpus_floor: float = DEFAULT_CORPUS_FLOOR,
    top_k: int = DEFAULT_TOP_K,
) -> AssembledContext:
    """Assemble a single relevance-gated context block for `query`.

    Only sources in `enable` with at least one hit clearing their floor contribute;
    a source whose best hit is below floor is omitted entirely (no drag). Hits are
    assumed pre-ranked by the caller (the recall path already applies the
    multi-factor importance model); the floor gates on relevance, not importance.
    """
    blocks: list[str] = []
    included: list[str] = []
    mem_kept: list[MemoryHit] = []
    corp_kept: list[CorpusHit] = []

    if "memory" in enable and memory_recall is not None:
        hits = memory_recall(query) or []
        mem_kept = [h for h in hits if h.relevance >= mem_floor][:top_k]
        if mem_kept:
            blocks.append(_render_memory(mem_kept))
            included.append("memory")

    if "corpus" in enable and corpus_retrieve is not None:
        hits = corpus_retrieve(query) or []
        corp_kept = [h for h in hits if h.score >= corpus_floor][:top_k]
        if corp_kept:
            blocks.append(_render_corpus(corp_kept))
            included.append("corpus")

    return AssembledContext(
        text="\n\n".join(blocks),
        sources_included=included,
        memory_hits=mem_kept,
        corpus_hits=corp_kept,
    )
