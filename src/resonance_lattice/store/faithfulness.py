"""faithfulness — machine grounding check for deep-search answers.

The entry gate of the confidence lifecycle (docs/internal/GROUNDING_MODEL.md).
A deep-search answer earns a provisional insight only if it is faithful:
every claim traces to a cited source, and the citations are on-topic for
the question. Faithfulness is a verifiable matching task — it does not need
to know whether the answer is *true*, only whether it honestly represents
its sources.

Two axes, both 0..1:

  - claim_support — fraction of the answer's atomic claims entailed by a
    cited passage.
  - question_relevance — how on-topic the cited passages are for the
    question.

A fluent answer that cites real, on-topic passages it does not actually
rest on fails claim_support. A faithfully-paraphrased answer to the WRONG
question (real citations, real support, off-topic) fails question_relevance
— the failure mode bare claim-support misses.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._llm import judge_json


FAITHFULNESS_SYSTEM = """\
You audit whether a synthesised answer is faithfully grounded in a set of
retrieved source passages. You are NOT judging whether the answer is true
in the world — only whether it honestly rests on the passages provided.

Steps:
1. Decompose the ANSWER into atomic factual claims. Ignore hedging,
   restated question text, and pure framing.
2. For each claim, decide whether one of the numbered PASSAGES entails it.
   A claim is supported only if a passage actually states or directly
   implies it — not merely shares vocabulary.
3. Rate, 0.0 to 1.0, how on-topic the passages are for the QUESTION —
   whether they address the thing actually asked, not an adjacent topic.

Answer with a single JSON object, nothing else:
{
  "claims": [
    {"claim": "<short>", "supported": true|false, "passage": <number|null>}
  ],
  "question_relevance": <0.0-1.0>,
  "reason": "<one short sentence>"
}
"""

# Gate floors. A provisional insight needs most claims grounded AND the
# evidence on-topic. Internal constants — not user knobs.
_CLAIM_SUPPORT_FLOOR = 0.8
_RELEVANCE_FLOOR = 0.6


@dataclass(frozen=True)
class ClaimCheck:
    claim: str
    supported: bool
    passage: int | None      # 1-based index into the passages shown, or None


@dataclass(frozen=True)
class FaithfulnessReport:
    """Result of a faithfulness audit. `faithful` is the gate verdict."""
    claim_support: float        # fraction of claims supported, 0..1
    question_relevance: float   # 0..1
    faithful: bool
    claims: tuple[ClaimCheck, ...]
    reason: str

    @property
    def score(self) -> float:
        """Single-number faithfulness — the weaker of the two axes. An
        answer is only as grounded as its least-grounded dimension."""
        return min(self.claim_support, self.question_relevance)


def _clamp01(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))


def _build_user_message(
    question: str, answer: str, passages: list[dict],
) -> str:
    blocks = []
    for i, p in enumerate(passages, start=1):
        anchor = p.get("source_file", "?")
        blocks.append(f"--- PASSAGE {i} ({anchor}) ---\n{p['text']}\n")
    return (
        f"QUESTION:\n{question}\n\n"
        f"ANSWER (audit this):\n{answer}\n\n"
        f"PASSAGES:\n\n" + "\n".join(blocks)
    )


def _unfaithful(reason: str) -> FaithfulnessReport:
    return FaithfulnessReport(
        claim_support=0.0, question_relevance=0.0, faithful=False,
        claims=(), reason=reason,
    )


def assess_faithfulness(
    question: str,
    answer: str,
    evidence_passages: list[dict],
    client,
    *,
    model: str | None = None,
) -> FaithfulnessReport:
    """Score a deep-search answer's grounding against its cited passages.

    `evidence_passages` is the deep-search result's evidence union — each a
    dict with at least `source_file` and `text`. `client` is an
    `anthropic.Anthropic`-shaped client; inject a stub for tests.

    Faithful means claim_support >= floor AND question_relevance >= floor.
    An answer with no checkable claims (empty, pure refusal) is NOT
    faithful — there is nothing to ground, so nothing to promote.
    """
    from .._pricing import SONNET_MODEL
    use_model = model or SONNET_MODEL

    usable = [p for p in evidence_passages if p.get("text")]
    if not answer.strip() or not usable:
        return _unfaithful("no answer text or no cited passages")

    verdict = judge_json(
        client, use_model, FAITHFULNESS_SYSTEM,
        _build_user_message(question, answer, usable),
        max_tokens=1500,
    )
    if verdict.get("_parse_error"):
        return _unfaithful(
            f"judge parse failed: {verdict['_parse_error'][:80]}"
        )

    claims = tuple(
        ClaimCheck(
            claim=str(c.get("claim", ""))[:300],
            supported=bool(c.get("supported", False)),
            passage=c["passage"] if isinstance(c.get("passage"), int) else None,
        )
        for c in (verdict.get("claims") or [])
        if isinstance(c, dict)
    )
    if not claims:
        return _unfaithful("no checkable claims in answer")

    claim_support = sum(c.supported for c in claims) / len(claims)
    relevance = _clamp01(verdict.get("question_relevance"))
    faithful = (
        claim_support >= _CLAIM_SUPPORT_FLOOR
        and relevance >= _RELEVANCE_FLOOR
    )
    return FaithfulnessReport(
        claim_support=claim_support,
        question_relevance=relevance,
        faithful=faithful,
        claims=claims,
        reason=str(verdict.get("reason", ""))[:200],
    )
