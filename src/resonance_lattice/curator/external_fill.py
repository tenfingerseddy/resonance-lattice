"""external_fill — author a VERIFIED claim from EXTERNAL sources for a TRUE gap.

`author.py` distils claims strictly from EXISTING corpus passages, so it fills SYNTHESIS gaps (scattered/implicit
corpus knowledge → one claim, the measured +0.082) but CANNOT fill a TRUE gap — one where the corpus genuinely
lacks the fact. The proven large value (+0.82, gap-fill grounding moat) came from serving a VERIFIED EXTERNAL fact
the model/corpus could not produce. This module is the missing source: the injected-capability path that fills a
true gap from outside.

Design invariants (match the project's North Star + the gap-fill safety lessons):
- **No required external service.** rlat stays dependency-free; the path is OFF unless the harness injects a
  `fetcher`. The library never imports a network client — the fetcher is a plain callable the caller supplies (the
  skill/harness wires WebSearch/WebFetch behind it).
- **Never serve an unverified fact (the safety guard).** A claim is authored ONLY if it is supported by at least
  `min_sources` INDEPENDENT sources that AGREE. Single-source, disagreeing, or thin evidence → no fill. This is the
  never-serve-wrong guard from the cross-source safety work, enforced structurally.
- **Same shape as a corpus fill.** Returns a `PendingFill` whose `evidence_passages` carry the fetched source text
  + a content hash + the URL as provenance, so the EXISTING faithfulness gate + promotion (`promote_if_faithful`)
  consume it unchanged — nothing downstream needs to know the source was external.
- **Born low-trust, earns by outcomes.** Like every fill; the only difference is provenance (`drift_status`
  "external" + the URL), so an inspector can see the claim came from outside the corpus.

Pure but for the two injected capabilities (the `fetcher` and the LLM `client`); never raises — a failed fetch or
author degrades to None, never breaks the growth loop.
"""
from __future__ import annotations

import hashlib

from .author import PendingFill

# A claim must be backed by at least this many independent, agreeing sources before it may be served. Two is the
# floor for "cross-source corroboration"; the harness can raise it for higher-stakes corpora.
_MIN_SOURCES = 2
_MAX_SOURCES = 5

EXTERNAL_AUTHOR_SYSTEM = """\
You distil ONE verified factual claim from independent external sources, for a question the local knowledge base
cannot answer.

Rules:
- The claim must rest ENTIRELY on the sources below, and be SUPPORTED BY AT LEAST TWO of them that AGREE on the
  fact. Add NO outside knowledge.
- If the sources disagree, are off-topic, or only one supports the fact, return an EMPTY claim — an unverified
  fact must NOT be served.
- Make it ONE self-contained factual claim (1-2 sentences) a reader can trust and trace to the cited sources.

Reply with a single JSON object, nothing else:
{"claim": "<the verified claim, or empty>", "supporting_sources": [<1-based source numbers that agree>], "agree": <true|false>}
"""


def _sources_message(sources: list[dict]) -> str:
    blocks = []
    for i, s in enumerate(sources, start=1):
        url = s.get("url", "?")
        blocks.append(f"--- SOURCE {i} ({url}) ---\n{s.get('text', '')}\n")
    return "SOURCES:\n\n" + "\n".join(blocks)


def author_external_fill(
    question: str,
    client,
    fetcher,
    *,
    model: str | None = None,
    min_sources: int = _MIN_SOURCES,
    max_sources: int = _MAX_SOURCES,
) -> PendingFill | None:
    """Author one VERIFIED, PENDING fill for `question` from external sources.

    `fetcher` is an injected callable `question -> list[{"url", "text"}]` (the harness wires WebSearch/WebFetch
    behind it; rlat itself never reaches the network). The function fetches candidate sources, requires at least
    `min_sources` usable ones, asks the LLM to distil ONE claim supported by ≥ `min_sources` AGREEING sources, and
    returns a `PendingFill` whose evidence is those sources — the same shape the corpus author returns, so the
    existing faithfulness gate + promotion consume it unchanged.

    Returns None when: no client / no fetcher / empty question; fewer than `min_sources` usable sources fetched;
    the author returns an empty claim, `agree=false`, or fewer than `min_sources` distinct supporting sources
    (the never-serve-an-unverified-fact guard). Never raises — any failure degrades to None."""
    if client is None or fetcher is None or not str(question).strip():
        return None
    try:
        fetched = fetcher(question) or []
        usable = [s for s in fetched if isinstance(s, dict) and str(s.get("text", "")).strip()][:max_sources]
        if len(usable) < min_sources:
            return None  # cannot cross-verify -> do not author

        from .._pricing import SONNET_MODEL
        from ..store._llm import judge_json

        out = judge_json(
            client, model or SONNET_MODEL, EXTERNAL_AUTHOR_SYSTEM,
            _sources_message(usable), max_tokens=800,
        )
        if not isinstance(out, dict) or out.get("_parse_error"):
            return None
        claim = str(out.get("claim", "")).strip()
        agree = bool(out.get("agree"))
        support = {i for i in out.get("supporting_sources", []) if isinstance(i, int) and 1 <= i <= len(usable)}
        # The never-serve-unverified guard: a real claim, the model asserts agreement, AND >= min_sources distinct
        # sources actually back it. Any failure -> no fill (silence beats an unverified served fact).
        if not claim or not agree or len(support) < min_sources:
            return None

        evidence = []
        for i in sorted(support):
            s = usable[i - 1]
            text = str(s.get("text", ""))
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            url = str(s.get("url", "external"))
            evidence.append({
                # synthetic non-corpus id (NOT in the registry) so the promote path's
                # passage_id+content_hash filter passes and the citation is flagged
                # external (skips the corpus drift check + the corpus-coverage gate).
                "passage_id": "external:" + digest[:16],
                "source_url": url,
                "source_file": url,
                "content_hash": "sha256:" + digest,
                "text": text,
                "score": 1.0,
                "drift_status": "external",
            })
        if len(evidence) < min_sources:
            return None
        return PendingFill(intent=str(question).strip(), claim=claim, evidence_passages=evidence)
    except Exception:
        return None
