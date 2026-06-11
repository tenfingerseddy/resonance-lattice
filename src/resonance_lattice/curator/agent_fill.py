"""agent_fill — LAND a fact the AGENT fetched + verified, with NO metered LLM gate (the free loop's landing).

The free skills (`rlat-curate`, the `rlat-gap-scan` create tier, `rlat-refresh-facts` re-author) fetch sources
with their own WebSearch/WebFetch and judge cross-source agreement THEMSELVES (agent-as-judge — proven as good as
a metered judge on this task). This is the clean entry to LAND that verified fact: it builds external evidence from
the fetched sources (the same shape `external_fill` produces) and lands it through `promote_if_faithful`'s
caller-verified path (`client=None` + an explicit faithfulness), so no Anthropic API key is needed.

SAFETY: the CALLER asserts grounding (it verified the claim traces to the sources + that ≥ `min_sources` of them
AGREE). This substitutes the agent's judgement for the metered faithfulness gate — valid because subscription
agents judge grounding as well as a metered model (measured). Everything else still applies structurally: ≥
`min_sources` DISTINCT sources are required here, and downstream the compression test + the ≥2-distinct-citation +
trust gates still gate the write. `provenance` sets the trust tier (default "verified_external" for a ≥2-source
agreeing fill). Never raises — any failure degrades to "not landed".
"""
from __future__ import annotations

import hashlib

_MIN_SOURCES = 2


def land_external_fact(
    km_path,
    question: str,
    claim: str,
    sources: list[dict],
    *,
    provenance: str = "verified_external",
    faithfulness: float = 0.9,
    min_sources: int = _MIN_SOURCES,
):
    """Land a verified external `claim` into the band from the agent-fetched `sources`.

    `sources` is a list of `{"url", "text"}` the agent fetched AND judged to agree. Requires ≥ `min_sources`
    DISTINCT (by url) usable sources and a non-empty claim, else `(False, [])` — the never-serve-an-unverified
    /single-source guard. Returns `(landed: bool, outcomes)`. Pure but for the archive read/write inside
    promotion; never raises."""
    try:
        from ..store.promotion import promote_if_faithful

        if not str(claim).strip():
            return False, []
        seen: set[str] = set()
        distinct: list[dict] = []
        for s in sources or []:
            if not (isinstance(s, dict) and str(s.get("text", "")).strip()):
                continue
            url = str(s.get("url", "")).strip()
            if url and url not in seen:
                seen.add(url)
                distinct.append({"url": url, "text": str(s["text"])})
        if len(distinct) < min_sources:
            return False, []  # cannot cross-verify → do not land

        evidence = []
        for s in distinct:
            digest = hashlib.sha256(s["text"].encode("utf-8")).hexdigest()
            evidence.append({
                # synthetic non-corpus id so promotion's external path (passed_external) consumes it: skips the
                # corpus-coverage gate + the corpus drift check, carries the URL as provenance.
                "passage_id": "external:" + digest[:16],
                "source_url": s["url"],
                "source_file": s["url"],
                "content_hash": "sha256:" + digest,
                "text": s["text"],
                "score": 1.0,
                "drift_status": "external",
            })
        report, outcomes = promote_if_faithful(
            km_path, question=str(question), answer=str(claim),
            evidence_passages=evidence, client=None, faithfulness=faithfulness,
            provenance=provenance)
        return bool(any(o.promoted for o in outcomes)), outcomes
    except Exception:
        return False, []
