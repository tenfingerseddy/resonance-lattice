"""external_freshness — detect external fills gone STALE vs the LIVE WORLD (the genuinely-useful staleness).

`reverification.py` handles CORPUS drift: a cited corpus passage changed on disk, and `rlat refresh` re-syncs it.
It explicitly SKIPS external claims — an external fact's truth does NOT live in the corpus, so a corpus re-sync can
never re-check it. THIS module is the missing check: re-fetch the external SOURCE (the cited URL) and ask whether
it STILL supports the claim. That is the staleness no `refresh` can catch — a fetched fact the world has since
moved past (a version number, a price, a result, a "current" anything).

Two capabilities, both INJECTED (rlat never reaches the network itself):
  - `url_fetcher`: `url -> str | None` — the CURRENT text at a cited URL (the harness/skill wires WebFetch behind
    it). Returning None/"" for a URL means "couldn't fetch" — that claim records as `unknown`, never wrongly stale.
  - the freshness judge — an Anthropic-shaped `client` (metered), or, in the free `rlat-refresh-facts` skill, the
    in-session agent reading the re-fetched text itself.

v1 SURFACES: it returns a per-claim freshness verdict; it does NOT auto-evict, auto-retire, or auto-demote (the
same surface-for-review posture as the contradiction act layer). Acting on a confirmed-stale fact — re-author from
the fresh source, or demote/retire it — is a higher-stakes, human/policy-gated step. Best-effort: a failed fetch or
judge degrades to `unknown`, never a crash and never a wrongful eviction.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._llm import judge_json


@dataclass(frozen=True)
class ExternalClaimRef:
    """An active external claim and the URLs it rests on — the FREE enumeration (no fetch, no LLM).

    The free skill consumes this: it knows which claims to re-fetch and where. `source_urls` are the distinct
    cited URLs (an external fill is backed by >= 2 agreeing sources)."""

    claim_id: str
    content: str
    state: str
    source_urls: tuple[str, ...]


@dataclass(frozen=True)
class FreshnessOutcome:
    """Per-claim freshness verdict. `status` is one of:
      - "fresh"   — the re-fetched sources still support the claim.
      - "stale"   — the re-fetched sources no longer support it (the world moved on) — SURFACE for re-author.
      - "unknown" — could not re-fetch any source, or the judge could not decide; left untouched.
    v1 never mutates the claim; `status` is a signal for a human/agent to act on."""

    claim_id: str
    status: str
    refetched_urls: tuple[str, ...]
    reason: str


FRESHNESS_SYSTEM = """\
You decide whether a fact is STILL SUPPORTED by the CURRENT content of the sources it was drawn from. The fact was
verified earlier from these sources; their content may since have changed (the world moves on).

Answer with a single JSON object, nothing else:
{"supports": true|false, "reason": "<one short sentence>"}

supports=true if the CURRENT source content still makes the claim true (wording may differ; additions that don't
contradict are fine).
supports=false if the current content contradicts the claim, no longer states it, or has moved past it (a newer
value, version, or result) — i.e. the claim is now STALE.
"""


def _is_external_claim(claim) -> bool:
    # Guard on source FIRST: only a corpus claim carries CorpusFacts.citations — an experience/attribute claim
    # (also stored active in the insight list) has no citations field, so reading it would AttributeError and
    # break the "never raises" contract. An external fill is always a corpus claim with all-external citations.
    from .insight import all_external
    return getattr(claim, "source", None) == "corpus" and all_external(claim.facts.citations)


def external_claims(contents) -> list[ExternalClaimRef]:
    """Enumerate the ACTIVE external claims + their cited URLs — pure, no fetch, no LLM (the free path's input).

    An external claim is one whose citations are ALL external (carry a `source_url` / `external:` id). Only
    `state == "active"` claims are returned (a retired/stale claim is not being served, so its world-freshness is
    moot). Never raises."""
    out: list[ExternalClaimRef] = []
    for claim in getattr(contents, "insights", []) or []:
        if getattr(claim, "state", None) != "active" or not _is_external_claim(claim):
            continue
        urls = []
        for c in claim.facts.citations:
            u = c.source_url or (c.passage_id if c.passage_id.startswith("external:") else None)
            if u and u not in urls:
                urls.append(u)
        out.append(ExternalClaimRef(
            claim_id=claim.claim_id, content=claim.content, state=claim.state,
            source_urls=tuple(urls)))
    return out


def _support_message(claim_content: str, fetched: list[dict]) -> str:
    blocks = [f"--- CURRENT CONTENT OF {s['url']} ---\n{s['text']}\n" for s in fetched]
    return f"CLAIM:\n{claim_content}\n\nThe claim was drawn from these sources; their CURRENT content is:\n\n" + \
           "\n".join(blocks)


def recheck_external_freshness(
    km_path: str | Path,
    url_fetcher,
    client,
    *,
    model: str | None = None,
    limit: int | None = None,
    cost_cap_usd: float | None = None,
) -> list[FreshnessOutcome]:
    """Re-fetch each active external claim's cited URLs and judge whether they STILL support the claim.

    `url_fetcher(url) -> str | None` supplies the current text (None/"" = couldn't fetch). `client` is an
    Anthropic-shaped judge (inject a stub for tests). For each external claim: re-fetch its URLs; if none come
    back, record `unknown`; otherwise ask the judge whether the current content still supports the claim and record
    `fresh`/`stale`. Returns one outcome per external claim. v1 does NOT write anything back — it SURFACES.

    `limit` caps the number of judged claims; `cost_cap_usd` bounds cumulative LLM spend (the pass stops issuing
    new judge calls once observed spend crosses the cap; unreached claims are simply omitted). Never raises — a bad
    fetch or judge for one claim degrades that claim to `unknown` and the pass continues."""
    from . import archive
    from .._pricing import SONNET_MODEL, CostMeter

    use_model = model or SONNET_MODEL
    meter = CostMeter(cap_usd=cost_cap_usd)
    if client is None or url_fetcher is None:
        return []

    try:
        contents = archive.read(Path(km_path))
    except Exception:
        return []
    refs = external_claims(contents)
    if limit is not None:
        refs = refs[:limit]

    outcomes: list[FreshnessOutcome] = []
    for ref in refs:
        if meter.has_exceeded_cap():
            break  # budget spent; leave the rest for the next pass
        fetched: list[dict] = []
        for url in ref.source_urls:
            try:
                text = url_fetcher(url)
            except Exception:
                text = None
            if text and str(text).strip():
                fetched.append({"url": url, "text": str(text)})
        if not fetched:
            outcomes.append(FreshnessOutcome(
                claim_id=ref.claim_id, status="unknown", refetched_urls=(),
                reason="no source could be re-fetched"))
            continue
        try:
            verdict = judge_json(client, use_model, FRESHNESS_SYSTEM,
                                 _support_message(ref.content, fetched), max_tokens=200, meter=meter,
                                 temperature=0.0)  # a classifier verdict — deterministic, reproducible
        except Exception:
            verdict = {"_parse_error": "judge call failed"}
        urls = tuple(s["url"] for s in fetched)
        if not isinstance(verdict, dict) or verdict.get("_parse_error"):
            outcomes.append(FreshnessOutcome(
                claim_id=ref.claim_id, status="unknown", refetched_urls=urls,
                reason="judge could not decide"))
            continue
        supports = bool(verdict.get("supports", False))
        reason = str(verdict.get("reason", ""))[:200]
        outcomes.append(FreshnessOutcome(
            claim_id=ref.claim_id, status=("fresh" if supports else "stale"),
            refetched_urls=urls, reason=reason))
    return outcomes
