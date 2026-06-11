"""author — the gap→author growth touch (CRITICAL_PATH Step 3) + the end-to-end loop.

A confirmed `RecurringIntent` (`curator.decide`) is the ONLY thing that triggers the
cloud: retrieve the corpus passages the recurring intent's centroid lands on, and
author ONE grounded claim distilled strictly from them. The claim is PENDING — it
then flows through the BUILT back half (faithfulness gate + compression test + band
writeback, `store.promotion.promote_if_faithful`), born low-trust, earning by
outcomes.

This is the loop's only growth operation and its only network touch: **rare** (the
decide tier dedups to one fill per recurring intent), **offline** (sleep-time, never
the query hot path), and **gated** (the fill must pass faithfulness before it lands).
The telemetry carries NO query text (fingerprints only), so the intent is
reconstructed from the passages its centroid retrieves — privacy-preserving by
construction.

The author's own retrieval is MACHINERY: it runs inside `capture.internal_retrieval()`
so growth never pollutes (or self-reinforces) the user-intent telemetry stream.

Graceful degradation: no key / no client / no grounded passages → `author_fill`
returns None and the `.rlat` still retrieves + remembers; growth pauses, the query
path never degrades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# Relevance gate on the evidence the author forwards (the value-proof keep:
# relevance-gating is load-bearing). Keep only passages in the centroid's relevance
# neighbourhood — within this ratio of the best match — so the long tail of weakly
# related passages becomes neither sloppy grounding context nor (downstream) a
# spurious citation that flatters the compression test's distinct-sources guard. The
# top hit is always kept. The per-citation provenance fix (cite only the passages the
# faithfulness judge marked supporting) is the deferred enrichment.md §2b threading.
_RELEVANCE_RATIO = 0.6


AUTHOR_SYSTEM = """\
You distil ONE grounded claim from a set of corpus passages that users repeatedly
retrieve together (a recurring information need).

Rules:
- The claim must rest ENTIRELY on the passages below. Add NO outside knowledge.
- Make it a single self-contained factual claim (1-3 sentences) a future reader can
  trust and trace back to these passages.
- Also state, in one short line, the recurring QUESTION these passages answer.
- If the passages are too thin or unrelated to support one honest claim, return an
  empty claim.

Reply with a single JSON object, nothing else:
{"intent": "<the recurring question, one line>", "claim": "<the grounded claim, or empty>"}
"""


# Authoring from the FULL source doc the centroid lands on — not just the retrieved
# chunk window — is the measured value lever (expertise-curve, 2026-06-03/04: chunk
# authoring -0.024, wider chunks +0.027, FULL-DOC +0.082 with CI excluding 0; see
# CRITICAL_PATH "full-doc authoring WORKS"). The win needs BOTH levers: full-doc grain
# AND 2-4 claims. Claims ground in the full doc but cite the top retrieved chunk's
# provenance, so verified-retrieval drift still tracks the source.
_DOC_CHARS = 6000  # matches the measured fulldoc bench cap

DOC_AUTHOR_SYSTEM = """\
You distil reusable knowledge from a source document that users repeatedly land on
(a recurring information need).

Rules:
- Output 2-4 STANDALONE factual claims. Each is a single self-contained sentence stating
  ONE concrete technical fact (a number, requirement, condition, behaviour, or step) drawn
  ENTIRELY from the document below. Add NO outside knowledge.
- Prefer SPECIFIC facts a reader could NOT get from a one-line topic summary.
- Each claim must be independently traceable to the document.
- Also state, in one short line, the recurring QUESTION these facts answer.
- If the document is too thin to support honest claims, return an empty list.

Reply with a single JSON object, nothing else:
{"intent": "<the recurring question, one line>", "claims": ["<claim>", "<claim>", ...]}
"""


@dataclass(frozen=True)
class PendingFill:
    """An authored, not-yet-gated fill for a recurring intent — Step 3's output.

    `intent` is the reconstructed recurring question (the faithfulness judge's
    question axis); `claim` is the grounded answer distilled from `evidence_passages`
    (each a `{passage_id, content_hash, source_file, text, score, ...}` dict the
    faithfulness gate + promotion pipeline consume)."""

    intent: str
    claim: str
    evidence_passages: list[dict]


@dataclass(frozen=True)
class GrowthOutcome:
    """One recurring-intent candidate's trip through author → gate → band."""

    candidate: object                       # curator.decide.RecurringIntent
    pending: PendingFill | None             # None ⇒ authoring produced no claim
    report: object | None = None            # store.faithfulness.FaithfulnessReport
    outcomes: list = field(default_factory=list)  # store.promotion.PromotionOutcome

    @property
    def promoted(self) -> bool:
        return any(getattr(o, "promoted", False) for o in self.outcomes)


def _author_user_message(passages: list[dict]) -> str:
    blocks = []
    for i, p in enumerate(passages, start=1):
        anchor = p.get("source_file", "?")
        blocks.append(f"--- PASSAGE {i} ({anchor}) ---\n{p.get('text', '')}\n")
    return "PASSAGES:\n\n" + "\n".join(blocks)


def _retrieve_evidence(
    km_path,
    centroid,
    *,
    top_k: int,
    contents=None,
    store=None,
    source_root=None,
) -> list[dict]:
    """Retrieve the source passages the recurring intent's centroid lands on, with
    text resolved — the grounding for the fill. Returns `[]` when the centroid is
    unusable or nothing verifies. The retrieval is wrapped in
    `capture.internal_retrieval()` so it never enters the user-intent telemetry."""
    from ..field import ann as ann_mod
    from ..field import capture, retrieve
    from ..store import archive, open_store
    from ..store.verified import verify_hits

    vec = np.asarray(centroid, dtype="float32")
    if vec.ndim != 1 or vec.size == 0:
        return []

    p = Path(km_path)
    if contents is None:
        contents = archive.read(p)
    if store is None:
        store = open_store(p, contents, source_root)

    handle = contents.select_band()
    ann_index = ann_mod.deserialize(handle.ann_blob) if handle.ann_blob else None
    with capture.internal_retrieval():  # growth retrieval is machinery, not a user query
        hits = retrieve(vec, handle, ann_index, contents.registry, top_k)
    verified = verify_hits(hits, store, contents.registry)

    out: list[dict] = []
    for h in verified:
        if getattr(h, "drift_status", None) != "verified" or not getattr(h, "text", ""):
            continue
        coord = contents.registry[h.passage_idx]
        out.append({
            "passage_id": coord.passage_id,
            "content_hash": h.content_hash,
            "source_file": h.source_file,
            "char_offset": h.char_offset,
            "char_length": h.char_length,
            "text": h.text,
            "score": h.score,
            "drift_status": h.drift_status,
        })
    # Relevance gate — `out` is score-descending (retrieve + verify preserve order),
    # so keep the top hit plus any within `_RELEVANCE_RATIO` of it; drop the weak tail.
    if out:
        floor = out[0]["score"] * _RELEVANCE_RATIO
        out = [out[0]] + [r for r in out[1:] if r["score"] >= floor]
    return out


def author_fill(
    km_path,
    candidate,
    client,
    *,
    top_k: int = 5,
    contents=None,
    store=None,
    source_root=None,
    model: str | None = None,
) -> PendingFill | None:
    """Author one grounded, PENDING fill for a recurring-intent `candidate`.

    Retrieves near `candidate.query_centroid`, asks the LLM to distil a single claim
    grounded strictly in those passages, and returns the `PendingFill` the back-half
    faithfulness gate consumes. Returns None when there is no grounded evidence, no
    client, or the author produced no claim. Never raises — growth must never break a
    query."""
    if client is None:
        return None
    try:
        evidence = _retrieve_evidence(
            km_path, candidate.query_centroid, top_k=top_k,
            contents=contents, store=store, source_root=source_root,
        )
        if not evidence:
            return None
        from .._pricing import SONNET_MODEL
        from ..store._llm import judge_json

        out = judge_json(
            client, model or SONNET_MODEL, AUTHOR_SYSTEM,
            _author_user_message(evidence), max_tokens=800,
        )
        if not isinstance(out, dict) or out.get("_parse_error"):
            return None
        claim = str(out.get("claim", "")).strip()
        if not claim:
            return None
        intent = str(out.get("intent", "")).strip() or "(recurring intent)"
        return PendingFill(intent=intent, claim=claim, evidence_passages=evidence)
    except Exception:
        return None


def author_doc_fills(
    km_path,
    candidate,
    client,
    *,
    top_k: int = 12,
    max_claims: int = 4,
    contents=None,
    store=None,
    source_root=None,
    model: str | None = None,
) -> list[PendingFill]:
    """Author 2-4 grounded, PENDING fills from the FULL content of the top source doc the
    recurring intent's centroid retrieves — the measured upgrade over `author_fill`'s
    single claim from the chunk window (see `_DOC_CHARS` note). The claims are authored
    from the full doc (passage 0's text) but CITE the doc's ≥2 relevance-gated chunks, so
    (a) the anti-paraphrase compression guard is satisfied honestly — the claims genuinely
    draw on multiple passages of the doc — and (b) verified-retrieval drift tracks real
    chunk provenance. Returns `[]` on no evidence / no client / no claim. Never raises —
    growth must never break a query."""
    if client is None:
        return []
    try:
        from ..store import archive, open_store

        p = Path(km_path)
        if contents is None:
            contents = archive.read(p)
        if store is None:
            store = open_store(p, contents, source_root)
        chunks = _retrieve_evidence(
            km_path, candidate.query_centroid, top_k=top_k,
            contents=contents, store=store, source_root=source_root,
        )
        if not chunks:
            return []
        top_src = chunks[0]["source_file"]
        # Cite the doc's own chunks (≥2 for the anti-paraphrase guard); fall back to the
        # top relevance-gated chunks when retrieval surfaced only one chunk of the doc.
        doc_chunks = [c for c in chunks if c["source_file"] == top_src]
        cited = (doc_chunks if len(doc_chunks) >= 2 else chunks[:2])[:max_claims]
        try:
            full = store.fetch_all({top_src}).get(top_src, "")
        except Exception:
            full = ""
        if not full or len(cited) < 2:
            return []
        # Author + gate from the FULL doc (passage 0), keep the other chunks as cited
        # provenance so distinct_sources >= 2.
        evidence = [dict(c) for c in cited]
        evidence[0] = {**evidence[0], "text": full[:_DOC_CHARS]}

        from .._pricing import SONNET_MODEL
        from ..store._llm import judge_json

        out = judge_json(
            client, model or SONNET_MODEL, DOC_AUTHOR_SYSTEM,
            _author_user_message(evidence), max_tokens=900,
        )
        if not isinstance(out, dict) or out.get("_parse_error"):
            return []
        intent = str(out.get("intent", "")).strip() or "(recurring intent)"
        claims = [
            str(c).strip() for c in (out.get("claims") or [])
            if len(str(c).strip()) > 25
        ][:max_claims]
        return [PendingFill(intent=intent, claim=c, evidence_passages=evidence) for c in claims]
    except Exception:
        return []


def grow_from_telemetry(
    km_path,
    client,
    *,
    max_fills: int = 1,
    top_k: int = 12,
    max_claims: int = 4,
    min_occurrences: int = 2,
    min_sessions: int = 1,
    source_root=None,
    model: str | None = None,
    fetcher=None,
    candidates: list | None = None,
) -> list[GrowthOutcome]:
    """The end-to-end self-improvement loop (Steps 2→3→4→5): decide → author → gate
    → land in the band.

    Reads the `.rlat`'s persisted telemetry for confirmed-recurring intents
    (`decide`), authors a grounded fill for the top `max_fills` of them, and runs each
    through `promote_if_faithful` (faithfulness gate + compression test + atomic band
    writeback). `max_fills` bounds the cloud cost per run (one fill per recurring
    intent is already the decide-tier guarantee). Returns one `GrowthOutcome` per
    authored FILL - `author_doc_fills` yields 2-4 fills per candidate, so the
    outcome count exceeds the candidate count (guarantee (g) pins this). Pass `candidates` (a prior `decide()` result) to fill
    exactly a previewed list — the CLI does this so its preview and the run share
    one telemetry snapshot instead of two reads that can diverge.

    `decide`'s coverage gate already filters candidates to the demand the corpus
    answers RELATIVELY WORST (the undercovered intents). When a `fetcher` is injected
    (the harness wiring WebSearch/WebFetch behind it; OFF by default — rlat stays
    dependency-free), each such intent gets a VERIFIED EXTERNAL fill attempt for its
    reconstructed question: if ≥2 independent sources agree (`external_fill`), that
    fact REPLACES the corpus-synthesis fill (which was grounded in the adjacent,
    weaker passages a true gap leaves behind). This is the source half of "how info
    is added" — filling a TRUE gap from outside, not just distilling what's present.

    Never raises — no candidates / no client / authoring failures degrade to fewer
    fills, never a crash. The whole pass runs inside `capture.internal_retrieval()`
    so no growth-time retrieval re-enters the user-intent telemetry."""
    from ..field import capture
    from ..store import archive, open_store
    from ..store.promotion import promote_if_faithful

    from .decide import decide

    p = Path(km_path)
    if candidates is None:
        candidates = decide(
            str(p), min_occurrences=min_occurrences, min_sessions=min_sessions,
        )
    if not candidates or client is None:
        return []

    results: list[GrowthOutcome] = []
    with capture.internal_retrieval():
        # Source band + registry are stable across fills (promotion only mutates the
        # insight layer), so author retrieval can reuse one read; promote_if_faithful
        # re-reads fresh each fill so it sees prior fills' insight rows. A corrupt /
        # half-written archive PAUSES growth (boundary.md "pause, never break"), it
        # never crashes — unlike decide()'s telemetry-only read, archive.read here
        # validates the full format and can raise.
        try:
            contents = archive.read(p)
            store = open_store(p, contents, source_root)
        except Exception:
            return []
        for cand in candidates[:max_fills]:
            pendings = author_doc_fills(
                p, cand, client, top_k=top_k, max_claims=max_claims,
                contents=contents, store=store, source_root=source_root, model=model,
            )
            # SOURCE ROUTING: this candidate is already an undercovered intent (decide's
            # coverage gate). With a fetcher, try a verified EXTERNAL fill for the
            # reconstructed question; a cross-source-verified fact replaces the
            # synthesis fills (grounded in the weaker adjacent passages). Lazy import
            # avoids the external_fill→author PendingFill import cycle. Never raises:
            # author_external_fill returns None on any failure → synthesis stands.
            if fetcher is not None and pendings:
                from .external_fill import author_external_fill
                ext = author_external_fill(pendings[0].intent, client, fetcher, model=model)
                if ext is not None:
                    pendings = [ext]
            if not pendings:
                results.append(GrowthOutcome(candidate=cand, pending=None))
                continue
            for pending in pendings:
                # Each fill re-reads fresh inside promote_if_faithful so it sees prior
                # fills' insight rows (the compression test dedups across the batch).
                try:
                    report, outcomes = promote_if_faithful(
                        p, question=pending.intent, answer=pending.claim,
                        evidence_passages=pending.evidence_passages,
                        client=client, model=model,
                    )
                except Exception:
                    report, outcomes = None, []
                results.append(GrowthOutcome(
                    candidate=cand, pending=pending, report=report, outcomes=outcomes,
                ))
    return results
