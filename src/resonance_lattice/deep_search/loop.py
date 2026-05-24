"""Deep-search orchestrator: plan → search → refine → synthesize.

Reuses the single-recipe retrieval path (`field.retrieve` against the
band the KM exposes via `store.archive.select_band`) and the existing
verified-hit shape, so a deep-search hop costs the same as one
`rlat search` invocation but stays in-process (no subprocess overhead,
encoder + ANN warm across hops).

Cost model: one planner call (~100 out tokens) + up to `max_hops - 1`
retrieve+refine pairs (each ~400 out tokens) + optional one synth call
when hops exhaust without an `answer` decision. Bench numbers (Fabric
corpus, 63 questions, 4 hops max): mean $0.010/q vs augment $0.004/q.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .._pricing import SONNET_MODEL, cost_usd
from ..cli import _namecheck
from ..field import ann, retrieve, retrieve_insight
from ..field.encoder import Encoder
from ..store import archive, open_store
from ..store.verified import InsightHit, VerifiedHit, verify_hits, verify_insight_hits
from .prompts import (
    GIVE_UP_ANSWER,
    NAME_MISMATCH_ANSWER,
    PLANNER_SYSTEM,
    REFINER_SYSTEM,
    SYNTHESIZER_SYSTEM,
)
from .types import DeepSearchHop, DeepSearchResult


def _render_evidence_block(
    query: str,
    source_hits: list[VerifiedHit],
    insight_hits: list[InsightHit],
) -> str:
    """Format a hop's hits the way the refiner expects.

    Source passages and earned insights render under distinct labels so
    the refiner weighs a verified primary source differently from a
    derived synthesis — trust contract: the two layers stay visibly
    distinct at every output surface.
    """
    parts = [f"--- Search query: {query!r} ---"]
    rendered = False
    for h in source_hits:
        if h.drift_status == "missing" or not h.text:
            continue
        anchor = f"{h.source_file}:{h.char_offset}+{h.char_length}"
        parts.append(
            f"[source {anchor}] (score {h.score:.3f}) {h.text.strip()}"
        )
        rendered = True
    for h in insight_hits:
        if not h.content:
            continue
        parts.append(
            f"[earned insight {h.insight_id} · verdict={h.verdict_state} "
            f"· confidence {h.confidence:.2f}] (score {h.score:.3f}) "
            f"{h.content.strip()}"
        )
        rendered = True
    if not rendered:
        parts.append("(no hits)")
    return "\n".join(parts)


def _retrieve_hop(
    *, encoder: Encoder, query: str, handle: Any, ann_index: Any,
    insight_handle: Any, insight_ann: Any, contents: Any, store: Any,
    top_k: int,
) -> tuple[list[VerifiedHit], list[InsightHit]]:
    """One in-process retrieve-and-verify call across source + insight layers.

    Insight retrieval is filtered to `accepted` rows (the
    `verify_insight_hits` default) — deep-search builds on earned
    synthesis, never on stale or candidate insights. A corpus with no
    insight layer takes the source-only path unchanged.
    """
    q_emb = np.asarray(encoder.encode([query])[0])
    src_raw = retrieve(q_emb, handle, ann_index, contents.registry, top_k)
    source_hits = verify_hits(src_raw, store, contents.registry)

    insight_hits: list[InsightHit] = []
    if insight_handle is not None:
        ins_raw = retrieve_insight(
            q_emb, insight_handle.band, insight_ann, top_k,
        )
        insight_hits = verify_insight_hits(ins_raw, contents.insights)
    return source_hits, insight_hits


def _llm_call(client: Any, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
    """Wrap the Anthropic SDK call into a (text, in_tokens, out_tokens) triple."""
    msg = client.messages.create(
        model=SONNET_MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return (
        msg.content[0].text.strip(),
        int(msg.usage.input_tokens),
        int(msg.usage.output_tokens),
    )


def _parse_refiner_action(raw: str) -> dict | None:
    """First `{...}` action object embedded in the refiner's output. None on parse fail.

    `JSONDecoder.raw_decode` honours strings + escapes, so the `answer`
    field can legitimately contain `{` / `}` (code placeholders, set
    literals) without truncating the match — unlike a flat regex.
    """
    decoder = json.JSONDecoder()
    i = raw.find("{")
    while i != -1:
        try:
            obj, _ = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            i = raw.find("{", i + 1)
            continue
        if isinstance(obj, dict) and "action" in obj:
            return obj
        i = raw.find("{", i + 1)
    return None


def _synthesize_from_evidence(
    client: Any, question: str, evidence_blocks: list[str],
) -> tuple[str, int, int]:
    """One synthesizer call over accumulated evidence.

    Used both when the hop budget exhausts without an `answer` decision
    and when the refiner's output fails to parse — either way the loop
    still owes the caller a grounded answer, not raw model output.
    """
    evidence = "\n\n".join(evidence_blocks)[:10000]
    synth_prompt = (
        f"Question: {question}\n\n"
        f"All evidence collected:\n\n{evidence}\n\n"
        f"Provide a concise answer based ONLY on the evidence above. "
        f"If the evidence doesn't cover the question, say so."
    )
    return _llm_call(client, SYNTHESIZER_SYSTEM, synth_prompt, max_tokens=1000)


def _dedupe_passages(
    verified: list[VerifiedHit], registry=None,
) -> list[dict]:
    """Project verified hits to citation-ready dicts, deduped on (source_file, char_offset).

    `registry` (optional `list[PassageCoord]`) lets the projection carry
    the stable `passage_id` + `content_hash` for each hit — needed by
    the lensed-knowledge synthesis_candidate writeback path. When `None`
    (legacy callers), those fields are omitted.
    """
    seen: set[tuple[str, int]] = set()
    out: list[dict] = []
    for h in verified:
        key = (h.source_file, h.char_offset)
        if key in seen:
            continue
        seen.add(key)
        entry = {
            "source_file": h.source_file,
            "char_offset": h.char_offset,
            "char_length": h.char_length,
            "score": h.score,
            "drift_status": h.drift_status,
            "text": h.text,
        }
        if registry is not None and h.passage_idx < len(registry):
            coord = registry[h.passage_idx]
            entry["passage_id"] = coord.passage_id
            entry["content_hash"] = coord.content_hash
        out.append(entry)
    return out


def deep_search(
    km_path: Path,
    question: str,
    *,
    client: Any,
    max_hops: int = 4,
    top_k: int = 5,
    source_root: str | None = None,
    strict_names: bool = False,
) -> DeepSearchResult:
    """Multi-hop research loop against `km_path`.

    Loads the KM once, encodes once, retrieves in-process per hop. The
    refiner decides at each hop whether to answer, search again, or
    give up; on `answer` the loop returns immediately. If `max_hops` is
    exhausted without an `answer`, a one-shot synth call summarises
    accumulated evidence.

    Name-check runs over the union of all retrieved passages against
    the original question after the loop terminates. When distinctive
    tokens are missing:
      - default: prepend a refusal directive to `result.answer`
      - `strict_names=True`: replace the answer with the standard
        name-mismatch refusal and skip any further work

    `client` is an instance of `anthropic.Anthropic`; the loop uses
    only `client.messages.create`. Inject a fake here for tests.
    """
    # Use the underlying archive/store APIs directly so library callers get
    # a raised exception instead of `sys.exit(1)`. CLI surfaces translate
    # the exception into a friendly error themselves.
    contents = archive.read(km_path)
    handle = contents.select_band()
    ann_index = ann.deserialize(handle.ann_blob) if handle.ann_blob else None
    store = open_store(km_path, contents, source_root)

    # Insight layer — retrieved alongside source per architecture §10
    # (the canonical promotion loop's T0). Handle + ANN resolved once;
    # None when the corpus has no insight layer (source-only path).
    insight_handle = contents.insight_band() if contents.insights else None
    insight_ann = (
        ann.deserialize(insight_handle.ann_blob)
        if insight_handle is not None and insight_handle.ann_blob
        else None
    )

    encoder = Encoder()
    result = DeepSearchResult(question=question, answer="")

    # Hop 1: planner generates the first query.
    plan_text, in_t, out_t = _llm_call(
        client, PLANNER_SYSTEM, f"Question: {question}", max_tokens=100,
    )
    result.input_tokens += in_t
    result.output_tokens += out_t
    current_query = plan_text.split("\n")[0].strip() or question
    result.hops.append(DeepSearchHop(n=1, kind="plan", query=current_query))

    all_verified: list[VerifiedHit] = []
    all_insight_hits: list[InsightHit] = []
    evidence_blocks: list[str] = []
    # Passage-text-only mirror of evidence_blocks. Used for name-check so
    # the check matches what the LLM saw FROM THE CORPUS, not what the
    # query header text echoed of the question itself. (A planner that
    # picks `"MVE default action"` as the search query would otherwise
    # cause namecheck to falsely pass on `MVE` because the query string
    # appears in the rendered evidence.)
    passage_blocks: list[str] = []
    queries_tried: list[str] = []
    # The slice of the evidence the LLM saw — same truncation point,
    # but applied to passage text only.
    passages_seen_by_llm = ""

    for hop_n in range(2, max_hops + 1):
        try:
            source_hits, insight_hits = _retrieve_hop(
                encoder=encoder, query=current_query, handle=handle,
                ann_index=ann_index, insight_handle=insight_handle,
                insight_ann=insight_ann, contents=contents, store=store,
                top_k=top_k,
            )
        except Exception as e:
            result.hops.append(DeepSearchHop(
                n=hop_n, kind="search_failed", query=current_query,
                error=str(e)[:200],
            ))
            break

        queries_tried.append(current_query)
        # name-check + candidate citations stay source-anchored: insights
        # are derived, so they neither satisfy a name-check nor become a
        # new candidate's provenance binding.
        all_verified.extend(source_hits)
        all_insight_hits.extend(insight_hits)
        evidence_blocks.append(
            _render_evidence_block(current_query, source_hits, insight_hits)
        )
        passage_blocks.append("\n".join(
            v.text for v in source_hits
            if v.drift_status != "missing" and v.text
        ))
        result.hops.append(DeepSearchHop(
            n=hop_n, kind="search", query=current_query,
            n_passages=len(source_hits), n_insights=len(insight_hits),
        ))

        # Refiner decides next action.
        evidence = "\n\n".join(evidence_blocks)
        evidence_for_llm = evidence[:8000]
        passages_seen_by_llm = "\n\n".join(passage_blocks)[:8000]
        prompt = (
            f"Question: {question}\n\n"
            f"Evidence collected so far ({len(queries_tried)} queries tried):\n\n"
            f"{evidence_for_llm}\n\n"
            f"What's your next action? (answer / search / give_up)"
        )
        # 400 was tight for `answer` actions whose JSON carries the full
        # synthesis; truncation mid-string bust parses.
        raw, in_t, out_t = _llm_call(
            client, REFINER_SYSTEM, prompt, max_tokens=1000,
        )
        result.input_tokens += in_t
        result.output_tokens += out_t

        action = _parse_refiner_action(raw)
        if action is None:
            # The refiner's output wasn't parseable JSON (commonly an
            # `answer` value with unescaped quotes). Recover with a synth
            # call rather than handing back raw model text.
            result.hops.append(DeepSearchHop(
                n=hop_n, kind="parse_failed", error=raw[:300],
            ))
            synth_text, in_t, out_t = _synthesize_from_evidence(
                client, question, evidence_blocks,
            )
            result.input_tokens += in_t
            result.output_tokens += out_t
            result.answer = synth_text
            result.hops.append(DeepSearchHop(
                n=hop_n + 1, kind="synth_after_parse_fail",
            ))
            break

        kind = action.get("action")
        if kind == "answer":
            result.hops.append(DeepSearchHop(
                n=hop_n, kind="decide_answer", action=kind,
            ))
            result.answer = action.get("answer", "")
            break
        if kind == "give_up":
            result.hops.append(DeepSearchHop(
                n=hop_n, kind="decide_give_up", action=kind,
            ))
            result.answer = GIVE_UP_ANSWER
            break
        if kind == "search":
            result.hops.append(DeepSearchHop(
                n=hop_n, kind="decide_search", action=kind,
                query=action.get("query", current_query),
            ))
            current_query = action.get("query", current_query)
            continue
        # Unknown action — fall through to synth.
        break
    else:
        # Hops exhausted without `answer`. Synthesise from accumulated evidence.
        passages_seen_by_llm = "\n\n".join(passage_blocks)[:10000]
        synth_text, in_t, out_t = _synthesize_from_evidence(
            client, question, evidence_blocks,
        )
        result.input_tokens += in_t
        result.output_tokens += out_t
        result.answer = synth_text
        result.hops.append(DeepSearchHop(
            n=max_hops + 1, kind="synth_after_max_hops",
        ))

    result.evidence_passages = _dedupe_passages(all_verified, contents.registry)
    # Insight ids engaged across all hops, deduped on best score, rank-ordered.
    best_insight_score: dict[str, float] = {}
    for ih in all_insight_hits:
        if (ih.insight_id not in best_insight_score
                or ih.score > best_insight_score[ih.insight_id]):
            best_insight_score[ih.insight_id] = ih.score
    result.insight_ids = [
        iid for iid, _ in sorted(
            best_insight_score.items(), key=lambda kv: -kv[1],
        )
    ]

    # Empty-answer paths (search_failed, unknown action) shouldn't return
    # CLI rc=0 with a silent empty string. Surface it as a refusal so the
    # consumer LLM / shell caller can act on it.
    if not result.answer:
        result.answer = (
            "I cannot produce an answer — the deep-search loop terminated "
            "before reaching a decision. See `hops` for details."
        )
        result.cost_usd = cost_usd(result.input_tokens, result.output_tokens)
        return result

    # Name-check against the passage text the LLM actually saw — same
    # truncation point as the evidence the refiner / synthesizer
    # consumed, but the QUERY HEADERS are stripped. A planner that
    # echoes question tokens into its query (e.g. picks `"MVE default
    # action"` for an MVE question) would otherwise cause namecheck to
    # falsely pass on a token that only appears in the query string,
    # never in the corpus.
    nc = _namecheck.verify_question_in_passages(question, passages_seen_by_llm)
    result.name_check_missing = nc.missing_tokens
    if nc.missing_tokens:
        if strict_names:
            result.answer = NAME_MISMATCH_ANSWER
            result.strict_names_aborted = True
        else:
            result.answer = (
                _namecheck.refusal_directive(nc.missing_tokens)
                + result.answer
            )

    result.cost_usd = cost_usd(result.input_tokens, result.output_tokens)
    return result
