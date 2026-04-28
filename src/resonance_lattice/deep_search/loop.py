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
from ..field import ann, retrieve
from ..field.encoder import Encoder
from ..store import archive, open_store
from ..store.verified import VerifiedHit, verify_hits
from .prompts import (
    GIVE_UP_ANSWER,
    NAME_MISMATCH_ANSWER,
    PLANNER_SYSTEM,
    REFINER_SYSTEM,
    SYNTHESIZER_SYSTEM,
)
from .types import DeepSearchHop, DeepSearchResult


def _render_evidence_block(query: str, hits: list[VerifiedHit]) -> str:
    """Format a hop's verified hits the way the refiner expects.

    Mirrors the bench-harness rendering — drift-tagged anchor + score +
    blockquoted text — minus the per-block grounding header (the
    refiner sees a sequence of evidence blocks, the directive is in the
    refiner's own system prompt).
    """
    if not hits:
        return f"--- Search query: {query!r} ---\n(no hits)"
    parts = [f"--- Search query: {query!r} ---"]
    for h in hits:
        if h.drift_status == "missing" or not h.text:
            continue
        anchor = f"{h.source_file}:{h.char_offset}+{h.char_length}"
        parts.append(
            f"[{anchor}] (score {h.score:.3f}) {h.text.strip()}"
        )
    return "\n".join(parts)


def _retrieve_hop(
    *, encoder: Encoder, query: str, handle: Any, ann_index: Any,
    contents: Any, store: Any, top_k: int,
) -> list[VerifiedHit]:
    """One in-process retrieve-and-verify call."""
    q_emb = encoder.encode([query])[0]
    hits = retrieve(
        np.asarray(q_emb), handle, ann_index, contents.registry, top_k
    )
    return verify_hits(hits, store, contents.registry)


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
    """Extract the JSON action object from the refiner's output. None on parse fail."""
    m = re.search(r'\{[^{}]*"action"[^{}]*\}', raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _dedupe_passages(verified: list[VerifiedHit]) -> list[dict]:
    """Project verified hits to citation-ready dicts, deduped on (source_file, char_offset)."""
    seen: set[tuple[str, int]] = set()
    out: list[dict] = []
    for h in verified:
        key = (h.source_file, h.char_offset)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "source_file": h.source_file,
            "char_offset": h.char_offset,
            "char_length": h.char_length,
            "score": h.score,
            "drift_status": h.drift_status,
            "text": h.text,
        })
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
            verified = _retrieve_hop(
                encoder=encoder, query=current_query, handle=handle,
                ann_index=ann_index, contents=contents, store=store,
                top_k=top_k,
            )
        except Exception as e:
            result.hops.append(DeepSearchHop(
                n=hop_n, kind="search_failed", query=current_query,
                error=str(e)[:200],
            ))
            break

        queries_tried.append(current_query)
        all_verified.extend(verified)
        evidence_blocks.append(_render_evidence_block(current_query, verified))
        passage_blocks.append("\n".join(
            v.text for v in verified
            if v.drift_status != "missing" and v.text
        ))
        result.hops.append(DeepSearchHop(
            n=hop_n, kind="search", query=current_query,
            n_passages=len(verified),
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
        raw, in_t, out_t = _llm_call(
            client, REFINER_SYSTEM, prompt, max_tokens=400,
        )
        result.input_tokens += in_t
        result.output_tokens += out_t

        action = _parse_refiner_action(raw)
        if action is None:
            result.hops.append(DeepSearchHop(
                n=hop_n, kind="parse_failed", error=raw[:300],
            ))
            result.answer = raw
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
        evidence = "\n\n".join(evidence_blocks)
        evidence_for_llm = evidence[:10000]
        passages_seen_by_llm = "\n\n".join(passage_blocks)[:10000]
        synth_prompt = (
            f"Question: {question}\n\n"
            f"All evidence collected:\n\n{evidence_for_llm}\n\n"
            f"Provide a concise answer based ONLY on the evidence above. "
            f"If the evidence doesn't cover the question, say so."
        )
        synth_text, in_t, out_t = _llm_call(
            client, SYNTHESIZER_SYSTEM, synth_prompt, max_tokens=500,
        )
        result.input_tokens += in_t
        result.output_tokens += out_t
        result.answer = synth_text
        result.hops.append(DeepSearchHop(
            n=max_hops + 1, kind="synth_after_max_hops",
        ))

    result.evidence_passages = _dedupe_passages(all_verified)

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
