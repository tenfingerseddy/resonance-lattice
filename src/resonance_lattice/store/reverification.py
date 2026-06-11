"""Re-verification of stale insights.

Architecture §4.4: when `rlat refresh` detects drift, every insight that
cites the drifted source flips to `state="stale"`. Without
re-verification, stale rows accumulate forever — the corpus grows a
shadow layer of retired content.

This module fixes that. For each stale insight:

  1. Fetch the (now-changed) cited source passages from the store.
  2. Ask an LLM: "Does this updated source still support the insight's
     content?"
  3. If yes → flip back to `active`, update `source_passage_hashes`
     to reflect the new content_hash, append a `verdict` signal with
     source="llm", polarity="accept".
  4. If no → flip to `retired` (final).

LLM-driven; requires an Anthropic API key. Costs ~1 LLM call per stale
insight, so the bill is bounded by drift volume. Default model: Sonnet
(reasoning quality matters more than throughput here).

Exposed via `rlat reverify <km.rlat>`. Run on your own cadence — after
`rlat refresh` is the natural place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..state.claim import Claim, evolve
from ..state.claim_lifecycle import (
    GateSignals,
    accumulate_outcome,
    consolidate_corpus,
    record_verdict,
)
from ._llm import judge_json

# A passing re-verification is a corroboration outcome — the cited sources
# changed and an LLM confirmed the insight still follows.
_REVERIFY_CORROBORATION = 1.0


REVERIFY_SYSTEM = """\
You evaluate whether an updated source passage still supports a prior
synthesised insight. The insight was generated when the source had a
different content; the source has since changed. You must decide whether
the insight remains supported by the NEW source content.

Answer with a single JSON object:
{
  "supports": true|false,
  "reason": "<one short sentence>"
}

supports=true if the updated source still makes the insight content
true (paraphrasing, minor wording changes, additions that don't
contradict).

supports=false if the updated source contradicts the insight, removes
the supporting claim entirely, or means the insight no longer follows.
"""


@dataclass(frozen=True)
class ReverificationOutcome:
    """Per-claim result of the re-verification pass."""
    claim_id: str
    new_state: Literal["active", "retired", "skipped"]
    reason: str
    refreshed_source_hashes: tuple[str, ...]


def _build_user_message(
    insight: Claim,
    cited_sources: list[dict],
) -> str:
    blocks = []
    for s in cited_sources:
        blocks.append(
            f"=== UPDATED SOURCE: {s['source_file']}:"
            f"{s['char_offset']}+{s['char_length']} ===\n"
            f"{s['text']}\n"
        )
    return (
        f"INSIGHT content (generated against the prior source state):\n\n"
        f"{insight.content}\n\n"
        f"INSIGHT cited these source passages — their UPDATED content is:\n\n"
        + "\n".join(blocks)
    )


def reverify_stale_insights(
    km_path: str | Path,
    client,
    *,
    model: str | None = None,
    limit: int | None = None,
    cost_cap_usd: float | None = None,
) -> list[ReverificationOutcome]:
    """Run the re-verification pass against every stale insight in the
    archive. Writes survivors back via `write_insight_layer_in_place`.

    Returns one outcome per processed insight. `limit` caps the number
    of LLM calls (hard count). `cost_cap_usd` caps cumulative spend in
    USD across all calls this invocation makes — the pass stops before
    the next call once observed spend crosses the cap. Stale insights
    not reached on this pass stay stale for the next one. A claim
    short-circuited by the cap appears in outcomes with
    `new_state="skipped"` and a reason naming the cap.
    """
    from . import archive, open_store
    from .._pricing import SONNET_MODEL, CostMeter
    use_model = model or SONNET_MODEL
    meter = CostMeter(cap_usd=cost_cap_usd)

    p = Path(km_path)
    contents = archive.read(p)
    stale = [
        (i, ins) for i, ins in enumerate(contents.insights)
        if ins.state == "stale"
    ]
    if not stale:
        return []
    if limit is not None:
        stale = stale[:limit]

    store = open_store(p, contents)
    coords_by_id = {c.passage_id: c for c in contents.registry}

    outcomes: list[ReverificationOutcome] = []
    updated_insights = list(contents.insights)

    # The `finally` block flushes any work that completed before a mid-pass
    # exception (network outage, SDK error, KeyboardInterrupt). Without it,
    # a flaky network can repeatedly waste already-metered LLM spend on
    # rows that successfully re-verified earlier in the pass but never
    # made it to disk. The check (`any non-skipped`) is the same gate the
    # post-loop path used; nothing is written for a pure-skipped pass.
    try:
        for idx, ins in stale:
            # An all-external claim has no CORPUS sources to re-verify; corpus
            # reverification — and the "all sources missing" orphan-retire below —
            # does not apply (its citations resolve to no registry coord by design).
            # Leave it as-is; its lifecycle is governed by outcomes, not corpus
            # hashes. (detect_drift already avoids staling it via drift; this guards
            # any other stale path so it isn't wrongly retired.)
            from .insight import all_external
            if all_external(ins.facts.citations):
                outcomes.append(ReverificationOutcome(
                    claim_id=ins.claim_id, new_state=ins.state,
                    reason="external claim — no corpus reverification",
                    refreshed_source_hashes=(),
                ))
                continue
            cited_sources: list[dict] = []
            refreshed_hashes: list[str] = []
            for cit in ins.facts.citations:
                coord = coords_by_id.get(cit.passage_id)
                if coord is None:
                    continue
                try:
                    text = store.fetch(coord.source_file, coord.char_offset,
                                       coord.char_length)
                except (FileNotFoundError, ValueError):
                    continue
                cited_sources.append({
                    "source_file": coord.source_file,
                    "char_offset": coord.char_offset,
                    "char_length": coord.char_length,
                    "text": text,
                })
                refreshed_hashes.append(coord.content_hash)

            if not cited_sources:
                # All citations orphan — a failed re-verification.
                committed = consolidate_corpus(
                    ins, signals=GateSignals(compression_test_pass=False),
                )
                updated_insights[idx] = committed
                outcomes.append(ReverificationOutcome(
                    claim_id=ins.claim_id, new_state=committed.state,
                    reason="all cited sources missing",
                    refreshed_source_hashes=(),
                ))
                continue

            if meter.has_exceeded_cap():
                # Pre-flight cap check — the previous call's observed spend
                # already crossed the budget, so the loop stops before
                # issuing the next call. Remaining stale insights record as
                # skipped (with the cap reason) and stay stale for next pass.
                outcomes.append(ReverificationOutcome(
                    claim_id=ins.claim_id, new_state="skipped",
                    reason=f"cost cap crossed "
                           f"(${meter.cost_so_far():.4f} of ${meter.cap_usd:.4f})",
                    refreshed_source_hashes=(),
                ))
                continue

            verdict = judge_json(
                client, use_model, REVERIFY_SYSTEM,
                _build_user_message(ins, cited_sources),
                max_tokens=200,
                meter=meter,
            )
            if verdict.get("_parse_error"):
                # Couldn't parse — skip this claim, leave stale for next pass.
                outcomes.append(ReverificationOutcome(
                    claim_id=ins.claim_id, new_state="skipped",
                    reason=f"judge parse failed: {verdict['_parse_error'][:80]}",
                    refreshed_source_hashes=tuple(refreshed_hashes),
                ))
                continue

            supports = bool(verdict.get("supports", False))
            reason = str(verdict.get("reason", "")[:200])
            if supports:
                # Record the LLM accept, refresh the drifted source hashes,
                # bank the corroboration, then let the spine commit the
                # transition. The corroboration lands before `consolidate_corpus`
                # so the retire-floor check sees the bumped trust. The spine
                # may still retire the claim — an outstanding user reject,
                # or trust still below the floor after the +1 — so the
                # reported `new_state` is the spine's actual verdict, not an
                # assumed `active`.
                recorded = record_verdict(ins, source="llm", polarity="accept")
                refreshed = evolve(
                    recorded, source_passage_hashes=tuple(refreshed_hashes),
                )
                corroborated = accumulate_outcome(
                    refreshed, corroboration=_REVERIFY_CORROBORATION,
                )
                committed = consolidate_corpus(
                    corroborated,
                    signals=GateSignals(compression_test_pass=True),
                )
            else:
                committed = consolidate_corpus(
                    ins, signals=GateSignals(compression_test_pass=False),
                )
            updated_insights[idx] = committed
            outcomes.append(ReverificationOutcome(
                claim_id=ins.claim_id, new_state=committed.state,
                reason=reason,
                refreshed_source_hashes=tuple(refreshed_hashes),
            ))
    finally:
        if any(o.new_state != "skipped" for o in outcomes):
            insight_band = contents.bands[archive.INSIGHT_BAND_NAME]
            archive.write_insight_layer_in_place(
                p, updated_insights, insight_band,
                mark_reverified_utc=_now_utc_iso(),
            )
    return outcomes


def _now_utc_iso() -> str:
    """ISO-8601 UTC stamp matching the project convention
    (`memory._common.utcnow_iso()` — Z-suffix). Inlined here to avoid a
    store→memory import arrow."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
