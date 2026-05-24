"""Re-verification of stale insights.

Architecture §4.4: when `rlat refresh` detects drift, every insight that
cites the drifted source flips to `verdict_state="stale"`. Without
re-verification, stale rows accumulate forever — the corpus grows a
shadow layer of retired content.

This module fixes that. For each stale insight:

  1. Fetch the (now-changed) cited source passages from the store.
  2. Ask an LLM: "Does this updated source still support the insight's
     content?"
  3. If yes → flip back to `accepted`, update `source_passage_hashes`
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

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from ._llm import judge_json
from .insight import InsightPassage, VerdictSignal
from .insight_lifecycle import accumulate_outcome

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
    """Per-insight result of the re-verification pass."""
    insight_id: str
    new_state: str               # "accepted" | "retired" | "skipped"
    reason: str
    refreshed_source_hashes: tuple[str, ...]


def _build_user_message(
    insight: InsightPassage,
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
) -> list[ReverificationOutcome]:
    """Run the re-verification pass against every stale insight in the
    archive. Writes survivors back via `write_insight_layer_in_place`.

    Returns one outcome per processed insight. `limit` caps the number
    of LLM calls (cost control); remaining stale insights stay stale
    for the next pass.
    """
    from . import archive, open_store
    from .._pricing import SONNET_MODEL
    use_model = model or SONNET_MODEL

    p = Path(km_path)
    contents = archive.read(p)
    stale = [
        (i, ins) for i, ins in enumerate(contents.insights)
        if ins.verdict_state == "stale"
    ]
    if not stale:
        return []
    if limit is not None:
        stale = stale[:limit]

    store = open_store(p, contents)
    coords_by_id = {c.passage_id: c for c in contents.registry}

    outcomes: list[ReverificationOutcome] = []
    updated_insights = list(contents.insights)

    for idx, ins in stale:
        cited_sources: list[dict] = []
        refreshed_hashes: list[str] = []
        for cit in ins.citations:
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
            # All citations orphan — retire.
            updated_insights[idx] = replace(ins, verdict_state="retired")
            outcomes.append(ReverificationOutcome(
                insight_id=ins.insight_id, new_state="retired",
                reason="all cited sources missing",
                refreshed_source_hashes=(),
            ))
            continue

        verdict = judge_json(
            client, use_model, REVERIFY_SYSTEM,
            _build_user_message(ins, cited_sources),
            max_tokens=200,
        )
        if verdict.get("_parse_error"):
            # Couldn't parse — skip this row, leave stale for next pass.
            outcomes.append(ReverificationOutcome(
                insight_id=ins.insight_id, new_state="skipped",
                reason=f"judge parse failed: {verdict['_parse_error'][:80]}",
                refreshed_source_hashes=tuple(refreshed_hashes),
            ))
            continue

        supports = bool(verdict.get("supports", False))
        reason = str(verdict.get("reason", "")[:200])
        if supports:
            sig = VerdictSignal(
                source="llm", polarity="accept",
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                lens_id=None,
            )
            promoted = replace(
                ins,
                verdict_state="accepted",
                source_passage_hashes=tuple(refreshed_hashes),
                verdict_signals=ins.verdict_signals + (sig,),
            )
            updated_insights[idx] = accumulate_outcome(
                promoted, corroboration=_REVERIFY_CORROBORATION,
            )
            outcomes.append(ReverificationOutcome(
                insight_id=ins.insight_id, new_state="accepted",
                reason=reason,
                refreshed_source_hashes=tuple(refreshed_hashes),
            ))
        else:
            updated_insights[idx] = replace(ins, verdict_state="retired")
            outcomes.append(ReverificationOutcome(
                insight_id=ins.insight_id, new_state="retired",
                reason=reason,
                refreshed_source_hashes=tuple(refreshed_hashes),
            ))

    if any(o.new_state != "skipped" for o in outcomes):
        insight_band = contents.bands[archive.INSIGHT_BAND_NAME]
        archive.write_insight_layer_in_place(p, updated_insights, insight_band)
    return outcomes
