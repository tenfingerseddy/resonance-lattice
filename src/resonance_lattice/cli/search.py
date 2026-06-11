"""`rlat search <knowledge_model.rlat> "<query>" [flags]`

Composed retrieval over source + insight layers (lensed knowledge Day 1).
Source-only path preserves the rlat v2.0 behaviour (the honest baseline,
foundation 5 of the trust contract); the default path composes the
insight layer alongside with visible labels.

Output formats:
  text      one line per hit, prefixed [SOURCE] or [INSIGHT]
  json      one JSON object per hit with `layer` discriminator
  context   concatenated passages within a token budget — synthesis-ready
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import MaterialiserConfig
from ..field import ann, retrieve, retrieve_insight
from ..field.encoder import Encoder
from ..store import telemetry
from ..rql.types import ConfidenceMetrics
from ..store.verified import (
    filter_verified,
    verify_hits,
    verify_insight_hits,
)
from . import _grounding, _namecheck
from ._grounding import Mode
from ._load import load_or_exit, open_store_or_exit

# Row-boundary char valve for the serve-ALL world block in `--format
# context` — a transport safety cap (a band would need hundreds of
# constraints to reach it), same size as the hook's injection cap.
_WORLD_BLOCK_MAX_CHARS = 6000


def _layer_tag(hit) -> str:
    """Visible layer label for trust-contract foundation 5. Source and
    insight must be distinguishable at every output surface; this is the
    single source of truth for the prefix string."""
    return "[INSIGHT]" if hit.layer == "insight" else "[SOURCE] "


def _hit_origin(hit) -> str:
    """Origin slug for the text formatter — source location or insight id.
    Lets the rendered output stay informative while distinguishing the
    layer semantics."""
    if hit.layer == "insight":
        return f"insight:{hit.claim_id} ({hit.kind})"
    return f"{hit.source_file}:{hit.char_offset}+{hit.char_length}"


def _hit_text(hit) -> str:
    """Body text for the text/context formatters — insight content for
    insight hits, source text for source hits."""
    return hit.content if hit.layer == "insight" else hit.text


def _format_text(hits: list, max_preview_chars: int = 100) -> str:
    if not hits:
        return "(no hits)"
    lines = []
    for h in hits:
        body = _hit_text(h).replace("\n", " ").strip()
        if len(body) > max_preview_chars:
            body = body[:max_preview_chars - 1] + "…"
        # `drift_status` is uniform across both layers (InsightHit's is
        # derived from its `state` via _INSIGHT_STATE_TO_DRIFT).
        lines.append(
            f"{_layer_tag(h)} {h.score:.3f}  {_hit_origin(h)}  "
            f"[{h.drift_status}]  {body}"
        )
    return "\n".join(lines)


def _format_json(hits: list) -> str:
    out: list[dict] = []
    for h in hits:
        if h.layer == "insight":
            out.append({
                "layer": "insight",
                "insight_idx": h.insight_idx,
                "claim_id": h.claim_id,
                "content_fingerprint": h.content_fingerprint,
                "kind": h.kind,
                "state": h.state,
                "confidence": h.confidence,
                "drift_status": h.drift_status,
                "score": h.score,
                "content": h.content,
                "citations": [
                    {
                        "passage_id": c.passage_id,
                        "char_span": list(c.char_span) if c.char_span else None,
                        "confidence": c.confidence,
                    }
                    for c in h.citations
                ],
                "source_passage_hashes": list(h.source_passage_hashes),
                "created_at": h.created_at,
                "intent_context": h.intent_context,
            })
        else:
            out.append({
                "layer": "source",
                "passage_idx": h.passage_idx,
                "source_file": h.source_file,
                "char_offset": h.char_offset,
                "char_length": h.char_length,
                "content_hash": h.content_hash,
                "drift_status": h.drift_status,
                "score": h.score,
                "text": h.text,
            })
    return json.dumps(out, indent=2)


def _format_context(
    hits: list, config: MaterialiserConfig, mode: Mode, band_name: str,
    query: str, world_block: str = "",
) -> tuple[str, list[str]]:
    """Concatenate verified hits up to the token budget. Higher-scored hits
    win when the budget runs out.

    Insight hits are rendered with an [INSIGHT] header showing kind,
    state, and confidence so the consumer LLM treats them
    appropriately. Source hits keep the v2.0 source-file:offset header.
    Both layers count against the same token budget; mixed lists are
    sorted by score descending before budgeting.

    `world_block` is the serve-ALL constraints/falsified section (already
    framed by `store.serve_framing`). It renders right after the grounding
    header, outside the passage budget and outside suppression — a standing
    rule of the world applies even when the corpus evidence is too thin to
    answer from.

    Confidence metrics for grounding-mode gating are computed against the
    source hits only — insights are derived, so they don't carry the same
    band-distribution signal that ConfidenceMetrics expects.
    """
    source_hits = [h for h in hits if h.layer == "source"]
    metrics = ConfidenceMetrics.from_verified(source_hits, band_name)
    header = _grounding.format_header(mode)
    if world_block:
        header = f"{header}\n\n{world_block}"

    if _grounding.should_suppress(metrics, mode):
        return f"{header}\n\n{_grounding.suppression_marker(metrics, mode)}\n", []

    char_budget = config.token_budget * config.chars_per_token
    parts: list[str] = []
    rendered_texts: list[str] = []
    used = 0
    for h in hits:
        if h.drift_status == "missing":
            continue
        if h.layer == "insight":
            block = (
                f"<!-- INSIGHT id={h.claim_id} kind={h.kind} "
                f"state={h.state} confidence={h.confidence:.2f} "
                f"score={h.score:.3f} -->\n"
                f"{h.content}\n"
            )
            rendered = h.content
        else:
            if not h.text:
                continue
            block = (
                f"<!-- SOURCE {h.source_file}:{h.char_offset}+{h.char_length} "
                f"score={h.score:.3f} {h.drift_status} -->\n"
                f"{h.text}\n"
            )
            rendered = h.text
        if used + len(block) > char_budget and parts:
            break
        parts.append(block)
        rendered_texts.append(rendered)
        used += len(block)

    body = "\n".join(parts)
    nc = _namecheck.verify_question_in_passages(query, "\n".join(rendered_texts))
    if nc.missing_tokens:
        body = _namecheck.refusal_directive(nc.missing_tokens) + body
    return f"{header}\n\n{body}", nc.missing_tokens


def cmd_search(args: argparse.Namespace) -> int:
    if str(args.knowledge_model).startswith("fabric://"):
        from . import _fabric
        return _fabric.cmd_search_fabric(args)

    if args.query is None:
        print(
            "error: query is required for path-style search "
            "(only `fabric://<alias>` discovery may omit it)",
            file=sys.stderr,
        )
        return 2

    km_path = Path(args.knowledge_model)
    contents = load_or_exit(km_path)

    # Source retrieval — always runs. This is the honest baseline that the
    # trust contract's foundation 5 protects: source-only never goes away,
    # never gets slow, and is always returned with layer labels.
    source_handle = contents.select_band()
    source_ann_index = (
        ann.deserialize(source_handle.ann_blob) if source_handle.ann_blob else None
    )

    encoder = Encoder()
    query_emb = encoder.encode([args.query])[0]
    source_hits_raw = retrieve(
        query_emb, source_handle, source_ann_index, contents.registry, args.top_k,
    )

    store = open_store_or_exit(km_path, contents, args.source_root)
    source_hits: list = list(verify_hits(source_hits_raw, store, contents.registry))

    # Insight retrieval — runs by default if the corpus has an insight band
    # and the user hasn't asked for source-only. The insight layer is opt-in
    # *visible* (--source-only opts out), never opt-in *hidden*.
    insight_hits: list = []
    insight_handle = contents.insight_band() if contents.insights else None
    if not args.source_only and insight_handle is not None:
        insight_ann_index = (
            ann.deserialize(insight_handle.ann_blob) if insight_handle.ann_blob else None
        )
        raw = retrieve_insight(
            query_emb, insight_handle.band, insight_ann_index, args.top_k,
            km_id=insight_handle.km_id,
        )
        insight_hits = list(verify_insight_hits(
            raw, contents.insights, include_stale=args.include_stale,
        ))

    if args.verified_only:
        source_hits = filter_verified(source_hits)
        insight_hits = filter_verified(insight_hits)

    # Lens overlay (optional). Re-rank source by trust_weights pattern
    # matches; re-rank insight by per-insight preferences. The lens is
    # the user's accumulated way of seeing — it never replaces source
    # ground truth, just adjusts where attention falls in the top-K.
    if args.lens and not args.source_only:
        from dataclasses import replace
        from ..lens import schema as lens_mod
        lens = lens_mod.load(Path(args.lens))
        source_hits = [
            replace(h, score=h.score * lens.trust_for_source(h.source_file))
            for h in source_hits
        ]
        # Experience hits carry an empty fingerprint — never lens-keyed (a
        # lens row keyed "" must not re-weight every experience hit at once).
        insight_hits = [
            replace(h, score=h.score
                    * lens.preference_for_insight(h.content_fingerprint))
            if h.content_fingerprint else h
            for h in insight_hits
        ]

    # Merge layers; sort by score descending. Layer labels travel with
    # each hit so the user always sees which layer surfaced what.
    all_hits = source_hits + insight_hits
    all_hits.sort(key=lambda h: -h.score)
    # Cap at top_k after the merge — the user asked for top_k results total,
    # not top_k per layer. (Per-layer top_k retrieval upstream gives the
    # merge a wider candidate pool to draw from.)
    all_hits = all_hits[: args.top_k]

    missing_names: list[str] = []
    if args.format == "text":
        print(_format_text(all_hits))
    elif args.format == "json":
        print(_format_json(all_hits))
    elif args.format == "context":
        # Serve-ALL world rules (standing constraints + falsified findings) —
        # query-independent by design (R1: no selection, zero over-blocking),
        # so they never compete with passages for retrieval rank or budget.
        # Lines are flattened (the serve_framing contract — a multi-line
        # claim must not forge its own framed section) and capped at a
        # row-boundary char valve, mirroring the hook channel.
        world_block = ""
        body_hits = all_hits
        if not args.source_only and contents.insights:
            from ..store.serve_framing import frame_claim_lines
            from ..store.verified import serve_band_constraints
            rows: list[tuple[str, str]] = []
            world_budget = _WORLD_BLOCK_MAX_CHARS
            for c in serve_band_constraints(contents.insights):
                text = c.content.replace("\n", " ").strip()
                if not text:
                    continue
                if world_budget - len(text) - 3 < 0:
                    break
                rows.append((c.kind, text))
                world_budget -= len(text) + 3
            world_block = frame_claim_lines(rows)
            if world_block:
                # Already served in full above — don't serve the same rows
                # again through the ranked, budgeted body.
                body_hits = [
                    h for h in all_hits
                    if not (h.layer == "insight"
                            and h.kind in ("constraint", "negation"))
                ]
        rendered, missing_names = _format_context(
            body_hits, MaterialiserConfig(), Mode(args.mode),
            source_handle.name, args.query, world_block,
        )
        print(rendered)

    if args.strict_names and missing_names:
        print(
            f"error: --strict-names and distinctive question tokens not "
            f"found in retrieved passages: {','.join(missing_names)}. The "
            f"question may be about an entity the corpus does not cover.",
            file=sys.stderr,
        )
        return 3

    n_source = sum(1 for h in all_hits if h.layer == "source")
    # Rank-ordered (all_hits is score-sorted) — list position is the rank.
    insight_ids = [h.claim_id for h in all_hits if h.layer == "insight"]
    n_insight = len(insight_ids)

    if not args.quiet:
        # Banner to stderr so it doesn't pollute json/context stdout consumers.
        total_insight = len(contents.insights) if contents.insights else 0
        print(
            f"[search] band={source_handle.name} "
            f"ann={'yes' if source_ann_index else 'no'} "
            f"hits={len(all_hits)} (source={n_source} insight={n_insight}) "
            f"corpus=(source={contents.metadata.bands[source_handle.name].passage_count} "
            f"insight={total_insight})",
            file=sys.stderr,
        )

    # The retrieval is already observed at the heart (field.retrieve /
    # retrieve_insight) — fingerprint + per-rank scores, keyed to the corpus.
    # Fold this process's observations INTO the .rlat itself (capture.md §3
    # persistence / CRITICAL_PATH step 1): the telemetry travels with the
    # portable file, no sidecar. Session-boundary cadence, gated + best-effort
    # (a no-op for a bare one-shot search; never raises). Runs after the hits
    # are already printed, so it never delays the user's results.
    telemetry.flush(str(contents.source_path) if contents.source_path else None)
    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("search", help="Top-k retrieval over source + insight layers")
    p.add_argument("knowledge_model",
                   help="Path to a .rlat knowledge model, or a fabric://<alias>[/<km>] URL")
    p.add_argument("query", nargs="?", default=None,
                   help="Query text. Optional only for `fabric://<alias>` discovery; "
                        "required for any path-style or `fabric://<alias>/<km>` search.")
    p.add_argument("--top-k", type=int, default=10, help="Number of hits (default: 10)")
    p.add_argument(
        "--format", default="text", choices=["text", "json", "context"],
        help="Output format (default: text)",
    )
    p.add_argument(
        "--source-only", action="store_true",
        help="Bypass the insight layer; return source passages only. The "
             "honest baseline — always available, always fast.",
    )
    p.add_argument(
        "--lens", default=None,
        help="Path to a .lens file. Applies trust_weights to source hits and "
             "insight_preferences to insight hits before the top-K merge. "
             "Ignored when --source-only is set.",
    )
    p.add_argument(
        "--include-stale", action="store_true",
        help="Include insight rows whose source has drifted (state=stale). "
             "Default: excluded from retrieval until re-verification passes.",
    )
    p.add_argument(
        "--source-root", default=None,
        help="Override recorded source_root (local mode only)",
    )
    p.add_argument(
        "--verified-only", action="store_true",
        help="Drop hits whose source has drifted or gone missing (applies to "
             "both source and insight layers)",
    )
    p.add_argument(
        "--strict-names", action="store_true",
        help="(--format context only) " + _namecheck.STRICT_NAMES_HELP,
    )
    p.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress the [search] banner on stderr",
    )
    p.add_argument(
        "--mode", default=_grounding.DEFAULT_MODE,
        choices=list(_grounding.MODE_CHOICES),
        help=f"Grounding mode for the consumer LLM, applied to "
             f"--format context only (default: {_grounding.DEFAULT_MODE}). "
             "augment = passages are primary context blended with the "
             "LLM's training (default; bench 2: 55%% accuracy, 4%% "
             "hallucination); constrain = passages are the only source "
             "of truth, refuse on thin evidence (2%% hallucination — "
             "pick for compliance / audit work); knowledge = passages "
             "supplement training, lighter gate.",
    )
    p.set_defaults(func=cmd_search)
