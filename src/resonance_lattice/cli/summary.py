"""`rlat summary <knowledge_model.rlat> [-o output.md] [--query "..." ...]`

Extractive primer generation — produces a structured markdown document that
captures what's in a knowledge model. Powers `.claude/resonance-context.md`
regeneration: when an AI assistant boots into a project, it reads the primer
to learn the corpus shape without spending tokens on a full search.

Three sections, each token-budgeted via `MaterialiserConfig`:

  ## Landscape    Most representative passages (top hits vs corpus centroid)
  ## Structure    Source-file breakdown — TOC of what's in the corpus
  ## Evidence     Per-query top hits (only when --query provided)

Single-recipe — no LLM, no synthesis, just retrieval. Consumers (your
assistant) layer interpretation on top of the extracted passages.

Phase 3 deliverable.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from ..config import MaterialiserConfig
from ..field import ann, retrieve
from ..field.algebra import centroid
from ..field.encoder import Encoder
from ..store.archive import ArchiveContents, BandHandle
from ..store.base import Store
from ..store.verified import VerifiedHit, verify_hits
from ._load import load_or_exit, open_store_or_exit


def _hits_within_budget(
    hits: list[VerifiedHit],
    char_budget: int,
) -> list[VerifiedHit]:
    """Take hits in score order until the running text length exceeds the
    section's character budget. The first hit is always included even if
    oversized — better to honour the section's purpose with one truncated
    primary passage than emit nothing."""
    out: list[VerifiedHit] = []
    used = 0
    for h in hits:
        if h.drift_status == "missing" or not h.text:
            continue
        if used + len(h.text) > char_budget and out:
            break
        out.append(h)
        used += len(h.text)
    return out


def _format_landscape(
    hits: list[VerifiedHit], char_budget: int,
) -> str:
    selected = _hits_within_budget(hits, char_budget)
    if not selected:
        return "_(no representative passages — corpus centroid had no verified neighbours)_"
    parts = []
    for h in selected:
        parts.append(
            f"- **{h.source_file}** (score {h.score:.3f}): "
            + h.text.strip().replace("\n", " ")[:600]
        )
    return "\n".join(parts)


def _format_structure(
    registry: list, char_budget: int,
) -> str:
    counts = collections.Counter(c.source_file for c in registry)
    # Sorted by passage count descending so the largest contributors lead.
    rows = counts.most_common()
    lines = []
    used = 0
    for source_file, n in rows:
        line = f"- `{source_file}` — {n} passages"
        if used + len(line) > char_budget and lines:
            lines.append(f"- _(…and {len(rows) - len(lines)} more files)_")
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


def _format_evidence(
    per_query: list[tuple[str, list[VerifiedHit]]], char_budget: int,
) -> str:
    if not per_query:
        return ""
    # Distribute the budget evenly across queries — one big query shouldn't
    # starve the others. Loop-invariant; computed once outside the loop.
    per_query_budget = max(200, char_budget // len(per_query))
    sections: list[str] = []
    used = 0
    for query, hits in per_query:
        selected = _hits_within_budget(hits, per_query_budget)
        if not selected:
            continue
        section = [f"### Query: _{query}_"]
        for h in selected:
            section.append(
                f"- **{h.source_file}** (score {h.score:.3f}): "
                + h.text.strip().replace("\n", " ")[:400]
            )
        block = "\n".join(section)
        if used + len(block) > char_budget and sections:
            break
        sections.append(block)
        used += len(block)
    return "\n\n".join(sections)


def _build_primer(
    km_path: Path,
    contents: ArchiveContents,
    handle: BandHandle,
    ann_index: object | None,
    store: Store,
    queries: list[str],
    config: MaterialiserConfig,
) -> str:
    corpus_centroid = centroid(handle.band)
    landscape_hits = verify_hits(
        retrieve(corpus_centroid, handle, ann_index, contents.registry, top_k=10),
        store, contents.registry,
    )

    evidence: list[tuple[str, list[VerifiedHit]]] = []
    if queries:
        # One batched encode call across all queries — Encoder.encode runs a
        # single tokenizer pass + a single runtime forward over the list.
        # The earlier per-query loop fired N tokenizer passes for typical
        # N=3-5 multi-topic primers; at N=10 it's a measurable cold-path
        # regression on Sonnet-class workflows that load + summarise.
        encoder = Encoder()
        query_embs = encoder.encode(list(queries))
        for q, q_emb in zip(queries, query_embs):
            hits = verify_hits(
                retrieve(q_emb, handle, ann_index, contents.registry, top_k=5),
                store, contents.registry,
            )
            evidence.append((q, hits))

    cpt = config.chars_per_token
    md_parts: list[str] = []
    md_parts.append(f"# {km_path.name} — context primer")
    md_parts.append("")
    md_parts.append(
        f"Knowledge model with **{len(contents.registry)} passages** across "
        f"**{len(set(c.source_file for c in contents.registry))} files**, "
        f"encoded by `{contents.metadata.backbone.name}` "
        f"(rev `{contents.metadata.backbone.revision[:12]}`)."
    )
    md_parts.append("")
    md_parts.append("## Landscape")
    md_parts.append("")
    md_parts.append(_format_landscape(landscape_hits, config.sections_landscape * cpt))
    md_parts.append("")
    md_parts.append("## Structure")
    md_parts.append("")
    md_parts.append(_format_structure(contents.registry, config.sections_structure * cpt))
    if evidence:
        md_parts.append("")
        md_parts.append("## Evidence")
        md_parts.append("")
        md_parts.append(_format_evidence(evidence, config.sections_evidence * cpt))
    md_parts.append("")
    return "\n".join(md_parts)


def cmd_summary(args: argparse.Namespace) -> int:
    km_path = Path(args.knowledge_model)
    contents = load_or_exit(km_path)
    handle = contents.select_band()
    ann_index = ann.deserialize(handle.ann_blob) if handle.ann_blob else None
    store = open_store_or_exit(km_path, contents, args.source_root)

    primer = _build_primer(
        km_path, contents, handle, ann_index, store,
        args.query or [], MaterialiserConfig(),
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(primer, encoding="utf-8")
        print(f"[summary] wrote {out_path} "
              f"({len(primer)} chars, {len(primer) // 4} tokens approx)",
              file=sys.stderr)
    else:
        print(primer)
    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("summary", help="Generate context primer")
    p.add_argument("knowledge_model", help="Path to a .rlat knowledge model")
    p.add_argument(
        "-o", "--output", default=None,
        help="Output path (markdown). Default: stdout.",
    )
    p.add_argument(
        "--query", action="append", default=None,
        help="Themed query (repeatable). Each --query flag adds an evidence "
             "block in the order provided.",
    )
    p.add_argument(
        "--source-root", default=None,
        help="Override recorded source_root (local mode only)",
    )
    p.set_defaults(func=cmd_summary)
