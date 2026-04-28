"""`rlat skill-context <km.rlat> --query Q [--query Q ...] [flags]`

Skill-friendly context block, optimised for the Anthropic skill `!command`
dynamic-injection primitive (https://code.claude.com/docs/en/skills.md
#inject-dynamic-context).

A skill body uses the primitive like this:

    !`rlat skill-context fabric-docs.rlat \
        --query "Fabric workspace fundamentals" \
        --query "$user_question" --top-k 5`

The shell command runs BEFORE the model sees the skill; stdout replaces the
placeholder. The output is markdown — citation-anchored passages plus a
header line carrying ConfidenceMetrics so the model can self-judge whether
to make strong claims (low source_diversity → hedge; high drift_fraction →
flag staleness).

Trust spine — three guarantees the integration enforces:

  1. Verified retrieval is the default, not a flag. Every passage carries
     `[source_file:char_offset+char_length]` plus `[drift_status]`.
  2. Drift gate. When `--strict`, any drifted/missing hit aborts non-zero
     so the skill loader surfaces an error instead of grounding on stale
     content. Default behaviour prepends a yellow ⚠ banner.
  3. Confidence header. ConfidenceMetrics for each query's evidence set is
     printed as an HTML comment header, machine- and human-readable.

Repeated `--query` flags retrieve a separate top-k for each query and
concatenate the blocks. Total output is hard-capped by `--token-budget`
so a runaway `!command` can't blow the consuming model's context window.

Phase 7 deliverable. Skill-integration spec: docs/internal/SKILL_INTEGRATION.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from ..config import MaterialiserConfig
from ..field import ann, retrieve
from ..field.encoder import Encoder
from ..rql.types import ConfidenceMetrics
from ..store.verified import VerifiedHit, verify_hits
from . import _grounding, _namecheck
from ._grounding import Mode
from ._load import load_or_exit, open_store_or_exit


def _format_query_block(
    query: str,
    verified: list[VerifiedHit],
    band_name: str,
    mode: Mode,
) -> tuple[str, list[str]]:
    """Render one query's hits as a markdown block.

    `missing` hits and hits with empty text are dropped from the rendered
    body but their drift_status is preserved in the metrics header — the
    caller does the drift-gate decision off `verified`, not off the
    rendered subset. When the grounding-mode gate fires, the body is
    replaced by a `suppression_marker` so the consumer LLM gets the
    directive without being grounded on low-confidence passages.

    Returns `(block, missing_name_tokens)`. `missing_name_tokens` is the
    list of distinctive proper-noun-like tokens from `query` that did not
    appear verbatim in the rendered passage text — the caller uses this
    to decide on `--strict-names` aborts. When non-empty, a refusal
    directive is prepended to the body so the consumer LLM is told the
    question may be about a different entity than the corpus describes.
    """
    metrics = ConfidenceMetrics.from_verified(verified, band_name)
    renderable = [v for v in verified if v.drift_status != "missing" and v.text]

    body_was_suppressed = False
    if not renderable:
        body = "*(no grounded passages — corpus may not cover this query)*"
        body_was_suppressed = True
    elif _grounding.should_suppress(metrics, mode):
        body = _grounding.suppression_marker(metrics, mode)
        body_was_suppressed = True
    else:
        parts: list[str] = []
        for v in renderable:
            anchor = f"{v.source_file}:{v.char_offset}+{v.char_length}"
            text = v.text.strip()
            parts.append(
                f"> **source: `{anchor}`** — score {v.score:.3f} "
                f"`[{v.drift_status}]`\n>\n"
                + "\n".join(f"> {line}" for line in text.split("\n"))
            )
        body = "\n\n".join(parts)

    # Name-check the query against the rendered passage text. Skip when
    # the body was already replaced by a suppression marker — the LLM is
    # already being told there's no grounded evidence, the namecheck
    # warning would just be noise.
    missing: list[str] = []
    if not body_was_suppressed:
        passages_text = "\n".join(v.text for v in renderable)
        nc = _namecheck.verify_question_in_passages(query, passages_text)
        missing = nc.missing_tokens
        if missing:
            body = _namecheck.refusal_directive(missing) + body

    header = (
        f"<!-- rlat skill-context query={query!r} band={band_name} "
        f"mode={mode.value} "
        f"top1_score={metrics.top1_score:.3f} "
        f"top1_top2_gap={metrics.top1_top2_gap:.3f} "
        f"source_diversity={metrics.source_diversity:.2f} "
        f"drift_fraction={metrics.drift_fraction:.2f} "
        f"missing_names={','.join(missing) if missing else '-'} -->"
    )
    return f"{header}\n## Context for: \"{query}\"\n\n{body}", missing


def _truncate_to_budget(blocks: list[str], char_budget: int) -> list[str]:
    """Greedy truncate from the end so the first query (typically the
    skill-author preset) is preserved over later queries (typically the
    user-supplied dynamic query). If even the first block overflows, it's
    kept whole — runaway-budget protection is a soft guarantee, not a
    hard one; an oversized single passage is the corpus author's bug to
    fix, not ours to silently drop."""
    if not blocks:
        return blocks
    kept: list[str] = []
    used = 0
    for block in blocks:
        if used + len(block) > char_budget and kept:
            break
        kept.append(block)
        used += len(block)
    return kept


def cmd_skill_context(args: argparse.Namespace) -> int:
    km_path = Path(args.knowledge_model)
    contents = load_or_exit(km_path)
    handle = contents.select_band()
    ann_index = ann.deserialize(handle.ann_blob) if handle.ann_blob else None
    store = open_store_or_exit(km_path, contents, args.source_root)

    mode = Mode(args.mode)
    encoder = Encoder()
    query_embeddings = encoder.encode(list(args.query))

    blocks: list[str] = []
    any_drift = False
    missing_by_query: dict[str, list[str]] = {}
    for q_text, q_emb in zip(args.query, query_embeddings):
        hits = retrieve(
            np.asarray(q_emb), handle, ann_index, contents.registry, args.top_k
        )
        verified = verify_hits(hits, store, contents.registry)
        block, missing = _format_query_block(q_text, verified, handle.name, mode)
        blocks.append(block)
        if missing:
            missing_by_query[q_text] = missing
        if any(v.drift_status != "verified" for v in verified):
            any_drift = True

    if args.strict and any_drift:
        print(
            f"error: --strict and drift detected in {km_path}. "
            f"Run: rlat refresh {km_path}",
            file=sys.stderr,
        )
        return 2

    if args.strict_names and missing_by_query:
        details = "; ".join(
            f"{q!r} missing {','.join(toks)}"
            for q, toks in missing_by_query.items()
        )
        print(
            f"error: --strict-names and distinctive question tokens not "
            f"found in retrieved passages: {details}. The question may be "
            f"about an entity the corpus does not cover.",
            file=sys.stderr,
        )
        return 3

    # Hard cap on total output. Chars-per-token heuristic borrowed from
    # MaterialiserConfig (conservative under-estimate so the consuming
    # model's window has headroom). The mode header is small and ships
    # outside the budget — the directive is non-negotiable.
    config = MaterialiserConfig()
    char_budget = args.token_budget * config.chars_per_token
    blocks = _truncate_to_budget(blocks, char_budget)

    print(_grounding.format_header(mode))
    print()
    if any_drift:
        banner = (
            f"> ⚠ **DRIFT WARNING**: at least one passage below has stale or "
            f"missing source. Treat content as advisory; refresh with "
            f"`rlat refresh {km_path.name}` for canonical results."
        )
        print(banner)
        print()

    print("\n\n".join(blocks))
    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "skill-context",
        help="Markdown context block for Anthropic-skill !command injection",
    )
    p.add_argument("knowledge_model", help="Path to a .rlat knowledge model")
    p.add_argument(
        "--query", action="append", required=True,
        help="Retrieval query (repeatable). Each --query flag adds a passage "
             "block in the order provided — typical pattern: skill-author "
             "preset queries first, then $user_query last.",
    )
    p.add_argument(
        "--top-k", type=int, default=5,
        help="Top-k passages per query (default: 5)",
    )
    p.add_argument(
        "--token-budget", type=int, default=4000,
        help="Hard cap on total output tokens (default: 4000). "
             "Lower-priority query blocks dropped first.",
    )
    p.add_argument(
        "--source-root", default=None,
        help="Override recorded source_root (local mode only)",
    )
    p.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero if any retrieved passage has drifted or "
             "missing source (default: warn-and-proceed)",
    )
    p.add_argument(
        "--strict-names", action="store_true",
        help=_namecheck.STRICT_NAMES_HELP,
    )
    p.add_argument(
        "--mode", default=_grounding.DEFAULT_MODE,
        choices=list(_grounding.MODE_CHOICES),
        help=f"Grounding mode for the consumer LLM (default: "
             f"{_grounding.DEFAULT_MODE}). augment = passages are primary "
             "context blended with the LLM's training (default; bench 2: "
             "55%% answerable accuracy, 4%% hallucination on Fabric "
             "docs); constrain = passages are the only source of truth, "
             "refuse on thin evidence (2%% hallucination — pick for "
             "compliance / audit work); knowledge = passages supplement "
             "training, lighter gate.",
    )
    p.set_defaults(func=cmd_skill_context)
