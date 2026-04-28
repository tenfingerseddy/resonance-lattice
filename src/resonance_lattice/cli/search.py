"""`rlat search <knowledge_model.rlat> "<query>" [flags]`

Single retrieval path. Optimised band used if present; base otherwise.
No --rerank, no --hybrid, no --retrieval-mode, no --cascade.

Output formats:
  text      one line per hit: "score  source_file:offset  drift_status  preview"
  json      one JSON object per hit (drift fields included)
  context   concatenated passages within a token budget — synthesis-ready

Phase 3 deliverable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import MaterialiserConfig
from ..field import ann, retrieve
from ..field.encoder import Encoder
from ..rql.types import ConfidenceMetrics
from ..store.verified import filter_verified, verify_hits
from . import _grounding, _namecheck
from ._grounding import Mode
from ._load import load_or_exit, open_store_or_exit


def _format_text(hits: list, max_preview_chars: int = 100) -> str:
    if not hits:
        return "(no hits)"
    lines = []
    for h in hits:
        preview = h.text.replace("\n", " ").strip()
        if len(preview) > max_preview_chars:
            preview = preview[:max_preview_chars - 1] + "…"
        lines.append(
            f"{h.score:.3f}  "
            f"{h.source_file}:{h.char_offset}+{h.char_length}  "
            f"[{h.drift_status}]  {preview}"
        )
    return "\n".join(lines)


def _format_json(hits: list) -> str:
    return json.dumps(
        [
            {
                "passage_idx": h.passage_idx,
                "source_file": h.source_file,
                "char_offset": h.char_offset,
                "char_length": h.char_length,
                "content_hash": h.content_hash,
                "drift_status": h.drift_status,
                "score": h.score,
                "text": h.text,
            }
            for h in hits
        ],
        indent=2,
    )


def _format_context(
    hits: list, config: MaterialiserConfig, mode: Mode, band_name: str,
    query: str,
) -> tuple[str, list[str]]:
    """Concatenate verified hits up to the token budget. Higher-scored hits
    win when the budget runs out. The `chars_per_token` heuristic on
    MaterialiserConfig is a conservative under-estimate so the materialised
    context fits the consuming model's window with headroom.

    The grounding-mode header is stamped at the top so the consumer LLM
    knows how to treat the passages. When the mode-gate fires (weak
    retrieval under `augment` or `knowledge`), the passage body is
    replaced by a suppression marker — the directive still ships.

    When the query references a distinctive proper noun / acronym / ID
    not present in any rendered passage, a refusal directive is
    prepended to the body. Returns `(rendered, missing_name_tokens)` so
    the caller can gate on `--strict-names`.
    """
    metrics = ConfidenceMetrics.from_verified(hits, band_name)
    header = _grounding.format_header(mode)

    if _grounding.should_suppress(metrics, mode):
        return f"{header}\n\n{_grounding.suppression_marker(metrics, mode)}\n", []

    char_budget = config.token_budget * config.chars_per_token
    parts: list[str] = []
    rendered_passage_texts: list[str] = []
    used = 0
    for h in hits:
        if h.drift_status == "missing" or not h.text:
            continue
        block = (
            f"<!-- {h.source_file}:{h.char_offset}+{h.char_length} "
            f"score={h.score:.3f} {h.drift_status} -->\n"
            f"{h.text}\n"
        )
        if used + len(block) > char_budget and parts:
            break
        parts.append(block)
        rendered_passage_texts.append(h.text)
        used += len(block)

    body = "\n".join(parts)
    # Name-check against the passage text the consumer LLM will actually
    # see — i.e. the text that survived the token-budget truncation.
    # Checking the full hits list would falsely pass when the missing
    # distinctive token only appears in a passage that was budget-
    # truncated out before reaching the consumer.
    nc = _namecheck.verify_question_in_passages(
        query, "\n".join(rendered_passage_texts)
    )
    if nc.missing_tokens:
        body = _namecheck.refusal_directive(nc.missing_tokens) + body
    return f"{header}\n\n{body}", nc.missing_tokens


def cmd_search(args: argparse.Namespace) -> int:
    km_path = Path(args.knowledge_model)
    contents = load_or_exit(km_path)
    handle = contents.select_band()
    ann_index = ann.deserialize(handle.ann_blob) if handle.ann_blob else None

    encoder = Encoder()  # auto runtime — query path picks ONNX/OpenVINO
    query_emb = encoder.encode([args.query])[0]
    hits = retrieve(query_emb, handle, ann_index, contents.registry, args.top_k)

    store = open_store_or_exit(km_path, contents, args.source_root)
    verified = verify_hits(hits, store, contents.registry)
    if args.verified_only:
        verified = filter_verified(verified)

    missing_names: list[str] = []
    if args.format == "text":
        print(_format_text(verified))
    elif args.format == "json":
        print(_format_json(verified))
    elif args.format == "context":
        rendered, missing_names = _format_context(
            verified, MaterialiserConfig(), Mode(args.mode), handle.name,
            args.query,
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

    if not args.quiet:
        # Banner to stderr so it doesn't pollute json/context stdout consumers.
        print(
            f"[search] band={handle.name} ann={'yes' if ann_index else 'no'} "
            f"hits={len(verified)} "
            f"({contents.metadata.bands[handle.name].passage_count} passages)",
            file=sys.stderr,
        )
    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("search", help="Top-k retrieval")
    p.add_argument("knowledge_model", help="Path to a .rlat knowledge model")
    p.add_argument("query", help="Query text")
    p.add_argument("--top-k", type=int, default=10, help="Number of hits (default: 10)")
    p.add_argument(
        "--format", default="text", choices=["text", "json", "context"],
        help="Output format (default: text)",
    )
    p.add_argument(
        "--source-root", default=None,
        help="Override recorded source_root (local mode only)",
    )
    p.add_argument(
        "--verified-only", action="store_true",
        help="Drop hits whose source has drifted or gone missing",
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
