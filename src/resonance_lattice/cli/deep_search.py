"""`rlat deep-search <km.rlat> "<question>" [flags]`

Multi-hop research loop: plan → search → refine → synthesize. The
launch-headline mode for v2.0. On the Microsoft Fabric 5-lane bench
(63 questions, relaxed rubric):

    augment       (single-shot): 74.5% acc / 3.9% halluc / $0.004/q
    deep-search   (multi-hop):   92.2% acc / 2.0% halluc / $0.010/q

Cost trade-off: ~2.5× the cost over single-shot augment for ~24 pp
accuracy lift and 2× lower hallucination. Worth it on questions where
correctness matters more than latency / spend.

Usage:

  export CLAUDE_API=sk-ant-...
  rlat deep-search project.rlat "What is the default action for MLV?"
  rlat deep-search project.rlat "$user_q" --max-hops 6 --strict-names \\
      --format json > result.json

Anthropic API key resolved via `RLAT_LLM_API_KEY_ENV` indirection,
then `CLAUDE_API`, then `ANTHROPIC_API_KEY` (matches the `optimise`
discovery order). Anthropic-only for v2.0; OpenAI / local-LLM
adapters land post-launch if there's demand.

Phase 7 deliverable. Spec: docs/internal/SKILL_INTEGRATION.md +
docs/internal/benchmarks/02_distractor_floor_analysis.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from ..deep_search import deep_search
from ..deep_search.types import DeepSearchResult
from . import _namecheck


def _format_text(result: DeepSearchResult) -> str:
    """Human-readable rendering for terminal use.

    Answer first (the user's actual deliverable), then a one-line cost
    summary, then a compact hops log so the user can audit which
    queries surfaced which evidence.
    """
    lines = [result.answer.rstrip(), ""]
    lines.append(
        f"[deep-search] hops={len(result.hops)} "
        f"in={result.input_tokens} out={result.output_tokens} "
        f"cost=${result.cost_usd:.4f} "
        f"evidence={len(result.evidence_passages)} passages"
    )
    if result.name_check_missing:
        lines.append(
            f"[deep-search] name-check missing: "
            f"{', '.join(result.name_check_missing)}"
            + (" (strict_names → refusal)" if result.strict_names_aborted else "")
        )
    for h in result.hops:
        if h.kind == "search":
            lines.append(f"  hop {h.n} search {h.query!r} → {h.n_passages} passages")
        elif h.kind in ("plan", "decide_search"):
            lines.append(f"  hop {h.n} {h.kind:14s} {h.query!r}")
        elif h.kind == "decide_answer":
            lines.append(f"  hop {h.n} answer (loop terminated)")
        elif h.kind == "decide_give_up":
            lines.append(f"  hop {h.n} give_up (corpus does not cover question)")
        elif h.kind == "synth_after_max_hops":
            lines.append(f"  hop {h.n} synth after max-hops")
        elif h.kind == "search_failed":
            lines.append(f"  hop {h.n} search FAILED: {h.error}")
        elif h.kind == "parse_failed":
            lines.append(f"  hop {h.n} refiner parse FAILED")
    return "\n".join(lines)


def _format_json(result: DeepSearchResult) -> str:
    """Machine-readable rendering. Hops + evidence flatten to JSON via asdict."""
    payload = {
        "question": result.question,
        "answer": result.answer,
        "hops": [asdict(h) for h in result.hops],
        "evidence_passages": result.evidence_passages,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": result.cost_usd,
        "name_check_missing": result.name_check_missing,
        "strict_names_aborted": result.strict_names_aborted,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _format_markdown(result: DeepSearchResult) -> str:
    """Markdown for piping into another LLM as a research artefact.

    Surfaces the answer, the evidence union, and the hops log in a
    shape another model can read directly. Matches the shape of `rlat
    skill-context` so a deep-search result can drop into the same
    consumer-LLM context window.
    """
    lines = [
        f"<!-- rlat deep-search question={result.question!r} "
        f"hops={len(result.hops)} cost_usd={result.cost_usd:.4f} "
        f"missing_names={','.join(result.name_check_missing) or '-'} -->",
        f"# Answer",
        "",
        result.answer.rstrip(),
        "",
        f"# Evidence ({len(result.evidence_passages)} passages)",
        "",
    ]
    for p in result.evidence_passages:
        anchor = f"{p['source_file']}:{p['char_offset']}+{p['char_length']}"
        lines.append(
            f"> **source: `{anchor}`** — score {p['score']:.3f} "
            f"`[{p['drift_status']}]`"
        )
        lines.append(">")
        for line in p.get("text", "").split("\n"):
            lines.append(f"> {line}")
        lines.append("")
    return "\n".join(lines)


def cmd_deep_search(args: argparse.Namespace) -> int:
    try:
        import anthropic
    except ImportError:
        print(
            "error: rlat deep-search requires the `anthropic` package. "
            "Install with `pip install rlat[optimise]` (which already "
            "pulls anthropic) or `pip install anthropic` standalone.",
            file=sys.stderr,
        )
        return 1

    from ..optimise.synth_queries import api_key_or_error
    try:
        api_key = api_key_or_error()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    client = anthropic.Anthropic(api_key=api_key)
    km_path = Path(args.knowledge_model)

    try:
        result = deep_search(
            km_path, args.question,
            client=client,
            max_hops=args.max_hops,
            top_k=args.top_k,
            source_root=args.source_root,
            strict_names=args.strict_names,
        )
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.format == "text":
        print(_format_text(result))
    elif args.format == "json":
        print(_format_json(result))
    elif args.format == "markdown":
        print(_format_markdown(result))

    if args.strict_names and result.strict_names_aborted:
        return 3
    # Empty-answer paths (search_failed, unknown refiner action, parse_failed
    # without raw text) return rc=2 so a shell caller can distinguish a
    # principled refusal from a loop that didn't produce one.
    if any(h.kind in ("search_failed", "parse_failed") for h in result.hops):
        return 2
    return 0


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "deep-search",
        help="Multi-hop research loop (plan → search → refine → synth) — "
             "v2.0 launch headline; ~24 pp accuracy lift over single-shot",
    )
    p.add_argument("knowledge_model", help="Path to a .rlat knowledge model")
    p.add_argument("question", help="The question to research")
    p.add_argument(
        "--max-hops", type=int, default=4,
        help="Maximum retrieve+refine cycles (default: 4). The planner "
             "counts as hop 1; a synth call is added if the budget "
             "exhausts without an `answer` decision.",
    )
    p.add_argument(
        "--top-k", type=int, default=5,
        help="Top-k passages per hop (default: 5)",
    )
    p.add_argument(
        "--format", default="text", choices=["text", "json", "markdown"],
        help="Output format (default: text). markdown = drop-in shape for "
             "piping into another LLM; json = machine-readable, includes "
             "full hops + evidence union.",
    )
    p.add_argument(
        "--source-root", default=None,
        help="Override recorded source_root (local mode only)",
    )
    p.add_argument(
        "--strict-names", action="store_true",
        help=_namecheck.STRICT_NAMES_HELP + " For deep-search, the answer "
             "is also replaced by the name-mismatch refusal text.",
    )
    p.set_defaults(func=cmd_deep_search)
