"""Benchmark 5 — Session-start primer effectiveness (MVP, 20 scenarios).

Tests whether loading `.claude/resonance-context.md` (the rlat-generated
primer) at session start measurably improves assistant answer quality on
the first turn. Each scenario is a 2-turn conversation: a turn-1 question
about the corpus + a turn-2 follow-up that builds on it. A separate
Sonnet judge grades each turn on a 4-state rubric (correct / partial /
wrong / refused).

MVP scope: 5 lanes (skipping `full_context` — known to be 67× more
expensive per the bench-1 token-spend result; we already have evidence
that full-context dominates cost without adding meaningful accuracy
on most questions).

Lanes:
  primer_loaded         — system prompt includes the code-base primer
                          (`.claude/resonance-context.md`, ~3 KB)
  memory_primer_loaded  — system prompt includes the memory primer
                          (synthesised summary of layered memory, ~2 KB)
  both_primers          — system prompt includes BOTH primers (the
                          state a user actually has at session start
                          when both `rlat summary` + `rlat memory primer`
                          have been run)
  rlat_search_v1        — no primer; system prompt includes `rlat
                          search --format context --top-k 5 --mode
                          augment` result for the turn's question
                          (per-turn retrieval, ~4 KB)
  cold                  — no corpus context at all

Methodology: docs/internal/benchmarks/05_primer_effectiveness.md

Usage:
    export CLAUDE_API=sk-ant-...
    python -m benchmarks.user_bench.primer_effectiveness.run \\
      --km resonance-lattice.rlat \\
      --primer .claude/resonance-context.md \\
      --tasks-file benchmarks/user_bench/primer_effectiveness/tasks.jsonl \\
      --output benchmarks/results/user_bench/primer_effectiveness.json \\
      --budget-usd 2
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent.parent.parent
_SRC = _REPO / "src"
if (_SRC / "resonance_lattice" / "__init__.py").exists():
    sys.path.insert(0, str(_SRC))

from resonance_lattice._pricing import SONNET_MODEL as MODEL, cost_usd as _cost_usd  # noqa: F401


@dataclass
class TurnResult:
    answer: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    judge_score: str = ""        # correct | partial | wrong | refused
    judge_reason: str = ""


@dataclass
class Trial:
    task_id: str
    tier: int
    topic: str
    approach: str
    turn1: TurnResult = field(default_factory=TurnResult)
    turn2: TurnResult = field(default_factory=TurnResult)
    cost_usd: float = 0.0
    wall_seconds: float = 0.0
    error: str | None = None


def _client():
    from resonance_lattice.optimise.synth_queries import discover_api_key
    import anthropic
    key = discover_api_key()
    if not key:
        raise RuntimeError("no LLM API key resolvable")
    return anthropic.Anthropic(api_key=key)


_ANSWER_SYSTEM = (
    "You are answering a question about a software project. The system "
    "prompt below may include corpus context the user has prepared for "
    "you. Answer concisely from whatever context you have plus your "
    "training knowledge. Cite source file paths inline when the context "
    "supports it. If you don't know the answer with confidence, say so."
)


def _rlat_search(km_path: Path, query: str, top_k: int = 5) -> str:
    """Run `rlat search --format context` and return the markdown."""
    cmd = [
        sys.executable, "-m", "resonance_lattice.cli.app",
        "search", str(km_path), query,
        "--top-k", str(top_k), "--format", "context",
        "--mode", "augment", "-q",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        return f"(rlat search failed: {proc.stderr[:300]})"
    return proc.stdout


def _build_system_prompt(approach: str, primer: str, memory_primer: str,
                          search_ctx: str) -> str:
    """Assemble the system prompt for a given approach + turn-context."""
    parts = [_ANSWER_SYSTEM]
    if approach == "primer_loaded":
        parts.append("\n\n--- PROJECT PRIMER (code-base) ---\n\n" + primer)
    elif approach == "memory_primer_loaded":
        parts.append("\n\n--- MEMORY PRIMER (cross-session knowledge) ---\n\n" + memory_primer)
    elif approach == "both_primers":
        parts.append("\n\n--- PROJECT PRIMER (code-base) ---\n\n" + primer)
        parts.append("\n\n--- MEMORY PRIMER (cross-session knowledge) ---\n\n" + memory_primer)
    elif approach == "rlat_search_v1":
        parts.append("\n\n--- RETRIEVED PASSAGES ---\n\n" + search_ctx)
    # cold: no extra context
    return "".join(parts)


def _ask_turn(client, system: str, prior_messages: list[dict], user: str,
              max_tokens: int = 600) -> TurnResult:
    """Run one conversational turn. Returns a TurnResult (answer + tokens).
    Judge is run separately by the caller."""
    msg = client.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system,
        messages=prior_messages + [{"role": "user", "content": user}],
    )
    return TurnResult(
        answer=msg.content[0].text.strip(),
        input_tokens=int(msg.usage.input_tokens),
        output_tokens=int(msg.usage.output_tokens),
    )


JUDGE_PROMPT = """You are grading a candidate answer against ground truth on a 4-state rubric:

  correct  — same factual claim as ground truth (paraphrasing OK; the
             load-bearing fact is captured)
  partial  — gist directionally right but misstates or misses a critical detail
  wrong    — different fact, contradicts ground truth, or invents content
  refused  — explicitly says \"I don't know\" / \"I cannot answer\" / \"the corpus
             doesn't have that\" — the right response when the context does
             not cover the question

Output exactly one JSON object on a single line: {\"score\": \"...\", \"reason\": \"...\"}
"""


def _judge(client, question: str, gt: str, candidate: str) -> tuple[str, str, int, int]:
    msg = client.messages.create(
        model=MODEL, max_tokens=200, system=JUDGE_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Ground truth: {gt}\n\n"
                f"Candidate answer: {candidate}\n\n"
                "Output the JSON now."
            ),
        }],
    )
    raw = msg.content[0].text.strip()
    m = re.search(r'\{[^{}]*"score"\s*:\s*"([^"]+)"[^{}]*\}', raw)
    if not m:
        return "wrong", f"unparseable: {raw[:200]}", msg.usage.input_tokens, msg.usage.output_tokens
    try:
        obj = json.loads(m.group(0))
        return (obj.get("score", "wrong"), obj.get("reason", ""),
                msg.usage.input_tokens, msg.usage.output_tokens)
    except json.JSONDecodeError:
        return m.group(1), raw[:200], msg.usage.input_tokens, msg.usage.output_tokens


def _run_scenario(client, task: dict, approach: str, km_path: Path,
                   primer_text: str, memory_primer_text: str) -> Trial:
    """Run a single 2-turn scenario for one approach."""
    t0 = time.perf_counter()
    tr = Trial(
        task_id=task["id"],
        tier=task["tier"],
        topic=task["topic"],
        approach=approach,
    )
    try:
        # Turn 1
        search_ctx = _rlat_search(km_path, task["q1"]) if approach == "rlat_search_v1" else ""
        sys1 = _build_system_prompt(approach, primer_text, memory_primer_text, search_ctx)
        tr.turn1 = _ask_turn(client, sys1, [], task["q1"])

        # Turn 2 (follow-up — keeps prior turn in conversation; new context for rlat_search_v1)
        prior = [
            {"role": "user", "content": task["q1"]},
            {"role": "assistant", "content": tr.turn1.answer},
        ]
        search_ctx2 = _rlat_search(km_path, task["q2"]) if approach == "rlat_search_v1" else ""
        sys2 = _build_system_prompt(approach, primer_text, memory_primer_text, search_ctx2)
        tr.turn2 = _ask_turn(client, sys2, prior, task["q2"])

        # Judge each turn
        s1, r1, ji1, jo1 = _judge(client, task["q1"], task["gt1"], tr.turn1.answer)
        s2, r2, ji2, jo2 = _judge(client, task["q2"], task["gt2"], tr.turn2.answer)
        tr.turn1.judge_score, tr.turn1.judge_reason = s1, r1
        tr.turn2.judge_score, tr.turn2.judge_reason = s2, r2

        # Cost = inference + judge tokens
        in_total = tr.turn1.input_tokens + tr.turn2.input_tokens + ji1 + ji2
        out_total = tr.turn1.output_tokens + tr.turn2.output_tokens + jo1 + jo2
        tr.cost_usd = _cost_usd(in_total, out_total)
    except Exception as e:
        tr.error = repr(e)
    tr.wall_seconds = time.perf_counter() - t0
    return tr


def _aggregate(trials: list[Trial]) -> dict[str, dict]:
    by_approach: dict[str, dict] = {}
    for tr in trials:
        agg = by_approach.setdefault(tr.approach, {
            "n": 0,
            "turn1_correct": 0, "turn1_partial": 0, "turn1_wrong": 0, "turn1_refused": 0,
            "turn2_correct": 0, "turn2_partial": 0, "turn2_wrong": 0, "turn2_refused": 0,
            "both_correct": 0,
            "total_cost_usd": 0.0,
            "wall_seconds_total": 0.0,
            "tier1": {"n": 0, "correct": 0},  # orientation
            "tier2": {"n": 0, "correct": 0},  # specific factual (rlat-search territory)
            "tier3": {"n": 0, "correct": 0},  # cross-reference
            "tier4": {"n": 0, "correct": 0},  # memory recall
        })
        agg["n"] += 1
        agg["total_cost_usd"] += tr.cost_usd
        agg["wall_seconds_total"] += tr.wall_seconds or 0.0
        s1 = tr.turn1.judge_score or "wrong"
        s2 = tr.turn2.judge_score or "wrong"
        agg[f"turn1_{s1}"] = agg.get(f"turn1_{s1}", 0) + 1
        agg[f"turn2_{s2}"] = agg.get(f"turn2_{s2}", 0) + 1
        if s1 == "correct" and s2 == "correct":
            agg["both_correct"] += 1
        tkey = f"tier{tr.tier}"
        agg[tkey]["n"] += 1
        if s1 == "correct":
            agg[tkey]["correct"] += 1
    for ap, agg in by_approach.items():
        n = max(agg["n"], 1)
        agg["turn1_correct_rate"] = agg.get("turn1_correct", 0) / n
        agg["turn2_correct_rate"] = agg.get("turn2_correct", 0) / n
        agg["both_correct_rate"] = agg["both_correct"] / n
        agg["mean_cost_usd"] = agg["total_cost_usd"] / n
        agg["mean_wall_seconds"] = agg["wall_seconds_total"] / n
    return by_approach


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--km", default="resonance-lattice.rlat")
    p.add_argument("--primer", default=".claude/resonance-context.md",
                   help="Path to the code-base primer (rlat summary output)")
    p.add_argument("--memory-primer",
                   default="benchmarks/user_bench/primer_effectiveness/fixtures/memory_primer.md",
                   help="Path to the memory primer (rlat memory primer output)")
    p.add_argument("--tasks-file",
                   default="benchmarks/user_bench/primer_effectiveness/tasks.jsonl")
    p.add_argument("--output",
                   default="benchmarks/results/user_bench/primer_effectiveness.json")
    p.add_argument("--budget-usd", type=float, default=3.0)
    p.add_argument("--n-tasks", type=int, default=0,
                   help="Limit to first N (0 = all)")
    p.add_argument(
        "--approaches",
        default="primer_loaded,memory_primer_loaded,both_primers,rlat_search_v1,cold",
        help="Comma-separated subset",
    )
    args = p.parse_args(argv)

    km_path = Path(args.km)
    if not km_path.exists():
        print(f"[primer-bench] FATAL: --km {km_path} not found", file=sys.stderr)
        return 1
    primer_text = Path(args.primer).read_text(encoding="utf-8")
    memory_primer_text = Path(args.memory_primer).read_text(encoding="utf-8") if Path(args.memory_primer).exists() else ""
    if not memory_primer_text and any(a in args.approaches for a in ("memory_primer_loaded", "both_primers")):
        print(f"[primer-bench] WARNING: --memory-primer {args.memory_primer} not found; "
              f"memory_primer_loaded and both_primers will fall back to empty primer",
              file=sys.stderr)
    tasks = [json.loads(line) for line in Path(args.tasks_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.n_tasks:
        tasks = tasks[:args.n_tasks]
    approaches = [a for a in args.approaches.split(",") if a]
    print(f"[primer-bench] {len(tasks)} scenarios × {len(approaches)} lanes "
          f"= {len(tasks) * len(approaches)} trials", flush=True)

    client = _client()
    trials: list[Trial] = []
    spent = 0.0
    for task in tasks:
        for approach in approaches:
            if spent >= args.budget_usd:
                print(f"[primer-bench] BUDGET CAP ${args.budget_usd:.2f} reached", flush=True)
                break
            tr = _run_scenario(client, task, approach, km_path, primer_text, memory_primer_text)
            spent += tr.cost_usd
            print(
                f"[primer-bench] {task['id']:5s}/T{task['tier']} {approach:18s} "
                f"t1={tr.turn1.judge_score:8s} t2={tr.turn2.judge_score:8s} "
                f"${tr.cost_usd:.4f} ({tr.wall_seconds:.1f}s) running=${spent:.2f}",
                flush=True,
            )
            trials.append(tr)
            # Incremental checkpoint
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "config": {"model": MODEL, "n_tasks": len(tasks),
                           "approaches": approaches, "km": str(km_path),
                           "primer": str(args.primer)},
                "trials": [asdict(t) for t in trials],
                "partial": True,
            }
            Path(args.output).write_text(json.dumps(checkpoint, indent=2),
                                          encoding="utf-8")
        else:
            continue
        break

    by_approach = _aggregate(trials)
    payload = {
        "config": {"model": MODEL, "n_tasks": len(tasks),
                   "approaches": approaches, "km": str(km_path),
                   "primer": str(args.primer)},
        "by_approach": by_approach,
        "trials": [asdict(t) for t in trials],
        "total_cost_usd": spent,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[primer-bench] wrote {args.output}", flush=True)
    print(f"[primer-bench] total spend: ${spent:.4f}", flush=True)
    print(f"\n[primer-bench] per-approach summary:", flush=True)
    for ap, agg in sorted(by_approach.items()):
        print(
            f"  {ap:20s}  t1_correct={agg['turn1_correct_rate']:.1%}  "
            f"t2_correct={agg['turn2_correct_rate']:.1%}  "
            f"both={agg['both_correct_rate']:.1%}  "
            f"mean_wall={agg['mean_wall_seconds']:.1f}s  "
            f"$/q={agg['mean_cost_usd']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
