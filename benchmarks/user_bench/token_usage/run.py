"""Benchmark 1 — token usage vs grep/glob/RAG patterns.

Measures token spend, $ cost, and wall-time to a correct grounded answer.
Compares four real-world approaches an AI-assistant developer might pick:

  A. rlat skill-context  — ship one markdown context block to Sonnet (1 LLM call).
  B. grep + Read tool loop — Sonnet uses grep/Read tools to locate content
     (multi-turn, multi-tool, capped at 6 turns).
  C. Full corpus in context — prepend all docs+src into the system prompt
     (1 LLM call, large input).
  D. No retrieval — ask Sonnet alone, no corpus access (1 LLM call, small input).

LLM judge (Sonnet 4.6 with locked rubric) scores each answer correct/partial/
wrong. Headline metric: tokens-per-correct-answer, $-per-correct-answer.

Reuses `discover_api_key` from synth_queries for env-var resolution.

Usage:
  export CLAUDE_API=sk-ant-...
  python -m benchmarks.user_bench.token_usage.run \\
      --output benchmarks/results/user_bench/token_usage.json \\
      --budget-usd 100

  # Pilot run with only 5 tasks (faster, cheaper)
  python -m benchmarks.user_bench.token_usage.run --n-tasks 5 --budget-usd 5
"""

from __future__ import annotations

import argparse
import json
import os
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


# Sonnet 4.6 — locked judge + agent model. Pricing as of 2026-04 (per Anthropic
# api docs, price-per-1M-tokens): $3 input, $15 output. If pricing changes, edit
# here only — every benchmark in this suite reads from these constants.
MODEL = "claude-sonnet-4-6"
PRICE_IN_PER_MTOK = 3.0
PRICE_OUT_PER_MTOK = 15.0


def _cost_usd(in_tok: int, out_tok: int) -> float:
    return (in_tok / 1_000_000.0) * PRICE_IN_PER_MTOK + (out_tok / 1_000_000.0) * PRICE_OUT_PER_MTOK


@dataclass
class Trial:
    task_id: str
    approach: str
    answer: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    wall_seconds: float = 0.0
    n_llm_calls: int = 0
    error: str | None = None
    judge_score: str = ""           # correct | partial | wrong | refused
    judge_explanation: str = ""


def _client():
    """Reuse the same env-var resolution chain the optimise pipeline uses."""
    from resonance_lattice.optimise.synth_queries import discover_api_key
    import anthropic
    key = discover_api_key()
    if not key:
        raise RuntimeError(
            "no LLM API key resolvable; set CLAUDE_API / ANTHROPIC_API_KEY"
        )
    return anthropic.Anthropic(api_key=key)


# ---------- Approach A: rlat skill-context -----------------------------------

def _run_rlat_skill(client, task: dict, km_path: Path, mode: str = "constrain") -> Trial:
    """Subprocess rlat skill-context, then 1 Sonnet call grounded in the markdown.

    Default mode `constrain`: the confidence-gate doesn't suppress the body
    (refusal is the LLM's job). With `augment` (rlat's default), close top-1/
    top-2 scores trigger the confidence-gate and the body is replaced with a
    "no confident evidence" marker — Sonnet faithfully refuses. For a
    benchmark whose explicit purpose is "did rlat-retrieved passages help
    answer the question?", we want the passages to actually reach Sonnet;
    `constrain` is the directive that lets that happen while still
    preserving the no-blend-with-training-knowledge contract.
    """
    t0 = time.perf_counter()
    tr = Trial(task_id=task["id"], approach=f"rlat-skill-context-{mode}")
    proc = subprocess.run(
        [
            "rlat", "skill-context", str(km_path),
            "--query", task["question"],
            "--top-k", "5",
            "--token-budget", "4000",
            "--mode", mode,
        ],
        capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode != 0:
        tr.error = f"rlat skill-context exit={proc.returncode}: {proc.stderr[:300]}"
        return tr
    context_md = proc.stdout
    msg = client.messages.create(
        model=MODEL, max_tokens=512,
        system=(
            "You are answering a single question grounded in the provided "
            "context block. Cite passages by their HTML-comment anchors when "
            "possible. Be concise."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"{context_md}\n\n---\n\nQuestion: {task['question']}\n\n"
                f"Answer concisely:"
            ),
        }],
    )
    tr.answer = msg.content[0].text.strip()
    tr.input_tokens = msg.usage.input_tokens
    tr.output_tokens = msg.usage.output_tokens
    tr.cost_usd = _cost_usd(tr.input_tokens, tr.output_tokens)
    tr.n_llm_calls = 1
    tr.wall_seconds = time.perf_counter() - t0
    return tr


# ---------- Approach B: grep + Read tool loop --------------------------------

_GREP_TOOL = {
    "name": "grep",
    "description": "Search file contents in the corpus root for a regex pattern. Returns up to 25 matching lines with file:line prefixes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to match"},
            "path_glob": {"type": "string", "description": "Optional glob (e.g. '*.md', '*.py'). Default: all files."},
        },
        "required": ["pattern"],
    },
}
_READ_TOOL = {
    "name": "read",
    "description": "Read a UTF-8 file from the corpus root and return its full contents.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the corpus root"},
        },
        "required": ["path"],
    },
}


def _tool_grep(corpus_root: Path, pattern: str, path_glob: str = "") -> str:
    cmd = ["rg", "-n", "--no-heading", "-S", "-m", "25", pattern]
    if path_glob:
        cmd += ["-g", path_glob]
    cmd += [str(corpus_root)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    out = proc.stdout
    if not out:
        return f"no matches for {pattern!r}"
    lines = out.splitlines()[:25]
    # Strip absolute paths down to corpus-relative for clarity.
    rel_lines = [
        ln.replace(str(corpus_root) + os.sep, "").replace(str(corpus_root) + "/", "")
        for ln in lines
    ]
    return "\n".join(rel_lines)


def _tool_read(corpus_root: Path, path: str) -> str:
    target = (corpus_root / path).resolve()
    if not str(target).startswith(str(corpus_root.resolve())):
        return f"refused: path {path!r} escapes corpus root"
    if not target.exists():
        return f"file not found: {path}"
    try:
        text = target.read_text(encoding="utf-8")
    except Exception as e:
        return f"read failed: {e}"
    if len(text) > 50000:
        text = text[:50000] + f"\n\n... [truncated, file is {len(text)} chars]"
    return text


def _run_grep_read_loop(client, task: dict, corpus_root: Path, max_turns: int = 6) -> Trial:
    t0 = time.perf_counter()
    tr = Trial(task_id=task["id"], approach="grep-read-loop")
    messages = [
        {"role": "user", "content": task["question"] + "\n\nUse the grep and read tools to find the answer in the corpus, then answer concisely."},
    ]
    system = (
        "You are answering a question grounded in a corpus on disk. Use the "
        "`grep` tool to locate relevant lines, then `read` to view full files. "
        "Stop tool-using as soon as you have enough evidence and write a concise "
        f"final answer. You have {max_turns} tool turns max."
    )
    total_in = 0
    total_out = 0
    n_calls = 0
    for turn in range(max_turns):
        msg = client.messages.create(
            model=MODEL, max_tokens=1024, system=system,
            tools=[_GREP_TOOL, _READ_TOOL], messages=messages,
        )
        n_calls += 1
        total_in += msg.usage.input_tokens
        total_out += msg.usage.output_tokens
        if msg.stop_reason != "tool_use":
            # Final answer — extract text and exit
            for block in msg.content:
                if getattr(block, "type", None) == "text":
                    tr.answer = block.text.strip()
                    break
            break
        # Append assistant turn + tool results, then loop.
        messages.append({"role": "assistant", "content": msg.content})
        tool_results: list[dict] = []
        for block in msg.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            name = block.name
            inp = block.input
            if name == "grep":
                result = _tool_grep(corpus_root, inp.get("pattern", ""), inp.get("path_glob", ""))
            elif name == "read":
                result = _tool_read(corpus_root, inp.get("path", ""))
            else:
                result = f"unknown tool: {name}"
            tool_results.append({
                "type": "tool_result", "tool_use_id": block.id, "content": result[:8000],
            })
        messages.append({"role": "user", "content": tool_results})
    if not tr.answer:
        tr.answer = "(no final answer within tool-turn budget)"
        tr.error = f"hit max_turns={max_turns}"
    tr.input_tokens = total_in
    tr.output_tokens = total_out
    tr.cost_usd = _cost_usd(total_in, total_out)
    tr.n_llm_calls = n_calls
    tr.wall_seconds = time.perf_counter() - t0
    return tr


# ---------- Approach C: Full corpus in context -------------------------------

def _build_full_context(corpus_root: Path, max_chars: int = 600_000) -> str:
    """Concatenate every text file under corpus_root, capped to max_chars.

    600 KB ≈ 150 K tokens — well within Sonnet 4.6's 200 K context window with
    headroom for the question + answer. Order: docs/ first (more reference-y),
    then src/ — same priority a developer would skim.
    """
    parts: list[str] = []
    total = 0
    text_exts = {".md", ".py", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml"}
    for top in ("docs", "src", "README.md", "CLAUDE.md"):
        root = corpus_root / top
        if root.is_file():
            paths = [root]
        elif root.is_dir():
            paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in text_exts)
        else:
            continue
        for p in paths:
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            rel = p.relative_to(corpus_root)
            chunk = f"\n\n=== {rel} ===\n{text}"
            if total + len(chunk) > max_chars:
                break
            parts.append(chunk)
            total += len(chunk)
        if total >= max_chars:
            break
    return "".join(parts)


def _run_full_context(client, task: dict, full_corpus: str) -> Trial:
    t0 = time.perf_counter()
    tr = Trial(task_id=task["id"], approach="full-corpus-in-context")
    msg = client.messages.create(
        model=MODEL, max_tokens=512,
        system=(
            "You are answering a question grounded in the corpus dump below. "
            "Be concise. Cite source files (e.g. 'docs/internal/FIELD.md') "
            "when possible."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"=== CORPUS ===\n{full_corpus}\n=== END CORPUS ===\n\n"
                f"Question: {task['question']}\n\nAnswer concisely:"
            ),
        }],
    )
    tr.answer = msg.content[0].text.strip()
    tr.input_tokens = msg.usage.input_tokens
    tr.output_tokens = msg.usage.output_tokens
    tr.cost_usd = _cost_usd(tr.input_tokens, tr.output_tokens)
    tr.n_llm_calls = 1
    tr.wall_seconds = time.perf_counter() - t0
    return tr


# ---------- Approach D: No retrieval -----------------------------------------

def _run_no_retrieval(client, task: dict) -> Trial:
    t0 = time.perf_counter()
    tr = Trial(task_id=task["id"], approach="no-retrieval")
    msg = client.messages.create(
        model=MODEL, max_tokens=512,
        system=(
            "Answer the question concisely from your training knowledge. The "
            "question may be about a specific software project; if you don't "
            "know the answer with confidence, say so explicitly."
        ),
        messages=[{"role": "user", "content": f"Question: {task['question']}\n\nAnswer concisely:"}],
    )
    tr.answer = msg.content[0].text.strip()
    tr.input_tokens = msg.usage.input_tokens
    tr.output_tokens = msg.usage.output_tokens
    tr.cost_usd = _cost_usd(tr.input_tokens, tr.output_tokens)
    tr.n_llm_calls = 1
    tr.wall_seconds = time.perf_counter() - t0
    return tr


# ---------- Locked judge -----------------------------------------------------

JUDGE_PROMPT = """\
You are grading a candidate answer against a known-good ground truth. Score one of:

  correct  — the candidate states the same factual claim as the ground truth
             (paraphrasing OK; minor extra detail OK; missing detail not OK)
  partial  — the candidate has the gist right but is missing a load-bearing
             detail OR contains a small factual error (e.g. wrong number, wrong
             flag name) that doesn't fully reverse the meaning
  wrong    — the candidate states a different fact, contradicts the ground
             truth, or is unrelated
  refused  — the candidate explicitly says "I don't know" or "I cannot answer"
             or refuses without attempting; this is distinct from "wrong"

Output exactly one JSON object on a single line: {"score": "...", "reason": "..."}
"""


def _judge(client, task: dict, candidate: str) -> tuple[str, str]:
    msg = client.messages.create(
        model=MODEL, max_tokens=200, system=JUDGE_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Question: {task['question']}\n\n"
                f"Ground truth: {task['ground_truth']}\n\n"
                f"Candidate answer: {candidate}\n\n"
                "Output the JSON now."
            ),
        }],
    )
    raw = msg.content[0].text.strip()
    # Extract first JSON object on a single line; tolerate stray markdown.
    m = re.search(r"\{[^{}]*\"score\"\s*:\s*\"([^\"]+)\"[^{}]*\}", raw)
    if not m:
        return "wrong", f"unparseable judge output: {raw[:200]}"
    try:
        obj = json.loads(m.group(0))
        return obj.get("score", "wrong"), obj.get("reason", "")
    except json.JSONDecodeError:
        return m.group(1), raw[:200]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/results/user_bench/token_usage.json")
    parser.add_argument("--km", default="resonance-lattice.rlat",
                        help="Path to a built knowledge model of the rlat repo")
    parser.add_argument("--corpus-root", default=str(_REPO),
                        help="Filesystem root for grep/read tools and full-context dump")
    parser.add_argument("--n-tasks", type=int, default=0,
                        help="Limit to first N tasks (default: all). Useful for pilots.")
    parser.add_argument("--budget-usd", type=float, default=100.0,
                        help="Hard cap on cumulative spend; aborts when exceeded.")
    parser.add_argument(
        "--approaches", default="rlat,grep_read,full_context,no_retrieval",
        help="Comma-separated subset of approaches to run."
    )
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip LLM-judge phase (use for harness debugging).")
    args = parser.parse_args(argv)

    tasks_path = _HERE.parent / "tasks.jsonl"
    tasks = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.n_tasks:
        tasks = tasks[:args.n_tasks]
    print(f"[bench1] {len(tasks)} tasks loaded", flush=True)

    km_path = Path(args.km)
    if not km_path.exists():
        print(f"[bench1] FATAL: --km {km_path} not found. Build with "
              f"`rlat build ./docs ./src -o resonance-lattice.rlat`.", file=sys.stderr)
        return 1
    corpus_root = Path(args.corpus_root)

    client = _client()
    enabled = set(args.approaches.split(","))

    # Pre-build full-corpus context once (cached across tasks for approach C).
    full_corpus = _build_full_context(corpus_root) if "full_context" in enabled else ""
    if full_corpus:
        print(f"[bench1] full-context bundle: {len(full_corpus)} chars", flush=True)

    trials: list[Trial] = []
    running_cost = 0.0
    for task in tasks:
        for approach_key, runner in (
            ("rlat", lambda t: _run_rlat_skill(client, t, km_path, mode="constrain")),
            ("rlat_augment", lambda t: _run_rlat_skill(client, t, km_path, mode="augment")),
            ("rlat_knowledge", lambda t: _run_rlat_skill(client, t, km_path, mode="knowledge")),
            ("grep_read", lambda t: _run_grep_read_loop(client, t, corpus_root)),
            ("full_context", lambda t: _run_full_context(client, t, full_corpus)),
            ("no_retrieval", lambda t: _run_no_retrieval(client, t)),
        ):
            if approach_key not in enabled:
                continue
            if running_cost >= args.budget_usd:
                print(f"[bench1] BUDGET CAP $${args.budget_usd:.2f} reached; aborting", flush=True)
                break
            try:
                tr = runner(task)
            except Exception as e:
                tr = Trial(task_id=task["id"], approach=approach_key, error=repr(e))
            running_cost += tr.cost_usd
            print(
                f"[bench1] {task['id']:5s} {tr.approach:24s}  "
                f"in={tr.input_tokens:6d} out={tr.output_tokens:4d}  "
                f"${tr.cost_usd:.4f}  ({tr.wall_seconds:.1f}s)  "
                f"running=${running_cost:.2f}",
                flush=True,
            )
            trials.append(tr)
            # Incremental checkpoint — preserves partial progress against
            # any later failure (typically API rate-limit / out-of-credits
            # in the judge phase). Writes a partial JSON every trial.
            _checkpoint = {
                "config": {"model": MODEL, "n_tasks": len(tasks),
                           "approaches": sorted(enabled), "km": str(km_path)},
                "trials": [asdict(t) for t in trials],
                "partial": True,
            }
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(_checkpoint, indent=2),
                                          encoding="utf-8")
        else:
            continue
        break  # outer break propagation when budget hit

    # Judge phase
    if not args.skip_judge:
        print(f"\n[bench1] judging {len(trials)} trials with {MODEL} ...", flush=True)
        for tr in trials:
            if tr.error or not tr.answer:
                tr.judge_score = "wrong"
                tr.judge_explanation = tr.error or "empty answer"
                continue
            task = next(t for t in tasks if t["id"] == tr.task_id)
            score, reason = _judge(client, task, tr.answer)
            tr.judge_score = score
            tr.judge_explanation = reason
            running_cost += 0.001  # judge calls are cheap; sum cost-of-judge into total

    # Aggregate
    by_approach: dict[str, dict] = {}
    for tr in trials:
        agg = by_approach.setdefault(tr.approach, {
            "n_trials": 0, "n_correct": 0, "n_partial": 0, "n_wrong": 0, "n_refused": 0,
            "total_input_tokens": 0, "total_output_tokens": 0, "total_cost_usd": 0.0,
            "total_wall_seconds": 0.0, "total_llm_calls": 0,
        })
        agg["n_trials"] += 1
        agg[f"n_{tr.judge_score}"] = agg.get(f"n_{tr.judge_score}", 0) + 1
        agg["total_input_tokens"] += tr.input_tokens
        agg["total_output_tokens"] += tr.output_tokens
        agg["total_cost_usd"] += tr.cost_usd
        agg["total_wall_seconds"] += tr.wall_seconds
        agg["total_llm_calls"] += tr.n_llm_calls
    for ap, agg in by_approach.items():
        n_ok = agg.get("n_correct", 0)
        agg["accuracy"] = n_ok / agg["n_trials"] if agg["n_trials"] else 0.0
        agg["tokens_per_correct"] = (
            (agg["total_input_tokens"] + agg["total_output_tokens"]) / n_ok
            if n_ok else float("inf")
        )
        agg["dollars_per_correct"] = agg["total_cost_usd"] / n_ok if n_ok else float("inf")

    payload = {
        "config": {
            "model": MODEL,
            "n_tasks": len(tasks),
            "approaches": sorted(enabled),
            "km": str(km_path),
            "corpus_root": str(corpus_root),
            "budget_usd": args.budget_usd,
        },
        "by_approach": by_approach,
        "trials": [asdict(t) for t in trials],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[bench1] wrote {out}", flush=True)
    print(f"[bench1] running total cost: $${running_cost:.2f}", flush=True)
    print(f"\n[bench1] per-approach summary:", flush=True)
    for ap, agg in sorted(by_approach.items()):
        print(
            f"  {ap:24s}  acc={agg['accuracy']:.1%}  "
            f"tokens/correct={agg['tokens_per_correct']:>8.0f}  "
            f"$/correct={agg['dollars_per_correct']:.4f}  "
            f"calls/q={agg['total_llm_calls']/max(agg['n_trials'],1):.2f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
