"""Benchmark 2 — Hallucination reduction.

Measures hallucination rate, refusal-correctness, citation accuracy across
five grounding modes against a 40-question test set (30 answerable + 10
distractors). Drifted-subset deferred to v2.0.1.

Approaches share the same retrieval (rlat top-K=5 against
`resonance-lattice.rlat`) and differ only in how the consumer LLM is
instructed:

  A. constrain_verified  — `--mode constrain --verified-only` (default)
  B. augment             — `--mode augment`
  C. knowledge           — `--mode knowledge`
  D. no_retrieval        — Sonnet alone, no rlat (the LLM-only baseline)
  D. plain_rag           — same passages, no grounding directive
  E. no_retrieval        — Sonnet alone

Locked-rubric judge scores each candidate answer correct/partial/wrong/
refused.

Methodology: docs/internal/benchmarks/02_hallucination.md.

Usage:
  export CLAUDE_API=sk-ant-...
  python -m benchmarks.user_bench.hallucination.run \\
      --output benchmarks/results/user_bench/hallucination.json \\
      --budget-usd 150
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent.parent.parent
_SRC = _REPO / "src"
if (_SRC / "resonance_lattice" / "__init__.py").exists():
    sys.path.insert(0, str(_SRC))


from resonance_lattice._pricing import SONNET_MODEL as MODEL, cost_usd as _cost_usd  # noqa: F401


@dataclass
class Trial:
    task_id: str
    kind: str            # answerable | distractor | drifted
    approach: str
    answer: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    wall_seconds: float = 0.0
    error: str | None = None
    judge_score: str = ""      # correct | partial | wrong | refused
    judge_explanation: str = ""


def _client():
    from resonance_lattice.optimise.synth_queries import discover_api_key
    import anthropic
    key = discover_api_key()
    if not key:
        raise RuntimeError("no LLM API key resolvable")
    return anthropic.Anthropic(api_key=key)


# Pre-baked system prompt suffixes for each approach. The rlat-mode header
# is included by `rlat search --format context --mode <m>`; the plain_rag
# approach strips that header and treats the same passages as a flat
# context block; no_retrieval gets nothing.

_RAG_SYSTEM = (
    "You are answering a question grounded in the corpus passages below. "
    "Be concise. Cite source files when possible."
)

_NORETRIEVAL_SYSTEM = (
    "Answer the question concisely from your training knowledge. The "
    "question may be about a specific software project; if you don't "
    "know the answer with confidence, say so explicitly."
)


def _retrieve(km_path: Path, query: str, mode: str | None, verified_only: bool, top_k: int = 5) -> tuple[str, str]:
    """Subprocess `rlat search --format context` and return (markdown, err).

    `mode` of None means we want plain passages without the directive (for
    `plain_rag` approach) — we strip the rlat-mode header lines.

    Invokes via `sys.executable -m resonance_lattice.cli.app` rather than
    the `rlat` console script so the bench always exercises the checked-out
    package on `sys.path`, never a stale globally-installed `rlat`.
    """
    cmd = [
        sys.executable, "-m", "resonance_lattice.cli.app",
        "search", str(km_path), query,
        "--top-k", str(top_k), "--format", "context",
    ]
    if mode:
        cmd += ["--mode", mode]
    if verified_only:
        cmd += ["--verified-only"]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        return "", f"rlat search exit={proc.returncode}: {proc.stderr[:500]}"
    text = proc.stdout
    if mode is None:
        # Strip the leading mode header (HTML comment + blockquote) so plain_rag
        # really doesn't see the directive.
        lines = text.splitlines()
        while lines and (
            lines[0].startswith("<!-- rlat-mode") or
            lines[0].startswith(">") or
            lines[0].strip() == ""
        ):
            lines.pop(0)
        text = "\n".join(lines)
    return text, ""


_QUERY_REWRITE_SYSTEM = """You are an expert at generating diverse retrieval queries.

Given a user question about a documentation corpus, output exactly 3 distinct
short query phrasings (each 6-15 words) that, between them, are likely to
surface the relevant passage from a dense-retrieval index. Vary the
vocabulary across the three. Output ONLY the three queries, one per line, no
numbering or explanation."""


def _multi_query_retrieve(client, km_path: Path, question: str,
                          mode: str = "augment", top_k: int = 5) -> tuple[str, int, int, list[str]]:
    """Generate 3 query variants, retrieve top_k for each, dedupe + return as one
    grounding-mode markdown block. `mode` parameterises the directive header
    + per-query retrieval call. Returns (ctx_markdown, in_tokens, out_tokens, queries)."""
    from resonance_lattice.cli._grounding import Mode, format_header
    rewrite_msg = client.messages.create(
        model=MODEL, max_tokens=200, system=_QUERY_REWRITE_SYSTEM,
        messages=[{"role": "user", "content": f"Question: {question}\n\nThree query phrasings:"}],
    )
    raw = rewrite_msg.content[0].text.strip()
    queries = [q.strip() for q in raw.splitlines() if q.strip() and not q.strip().startswith("#")]
    if not queries:
        queries = [question]
    queries = queries[:3]
    seen_anchors: set[str] = set()
    blocks: list[str] = []
    header = format_header(Mode(mode)) + "\n"
    for q in queries:
        ctx, err, _ = _retrieve(km_path, q, mode, False, top_k=top_k)
        if err:
            continue
        # Strip the per-query mode header (we'll write one combined header)
        body_lines = []
        for line in ctx.splitlines():
            if line.startswith("<!-- rlat-mode") or line.startswith(">") or line.startswith("# "):
                continue
            body_lines.append(line)
        body = "\n".join(body_lines)
        # Dedupe on the source-file:offset anchor
        unique_block_lines = []
        skip_until_blank = False
        for line in body.splitlines():
            if line.startswith("<!--") and "score=" in line:
                anchor = line.split("score=")[0]
                if anchor in seen_anchors:
                    skip_until_blank = True
                    continue
                seen_anchors.add(anchor)
                skip_until_blank = False
            if skip_until_blank and line.strip() == "":
                skip_until_blank = False
                continue
            if not skip_until_blank:
                unique_block_lines.append(line)
        blocks.append(f"<!-- query: {q} -->\n" + "\n".join(unique_block_lines))
    combined = header + "\n" + "\n".join(blocks)
    return combined, rewrite_msg.usage.input_tokens, rewrite_msg.usage.output_tokens, queries


_DR_PLANNER_SYSTEM = """You are a research planner for fact extraction from a documentation corpus.

Given a question, output a SHORT initial search query (6-15 words) that's
likely to surface a relevant passage. Output ONLY the query, no preamble."""


_DR_REFINE_SYSTEM = """You are a research agent answering a question from a documentation corpus.

You see: the original question, queries you've tried, and the retrieved
passages from each. Decide your next action.

Output exactly one line of JSON, nothing else:
- {"action": "answer", "answer": "<final answer>"}  if you have enough evidence
- {"action": "search", "query": "<next short query>"}  if you need more
- {"action": "give_up"}  if the corpus clearly doesn't have the answer

Stop searching when you have a clear answer — don't burn hops on confirmation."""


def _deep_research_retrieve(client, km_path: Path, question: str,
                             mode: str = "augment", max_hops: int = 4,
                             top_k: int = 5) -> tuple[str, int, int, list[dict]]:
    """Plan → search → refine → synthesize loop. Returns (final_answer, in_tokens, out_tokens, hops_log)."""
    in_tokens = 0
    out_tokens = 0
    hops_log: list[dict] = []

    # Hop 1: planner generates first query.
    plan_msg = client.messages.create(
        model=MODEL, max_tokens=100, system=_DR_PLANNER_SYSTEM,
        messages=[{"role": "user", "content": f"Question: {question}"}],
    )
    in_tokens += plan_msg.usage.input_tokens
    out_tokens += plan_msg.usage.output_tokens
    current_query = plan_msg.content[0].text.strip().split("\n")[0].strip()
    hops_log.append({"hop": 1, "kind": "plan", "query": current_query})

    evidence_blocks: list[str] = []
    queries_tried: list[str] = []
    for hop in range(2, max_hops + 1):
        ctx, err = _retrieve(km_path, current_query, mode, False, top_k=top_k)
        if err:
            hops_log.append({"hop": hop, "kind": "search_failed", "query": current_query, "error": err[:200]})
            break
        queries_tried.append(current_query)
        evidence_blocks.append(f"--- Search query: {current_query!r} ---\n{ctx}")
        hops_log.append({"hop": hop, "kind": "search", "query": current_query})

        # Refiner decides next action
        evidence = "\n\n".join(evidence_blocks)
        prompt = (
            f"Question: {question}\n\n"
            f"Evidence collected so far ({len(queries_tried)} queries tried):\n\n"
            f"{evidence[:8000]}\n\n"
            f"What's your next action? (answer / search / give_up)"
        )
        refine_msg = client.messages.create(
            model=MODEL, max_tokens=400, system=_DR_REFINE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        in_tokens += refine_msg.usage.input_tokens
        out_tokens += refine_msg.usage.output_tokens
        raw = refine_msg.content[0].text.strip()
        m = re.search(r'\{[^{}]*"action"[^{}]*\}', raw)
        if not m:
            hops_log.append({"hop": hop, "kind": "parse_failed", "raw": raw[:300]})
            return raw, in_tokens, out_tokens, hops_log
        try:
            action = json.loads(m.group(0))
        except json.JSONDecodeError:
            return raw, in_tokens, out_tokens, hops_log
        hops_log.append({"hop": hop, "kind": "decide", "action": action.get("action")})
        if action.get("action") == "answer":
            return action.get("answer", ""), in_tokens, out_tokens, hops_log
        if action.get("action") == "give_up":
            return ("I cannot find the answer to this question in the documentation corpus."
                   ), in_tokens, out_tokens, hops_log
        if action.get("action") == "search":
            current_query = action.get("query", current_query)
            continue

    # Out of hops — synthesise from whatever evidence we have
    evidence = "\n\n".join(evidence_blocks)
    synth_prompt = (
        f"Question: {question}\n\n"
        f"All evidence collected:\n\n{evidence[:10000]}\n\n"
        f"Provide a concise answer based ONLY on the evidence above. "
        f"If the evidence doesn't cover the question, say so."
    )
    synth_msg = client.messages.create(
        model=MODEL, max_tokens=500,
        system="You synthesise a concise answer from retrieved evidence. Cite source files in parentheses.",
        messages=[{"role": "user", "content": synth_prompt}],
    )
    in_tokens += synth_msg.usage.input_tokens
    out_tokens += synth_msg.usage.output_tokens
    hops_log.append({"hop": max_hops + 1, "kind": "synth_after_max_hops"})
    return synth_msg.content[0].text.strip(), in_tokens, out_tokens, hops_log


_GREP_GLOB_SYSTEM = (
    "You answer a question by exploring a documentation corpus with grep, "
    "glob, and read_file tools. The corpus is rooted at `.` — paths are "
    "relative. Strategy: glob to find candidate files, grep to narrow to "
    "lines, read_file to confirm. Make at most a handful of tool calls — "
    "stop and answer once you have evidence. If the corpus does not cover "
    "the question, say so explicitly. Cite source paths in your answer."
)


def _grep_glob_tools() -> list[dict]:
    """Anthropic tool-use schema for the grep/glob baseline."""
    return [
        {
            "name": "glob",
            "description": "List files in the corpus whose path matches a glob "
                           "pattern (e.g. `**/*.md`, `docs/fabric/*.md`).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern."},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "grep",
            "description": "Search file contents for a regex. Returns up to 50 "
                           "matching lines as `path:line:content`.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern."},
                    "path_glob": {"type": "string", "description": "Optional path "
                                  "glob to scope the search (default `**/*.md`)."},
                },
                "required": ["pattern"],
            },
        },
        {
            "name": "read_file",
            "description": "Read the full contents of one file. Truncates to 6000 chars.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to corpus root."},
                },
                "required": ["path"],
            },
        },
    ]


# Per-bench process: maps km_path → (extracted_root, [(rel_path, full_path), ...])
# The file-list is captured once at extraction time so the grep/glob tools
# don't re-walk the tree on every dispatch (8 tool calls × 63 questions
# = 504 walks otherwise; ~2.5s/q on a 2k-file Fabric corpus).
_CORPUS_CACHE: dict[Path, tuple[Path, list[tuple[str, Path]]]] = {}


def _ensure_corpus_extracted(km_path: Path) -> tuple[Path, list[tuple[str, Path]]]:
    """Extract bundled `source/` from the .rlat into a temp dir.

    Returns `(root_dir, files)` where `files` is the pre-walked list of
    `(rel_path, full_path)` tuples. Caches per `km_path` so repeated
    grep_glob trials in the same bench process hit zero extraction +
    zero rewalk cost.
    """
    if km_path in _CORPUS_CACHE:
        return _CORPUS_CACHE[km_path]
    import tempfile, zipfile
    import zstandard as zstd
    out = Path(tempfile.mkdtemp(prefix="rlat-grep-glob-"))
    dctx = zstd.ZstdDecompressor()
    files: list[tuple[str, Path]] = []
    with zipfile.ZipFile(km_path, "r") as zf:
        for name in zf.namelist():
            if not name.startswith("source/"):
                continue
            rel = name[len("source/"):]
            if not rel:
                continue
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            blob = zf.read(name)
            try:
                target.write_bytes(dctx.decompress(blob))
            except Exception:
                # Fallback: not zstd-framed (older bundled or partial).
                target.write_bytes(blob)
            files.append((rel.replace("\\", "/"), target))
    _CORPUS_CACHE[km_path] = (out, files)
    return out, files


def _grep_glob_tool_dispatch(tool_name: str, tool_input: dict,
                              files: list[tuple[str, Path]]) -> str:
    """Execute one tool call against the pre-walked corpus file list."""
    import fnmatch
    if tool_name == "glob":
        pattern = tool_input.get("pattern", "**/*")
        matches = [rel for rel, _ in files if fnmatch.fnmatch(rel, pattern)]
        if not matches:
            return f"(no files match {pattern!r})"
        return "\n".join(matches[:200])
    if tool_name == "grep":
        pattern = tool_input.get("pattern", "")
        path_glob = tool_input.get("path_glob", "**/*.md")
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"(invalid regex: {e})"
        results: list[str] = []
        for rel, full in files:
            if not fnmatch.fnmatch(rel, path_glob):
                continue
            try:
                text = full.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for ln, line in enumerate(text.split("\n"), 1):
                if regex.search(line):
                    results.append(f"{rel}:{ln}:{line.strip()[:200]}")
                    if len(results) >= 50:
                        break
            if len(results) >= 50:
                break
        if not results:
            return f"(no matches for {pattern!r} in {path_glob})"
        return "\n".join(results)
    if tool_name == "read_file":
        path = tool_input.get("path", "")
        target = (root / path).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError:
            return f"(refused: path {path!r} outside corpus root)"
        if not target.is_file():
            return f"(file {path!r} not found)"
        text = target.read_text(encoding="utf-8", errors="ignore")
        if len(text) > 6000:
            return text[:6000] + f"\n... [truncated; full file is {len(text)} chars]"
        return text
    return f"(unknown tool: {tool_name})"


def _llm_grep_glob(client, km_path: Path, question: str,
                   max_tool_calls: int = 8) -> tuple[str, int, int, int]:
    """LLM-with-grep/glob baseline: Sonnet sees the question and a tool
    surface, makes up to N tool calls against the extracted corpus, then
    answers. Returns (answer, in_tokens, out_tokens, n_tool_calls)."""
    root = _ensure_corpus_extracted(km_path)
    tools = _grep_glob_tools()
    messages: list[dict] = [{
        "role": "user",
        "content": (
            f"Corpus root: ./ (relative paths). Question: {question}\n\n"
            f"Use the tools to search the corpus, then answer concisely. "
            f"Cite source paths."
        ),
    }]
    in_tokens = 0
    out_tokens = 0
    n_tool_calls = 0

    while n_tool_calls <= max_tool_calls:
        msg = client.messages.create(
            model=MODEL, max_tokens=1024, system=_GREP_GLOB_SYSTEM,
            tools=tools, messages=messages,
        )
        in_tokens += msg.usage.input_tokens
        out_tokens += msg.usage.output_tokens

        tool_uses = [b for b in msg.content if getattr(b, "type", None) == "tool_use"]
        if not tool_uses:
            text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
            return "\n".join(text_blocks).strip() or "(no answer produced)", in_tokens, out_tokens, n_tool_calls

        # Append the assistant's tool-use turn as-is (Anthropic's API requires this).
        messages.append({"role": "assistant", "content": msg.content})
        tool_results = []
        for tu in tool_uses:
            n_tool_calls += 1
            result_text = _grep_glob_tool_dispatch(tu.name, tu.input, root)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result_text[:8000],
            })
        messages.append({"role": "user", "content": tool_results})

    # Out of tool budget — force an answer.
    messages.append({
        "role": "user",
        "content": "Tool budget exhausted. Provide your best answer now or say you cannot.",
    })
    msg = client.messages.create(
        model=MODEL, max_tokens=512, system=_GREP_GLOB_SYSTEM,
        messages=messages,
    )
    in_tokens += msg.usage.input_tokens
    out_tokens += msg.usage.output_tokens
    text_blocks = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    return "\n".join(text_blocks).strip() or "(no answer produced)", in_tokens, out_tokens, n_tool_calls


def _ask(client, system: str, user: str, max_tokens: int = 512) -> tuple[str, int, int]:
    msg = client.messages.create(
        model=MODEL, max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text.strip(), msg.usage.input_tokens, msg.usage.output_tokens


# Approach taxonomy:
#
#   single_<mode>     rlat search --format context --mode <mode>, top-k=5
#   multi_<mode>      3-query rewrite + dedupe + grounding header for <mode>
#   deep_<mode>       plan→search→refine→synth loop (max-hops=4) under <mode>
#   no_retrieval      LLM-only baseline
#   llm_grep_glob     LLM with glob/grep/read_file tools over the source dir
#   plain_rag         retrieved passages without any grounding directive
#                     (kept for backwards compatibility / ablation)
#   constrain_verified   constrain mode + --verified-only (drift gate)
#   augment_topk_10      ablation: augment with top-k=10
#
# Default is the 11-lane matrix Kane requested:
#   3 modes × {single, multi, deep} + no_retrieval + llm_grep_glob.
_DEFAULT_APPROACHES = (
    "single_augment,single_constrain,single_knowledge,"
    "multi_augment,multi_constrain,multi_knowledge,"
    "deep_augment,deep_constrain,deep_knowledge,"
    "no_retrieval,llm_grep_glob"
)

_RLAT_RETRIEVAL_MODES = {"augment", "constrain", "knowledge"}

# Legacy approach names from earlier bench result JSONs map to current
# approach keys. Resolved at the top of `_run_approach`; never proliferates
# through the dispatch.
_LEGACY_ALIASES: dict[str, str] = {
    "augment":          "single_augment",
    "constrain":        "single_constrain",
    "knowledge":        "single_knowledge",
    "rlat_multi_query": "multi_augment",
    "deep_research":    "deep_augment",
}

# Approaches that take a non-rlat retrieval shape (no mode parameter).
_SPECIAL_APPROACHES: dict[str, str] = {
    "constrain_verified": "single",   # constrain mode + --verified-only
    "augment_topk_10":    "single",   # augment, top-k=10 ablation
    "plain_rag":          "single",   # passages without grounding directive
}


def _run_approach(client, task: dict, km_path: Path, approach: str) -> Trial:
    t0 = time.perf_counter()
    tr = Trial(task_id=task["id"], kind=task["kind"], approach=approach)
    canonical = _LEGACY_ALIASES.get(approach, approach)
    try:
        if canonical == "no_retrieval":
            answer, ti, to = _ask(client, _NORETRIEVAL_SYSTEM, f"Question: {task['question']}\n\nAnswer concisely:")

        elif canonical == "llm_grep_glob":
            answer, ti, to, n_calls = _llm_grep_glob(client, km_path, task["question"], max_tool_calls=8)

        elif canonical == "constrain_verified":
            ctx, err = _retrieve(km_path, task["question"], "constrain", True, top_k=5)
            if err:
                tr.error = err; tr.wall_seconds = time.perf_counter() - t0; return tr
            user = f"{ctx}\n\n---\n\nQuestion: {task['question']}\n\nAnswer concisely:"
            answer, ti, to = _ask(client, _RAG_SYSTEM, user)

        elif canonical == "augment_topk_10":
            ctx, err = _retrieve(km_path, task["question"], "augment", False, top_k=10)
            if err:
                tr.error = err; tr.wall_seconds = time.perf_counter() - t0; return tr
            user = f"{ctx}\n\n---\n\nQuestion: {task['question']}\n\nAnswer concisely:"
            answer, ti, to = _ask(client, _RAG_SYSTEM, user)

        elif canonical == "plain_rag":
            ctx, err = _retrieve(km_path, task["question"], None, False, top_k=5)
            if err:
                tr.error = err; tr.wall_seconds = time.perf_counter() - t0; return tr
            user = f"{ctx}\n\n---\n\nQuestion: {task['question']}\n\nAnswer concisely:"
            answer, ti, to = _ask(client, _RAG_SYSTEM, user)

        elif canonical.startswith(("single_", "multi_", "deep_")):
            prefix, _, mode = canonical.partition("_")
            if mode not in _RLAT_RETRIEVAL_MODES:
                raise ValueError(
                    f"unknown approach {approach!r} (canonical={canonical!r}): "
                    f"mode {mode!r} not in {sorted(_RLAT_RETRIEVAL_MODES)}"
                )
            if prefix == "single":
                ctx, err = _retrieve(km_path, task["question"], mode, False, top_k=5)
                if err:
                    tr.error = err; tr.wall_seconds = time.perf_counter() - t0; return tr
                user = f"{ctx}\n\n---\n\nQuestion: {task['question']}\n\nAnswer concisely:"
                answer, ti, to = _ask(client, _RAG_SYSTEM, user)
            elif prefix == "multi":
                ctx, mq_in, mq_out, _ = _multi_query_retrieve(client, km_path, task["question"], mode=mode, top_k=5)
                user = f"{ctx}\n\n---\n\nQuestion: {task['question']}\n\nAnswer concisely:"
                answer, ti_ans, to_ans = _ask(client, _RAG_SYSTEM, user, max_tokens=512)
                ti, to = mq_in + ti_ans, mq_out + to_ans
            else:  # prefix == "deep"
                answer, ti, to, _ = _deep_research_retrieve(client, km_path, task["question"], mode=mode, max_hops=4, top_k=5)

        else:
            raise ValueError(f"unknown approach {approach!r} (canonical={canonical!r})")

        tr.answer = answer
        tr.input_tokens = ti
        tr.output_tokens = to
        tr.cost_usd = _cost_usd(ti, to)
    except Exception as e:
        tr.error = repr(e)
    tr.wall_seconds = time.perf_counter() - t0
    return tr


JUDGE_PROMPT = """\
You are grading a candidate answer against ground truth on a 4-state rubric:

  correct  — same factual claim as ground truth (paraphrasing OK; minor extra
             detail OK; missing load-bearing detail not OK)
  partial  — gist right but missing a critical detail OR small error that
             doesn't fully reverse the meaning
  wrong    — different fact, contradicts ground truth, OR confidently states
             a false fact about a topic that doesn't exist in the corpus
             (this is the hallucination signal)
  refused  — explicitly says "I don't know", "I cannot answer", "this corpus
             doesn't have that", "the project doesn't include X" — distinct
             from "wrong"; this is the correct response to an unanswerable
             distractor question

Output exactly one JSON object on a single line: {"score": "...", "reason": "..."}
"""


def _judge(client, task: dict, candidate: str) -> tuple[str, str]:
    msg = client.messages.create(
        model=MODEL, max_tokens=200, system=JUDGE_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Question kind: {task['kind']}\n"
                f"Question: {task['question']}\n\n"
                f"Ground truth: {task['ground_truth']}\n\n"
                f"Candidate answer: {candidate}\n\n"
                "Output the JSON now."
            ),
        }],
    )
    raw = msg.content[0].text.strip()
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
    parser.add_argument("--output", default="benchmarks/results/user_bench/hallucination.json")
    parser.add_argument("--km", default="resonance-lattice.rlat")
    parser.add_argument("--n-tasks", type=int, default=0,
                        help="Limit to first N tasks (default: all).")
    parser.add_argument("--budget-usd", type=float, default=150.0)
    parser.add_argument(
        "--approaches",
        default=_DEFAULT_APPROACHES,
        help="Comma-separated subset. Default is the 11-lane matrix: "
             "single_<mode> + multi_<mode> + deep_<mode> for each of "
             "augment/constrain/knowledge, plus no_retrieval + "
             "llm_grep_glob. Legacy aliases also accepted: "
             "constrain_verified, augment, knowledge, augment_topk_10, "
             "rlat_multi_query, deep_research, plain_rag.",
    )
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument(
        "--tasks-file", default=None,
        help="Override tasks.jsonl path (default: bench dir tasks.jsonl). "
             "Use to run against alternate corpora (e.g. fabric_tasks.jsonl).",
    )
    args = parser.parse_args(argv)

    tasks_path = Path(args.tasks_file) if args.tasks_file else (_HERE.parent / "tasks.jsonl")
    tasks = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.n_tasks:
        tasks = tasks[:args.n_tasks]
    print(f"[bench2] {len(tasks)} tasks loaded ("
          f"{sum(1 for t in tasks if t['kind']=='answerable')} answerable, "
          f"{sum(1 for t in tasks if t['kind']=='distractor')} distractors)",
          flush=True)

    km_path = Path(args.km)
    if not km_path.exists():
        print(f"[bench2] FATAL: --km {km_path} not found.", file=sys.stderr)
        return 1

    client = _client()
    enabled = [a for a in args.approaches.split(",") if a]

    trials: list[Trial] = []
    running_cost = 0.0
    for task in tasks:
        for approach in enabled:
            if running_cost >= args.budget_usd:
                print(f"[bench2] BUDGET CAP reached; aborting", flush=True)
                break
            tr = _run_approach(client, task, km_path, approach)
            running_cost += tr.cost_usd
            print(
                f"[bench2] {task['id']:5s}/{task['kind']:11s} {approach:24s} "
                f"in={tr.input_tokens:6d} out={tr.output_tokens:4d} "
                f"${tr.cost_usd:.4f}  ({tr.wall_seconds:.1f}s)  "
                f"running=${running_cost:.2f}",
                flush=True,
            )
            trials.append(tr)
            # Incremental checkpoint — preserves partial progress.
            _checkpoint = {
                "config": {"model": MODEL, "n_tasks": len(tasks),
                           "approaches": enabled, "km": str(km_path)},
                "trials": [asdict(t) for t in trials],
                "partial": True,
            }
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(_checkpoint, indent=2),
                                          encoding="utf-8")
        else:
            continue
        break

    if not args.skip_judge:
        print(f"\n[bench2] judging {len(trials)} trials with {MODEL} ...", flush=True)
        for tr in trials:
            if tr.error or not tr.answer:
                tr.judge_score = "wrong"
                tr.judge_explanation = tr.error or "empty answer"
                continue
            task = next(t for t in tasks if t["id"] == tr.task_id)
            score, reason = _judge(client, task, tr.answer)
            tr.judge_score = score
            tr.judge_explanation = reason

    # Aggregate per (approach × kind)
    by_approach: dict[str, dict] = {}
    for tr in trials:
        key = tr.approach
        agg = by_approach.setdefault(key, {
            "n_answerable": 0, "n_distractor": 0,
            "answerable_correct": 0, "answerable_partial": 0,
            "answerable_wrong": 0, "answerable_refused": 0,
            "distractor_correct": 0, "distractor_refused": 0,
            "distractor_hallucinated": 0,  # = distractor_wrong + distractor_partial + distractor_correct (any non-refusal counts as hallucination on a distractor)
            "total_cost_usd": 0.0,
            "wall_seconds_total": 0.0,
            "_n_trials": 0,
        })
        agg["total_cost_usd"] += tr.cost_usd
        agg["wall_seconds_total"] += tr.wall_seconds or 0.0
        agg["_n_trials"] += 1
        if tr.kind == "answerable":
            agg["n_answerable"] += 1
            agg[f"answerable_{tr.judge_score}"] = agg.get(f"answerable_{tr.judge_score}", 0) + 1
        elif tr.kind == "distractor":
            agg["n_distractor"] += 1
            if tr.judge_score == "refused":
                agg["distractor_refused"] += 1
            else:
                agg["distractor_hallucinated"] += 1
                agg[f"distractor_{tr.judge_score}"] = agg.get(f"distractor_{tr.judge_score}", 0) + 1
    for ap, agg in by_approach.items():
        n_ans = max(agg["n_answerable"], 1)
        n_dis = max(agg["n_distractor"], 1)
        n_trials = max(agg.pop("_n_trials", 1), 1)
        agg["answerable_accuracy"] = agg.get("answerable_correct", 0) / n_ans
        agg["answerable_hallucination_rate"] = agg.get("answerable_wrong", 0) / n_ans
        agg["distractor_correct_refusal_rate"] = agg.get("distractor_refused", 0) / n_dis
        agg["distractor_hallucination_rate"] = agg.get("distractor_hallucinated", 0) / n_dis
        agg["mean_wall_seconds"] = agg["wall_seconds_total"] / n_trials
        agg["mean_cost_usd"] = agg["total_cost_usd"] / n_trials

    payload = {
        "config": {
            "model": MODEL,
            "n_tasks": len(tasks),
            "approaches": enabled,
            "km": str(km_path),
            "budget_usd": args.budget_usd,
        },
        "by_approach": by_approach,
        "trials": [asdict(t) for t in trials],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[bench2] wrote {out}", flush=True)
    print(f"[bench2] running total cost: ${running_cost:.2f}", flush=True)
    print(f"\n[bench2] per-approach summary:", flush=True)
    for ap, agg in sorted(by_approach.items()):
        print(
            f"  {ap:24s}  ans_acc={agg['answerable_accuracy']:.1%}  "
            f"ans_halluc={agg['answerable_hallucination_rate']:.1%}  "
            f"dis_refuse={agg['distractor_correct_refusal_rate']:.1%}  "
            f"dis_halluc={agg['distractor_hallucination_rate']:.1%}  "
            f"wall={agg['mean_wall_seconds']:.1f}s/q  "
            f"$/q={agg['mean_cost_usd']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
