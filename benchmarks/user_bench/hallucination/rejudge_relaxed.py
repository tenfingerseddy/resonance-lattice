"""Re-judge an existing bench-2 result with a relaxed rubric.

The strict rubric used in the original bench treats "missing critical
detail" as `partial`. Failure analysis (analyze_fabric.py) showed many
"partials" are answers that capture the load-bearing fact but miss a
secondary detail — i.e., they're substantively correct.

Relaxed rubric: a `correct_relaxed` score is granted when the answer
captures the SINGLE most-load-bearing fact in the ground truth, even if
secondary details are missing or imprecise. Wrong/refused/distractor-
hallucination semantics are unchanged.

Run after a bench has completed (--output JSON exists). Writes a parallel
*_relaxed.json with all trial.judge_score values rewritten and aggregates
recomputed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from anthropic import Anthropic

MODEL = "claude-sonnet-4-6"
INPUT_PRICE_PER_M = 3.0
OUTPUT_PRICE_PER_M = 15.0


RELAXED_PROMPT = """\
You are grading a candidate answer with a RELAXED rubric. Compare the
candidate to the ground truth and output one of these scores:

  correct  — the candidate captures the SINGLE most load-bearing fact
             from the ground truth. Secondary details, exact phrasing,
             additional caveats, or extra context are NOT required.
             A correct identification + reasonable elaboration counts.
  partial  — the candidate gets the gist DIRECTIONALLY right but
             materially misstates or contradicts the load-bearing fact.
  wrong    — the candidate states a different fact, contradicts the
             ground truth, or invents content. This is the hallucination
             signal.
  refused  — the candidate explicitly says it can't answer, doesn't have
             the information, or that the corpus doesn't cover the
             question. Distinct from wrong; this is the right response
             for distractors and for genuinely unanswerable cases.

Output exactly one JSON object on a single line:
{"score": "...", "reason": "..."}
"""


def _cost_usd(in_tok: int, out_tok: int) -> float:
    return (in_tok / 1e6) * INPUT_PRICE_PER_M + (out_tok / 1e6) * OUTPUT_PRICE_PER_M


def _judge(client: Anthropic, task: dict, candidate: str) -> tuple[str, str, int, int]:
    msg = client.messages.create(
        model=MODEL, max_tokens=200, system=RELAXED_PROMPT,
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
    m = re.search(r'\{[^{}]*"score"\s*:\s*"([^"]+)"[^{}]*\}', raw)
    if not m:
        return "wrong", f"unparseable: {raw[:200]}", msg.usage.input_tokens, msg.usage.output_tokens
    try:
        obj = json.loads(m.group(0))
        return (obj.get("score", "wrong"), obj.get("reason", ""),
                msg.usage.input_tokens, msg.usage.output_tokens)
    except json.JSONDecodeError:
        return m.group(1), raw[:200], msg.usage.input_tokens, msg.usage.output_tokens


def _aggregate(trials: list[dict]) -> dict[str, dict]:
    by_approach: dict[str, dict] = {}
    for tr in trials:
        key = tr["approach"]
        agg = by_approach.setdefault(key, {
            "n_answerable": 0, "n_distractor": 0,
            "answerable_correct": 0, "answerable_partial": 0,
            "answerable_wrong": 0, "answerable_refused": 0,
            "distractor_correct": 0, "distractor_refused": 0,
            "distractor_hallucinated": 0,
            "wall_seconds_total": 0.0, "_n": 0,
        })
        agg["wall_seconds_total"] += tr.get("wall_seconds") or 0.0
        agg["_n"] += 1
        score = tr.get("judge_score") or "wrong"
        if tr["kind"] == "answerable":
            agg["n_answerable"] += 1
            agg[f"answerable_{score}"] = agg.get(f"answerable_{score}", 0) + 1
        else:
            agg["n_distractor"] += 1
            if score == "refused":
                agg["distractor_refused"] += 1
            else:
                agg["distractor_hallucinated"] += 1
    for ap, agg in by_approach.items():
        n_ans = max(agg["n_answerable"], 1)
        n_dis = max(agg["n_distractor"], 1)
        n = max(agg.pop("_n", 1), 1)
        agg["answerable_accuracy"] = agg.get("answerable_correct", 0) / n_ans
        agg["answerable_hallucination_rate"] = agg.get("answerable_wrong", 0) / n_ans
        agg["distractor_correct_refusal_rate"] = agg.get("distractor_refused", 0) / n_dis
        agg["distractor_hallucination_rate"] = agg.get("distractor_hallucinated", 0) / n_dis
        agg["mean_wall_seconds"] = agg["wall_seconds_total"] / n
    return by_approach


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="bench result JSON to re-judge")
    p.add_argument("--output", required=True, help="output JSON path")
    p.add_argument("--tasks-file", required=True)
    p.add_argument("--budget-usd", type=float, default=2.0)
    args = p.parse_args(argv)

    client = Anthropic(api_key=os.environ.get("CLAUDE_API") or os.environ["ANTHROPIC_API_KEY"])
    tasks = {json.loads(line)["id"]: json.loads(line)
             for line in Path(args.tasks_file).read_text(encoding="utf-8").splitlines() if line.strip()}
    data = json.load(open(args.input, encoding="utf-8"))
    trials = data["trials"]
    print(f"[rejudge] re-judging {len(trials)} trials with relaxed rubric", flush=True)

    spent = 0.0
    n_changed = Counter()
    for i, tr in enumerate(trials, 1):
        if not tr.get("answer") or tr.get("error"):
            continue
        task = tasks.get(tr["task_id"])
        if not task:
            continue
        old = tr.get("judge_score")
        score, reason, ti, to = _judge(client, task, tr["answer"])
        cost = _cost_usd(ti, to)
        spent += cost
        if old != score:
            n_changed[(old, score)] += 1
        tr["judge_score_strict"] = old
        tr["judge_explanation_strict"] = tr.get("judge_explanation")
        tr["judge_score"] = score
        tr["judge_explanation"] = reason
        if i % 25 == 0:
            print(f"[rejudge] {i}/{len(trials)} done — spent=${spent:.4f}", flush=True)
        if spent >= args.budget_usd:
            print(f"[rejudge] BUDGET CAP ${args.budget_usd:.2f} reached at trial {i}", flush=True)
            break

    by_approach = _aggregate(trials)
    payload = {
        "config": {**data.get("config", {}), "rubric": "relaxed"},
        "by_approach": by_approach,
        "trials": trials,
        "rejudge_meta": {
            "transitions": {f"{a}->{b}": n for (a, b), n in n_changed.most_common()},
            "rejudge_cost_usd": spent,
        },
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[rejudge] wrote {args.output}", flush=True)
    print(f"[rejudge] re-judge spend: ${spent:.4f}", flush=True)
    print(f"[rejudge] score transitions:", flush=True)
    for (a, b), n in n_changed.most_common():
        print(f"  {a} -> {b}: {n}", flush=True)
    print(f"\n[rejudge] per-approach (relaxed rubric):", flush=True)
    for ap, agg in sorted(by_approach.items()):
        print(
            f"  {ap:22s}  ans_acc={agg['answerable_accuracy']:.1%}  "
            f"halluc={agg['answerable_hallucination_rate']:.1%}  "
            f"dis_refuse={agg['distractor_correct_refusal_rate']:.1%}  "
            f"dis_halluc={agg['distractor_hallucination_rate']:.1%}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
