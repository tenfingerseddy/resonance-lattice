"""Run the strict-rubric judge phase against an existing bench result JSON.

Used when the main bench harness crashed after inference but before the
in-process judge loop completed. Loads trials, judges each one with the
SAME rubric the main harness uses, writes back to the same path with
per-approach aggregates filled in.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from anthropic import Anthropic

MODEL = "claude-sonnet-4-6"
INPUT_PRICE_PER_M = 3.0
OUTPUT_PRICE_PER_M = 15.0


JUDGE_PROMPT = """You are grading a candidate answer against ground truth on a 4-state rubric:

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


def _cost_usd(in_tok: int, out_tok: int) -> float:
    return (in_tok / 1e6) * INPUT_PRICE_PER_M + (out_tok / 1e6) * OUTPUT_PRICE_PER_M


def _judge(client: Anthropic, task: dict, candidate: str) -> tuple[str, str, int, int]:
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
            "total_cost_usd": 0.0,
            "wall_seconds_total": 0.0, "_n": 0,
        })
        agg["total_cost_usd"] += tr.get("cost_usd") or 0.0
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
                agg[f"distractor_{score}"] = agg.get(f"distractor_{score}", 0) + 1
    for ap, agg in by_approach.items():
        n_ans = max(agg["n_answerable"], 1)
        n_dis = max(agg["n_distractor"], 1)
        n = max(agg.pop("_n", 1), 1)
        agg["answerable_accuracy"] = agg.get("answerable_correct", 0) / n_ans
        agg["answerable_hallucination_rate"] = agg.get("answerable_wrong", 0) / n_ans
        agg["distractor_correct_refusal_rate"] = agg.get("distractor_refused", 0) / n_dis
        agg["distractor_hallucination_rate"] = agg.get("distractor_hallucinated", 0) / n_dis
        agg["mean_wall_seconds"] = agg["wall_seconds_total"] / n
        agg["mean_cost_usd"] = agg["total_cost_usd"] / n
    return by_approach


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--tasks-file", required=True)
    p.add_argument("--budget-usd", type=float, default=2.0)
    args = p.parse_args(argv)

    client = Anthropic(api_key=os.environ.get("CLAUDE_API") or os.environ["ANTHROPIC_API_KEY"])
    tasks = {json.loads(line)["id"]: json.loads(line)
             for line in Path(args.tasks_file).read_text(encoding="utf-8").splitlines() if line.strip()}
    data = json.load(open(args.input, encoding="utf-8"))
    trials = data["trials"]
    print(f"[judge] judging {len(trials)} trials", flush=True)

    spent = 0.0
    for i, tr in enumerate(trials, 1):
        if tr.get("judge_score"):
            continue
        if not tr.get("answer") or tr.get("error"):
            tr["judge_score"] = "wrong"
            tr["judge_explanation"] = "error or empty answer"
            continue
        task = tasks.get(tr["task_id"])
        if not task:
            continue
        score, reason, ti, to = _judge(client, task, tr["answer"])
        tr["judge_score"] = score
        tr["judge_explanation"] = reason
        spent += _cost_usd(ti, to)
        if i % 25 == 0:
            print(f"[judge] {i}/{len(trials)} done — spent=${spent:.4f}", flush=True)
        if spent >= args.budget_usd:
            print(f"[judge] BUDGET CAP ${args.budget_usd:.2f} at trial {i}", flush=True)
            break

    by_approach = _aggregate(trials)
    payload = {
        **data,
        "by_approach": by_approach,
        "trials": trials,
        "judge_spend_usd": spent,
    }
    payload.pop("partial", None)
    Path(args.input).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[judge] wrote {args.input}", flush=True)
    print(f"[judge] judge spend: ${spent:.4f}", flush=True)
    print(f"\n[judge] per-approach (strict rubric):", flush=True)
    for ap, agg in sorted(by_approach.items()):
        print(
            f"  {ap:24s}  ans_acc={agg['answerable_accuracy']:.1%}  "
            f"halluc={agg['answerable_hallucination_rate']:.1%}  "
            f"dis_refuse={agg['distractor_correct_refusal_rate']:.1%}  "
            f"dis_halluc={agg['distractor_hallucination_rate']:.1%}  "
            f"wall={agg['mean_wall_seconds']:.1f}s/q  "
            f"$/q={agg['mean_cost_usd']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
