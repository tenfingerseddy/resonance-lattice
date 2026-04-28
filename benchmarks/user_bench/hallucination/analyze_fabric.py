"""Failure analysis for the bench 2 v3 Fabric run.

For every question, surface:
- Ground-truth source file
- Per-approach answer + judge_score + judge reason
- Disagreement classification (judge variance, retrieval miss, hallucination,
  refusal, etc.)

Run: python -m benchmarks.user_bench.hallucination.analyze_fabric
Output goes to docs/internal/benchmarks/02_fabric_failure_analysis.md
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).parent
_RESULT = _HERE.parent.parent / "results" / "user_bench" / "hallucination_fabric_v3.json"
_TASKS = _HERE / "fabric_tasks.jsonl"
_OUT = _HERE.parent.parent.parent / "docs" / "internal" / "benchmarks" / "02_fabric_failure_analysis.md"


def _load_tasks() -> dict[str, dict]:
    return {
        json.loads(line)["id"]: json.loads(line)
        for line in _TASKS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _classify(task: dict, approach: str, trial: dict) -> tuple[str, str]:
    """Return (category, brief_reason)."""
    score = trial.get("judge_score", "?")
    answer = (trial.get("answer") or "").lower()
    if task["kind"] == "distractor":
        if score == "refused":
            return "ok_refused", "correctly refused"
        if score in ("correct", "partial"):
            return "ok_partial_refusal", "partially-refused with hedging"
        return "DISTRACTOR_HALLUCINATION", "answered fake-feature question"
    # answerable
    if score == "correct":
        return "ok", ""
    if score == "partial":
        if "doesn't include" in answer or "doesn't cover" in answer or "no information" in answer:
            return "PARTIAL_OVERREFUSAL", "judge graded partial after constrain refused"
        return "JUDGE_PARTIAL", "judge says partial — answer may be near-correct"
    if score == "refused":
        if approach == "constrain_verified":
            return "CONSTRAIN_OVER_REFUSE", "constrain refused on weak retrieval"
        if approach == "no_retrieval":
            return "ok_no_knowledge", "LLM correctly refused (no corpus, no knowledge)"
        return "AUGMENT_OVER_REFUSE", "non-constrain mode refused"
    if score == "wrong":
        return "HALLUCINATION", "wrong answer — invention or stale training"
    return "OTHER", f"score={score}"


def _build_trial_index(trials: list[dict]) -> dict[tuple[str, str], dict]:
    return {(t["task_id"], t["approach"]): t for t in trials}


def main() -> None:
    data = json.load(open(_RESULT, encoding="utf-8"))
    tasks = _load_tasks()
    trials = data["trials"]
    by_pair = _build_trial_index(trials)

    approaches = ["constrain_verified", "augment", "knowledge", "no_retrieval"]

    # Build per-task-per-approach summary
    rows: list[dict] = []
    for tid, task in sorted(tasks.items()):
        row = {
            "id": tid,
            "kind": task["kind"],
            "tier": task.get("tier", "?"),
            "topic": task.get("topic", "?"),
            "question": task["question"],
            "ground_truth": task["ground_truth"],
            "source_file": task.get("source_file", "?"),
            "source_date": task.get("source_date", "?"),
        }
        for app in approaches:
            t = by_pair.get((tid, app))
            if t is None:
                row[app] = {"score": "MISSING", "category": "MISSING"}
                continue
            cat, reason = _classify(task, app, t)
            row[app] = {
                "score": t.get("judge_score"),
                "answer": (t.get("answer") or "").strip(),
                "judge": t.get("judge_explanation") or "",
                "category": cat,
                "category_reason": reason,
            }
        rows.append(row)

    # Per-approach failure category histogram (for answerable only)
    print("=== Failure category distribution (answerable questions) ===\n")
    for app in approaches:
        cats = Counter(
            r[app]["category"]
            for r in rows
            if r["kind"] == "answerable" and isinstance(r[app], dict)
        )
        print(f"  {app}:")
        for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"    {cat:35s}  {n}")
        print()

    # Identify questions where rlat-constrain failed but augment got it right (or vice versa)
    constrain_wins = []
    augment_wins = []
    both_failed = []
    both_correct = []
    for r in rows:
        if r["kind"] != "answerable":
            continue
        c_score = r["constrain_verified"]["score"]
        a_score = r["augment"]["score"]
        if c_score == "correct" and a_score != "correct":
            constrain_wins.append(r)
        elif a_score == "correct" and c_score != "correct":
            augment_wins.append(r)
        elif c_score == "correct" and a_score == "correct":
            both_correct.append(r)
        else:
            both_failed.append(r)
    print(f"=== rlat constrain vs augment on answerable ({sum(1 for r in rows if r['kind']=='answerable')} questions) ===")
    print(f"  Both correct:    {len(both_correct)}")
    print(f"  Both failed:     {len(both_failed)}")
    print(f"  Augment-only correct (constrain partial/refused/wrong): {len(augment_wins)}")
    print(f"  Constrain-only correct (augment partial/refused/wrong): {len(constrain_wins)}")
    print()

    # Write the full report
    out_lines: list[str] = []
    out_lines.append("# Bench 2 v3 — Fabric corpus failure analysis\n")
    out_lines.append("Generated by `benchmarks/user_bench/hallucination/analyze_fabric.py`. ")
    out_lines.append("Per-question per-approach trial inspection. The aim is to surface ")
    out_lines.append("the failure modes that drive sub-100% accuracy and pick the highest-")
    out_lines.append("impact fixes for v2.0.1.\n\n")

    out_lines.append("## Aggregate (recap)\n\n")
    agg = data["by_approach"]
    out_lines.append("| Approach | ans_acc | ans_halluc | ans_partial | ans_refused | dis_refuse | dis_halluc |\n")
    out_lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for app in approaches:
        m = agg[app]
        n_ans = m["n_answerable"]
        n_dis = m["n_distractor"]
        partial_pct = m.get("answerable_partial", 0) / n_ans * 100
        refused_pct = m.get("answerable_refused", 0) / n_ans * 100
        out_lines.append(
            f"| {app} | {m['answerable_accuracy']*100:.1f}% | "
            f"{m['answerable_hallucination_rate']*100:.1f}% | "
            f"{partial_pct:.1f}% | {refused_pct:.1f}% | "
            f"{m['distractor_correct_refusal_rate']*100:.1f}% | "
            f"{m['distractor_hallucination_rate']*100:.1f}% |\n"
        )
    out_lines.append("\n")

    out_lines.append("## Failure category distribution (answerable, 51 questions)\n\n")
    out_lines.append("| Category | constrain | augment | knowledge | no_retrieval |\n")
    out_lines.append("|---|---:|---:|---:|---:|\n")
    all_cats = set()
    cat_by_app = {}
    for app in approaches:
        cats = Counter(
            r[app]["category"]
            for r in rows
            if r["kind"] == "answerable" and isinstance(r[app], dict)
        )
        cat_by_app[app] = cats
        all_cats.update(cats.keys())
    for cat in sorted(all_cats):
        out_lines.append(
            f"| {cat} | {cat_by_app['constrain_verified'].get(cat, 0)} | "
            f"{cat_by_app['augment'].get(cat, 0)} | {cat_by_app['knowledge'].get(cat, 0)} | "
            f"{cat_by_app['no_retrieval'].get(cat, 0)} |\n"
        )
    out_lines.append("\n")

    out_lines.append("## constrain vs augment crosstab (answerable)\n\n")
    out_lines.append(f"- **Both correct**: {len(both_correct)} ({len(both_correct)*100/sum(1 for r in rows if r['kind']=='answerable'):.0f}%)\n")
    out_lines.append(f"- **Both failed (partial/wrong/refused for both)**: {len(both_failed)}\n")
    out_lines.append(f"- **Augment-only correct**: {len(augment_wins)} questions where augment got correct but constrain didn't\n")
    out_lines.append(f"- **Constrain-only correct**: {len(constrain_wins)} questions where constrain got correct but augment didn't\n\n")

    # Write the every-question table
    out_lines.append("## Per-question detail (answerable)\n\n")
    out_lines.append("Each row is one question. Score legend: `c` = correct, `p` = partial, ")
    out_lines.append("`w` = wrong (hallucination), `r` = refused.\n\n")
    out_lines.append("| id | tier | constrain | augment | knowledge | no-retr | source |\n")
    out_lines.append("|---|---|:-:|:-:|:-:|:-:|---|\n")
    score_letter = {"correct": "c", "partial": "p", "wrong": "**w**", "refused": "r"}
    for r in rows:
        if r["kind"] != "answerable":
            continue
        cells = [r["id"], r["tier"]]
        for app in approaches:
            s = r[app]["score"]
            cells.append(score_letter.get(s, s or "-"))
        cells.append(f"`{Path(r['source_file']).name}`")
        out_lines.append(f"| {' | '.join(cells)} |\n")
    out_lines.append("\n")

    # Distractor table
    out_lines.append("## Per-question detail (distractor)\n\n")
    out_lines.append("Distractors are questions about things that don't exist; correct response = refused.\n\n")
    out_lines.append("| id | constrain | augment | knowledge | no-retr | trap |\n")
    out_lines.append("|---|:-:|:-:|:-:|:-:|---|\n")
    for r in rows:
        if r["kind"] != "distractor":
            continue
        cells = [r["id"]]
        for app in approaches:
            s = r[app]["score"]
            cells.append(score_letter.get(s, s or "-"))
        trap = (r.get("ground_truth", "")[:80] + "...") if len(r.get("ground_truth", "")) > 80 else r.get("ground_truth", "")
        cells.append(trap.replace("|", " "))
        out_lines.append(f"| {' | '.join(cells)} |\n")
    out_lines.append("\n")

    # Detailed failure dives — every answerable question where constrain failed
    out_lines.append("## Constrain failure deep-dive (every non-correct answerable)\n\n")
    out_lines.append("Each block shows what the question asked, the ground-truth fact, what ")
    out_lines.append("rlat-constrain answered, what the LLM-only answered, and the judge's ")
    out_lines.append("reasoning. The category column at the top of each block summarises the ")
    out_lines.append("failure mode.\n\n")
    fail_count = 0
    for r in rows:
        if r["kind"] != "answerable":
            continue
        c = r["constrain_verified"]
        if c["score"] == "correct":
            continue
        fail_count += 1
        out_lines.append(f"### {r['id']} — {r['topic']} ({r['tier']}) — {c['category']}\n\n")
        out_lines.append(f"**Question**: {r['question']}\n\n")
        out_lines.append(f"**Ground truth** (`{r['source_file']}`, {r['source_date']}):\n> {r['ground_truth']}\n\n")
        out_lines.append(f"**rlat constrain** [`{c['score']}`]:\n```\n{c['answer'][:1200]}\n```\n")
        out_lines.append(f"_judge_: {c['judge'][:600]}\n\n")
        a = r["augment"]
        out_lines.append(f"**rlat augment** [`{a['score']}`]:\n```\n{a['answer'][:800]}\n```\n")
        out_lines.append(f"_judge_: {a['judge'][:400]}\n\n")
        n = r["no_retrieval"]
        out_lines.append(f"**LLM-only** [`{n['score']}`]:\n```\n{n['answer'][:800]}\n```\n")
        out_lines.append(f"_judge_: {n['judge'][:400]}\n\n")
        out_lines.append("---\n\n")

    out_lines.append(f"\n**Total non-correct constrain answers analyzed: {fail_count} of 51 answerable**\n")

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text("".join(out_lines), encoding="utf-8")
    print(f"\nWrote: {_OUT}")
    print(f"  {len(out_lines)} lines")


if __name__ == "__main__":
    main()
