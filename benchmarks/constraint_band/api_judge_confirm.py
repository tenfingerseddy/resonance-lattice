"""R1 Constraint Band — API-judge confirmation (calibration step).

Re-scores the run-1 transcripts with an independent API judge
(claude-haiku-4-5) using the EXACT prompt text the subscription judges
saw (r1-constraint-band workflow, run wf_fcf0f69e). No new answers are
generated — same transcripts, same pre-registered bars, different judge.

Judge-parity caveat (stated, not hidden): subscription judges could
search the web for GA/preview status; the API judge works from the
answer text plus training knowledge only. Disagreements on C1 (preview)
items may reflect world-knowledge drift, not transcript content.

Usage:
    python benchmarks/constraint_band/api_judge_confirm.py
Writes run_results/api_confirm_run1.json. Key: CLAUDE_API_2 / CLAUDE_API
/ ANTHROPIC_API_KEY (same chain as the product).
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from resonance_lattice._anthropic import api_key_or_error  # noqa: E402

MODEL = "claude-haiku-4-5"
RUN = HERE / "run_results" / "run1_2026-06-10.json"
ITEMS = HERE / "items.json"
OUT = HERE / "run_results" / "api_confirm_run1.json"

VIOLATION_TOOL = {
    "name": "verdict",
    "description": "Binary violation verdict with quoted-span evidence.",
    "input_schema": {
        "type": "object",
        "properties": {
            "violates": {"type": "boolean"},
            "evidence": {
                "type": "string",
                "description": "short quoted span from the answer, or why not",
            },
        },
        "required": ["violates", "evidence"],
    },
}

SUBSTANTIVE_TOOL = {
    "name": "verdict",
    "description": "Binary substantive-answer verdict.",
    "input_schema": {
        "type": "object",
        "properties": {
            "substantive": {"type": "boolean"},
            "note": {"type": "string"},
        },
        "required": ["substantive", "note"],
    },
}


# Prompt text below is byte-identical to the subscription run's
# violationPrompt / substantivePrompt — the judge changes, nothing else.
def violation_prompt(constraint: str, answer: str) -> str:
    return (
        "You are a strict compliance checker. CONSTRAINT (a hard rule for "
        f"this environment):\n{constraint}\n\nANSWER under review:\n{answer}"
        "\n\nDoes the answer RECOMMEND, instruct, or rely on anything that "
        "violates the constraint? Mere MENTIONS that correctly steer away "
        '(e.g. "avoid X since it\'s preview") are NOT violations. '
        "Recommending a violating option among alternatives IS a violation. "
        "Return violates true/false with a short quoted span as evidence."
    )


def substantive_prompt(question: str, answer: str) -> str:
    return (
        f"QUESTION: {question}\n\nANSWER: {answer}\n\nDoes the answer "
        "SUBSTANTIVELY address the question — concrete, usable guidance — "
        "as opposed to refusing, hedging into nothing, or derailing onto "
        "unrelated compliance talk? Return substantive true/false."
    )


def judge(client, prompt: str, tool: dict) -> dict:
    last_err: Exception | None = None
    for _ in range(3):
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=300,
                temperature=0.0,
                tools=[tool],
                tool_choice={"type": "tool", "name": "verdict"},
                messages=[{"role": "user", "content": prompt}],
            )
            for block in msg.content:
                if block.type == "tool_use":
                    return dict(block.input)
            raise RuntimeError("no tool_use block in response")
        except Exception as e:  # noqa: BLE001 — retry then surface
            last_err = e
            time.sleep(2)
    raise RuntimeError(f"judge failed after 3 attempts: {last_err}")


def main() -> int:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key_or_error())
    run = json.loads(RUN.read_text(encoding="utf-8"))
    items = json.loads(ITEMS.read_text(encoding="utf-8"))
    cons = items["constraints"]
    constraint_of = {it["id"]: it["constraint"] for it in items["items"]}
    question_of = {it["id"]: it["question"] for it in items["collateral"]}

    calls: list[tuple[str, str, str, dict]] = []  # (kind, id, prompt, tool)
    for it in run["gated"]:
        calls.append(
            ("blind", it["id"],
             violation_prompt(cons[it["constraint"]], it["blind_answer"]),
             VIOLATION_TOOL))
    for arm in ("served", "placebo"):
        for a in run[f"{arm}_answers"]:
            calls.append(
                (arm, a["id"],
                 violation_prompt(cons[constraint_of[a["id"]]], a["answer"]),
                 VIOLATION_TOOL))
    for it in run["collateral_blind"]:
        calls.append(
            ("coll_blind", it["id"],
             substantive_prompt(it["question"], it["blind_answer"]),
             SUBSTANTIVE_TOOL))
    for a in run["collateral_served"]:
        calls.append(
            ("coll_served", a["id"],
             substantive_prompt(question_of[a["id"]], a["answer"]),
             SUBSTANTIVE_TOOL))

    print(f"{len(calls)} judge calls -> {MODEL}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda c: {"kind": c[0], "id": c[1], **judge(client, c[2], c[3])},
            calls))

    by = lambda kind: [r for r in results if r["kind"] == kind]  # noqa: E731
    decisive = set(run["decisive_ids"])  # pre-registered subset, unchanged
    blind = by("blind")
    blind_v = {r["id"]: r["violates"] for r in blind}
    served_v = [r["violates"] for r in by("served") if r["id"] in decisive]
    placebo_v = [r["violates"] for r in by("placebo") if r["id"] in decisive]
    coll_b = [r["substantive"] for r in by("coll_blind")]
    coll_s = [r["substantive"] for r in by("coll_served")]

    # Agreement with the subscription judges, per arm.
    sub_blind = {it["id"]: it["blind_violates"] for it in run["gated"]}
    sub_arm = {(v["kind"], v["id"]): v.get("violates", v.get("substantive"))
               for v in run["verdicts"]}
    agree_blind = sum(blind_v[i] == sub_blind[i] for i in blind_v)
    agree_arms = {}
    for kind, sub_kind in (("served", "served"), ("placebo", "placebo"),
                           ("coll_blind", "coll_blind"),
                           ("coll_served", "coll_served")):
        rs = by(kind)
        hits = sum(
            r.get("violates", r.get("substantive"))
            == sub_arm.get((sub_kind, r["id"]))
            for r in rs)
        agree_arms[kind] = f"{hits}/{len(rs)}"

    n_dec = len(decisive)
    # Bars are relative to the BLIND rate (pre-registration). Under the
    # subscription judge blind=100% on the decisive subset by construction;
    # the API judge gets its own blind reference on the same subset.
    blind_dec_rate = sum(blind_v[i] for i in decisive) / n_dec
    served_rate = sum(served_v) / len(served_v)
    placebo_rate = sum(placebo_v) / len(placebo_v)
    summary = {
        "model": MODEL,
        "decisive_subset": "pre-registered run-1 subset (unchanged)",
        "blind_full": f"{sum(blind_v.values())}/{len(blind_v)}",
        "blind_decisive_confirmed":
            f"{sum(blind_v[i] for i in decisive)}/{n_dec}",
        "served_rate": served_rate,
        "placebo_rate": placebo_rate,
        "collateral_blind": sum(coll_b) / len(coll_b),
        "collateral_served": sum(coll_s) / len(coll_s),
        "bars": {
            "served_le_third_of_blind": served_rate <= blind_dec_rate / 3,
            "placebo_within_10pp_of_blind":
                abs(placebo_rate - blind_dec_rate) <= 0.10,
            "collateral_drop_lt_10pp":
                (sum(coll_b) - sum(coll_s)) / len(coll_b) < 0.10,
        },
        "agreement_with_subscription": {
            "blind": f"{agree_blind}/{len(blind_v)}", **agree_arms},
    }
    OUT.write_text(
        json.dumps({"summary": summary, "verdicts": results}, indent=1),
        encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
