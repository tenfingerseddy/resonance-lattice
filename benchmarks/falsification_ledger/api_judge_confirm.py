"""R2 Falsification Ledger — API-judge confirmation (calibration step).

Re-scores the run-2 + run-2b transcripts with an independent API judge
(claude-haiku-4-5) using the EXACT prompt text the subscription judges
saw. No new answers are generated — same transcripts, same pre-registered
bars, different judge. Mirrors benchmarks/constraint_band/api_judge_confirm.py.

Usage:
    python benchmarks/falsification_ledger/api_judge_confirm.py
Writes run_results/api_confirm_run2.json. Key: CLAUDE_API_2 / CLAUDE_API
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
RUN2 = HERE / "run_results" / "run2_2026-06-10.json"
RUN2B = HERE / "run_results" / "run2b_2026-06-10.json"
ITEMS = HERE / "items_run2.json"
OUT = HERE / "run_results" / "api_confirm_run2.json"

RECOMMENDS_TOOL = {
    "name": "verdict",
    "description": "Binary recommends verdict with quoted-span evidence.",
    "input_schema": {
        "type": "object",
        "properties": {
            "recommends": {"type": "boolean"},
            "evidence": {
                "type": "string",
                "description": "short quoted span from the answer, or why not",
            },
        },
        "required": ["recommends", "evidence"],
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
# recommendsPrompt / substantivePrompt — the judge changes, nothing else.
def recommends_prompt(approach: str, answer: str) -> str:
    return (
        "You are a strict reviewer checking whether an answer re-recommends "
        f"an approach.\n\nAPPROACH UNDER TEST: {approach}\n\nANSWER under "
        f"review:\n{answer}\n\nDoes the answer RECOMMEND trying or adopting "
        "this approach — as its primary recommendation or an endorsed option "
        "— WITHOUT acknowledging that it was already tried and failed at "
        "this organisation? Steering away from the approach, or proposing "
        "it only while explicitly acknowledging the prior local failure, is "
        "NOT a recommendation. Return recommends true/false with a short "
        "quoted span as evidence."
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
    run2 = json.loads(RUN2.read_text(encoding="utf-8"))
    run2b = json.loads(RUN2B.read_text(encoding="utf-8"))
    items = json.loads(ITEMS.read_text(encoding="utf-8"))
    approach_of = {it["id"]: items["atoms"][it["atom"]]["approach"]
                   for it in items["items"]}
    question_of = {it["id"]: it["question"] for it in items["collateral"]}

    calls: list[tuple[str, str, str, dict]] = []  # (kind, id, prompt, tool)
    for it in run2["gated"]:
        calls.append(
            ("blind", it["id"],
             recommends_prompt(approach_of[it["id"]], it["blind_answer"]),
             RECOMMENDS_TOOL))
    for arm in ("ledger", "topical", "placebo"):
        for a in run2[f"{arm}_answers"]:
            calls.append(
                (arm, a["id"],
                 recommends_prompt(approach_of[a["id"]], a["answer"]),
                 RECOMMENDS_TOOL))
    for a in run2b["blind2_answers"]:
        calls.append(
            ("blind2", a["id"],
             recommends_prompt(approach_of[a["id"]], a["answer"]),
             RECOMMENDS_TOOL))
    for it in run2["collateral_blind"]:
        calls.append(
            ("coll_blind", it["id"],
             substantive_prompt(it["question"], it["blind_answer"]),
             SUBSTANTIVE_TOOL))
    for a in run2["collateral_served"]:
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
    decisive = set(run2["decisive_ids"])  # pre-registered subset, unchanged
    blind_v = {r["id"]: r["recommends"] for r in by("blind")}
    n_dec = len(decisive)
    blind_dec_rate = sum(blind_v[i] for i in decisive) / n_dec
    rates = {}
    for arm in ("ledger", "topical", "placebo", "blind2"):
        rs = [r["recommends"] for r in by(arm) if r["id"] in decisive]
        rates[arm] = sum(rs) / len(rs)
    coll_b = [r["substantive"] for r in by("coll_blind")]
    coll_s = [r["substantive"] for r in by("coll_served")]

    # Agreement with the subscription judges, per arm.
    sub_blind = {it["id"]: it["blind_recommends"] for it in run2["gated"]}
    sub_arm = {(v["kind"], v["id"]): v.get("recommends", v.get("substantive"))
               for v in run2["verdicts"]}
    for v in run2b["verdicts"]:
        sub_arm[("blind2", v["id"])] = v["recommends"]
    agreement = {
        "blind": f"{sum(blind_v[i] == sub_blind[i] for i in blind_v)}"
                 f"/{len(blind_v)}"}
    for kind in ("ledger", "topical", "placebo", "blind2",
                 "coll_blind", "coll_served"):
        rs = by(kind)
        hits = sum(
            r.get("recommends", r.get("substantive"))
            == sub_arm.get((kind, r["id"]))
            for r in rs)
        agreement[kind] = f"{hits}/{len(rs)}"

    placebo_flips = [r["id"] for r in by("placebo")
                     if r["id"] in decisive and not r["recommends"]]
    blind2_flips = [r["id"] for r in by("blind2")
                    if r["id"] in decisive and not r["recommends"]]
    summary = {
        "model": MODEL,
        "decisive_subset": "pre-registered run-2 subset (unchanged)",
        "blind_full": f"{sum(blind_v.values())}/{len(blind_v)}",
        "blind_decisive_confirmed":
            f"{sum(blind_v[i] for i in decisive)}/{n_dec}",
        "rates_on_decisive": rates,
        "collateral_blind": sum(coll_b) / len(coll_b),
        "collateral_served": sum(coll_s) / len(coll_s),
        "bars": {
            "ledger_le_third_of_blind":
                rates["ledger"] <= blind_dec_rate / 3,
            "active_ingredient_topical_minus_ledger_ge_25pp":
                rates["topical"] - rates["ledger"] >= 0.25,
            "placebo_flips_le_blind2_flips":
                len(placebo_flips) <= len(blind2_flips),
            "collateral_drop_lt_10pp":
                (sum(coll_b) - sum(coll_s)) / len(coll_b) < 0.10,
        },
        "placebo_flip_ids": placebo_flips,
        "blind2_flip_ids": blind2_flips,
        "agreement_with_subscription": agreement,
    }
    OUT.write_text(
        json.dumps({"summary": summary, "verdicts": results}, indent=1),
        encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
