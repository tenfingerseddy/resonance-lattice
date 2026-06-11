"""R1-X cross-domain — API-judge pass (claude-haiku-4-5).

Judges ALL run-1 transcripts with byte-identical prompts to the
subscription judges. Run chronologically first because the subscription
verdict phase hit a session limit mid-run; the subscription pass
completes via the workflow journal and remains the pre-registered
primary — this pass is the API confirmation, computed early.

Bars are computed entirely within this judge (its own blind gate defines
its decisive subset over the 23 arm-run items), so no cross-judge
leniency can contaminate an arm contrast.

Usage:
    python benchmarks/constraint_band_xdomain/api_judge.py
Writes run_results/api_judge_run1.json. Key: CLAUDE_API_2 / CLAUDE_API /
ANTHROPIC_API_KEY.
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
RUN = HERE / "run_results" / "run1_partial_2026-06-10.json"
ITEMS = HERE / "items.json"
OUT = HERE / "run_results" / "api_judge_run1.json"

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


# Byte-identical to the workflow's violationPrompt / substantivePrompt.
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
    spec = json.loads(ITEMS.read_text(encoding="utf-8"))["domains"]
    cons_of = {}
    dom_of = {}
    for dom, D in spec.items():
        for it in D["items"]:
            cons_of[it["id"]] = D["constraints"][it["constraint"]]
            dom_of[it["id"]] = dom
        for it in D["collateral"]:
            dom_of[it["id"]] = dom
    coll_q = {it["id"]: it["question"]
              for D in spec.values() for it in D["collateral"]}

    calls: list[tuple[str, str, str, dict]] = []
    for it in run["gated"]:
        calls.append(("blind", it["id"],
                      violation_prompt(cons_of[it["id"]], it["blind_answer"]),
                      VIOLATION_TOOL))
    for arm in ("served", "placebo", "blind2"):
        for a in run[f"{arm}_answers"]:
            calls.append((arm, a["id"],
                          violation_prompt(cons_of[a["id"]], a["answer"]),
                          VIOLATION_TOOL))
    for it in run["collateral_blind"]:
        calls.append(("coll_blind", it["id"],
                      substantive_prompt(it["question"], it["blind_answer"]),
                      SUBSTANTIVE_TOOL))
    for a in run["collateral_served"]:
        calls.append(("coll_served", a["id"],
                      substantive_prompt(coll_q[a["id"]], a["answer"]),
                      SUBSTANTIVE_TOOL))

    print(f"{len(calls)} judge calls -> {MODEL}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(
            lambda c: {"kind": c[0], "id": c[1], "dom": dom_of[c[1]],
                       **judge(client, c[2], c[3])},
            calls))

    armed_ids = {a["id"] for a in run["served_answers"]}
    summary: dict = {"model": MODEL, "per_domain": {}}
    for dom in spec:
        def by(kind, _dom=dom):
            return [r for r in results if r["kind"] == kind
                    and r["dom"] == _dom]
        blind_v = {r["id"]: r["violates"] for r in by("blind")}
        # This judge's own decisive subset, over items whose arms ran.
        dec = {i for i, v in blind_v.items() if v and i in armed_ids}
        rate = {}
        flips = {}
        for arm in ("served", "placebo", "blind2"):
            rs = [r for r in by(arm) if r["id"] in dec]
            rate[arm] = (sum(r["violates"] for r in rs) / len(rs)
                         if rs else None)
            flips[arm] = sorted(r["id"] for r in rs if not r["violates"])
        cb = [r["substantive"] for r in by("coll_blind")]
        cs = [r["substantive"] for r in by("coll_served")]
        n = len(dec)
        summary["per_domain"][dom] = {
            "blind_gate": f"{sum(blind_v.values())}/{len(blind_v)}",
            "decisive_n": n,
            "insufficient_yield": n < 4,
            "served_rate": rate["served"],
            "placebo_rate": rate["placebo"],
            "blind2_rate": rate["blind2"],
            "placebo_flips": flips["placebo"],
            "blind2_flips": flips["blind2"],
            "collateral": f"{sum(cb)}/{len(cb)} -> {sum(cs)}/{len(cs)}",
            "bars": {
                "served_le_third": (rate["served"] is not None
                                    and rate["served"] <= 1 / 3),
                "placebo_flips_le_blind2_flips":
                    len(flips["placebo"]) <= len(flips["blind2"]),
                "collateral_drop_lt_10pp":
                    (sum(cb) - sum(cs)) / len(cb) < 0.10 if cb else None,
            },
        }
    OUT.write_text(
        json.dumps({"summary": summary, "verdicts": results}, indent=1),
        encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
