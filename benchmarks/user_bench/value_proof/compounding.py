"""Compounding curve — does answer quality rise as earned state accumulates?

The North Star: "measurably better at session N+1 than N." We re-seed the scratch
store with a GROWING set of earned lessons (simulated sessions) and measure the
memory-assembled lift on a FIXED held-out set at each checkpoint. The held-out set
and the lesson order are fixed in advance, so the curve rises only if accumulated
relevant state compounds into better answers FASTER than recall-noise degrades it
(more lessons = more chance recall surfaces the wrong one; the relevance gate must
overcome that). It is therefore not trivially monotonic.

HONEST LIMIT: *which* lessons get earned is synthetic here (we seed them in a fixed
order). This proxies the accumulation/coverage half of the moat; the *learning* half
— that the live loop earns the RIGHT lessons on its own — needs the capture→distil
loop and is not tested here.

Usage:
    PYTHONUTF8=1 python -m benchmarks.user_bench.value_proof.compounding --budget-usd 2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

_HERE = Path(__file__).resolve()
_SRC = _HERE.parent.parent.parent.parent / "src"
if (_SRC / "resonance_lattice" / "__init__.py").exists():
    sys.path.insert(0, str(_SRC))

from resonance_lattice._pricing import cost_usd as _cost_usd
from resonance_lattice.assembler import assemble

from .run import _ANSWER_SYSTEM, MemorySource, _ask, _client, _judge
from .seed_memory import seed

COMPOUND_USER = "value_proof_compound"


def _lesson_ids() -> list[str]:
    f = _HERE.parent / "lessons.jsonl"
    return [json.loads(l)["id"] for l in f.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _coverage(scns: list[dict], available: set[str]) -> float:
    """Fraction of held-out scenarios whose relevant lesson is available."""
    lb = [s for s in scns if s["kind"] == "load_bearing" and s.get("relevant_lessons")]
    if not lb:
        return 0.0
    covered = sum(1 for s in lb
                  if set(s["relevant_lessons"]) & available)
    return covered / len(lb)


def _score_lane(client, scn, system, repeats):
    ans, ai, ao = _ask(client, system, scn["question"])
    comps, accs, jin, jout = [], [], 0, 0
    for _ in range(repeats):
        sc, ji, jo = _judge(client, scn, ans)
        comps.append(sc.composite); accs.append(sc.accuracy)
        jin += ji; jout += jo
    return (sum(comps) / len(comps), sum(accs) / len(accs),
            _cost_usd(ai + jin, ao + jout))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios", default=str(_HERE.parent / "scenarios.jsonl"))
    p.add_argument("--checkpoints", default="8,16,24,32",
                   help="cumulative lesson counts")
    p.add_argument("--judge-repeats", type=int, default=1)
    p.add_argument("--budget-usd", type=float, default=2.5)
    p.add_argument("--output",
                   default="benchmarks/results/user_bench/value_proof_compounding.json")
    args = p.parse_args(argv)

    scns = [json.loads(l) for l in
            Path(args.scenarios).read_text(encoding="utf-8").splitlines() if l.strip()]
    held = [s for s in scns if s["kind"] == "load_bearing"]
    all_ids = _lesson_ids()
    checkpoints = [int(x) for x in args.checkpoints.split(",")]
    client = _client()
    spent = 0.0

    # cold baseline (memory-independent) — measured once
    print(f"[compounding] cold baseline over {len(held)} held-out scenarios", flush=True)
    cold = {}
    for scn in held:
        comp, acc, c = _score_lane(client, scn, _ANSWER_SYSTEM, args.judge_repeats)
        cold[scn["id"]] = comp
        spent += c
    cold_mean = sum(cold.values()) / len(cold)
    print(f"[compounding] cold mean composite = {cold_mean:.3f}  (${spent:.3f})", flush=True)

    curve = []
    for k in checkpoints:
        ids = all_ids[:k]
        seed(reset=True, user_id=COMPOUND_USER, lesson_ids=ids)
        mem = MemorySource(user_id=COMPOUND_USER)
        cov = _coverage(held, set(ids))
        comps, lifts = [], []
        for scn in held:
            if spent >= args.budget_usd:
                print(f"[compounding] BUDGET CAP ${args.budget_usd}", flush=True)
                break
            hits = mem.recall_hits(scn["question"])
            ctx = assemble(scn["question"], memory_recall=lambda _q: hits,
                           enable=("memory",))
            system = _ANSWER_SYSTEM + ("\n\n" + ctx.text if ctx.text else "")
            comp, acc, c = _score_lane(client, scn, system, args.judge_repeats)
            spent += c
            comps.append(comp)
            lifts.append(comp - cold[scn["id"]])
        mean_comp = sum(comps) / len(comps) if comps else 0.0
        mean_lift = sum(lifts) / len(lifts) if lifts else 0.0
        curve.append({"n_lessons": k, "coverage": round(cov, 3),
                      "memory_composite": round(mean_comp, 4),
                      "lift_vs_cold": round(mean_lift, 4)})
        print(f"[compounding] N={k:2d} lessons  coverage={cov:.2f}  "
              f"memory={mean_comp:.3f}  lift={mean_lift:+.3f}  (${spent:.3f})",
              flush=True)

    payload = {"config": {"held_out_n": len(held), "checkpoints": checkpoints,
                          "judge_repeats": args.judge_repeats},
               "cold_mean": round(cold_mean, 4), "curve": curve,
               "total_cost_usd": spent}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[compounding] wrote {args.output}  (${spent:.3f})", flush=True)
    print("[compounding] curve (lift vs cold should RISE with accumulated lessons):",
          flush=True)
    for pt in curve:
        bar = "#" * int(max(pt["lift_vs_cold"], 0) * 40)
        print(f"  N={pt['n_lessons']:2d}  cov={pt['coverage']:.2f}  "
              f"lift={pt['lift_vs_cold']:+.3f}  {bar}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
