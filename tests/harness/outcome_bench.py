"""outcome_bench — the STEP-0 decompose/arm logic of the outcome harness, no cloud.

Validates the novel parts (per-fact decompose: added vs flipped-wrong->right vs lost;
coverage; the recipe plug; one full run() row) with a stubbed LLM seam, so the logic is
proven before any credits are spent.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))


class _StubCli:
    """Returns preset answers + fact-judge verdicts in call order."""

    def __init__(self, answers, verdicts):
        self._answers, self._verdicts = list(answers), list(verdicts)

    def text(self, system, user, *, max_tokens=700):
        return self._answers.pop(0) if self._answers else "ans"

    def json(self, system, user, *, max_tokens=400):
        return {"verdicts": self._verdicts.pop(0)}


def run() -> int:
    import outcome_bench as ob

    f = 0

    # (a) coverage = present_correct fraction
    if abs(ob.coverage(["present_correct", "absent", "present_wrong"]) - 1 / 3) > 1e-9:
        print("[outcome_bench] FAIL a: coverage", file=sys.stderr); f += 1
    else:
        print("[outcome_bench] (a) coverage OK", file=sys.stderr)

    # (b) decompose: fact0 wrong->right (added+flipped), fact1 absent->right (added),
    #     fact2 right->absent (lost)
    d = ob.decompose(["present_wrong", "absent", "present_correct"],
                     ["present_correct", "present_correct", "absent"])
    if d != {"added": 2, "flipped_wrong_to_right": 1, "lost": 1}:
        print(f"[outcome_bench] FAIL b: decompose {d}", file=sys.stderr); f += 1
    else:
        print("[outcome_bench] (b) decompose added/flipped/lost OK", file=sys.stderr)

    # (c) recipes
    if (ob.recipe_bridge(None, {"crosscut_fact": "X joins Y"}) != ["X joins Y"]
            or ob.recipe_none(None, {}) != []
            or ob.recipe_pointer(None, {"relevant_docs": ["a.md", "b.md"]}) == []):
        print("[outcome_bench] FAIL c: recipes", file=sys.stderr); f += 1
    else:
        print("[outcome_bench] (c) recipes (bridge/none/pointer) OK", file=sys.stderr)

    # (d) a full run() row: arm2 covers 1/2, arm3 (with bridge claim) covers 2/2,
    #     fact0 flipped wrong->right.
    q = {"id": "t1", "question": "?", "key_facts": ["f0", "f1"],
         "crosscut_fact": "the join", "_context": ["ctx"]}
    stub = _StubCli(answers=["a2", "a3"],
                    verdicts=[["present_wrong", "present_correct"],
                              ["present_correct", "present_correct"]])
    rows = ob.run([q], stub, ob.recipe_bridge, perfect_retrieval=False)
    r = rows[0]
    ok = (r["n_claims"] == 1 and abs(r["arm2"] - 0.5) < 1e-9 and abs(r["arm3"] - 1.0) < 1e-9
          and abs(r["delta"] - 0.5) < 1e-9 and r["added"] == 1 and r["flipped_wrong_to_right"] == 1)
    if not ok:
        print(f"[outcome_bench] FAIL d: run row {r}", file=sys.stderr); f += 1
    else:
        print("[outcome_bench] (d) run() row: arm2=0.5 arm3=1.0 delta=+0.5 flipped=1 OK",
              file=sys.stderr)

    # (e) summarize: CI present, win counted
    s = ob.summarize(rows)
    if s["n"] != 1 or s["wins"] != 1 or "ci95" not in s or s["total_flipped_wrong_to_right"] != 1:
        print(f"[outcome_bench] FAIL e: summarize {s}", file=sys.stderr); f += 1
    else:
        print("[outcome_bench] (e) summarize (CI + flipped channel) OK", file=sys.stderr)

    # (f) placebo arm: 2 userfact questions; q0 gets its OWN fact (lifts to 1.0), the placebo
    #     serves q1's fact (no lift, stays 0.5). specificity_gap must be > 0.
    q0 = {"id": "p0", "question": "?", "key_facts": ["f0", "f1"],
          "user_fact": "right", "_context": ["ctx"]}
    q1 = {"id": "p1", "question": "?", "key_facts": ["g0", "g1"],
          "user_fact": "wrong", "_context": ["ctx"]}
    # call order per q: text(a2), text(a3), json(v2), json(v3), text(a3p), json(v3p)
    stub2 = _StubCli(
        answers=["a2", "a3", "a3p", "b2", "b3", "b3p"],
        verdicts=[["present_correct", "absent"], ["present_correct", "present_correct"],
                  ["present_correct", "absent"],            # q0 placebo: no gain
                  ["present_correct", "absent"], ["present_correct", "present_correct"],
                  ["present_correct", "absent"]])
    prows = ob.run([q0, q1], stub2, ob.recipe_userfact, perfect_retrieval=False, placebo=True)
    ps = ob.summarize(prows)
    ok_p = ("delta_placebo" in prows[0] and abs(prows[0]["delta_placebo"]) < 1e-9
            and abs(prows[0]["delta"] - 0.5) < 1e-9 and ps.get("specificity_gap", 0) > 0.49)
    if not ok_p:
        print(f"[outcome_bench] FAIL f: placebo {prows} / {ps.get('specificity_gap')}", file=sys.stderr); f += 1
    else:
        print("[outcome_bench] (f) placebo specificity (real +0.5 vs wrong-fact 0.0) OK", file=sys.stderr)

    if f:
        print(f"[outcome_bench] {f} check(s) failed", file=sys.stderr); return 1
    print("[outcome_bench] PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
