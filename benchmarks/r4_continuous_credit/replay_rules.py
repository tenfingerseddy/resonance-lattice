"""R4 — offline suppression-rule replay over the recorded per-serve stream.

Replays the pre-registered R4 confidence-sequence rule and the three
prior rules (point, neverhelped, wilson2) over the instrumented run's
`nolearn` streams (unbiased observation — no suppression ever fired, so
every fact's full evidence sequence was recorded), and applies the
pre-registered bars from DESIGN.md. Zero LLM calls.

Replay semantics: events are processed in recorded order (round-major,
as logged). Each rule sees, per fact, the same evidence prefix it would
have seen live under nolearn serving. CUT decisions are evaluated at
end-of-round boundaries, mirroring the live loop's per-round suppression
sweep.

Usage:
    python benchmarks/r4_continuous_credit/replay_rules.py
Reads  benchmarks/r4_continuous_credit/run_results/instrumented_stream.json
Writes benchmarks/r4_continuous_credit/run_results/replay_run2_v2.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).parent
RUN = HERE / "run_results" / "instrumented_stream.json"   # the COMMITTED stream — reproducible from a fresh checkout
OUT = HERE / "run_results" / "replay_run2_v2.json"  # versioned — never overwrite the locked run-1 artifact

# --- pre-registered R4 rule constants (DESIGN.md; locked before the stream existed)
ALPHA = 0.05
CLIP = 1.0
CS_CONST = 1.7          # conservative stitched-bound constant, pre-chosen
CUT_DEADBAND = 0.05     # the corroboration tol — "confidently below the deadband"
MIN_SERVES = 3

# --- prior rules' locked parameters (bench_closed_loop_v2.py defaults / locked runs)
HELP_RATE = 0.34
TOL = 0.05
WILSON_Z = 1.28
PROTECT_FLOOR = 0.10
PRIOR_MIN_SERVES = 2


def _radius(n: int) -> float:
    return math.sqrt(math.log(2 / ALPHA) * 2 / n) * CS_CONST / 2


def _wilson_bounds(c: int, n: int, z: float) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = c / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom, (centre + margin) / denom


class R4State:
    def __init__(self):
        self.n = 0
        self.total = 0.0
        self.protected = False

    def observe(self, delta: float):
        self.n += 1
        self.total += max(-CLIP, min(CLIP, delta))

    def verdict(self) -> str:
        if self.n == 0:
            return "observe"
        mean = self.total / self.n
        r = _radius(self.n)
        if mean - r > 0:
            self.protected = True
        if self.protected:
            return "protect"
        if self.n >= MIN_SERVES and mean + r < CUT_DEADBAND:
            return "cut"
        return "observe"


class R4CState:
    """v2 (DESIGN_V2.md): context-conditioned credit. A fact is judged by
    its best per-item record - protect on a repeated on-item lift, cut
    only when well-exposed with no home turf anywhere."""

    PROTECT_MEAN = 0.30   # >= ~60% of the published mean oracle-floor gap
    CUT_MEAN = 0.15       # half the protect anchor
    MIN_CELL = 2
    MIN_TOTAL = 8

    def __init__(self):
        self.cells = {}            # item -> (n, total_delta)
        self.total_serves = 0
        self.protected = False

    def observe(self, item: int, delta: float):
        self.total_serves += 1
        n, t = self.cells.get(item, (0, 0.0))
        self.cells[item] = (n + 1, t + delta)

    def verdict(self) -> str:
        best = max((t / n for n, t in self.cells.values()
                    if n >= self.MIN_CELL), default=None)
        if best is not None and best >= self.PROTECT_MEAN:
            self.protected = True
        if self.protected:
            return "protect"
        if (self.total_serves >= self.MIN_TOTAL
                and (best is None or best < self.CUT_MEAN)):
            return "cut"
        return "observe"


class BinaryState:
    """The prior rules share the (corr, fals, served) counters with the
    live loop's ±tol binarization."""

    def __init__(self):
        self.c = 0
        self.f = 0
        self.s = 0

    def observe(self, delta: float):
        self.s += 1
        if delta > TOL:
            self.c += 1
        elif delta < -TOL:
            self.f += 1

    def verdict(self, stat: str) -> str:
        if self.s < PRIOR_MIN_SERVES:
            return "observe"
        if stat == "point":
            return "cut" if (self.c / self.s) < HELP_RATE else "observe"
        if stat == "neverhelped":
            return "cut" if self.c == 0 else "observe"
        if stat == "wilson2":
            lo, hi = _wilson_bounds(self.c, self.s, WILSON_Z)
            if lo > PROTECT_FLOOR:
                return "protect"
            return "cut" if hi < HELP_RATE else "observe"
        raise ValueError(stat)


def replay_seed(events: list[dict]) -> dict:
    """Replay the rule set over one seed's nolearn stream."""
    facts: dict[str, dict] = {}
    rounds = sorted({e["round"] for e in events})
    # per-rule: fact -> cut round (None = never)
    cut_round = {rule: {} for rule in ("r4", "r4c", "point", "neverhelped", "wilson2")}
    states_r4: dict[str, R4State] = {}
    states_r4c: dict[str, R4CState] = {}
    states_bin: dict[str, BinaryState] = {}

    for r in rounds:
        for e in [x for x in events if x["round"] == r]:
            f = e["fact"]
            # GLOBAL identity: is_wrong is band-level (fact ∉ golds_set);
            # the logged is_gold is per-serve (right answer for the item
            # served) and misclassifies a gold first observed off-item.
            facts.setdefault(f, {"is_gold": not e["is_wrong"], "serves": 0})
            facts[f]["serves"] += 1
            states_r4.setdefault(f, R4State()).observe(e["delta"])
            states_r4c.setdefault(f, R4CState()).observe(e["item"], e["delta"])
            states_bin.setdefault(f, BinaryState()).observe(e["delta"])
        # end-of-round suppression sweep (mirrors the live loop)
        for f in facts:
            if f not in cut_round["r4"] and states_r4[f].verdict() == "cut":
                cut_round["r4"][f] = r
            if f not in cut_round["r4c"] and states_r4c[f].verdict() == "cut":
                cut_round["r4c"][f] = r
            for stat in ("point", "neverhelped", "wilson2"):
                if (f not in cut_round[stat]
                        and states_bin[f].verdict(stat) == "cut"):
                    cut_round[stat][f] = r

    out = {}
    golds = {f for f, m in facts.items() if m["is_gold"]}
    wrongs = {f for f in facts if f not in golds}
    for rule, cuts in cut_round.items():
        g_cut = sorted(f for f in cuts if f in golds)
        w_cut = sorted(f for f in cuts if f in wrongs)
        out[rule] = {
            "golds_cut": len(g_cut), "golds_cut_facts": g_cut,
            "wrongs_cut": len(w_cut),
            "first_correct_cut_round": (min(cuts[f] for f in w_cut)
                                        if w_cut else None),
            "wrongs_observed": len(wrongs), "golds_observed": len(golds),
        }
    out["_facts_observed"] = len(facts)
    out["_serves"] = len(events)
    return out


def main() -> int:
    run = json.loads(RUN.read_text(encoding="utf-8"))
    streams = run["serve_log"]["nolearn"]
    per_seed = [replay_seed(s) for s in streams]

    n_seeds = len(per_seed)

    def rule_bars(rule):
        safe = all(s[rule]["golds_cut"] == 0 for s in per_seed)
        ge = all(s[rule]["wrongs_cut"] >= s["wilson2"]["wrongs_cut"]
                 for s in per_seed)
        gt = sum(s[rule]["wrongs_cut"] > s["wilson2"]["wrongs_cut"]
                 for s in per_seed)
        return {"safe_zero_golds_every_seed": safe,
                "effective_ge_wilson2_every_seed": ge,
                "strictly_more_in_2_of_3": gt >= 2,
                "pass": safe and ge and gt >= 2}

    r4_safe = all(s["r4"]["golds_cut"] == 0 for s in per_seed)
    r4_ge = all(s["r4"]["wrongs_cut"] >= s["wilson2"]["wrongs_cut"]
                for s in per_seed)
    r4_gt = sum(s["r4"]["wrongs_cut"] > s["wilson2"]["wrongs_cut"]
                for s in per_seed)
    summary = {
        "seeds": n_seeds,
        "per_seed": per_seed,
        "bars_v2_r4c": rule_bars("r4c"),
        "bars": {
            "safe_zero_golds_every_seed": r4_safe,
            "effective_ge_wilson2_every_seed": r4_ge,
            "strictly_more_in_2_of_3": r4_gt >= 2,
            "pass": r4_safe and r4_ge and r4_gt >= 2,
            "tie_note": ("tie (>= in all seeds but strictly more in "
                         f"{r4_gt}/{n_seeds})" if r4_safe and r4_ge
                         and r4_gt < 2 else None),
        },
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
