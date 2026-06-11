"""Value proof (Stage 1 smoke) — does real personal memory lift answer quality?

The metric of record (value-proof.md): answer quality + tailoring + action value,
NOT retrieval/recall. This smoke runs the cheapest decisive probe of the bet's
core: `cold` vs `+memory`, where `+memory` injects the lessons the real rlat
recall path surfaces for the question (from the scratch store seeded by
seed_memory.py). Half the scenarios are load-bearing (a real lesson changes the
best answer/action); half are specificity controls (no relevant lesson — these
MUST show ~no lift; the SHARPEN anti-rig).

Judge: accuracy/grounding (a hard gate) + tailoring + action value, scored blind
(no lane label) and repeated (report variance).

Headline = mean composite(+memory) − mean composite(cold), split by scenario kind.
PASS-smoke = load_bearing shows positive lift AND control stays ~flat.

Usage:
    PYTHONUTF8=1 python -m benchmarks.user_bench.value_proof.seed_memory --reset
    PYTHONUTF8=1 python -m benchmarks.user_bench.value_proof.run --budget-usd 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent.parent.parent
_SRC = _REPO / "src"
if (_SRC / "resonance_lattice" / "__init__.py").exists():
    sys.path.insert(0, str(_SRC))

from resonance_lattice._pricing import SONNET_MODEL as MODEL, cost_usd as _cost_usd
from resonance_lattice.assembler import (
    CorpusHit,
    MemoryHit,
    assemble,
    DEFAULT_MEM_FLOOR,
    DEFAULT_CORPUS_FLOOR,
)

from .seed_memory import SCRATCH_USER

_LANE_SOURCES = {
    "cold": (),
    "memory": ("memory",),
    "corpus": ("corpus",),
    "assembled": ("memory", "corpus"),
}

_LANES = ("cold", "memory", "corpus", "assembled")


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
def _client():
    import anthropic
    key = os.environ.get("CLAUDE_API_2") or os.environ.get("CLAUDE_API")
    if not key:
        from resonance_lattice._anthropic import discover_api_key
        key = discover_api_key()
    if not key:
        raise RuntimeError("no LLM API key (set CLAUDE_API_2)")
    return anthropic.Anthropic(api_key=key)


# --------------------------------------------------------------------------- #
# Memory source — the real recall path over the scratch store
# --------------------------------------------------------------------------- #
class MemorySource:
    """Wraps the real rlat recall path over the seeded scratch store."""

    def __init__(self, user_id: str = SCRATCH_USER, top_k: int = 5):
        from resonance_lattice.memory.claim_store import ExperienceClaimStore
        self.store = ExperienceClaimStore(user_id=user_id)
        self.top_k = top_k
        # Fail loud if the store is empty — the smoke is meaningless without it.
        claims, _ = self.store.read_all_with_band()
        if not claims:
            raise RuntimeError(
                f"scratch memory store (user={user_id}) is empty — "
                "run seed_memory.py --reset first")

    def recall_hits(self, query: str) -> list[MemoryHit]:
        """Recalled lessons as assembler MemoryHits (relevance = cosine)."""
        from resonance_lattice.memory.recall import recall
        hits = recall(query, store=self.store, top_k=self.top_k,
                      auto_tune_cold_start=True)
        out = []
        for h in hits:
            pol = next((p for p in h.claim.facts.polarity
                        if p in ("prefer", "avoid", "factual")), "factual")
            out.append(MemoryHit(content=h.claim.content, polarity=pol,
                                 relevance=float(h.cosine)))
        return out


# --------------------------------------------------------------------------- #
# Corpus source — rlat retrieval over a domain knowledge model
# --------------------------------------------------------------------------- #
class CorpusSource:
    """Runs `rlat search` over a domain corpus and renders a context block.

    Uses --mode knowledge (treat corpus as authoritative; returns passages even
    when the local source has drifted — we want the build-time passage text)."""

    def __init__(self, km_path: str, top_k: int = 5):
        self.km_path = km_path
        self.top_k = top_k

    def retrieve_hits(self, query: str) -> list[CorpusHit]:
        """Retrieved passages as assembler CorpusHits (relevance = score)."""
        cmd = [sys.executable, "-m", "resonance_lattice.cli.app", "search",
               self.km_path, query, "--top-k", str(self.top_k),
               "--format", "json", "--mode", "knowledge", "-q"]
        import subprocess
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8")
        if proc.returncode != 0:
            return []
        try:
            hits = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return []
        out = []
        for h in hits[: self.top_k]:
            txt = (h.get("text") or "").strip()
            if not txt:
                continue
            out.append(CorpusHit(text=txt, source=h.get("source_file", "?"),
                                  score=float(h.get("score", 0.0))))
        return out


_ANSWER_SYSTEM = (
    "You are an assistant helping a developer work on a software project "
    "(the rlat / Resonance Lattice retrieval library). Answer the user's "
    "question concisely and correctly. If the system prompt includes earned "
    "lessons from past sessions, weigh them when they apply. If it includes "
    "retrieved domain passages, ground your answer in them. When the question "
    "is about what to do, name the concrete next action."
)


# --------------------------------------------------------------------------- #
# Judge — accuracy (gate) + tailoring + action value
# --------------------------------------------------------------------------- #
JUDGE_SYSTEM = """You grade a candidate answer to a developer's question on three independent 0-1 dimensions. You are given a GENERIC gold (the correct-but-generic answer) and a TAILORED gold (what a good answer FOR THIS USER adds — the project-specific detail and the right next action). You do NOT know which system produced the candidate.

Score each dimension in [0,1]:
- accuracy: is the answer factually correct and free of invented/contradictory claims? (1.0 fully correct, 0.0 wrong/hallucinated). This is a GATE — if the answer is inaccurate, the other dimensions do not matter.
- tailoring: does it capture what the TAILORED gold adds beyond the GENERIC gold (the project-specific, user-specific detail)? 1.0 = captures the tailored specifics; 0.5 = partial; 0.0 = only the generic answer, no tailoring. If the generic and tailored golds are identical (a control scenario), score 1.0 only if the answer is correct and does NOT bolt on irrelevant/invented project-specifics; an answer that fabricates project-specific tailoring here scores LOW.
- action_value: does it name a clear, correct, high-value next action for this user? 1.0 = a concrete correct action; 0.5 = vague gesture; 0.0 = none or wrong action. For a pure-fact question with no action, 1.0 if the answer is complete.

Output exactly one JSON object on one line: {"accuracy": 0.0, "tailoring": 0.0, "action_value": 0.0, "reason": "..."}"""


@dataclass
class Score:
    accuracy: float = 0.0
    tailoring: float = 0.0
    action_value: float = 0.0
    reason: str = ""

    @property
    def composite(self) -> float:
        # accuracy is a hard gate at 0.5; below it the answer scores 0.
        if self.accuracy < 0.5:
            return 0.0
        return (self.tailoring + self.action_value) / 2.0


@dataclass
class AnswerResult:
    scenario_id: str
    kind: str
    lane: str
    answer: str = ""
    surfaced: list[str] = field(default_factory=list)
    sources_included: list[str] = field(default_factory=list)
    scores: list[Score] = field(default_factory=list)  # repeated judge runs
    in_tokens: int = 0
    out_tokens: int = 0

    def mean(self, attr: str) -> float:
        vals = [getattr(s, attr) for s in self.scores] if attr != "composite" \
            else [s.composite for s in self.scores]
        return sum(vals) / len(vals) if vals else 0.0


def _ask(client, system: str, question: str, max_tokens: int = 500):
    msg = client.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": question}],
    )
    return (msg.content[0].text.strip(),
            int(msg.usage.input_tokens), int(msg.usage.output_tokens))


def _judge(client, scn: dict, candidate: str) -> tuple[Score, int, int]:
    user = (
        f"Question: {scn['question']}\n\n"
        f"GENERIC gold: {scn['generic_gold']}\n\n"
        f"TAILORED gold: {scn['tailored_gold']}\n\n"
        f"Candidate answer: {candidate}\n\n"
        "Output the JSON now."
    )
    msg = client.messages.create(
        model=MODEL, max_tokens=250, system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    raw = msg.content[0].text.strip()
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    sc = Score()
    if m:
        try:
            obj = json.loads(m.group(0))
            sc = Score(
                accuracy=float(obj.get("accuracy", 0.0)),
                tailoring=float(obj.get("tailoring", 0.0)),
                action_value=float(obj.get("action_value", 0.0)),
                reason=str(obj.get("reason", ""))[:200],
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            sc = Score(reason=f"unparseable: {raw[:120]}")
    else:
        sc = Score(reason=f"no-json: {raw[:120]}")
    return sc, int(msg.usage.input_tokens), int(msg.usage.output_tokens)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def _paired_bootstrap(lifts: list[float], n_boot: int = 10000,
                      seed: int = 17) -> dict:
    """Percentile CI on the mean of per-scenario paired lifts."""
    import numpy as np
    a = np.asarray(lifts, dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n": 0, "excludes_zero": False}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    means = a[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(a.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "n": int(a.size), "excludes_zero": bool(lo > 0 or hi < 0)}


def _aggregate(results: list[AnswerResult], lanes: list[str]) -> dict:
    # Per-scenario composite per lane (paired vs cold).
    by_scn: dict[str, dict] = {}
    for r in results:
        d = by_scn.setdefault(r.scenario_id, {"kind": r.kind})
        d[r.lane] = r.mean("composite")

    out: dict = {}
    for kind in ("load_bearing", "control"):
        out[kind] = {}
        for lane in lanes:
            rs = [r for r in results if r.kind == kind and r.lane == lane]
            if not rs:
                continue
            out[kind][lane] = {
                "n": len(rs),
                "accuracy": sum(r.mean("accuracy") for r in rs) / len(rs),
                "tailoring": sum(r.mean("tailoring") for r in rs) / len(rs),
                "action_value": sum(r.mean("action_value") for r in rs) / len(rs),
                "composite": sum(r.mean("composite") for r in rs) / len(rs),
            }
        # paired lift of each non-cold lane vs cold
        if "cold" in out[kind]:
            out[kind]["lift_vs_cold"] = {}
            for lane in lanes:
                if lane == "cold":
                    continue
                pairs = [d for d in by_scn.values()
                         if d["kind"] == kind and "cold" in d and lane in d]
                lifts = [d[lane] - d["cold"] for d in pairs]
                out[kind]["lift_vs_cold"][lane] = {
                    "lift": sum(lifts) / len(lifts) if lifts else 0.0,
                    "bootstrap": _paired_bootstrap(lifts),
                }
    return out


def _resolve_corpus_path(km: str) -> str:
    """Resolve a corpus name to a path (checks worktree _hf_corpora + main dir)."""
    if Path(km).exists():
        return km
    for cand in (_REPO / "_hf_corpora" / km,
                 _REPO / km,
                 _REPO.parent.parent / km):  # heuristic for the OneDrive main dir
        if cand.exists():
            return str(cand)
    return km


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenarios",
                   default=str(_HERE.parent / "scenarios.jsonl"))
    p.add_argument("--output",
                   default="benchmarks/results/user_bench/value_proof_smoke.json")
    p.add_argument("--lanes", default="cold,memory")
    p.add_argument("--km", default="", help="corpus .rlat for +corpus/+assembled lanes")
    p.add_argument("--corpus-label", default="", help="name for the corpus in output")
    p.add_argument("--judge-repeats", type=int, default=2)
    p.add_argument("--budget-usd", type=float, default=1.5)
    p.add_argument("--n", type=int, default=0, help="limit scenarios (0=all)")
    p.add_argument("--user-id", default=SCRATCH_USER)
    p.add_argument("--mem-floor", type=float, default=DEFAULT_MEM_FLOOR,
                   help="relevance gate for memory (assembler)")
    p.add_argument("--corpus-floor", type=float, default=DEFAULT_CORPUS_FLOOR,
                   help="relevance gate for corpus (assembler)")
    p.add_argument("--no-gate", action="store_true",
                   help="disable the relevance gate (naive concat — for A/B vs gated)")
    args = p.parse_args(argv)

    mem_floor = -1.0 if args.no_gate else args.mem_floor
    corpus_floor = -1.0 if args.no_gate else args.corpus_floor

    scns = [json.loads(l) for l in
            Path(args.scenarios).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.n:
        scns = scns[:args.n]
    lanes = [l for l in args.lanes.split(",") if l in _LANES]

    mem = MemorySource(user_id=args.user_id) if any(
        l in lanes for l in ("memory", "assembled")) else None
    corpus = None
    if any(l in lanes for l in ("corpus", "assembled")):
        if not args.km:
            print("[value-proof] FATAL: --km required for corpus/assembled lanes",
                  file=sys.stderr)
            return 1
        corpus = CorpusSource(_resolve_corpus_path(args.km))
    client = _client()

    results: list[AnswerResult] = []
    spent = 0.0
    print(f"[value-proof] {len(scns)} scenarios x {len(lanes)} lanes, "
          f"judge x{args.judge_repeats}, corpus={args.corpus_label or args.km or '-'}",
          flush=True)
    for scn in scns:
        # Retrieve each source's hits ONCE per scenario; the assembler gates
        # them per lane (so a lane only sees the sources it enables).
        mem_hits = mem.recall_hits(scn["question"]) if mem else []
        corp_hits = corpus.retrieve_hits(scn["question"]) if corpus else []
        for lane in lanes:
            if spent >= args.budget_usd:
                print(f"[value-proof] BUDGET CAP ${args.budget_usd}", flush=True)
                break
            ctx = assemble(
                scn["question"],
                memory_recall=lambda _q: mem_hits,
                corpus_retrieve=lambda _q: corp_hits,
                enable=_LANE_SOURCES[lane],
                mem_floor=mem_floor, corpus_floor=corpus_floor,
            )
            system = _ANSWER_SYSTEM + ("\n\n" + ctx.text if ctx.text else "")
            ans, ai, ao = _ask(client, system, scn["question"])
            res = AnswerResult(scenario_id=scn["id"], kind=scn["kind"], lane=lane,
                               answer=ans,
                               surfaced=[h.content for h in ctx.memory_hits],
                               sources_included=ctx.sources_included,
                               in_tokens=ai, out_tokens=ao)
            jin = jout = 0
            for _ in range(args.judge_repeats):
                sc, ji, jo = _judge(client, scn, ans)
                res.scores.append(sc)
                jin += ji; jout += jo
            spent += _cost_usd(ai + jin, ao + jout)
            results.append(res)
            print(f"  {scn['id']:4s} {scn['kind']:12s} {lane:9s} "
                  f"acc={res.mean('accuracy'):.2f} tail={res.mean('tailoring'):.2f} "
                  f"act={res.mean('action_value'):.2f} comp={res.mean('composite'):.2f} "
                  f"(${spent:.3f})", flush=True)
        else:
            continue
        break

    agg = _aggregate(results, lanes)
    payload = {
        "config": {"model": MODEL, "lanes": lanes, "km": args.km,
                   "corpus_label": args.corpus_label,
                   "judge_repeats": args.judge_repeats, "n": len(scns),
                   "mem_floor": mem_floor, "corpus_floor": corpus_floor,
                   "gated": not args.no_gate},
        "aggregate": agg,
        "results": [asdict(r) for r in results],
        "total_cost_usd": spent,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n[value-proof] wrote {args.output}  (${spent:.3f})", flush=True)
    print(f"\n[value-proof] corpus={args.corpus_label or args.km or '-'}  "
          "composite (tailoring+action, accuracy-gated); lift = lane − cold:", flush=True)
    for kind in ("load_bearing", "control"):
        k = agg.get(kind, {})
        if not k or "cold" not in k:
            continue
        cold = k["cold"]["composite"]
        print(f"  [{kind}]  cold={cold:.3f}", flush=True)
        for lane, lv in k.get("lift_vs_cold", {}).items():
            comp = k.get(lane, {}).get("composite", float("nan"))
            bs = lv["bootstrap"]
            ex = "  EXCLUDES 0" if bs.get("excludes_zero") else ""
            print(f"      {lane:9s} {comp:.3f}  lift={lv['lift']:+.3f}  "
                  f"95%CI[{bs['ci_lo']:+.3f},{bs['ci_hi']:+.3f}]{ex}", flush=True)
    print("\n[value-proof] read per corpus: a source 'lifts' if its load_bearing "
          "lift CI excludes 0 AND its control lift ~ 0. The bet's headline = "
          "assembled beats cold AND beats the best single source.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
