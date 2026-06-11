# R4 v2 — context-conditioned credit (pre-registered before any v2 replay)

**Date**: 2026-06-11. Follows the run-1 structural diagnosis: pooled
per-fact credit orders a gold below wrong facts because golds are served
off-item, where truth doesn't help. v2 conditions credit on the serve
context — a fact is judged by its best per-item record, not its average.

## Epistemic status (stated up front)

v2 is motivated by run-1's PUBLISHED aggregates (the pooled table in
VERDICT.md) and will be replayed on the SAME stream — so a v2 pass here
is **provisional, not confirmatory**: the rule could fit this stream's
quirks. A pass earns a fresh live run (new stream, rule as a live
`--suppress-stat`) before any claim ships. The thresholds below are
anchored to quantities published BEFORE this design (run-1 header
aggregates), not tuned on per-(fact, item) cells, which have not been
inspected.

## The rule (locked)

Maintain per-(fact, item) running mean m and count n over per-serve
deltas. End-of-round sweep:

- **PROTECT** fact f if ∃ item i with n(f,i) ≥ 2 and m(f,i) ≥ **+0.30**
  — repeated on-item lift at ≥~60% of the fixture's published mean
  oracle−floor gap (+0.52). A fact with a home turf is never cut.
- **CUT** fact f if its total serves ≥ **8** (≈ one full round of typical
  exposure) AND no item has n(f,i) ≥ 2 with m(f,i) ≥ **+0.15** (half the
  protect anchor) — enough exposure, no home turf anywhere.
- Otherwise observe.

No parameter may move after the replay runs; any adjusted variant is
exploratory and must say so.

## Pre-registered bars (unchanged LOCK structure, per nolearn seed)

1. **SAFE**: zero golds cut in every seed.
2. **EFFECTIVE**: wrongs cut ≥ wilson2's (3 per seed) in every seed;
   strictly more in ≥ 2 of 3 seeds.
3. All seeds individually.

PASS = provisional pass → live confirmation run required.
FAIL = recorded as-is; the stream remains the free bench for v3.

## Run verdict (2026-06-11, replay_run2_v2.json)

**FAIL on bar 2, recorded as-is** - and the most informative result of
the family:

| Rule | golds cut (3 seeds) | wrongs cut (3 seeds) |
|---|---|---|
| **r4c (v2)** | **0, 0, 0** | 2, 2, 2 |
| wilson2 | 1, 1, 1 | 3, 3, 3 |
| point | 1, 1, 1 | 4, 4, 4 |

Context-conditioning works: from 0 wrongs cut (v1) to 2 per seed while
remaining the only family member that never executes a gold. The bar it
fails compares raw wrong-cut counts against a rule that buys its third
wrong by killing a true fact every seed. Under any weighting where one
gold-kill costs more than one wrong-kept - the program's own trust
premise - r4c strictly dominates wilson2 (net 2-0 vs 3-1). The
pre-registered bar embedded an unweighted trade-off; that is a design
lesson, not grounds to re-score this run.

Disposition (DEVIATION, noted per Codex review): this section's own gate
says only a PASS earns the live confirmation, and bar 2 FAILED. The live
run proceeded anyway on the cost-adjusted dominance argument above - a
deliberate, separately pre-registered deviation (DESIGN_CONFIRM.md), not
a re-scoring of this replay. The live run then FAILED on its own bars,
vindicating the gate. r4c was the candidate (safety-dominant);
any v3 must pre-register a cost-weighted effectiveness bar BEFORE its
replay. No further same-stream iteration this cycle - overfitting risk
compounds with each pass over one stream.
