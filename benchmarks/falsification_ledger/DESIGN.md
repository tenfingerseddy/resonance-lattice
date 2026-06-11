# R2 — The Falsification Ledger (pre-registered design)

**Status**: design locked before any results (pre-registration discipline).
**Date**: 2026-06-10. Roadmap Stage 6, bet R2 (sharpened per the review:
phase 1 = internal tried-and-falsified claims, where the benchmark file
itself is the evidence — no freshness loop needed).

## The bet, in plain language

A project accumulates hard-won negative results: "we tried X, it made
things worse, here's the benchmark." The standard wisdom an assistant
carries ("add a reranker", "fine-tune on MS MARCO") is exactly what those
local results falsified. Serving tried-and-falsified claims as
first-class atoms should stop the assistant from re-recommending dead
ends — measurable as a binary per item: did the answer recommend the
falsified approach without acknowledging the prior local failure, yes/no.

This project's own ledger is the phase-1 corpus: ten real falsified bets,
each citing its benchmark evidence. World-absence claims ("library X has
no feature Y") are phase 2 — they need the R7 freshness loop and are out
of scope here.

## Why this isn't the "near-certain pass" version

Same discipline as R1, plus one guard R1 didn't need:

1. **Recommendation-decisive gate (STEP 0)** — an item enters only if the
   BLIND answer actually recommends the falsified approach. The claim is
   conditional on this subset, stated as such.
2. **Topical-mention control (the new guard)** — a verdict-free,
   purely descriptive sentence about the same approach, served in the
   identical framing. If the drop appears here too, topical priming —
   not the falsification verdict — is the active ingredient, and the
   ledger content class is NOT proven.
3. **Irrelevant-ledger placebo** — a falsified claim from an unrelated
   topic, manually audited to be non-bearing on the question. Catches
   "any served negative result makes the model conservative."
4. **Collateral set** — 8 questions untouched by every atom, answered
   with the FULL 10-atom ledger served (the realistic serve-all shape
   for a small ledger). Over-blocking — a pile of negative results
   making the model gun-shy or derailing answers — is measured, not
   assumed away.

## Items

- Domain: this project (dense-retrieval Python library; the answerer gets
  a neutral one-line context, no hint toward minimalism).
- 10 ledger atoms, each a real falsified bet with its evidence pointer
  (HONEST_CLAIMS.md / benchmark memories / committed results).
- 20 questions (2 per atom) phrased so the natural, literature-informed
  answer IS the falsified approach.
- Each item carries a hand-audited `placebo_atom` (non-bearing) and the
  atom's `approach` summary (what the judge tests for).
- 8 collateral questions orthogonal to every atom.

## Arms (paired, within-item — CHARTER §3)

All served arms use the IDENTICAL framing header ("Context notes for this
project") so the only delta between arms is the content of the note:

| Arm | Note content |
|---|---|
| A blind | none |
| B ledger | the matched atom: "Tried and falsified in this project (evidence: …): <finding>" |
| C topical | a verdict-free descriptive sentence about the same approach |
| D placebo | an unrelated atom in full ledger form |

Collateral: blind + full-ledger-served (all 10 atoms).

Same answerer for all arms (fresh subscription subagent per call).
Judge: fresh subagent per item-arm — binary `recommends: yes/no` with a
quoted span. Rule: endorsing the approach (primary or suggested option)
without acknowledging the prior local failure = recommends. Steering away
from it, or proposing a variant while explicitly acknowledging the local
failure, is NOT a recommendation. Collateral judged
`substantive: yes/no` (same prompt as R1).

## Pre-registered pass bar

On the recommendation-decisive subset:

- **PASS** requires ALL of:
  1. Ledger arm (B) recommend rate ≤ ⅓ × blind rate.
  2. Active ingredient: C rate − B rate ≥ 25pp (the verdict does work a
     topical mention doesn't).
  3. Placebo (D) within ±10pp of blind.
  4. Collateral substantive rate drops < 10pp vs blind.
- **FAIL / NEEDS-REDESIGN**: any guard breached, even with a headline
  drop. In particular, if C also collapses, the honest verdict is
  "topical priming suffices" — informative, but not the ledger bet.
- Judges: free subscription subagents first pass; a promising PASS gets
  the API-judge confirmation on the same transcripts (the R1
  `api_judge_confirm.py` pattern) before any claim ships.

## What a PASS buys / what it doesn't

Buys: a second band content class with internal verification — ledger
atoms never go stale the way world-facts do (the evidence is a committed
benchmark file in the repo) — plus the product story that rlat's own
falsification record becomes serve-able context. Doesn't buy: anything
about *capturing* falsifications automatically (that's the telemetry/
curator track), or world-absence claims (phase 2, needs R7).

## Artifacts

- `items.json` — atoms, questions, controls, placebo mapping, collateral
  (committed before any arm runs).
- `run_results/` — per-item per-arm answers + binary verdicts.
- `VERDICT.md` — the pre-registered bar applied, written last.

## Run-1 invalidation + run-2 amendment (2026-06-10, committed before run 2)

Run 1 (items committed at f31d6836; results in
`run_results/run1_2026-06-10_INVALID.json`) is **INVALID as a test of the
bars: gate yield 0/20**. The blind arm was not blind — answerer subagents
execute inside this repository, where the project context (CLAUDE.md's
thesis line: "no rerank, no lexical sidecar…") and the committed
falsification record itself are visible. 9/20 blind answers cite the
project's own record explicitly; several quote exact benchmark numbers.

Worth keeping from the wreckage: an in-repo agent **self-serves a
committed falsification ledger and follows it correctly** — strong for
the product story (commit your negative results; agents will find them),
fatal for this experiment's blind arm. Phase-1's "evidence is internal"
property was the very thing that leaked.

Run-2 changes (bars, arms, judge prompts, collateral mechanics all
unchanged):

- The advised project is **fictional** ("Lumera": e5-large-v2 1024d dense
  retrieval over internal news/docs corpora) — its ledger exists nowhere
  the answerer can reach, restoring a true blind arm. Same move R1 made
  with a fictional tenant's constraints.
- Same 10 approaches; atom numbers realistic but varied from this repo's
  real record; evidence pointers are Lumera-fictional paths.
- Answer prompts add: answer from general expertise; Lumera is unrelated
  to any repository or local files you can see — do not consult them.
- Items: `items_run2.json`.
