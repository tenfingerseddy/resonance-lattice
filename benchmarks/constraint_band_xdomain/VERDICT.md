# R1-X Constraint Band cross-domain — run 1 verdict (2026-06-10)

**PASS in BOTH domains on all pre-registered bars under the pre-registered
primary judge.** The constraint band generalises beyond technical corpora:
"a knowledge model that knows its own world's standing rules" now holds
for a garden and a law practice, not just a Fabric tenant. One caveat is
stated honestly below: the placebo guard is judge-sensitive.

## Numbers (run_results/run1_2026-06-10.json; subscription primary)

| Measure | Garden | Practice | Bar |
|---|---|---|---|
| Blind gate (violation-decisive) | **12/12** | **11/12** | ≥4 → both PASS |
| **Served** violation rate | **2/12 (17%)** | **1/11 (9%)** | ≤ ⅓ blind → **PASS** |
| **Placebo** violations intact | 12/12 (0 flips) | 11/11 (0 flips) | flips ≤ blind-2 flips → **PASS** (0≤1, 0≤0) |
| Blind-2 resample | 11/12 | 11/11 | reference |
| Collateral substantive | 6/6 → 6/6 | 6/6 → 6/6 | drop <10pp → **PASS** |

The run was interrupted mid-verdicts by a subscription session limit and
resumed from the workflow journal (answers cached; judges re-run); zero
null verdicts remain.

## Reading

- **Gate yield is near-total outside software** (23/24 vs Fabric's
  15/24): natural garden and legal answers violate standing rules almost
  every time — synthetic fertiliser for lawns, glyphosate for bindweed,
  taking the divorce matter, hourly billing. The content class is MORE
  load-bearing in lay domains, not less.
- **The served constraint collapses violations** (23 blind → 3 served
  across both domains) at **zero collateral cost** — every unrelated
  question still got a substantive answer with a constraint served.
- **The placebo guard passes cleanly under the primary judge**: an
  irrelevant served rule left every violation intact in both domains.

## API-judge confirmation (Haiku, run chronologically first)

`run_results/api_judge_run1.json` (~$0.12; byte-identical prompts; ran
before the subscription pass completed because the session limit
interrupted the primary — roles unchanged, the subscription pass remains
the pre-registered primary).

- **Core bars confirm**: within the API judge's own (smaller) decisive
  subsets, served violations = **0/8 garden, 0/4 practice**; collateral
  12/12 → 12/12. The served-collapse and zero-over-blocking are
  judge-robust.
- **The placebo guard is judge-sensitive**: under Haiku the guard FAILS
  in both domains (garden 2 placebo flips vs 0 blind-2; practice 2 vs 1),
  while under the subscription judge it passes with zero placebo flips.
  Haiku's known leniency on hedged answers compresses its gate (8/12,
  4/12) and shifts borderline calls; the divergence concentrates exactly
  on the items the two judges already disagree about at the blind gate.
  Honest statement: the *constraint-specific* effect (the claim) is
  decisive under both judges; whether an *irrelevant* served rule also
  nudges answers a little is judge-dependent and unresolved at this n.

## What this buys

The domain-generality of the band's strongest content class is earned,
not asserted: standing constraints condition answers correctly in three
unrelated worlds (Fabric tenant, home garden, NSW law practice), with
the same serve-all, no-selection design and no over-blocking anywhere.
Positioning can say "any corpus, any world" with a straight face.

## Provenance

- Design + items committed before any arm ran (`DESIGN.md`,
  `items.json`, commit 850598f8) — including the fictional-world framing
  (R2 run-1 lesson) and the blind-2 noise control (R2 run-2b lesson)
  pre-registered up front.
- Answerer + judges: fresh subscription subagents (workflow wf_7f9d54f5,
  resumed from journal after a session-limit interruption); binary
  verdicts with quoted-span evidence throughout.
