# R1 — The Constraint Band (pre-registered design)

**Status**: design locked before any results (pre-registration discipline).
**Date**: 2026-06-10. Roadmap Stage 6, bet R1 (survived adversarial critique;
this design incorporates the critic's re-aim).

## The bet, in plain language

A user has a handful of standing hard rules — "never preview features",
"EU data residency only", "service-principal auth only". A realistic answer
that ignores one of them isn't slightly worse, it's wrong-and-costly.
Because there are only ~3–10 such rules, the band can ALWAYS serve all of
them — **no retrieval, no selection step** — which sidesteps the program's
two measured walls (selection noise; outcome-judge noise) entirely. The
metric is binary per item: did the answer violate the constraint, yes/no.

## Why this isn't the "near-certain pass" version

The critic's finding: "LLM follows an instruction in its prompt" would be
uninformative. The information lives in the three guards this design adds:

1. **Violation-decisive item gate (STEP 0)** — an item only enters the test
   set if the BLIND answer actually violates the constraint. Items where
   the model already behaves are discarded (same build-gate-selection as
   the locked user-world proof). The claim is conditional on this subset,
   stated as such.
2. **Placebo arm** — an IRRELEVANT constraint served the same way must not
   change the answer materially. Catches "any served rule changes
   behaviour" (a recency/compliance artifact, not constraint value).
3. **Collateral set** — 10 questions where the constraint is irrelevant,
   answered with the constraint served. Over-blocking (refusing or
   degrading legitimate answers) is the known failure mode of served
   rules; we measure it instead of assuming it away.

## Items

- Domain: Microsoft Fabric / Power BI (the wedge corpus; rich in
  preview-gated, SKU-gated, region-gated features).
- 24 candidate (constraint, question) pairs across 5 standing constraints:
  no-preview-features, EU-only data residency, service-principal-auth only,
  no-capacity-above-F8, air-gapped (no external API calls).
- STEP 0 keeps the violation-decisive subset (expected ~12–18 items based
  on the userworld build-gate yield).
- Collateral set: 10 additional questions orthogonal to every constraint.

## Arms (paired, within-item — CHARTER §3)

| Arm | Context |
|---|---|
| A blind | question only |
| B served | question + "Standing constraints for this environment: <true constraint>" |
| C placebo | question + an irrelevant-but-plausible constraint |

Same answerer model for all arms (subscription subagent, fresh context per
call, temperature as-is). Judge: a separate fresh subagent per item — sees
the constraint + the answer, returns a BINARY verdict `violates: yes/no`
with a quoted span as evidence. Binary checks are the noise-robust grain
(the ±0.18 outcome-judge noise lives in scalar quality scores, not binary
properties). Collateral set judged for refusal/degradation: `answers the
question substantively: yes/no`.

## Pre-registered pass bar

- **PASS**: served violation rate ≤ ⅓ of blind violation rate on the
  violation-decisive subset (e.g. ≥60% → ≤20%), AND placebo arm's
  violation rate within ±10pp of blind, AND collateral substantive-answer
  rate drops < 10pp vs blind.
- **FAIL / NEEDS-REDESIGN**: any guard breached (placebo moves answers, or
  collateral damage ≥ 10pp) even if the headline rate drops.
- Judges: free subscription subagents (aggregate-faithful per
  feedback_subscription_judge_viability); a promising PASS gets a $-capped
  API-judge confirmation on the same transcripts before any claim ships.

## What a PASS buys / what it doesn't

Buys: the highest value-per-byte content class for the band (one rule
conditions every future answer that touches it), with a serve design that
needs no selector. Doesn't buy: anything about *capturing* constraints
(that's the attribute-miner track) — this isolates serve value, oracle-
style, exactly like the locked env-premise proof did.

## Artifacts

- `items.json` — constraints, questions, collateral set (committed before
  any arm runs).
- `run_results/` — per-item per-arm answers + binary verdicts (committed).
- `VERDICT.md` — the pre-registered bar applied, written last.
