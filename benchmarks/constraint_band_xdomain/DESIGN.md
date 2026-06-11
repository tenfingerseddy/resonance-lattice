# R1-X — Constraint Band, cross-domain replication (pre-registered design)

**Status**: design locked before any results (pre-registration discipline).
**Date**: 2026-06-10. Follows Kane's direction: rlat is a knowledge model
for ANY corpus — gardening or legal as much as software. Every content-
class proof so far ran on technical corpora; per the standing
validate-across-corpus-types rule, the domain-general claim is unearned
until a non-technical replication passes.

## The bet, in plain language

R1 proved (Fabric, API-confirmed): serving a handful of standing hard
rules eliminates rule-violating answers at zero collateral cost. If the
"knowledge model that knows its own world" framing is real, the same
design must work where the world is a garden or a suburban law practice —
no software anywhere. Same paired arms, same binary verdicts, two new
domains:

- **Garden** — a home garden with standing rules (organic-only, water
  restrictions, no-dig, dogs roam, natives-only).
- **Practice** — a small NSW law practice with standing practice policies
  (NSW matters only, no family law, transactional only, employers only,
  fixed-fee only).

## Carried-forward methodology (improvements locked in up front)

1. **Fictional-world framing** (R2 run-1 lesson): both worlds are
   fictional clients; answer prompts instruct "answer from your own
   general expertise; do not consult local files." No repo fact can leak
   into the blind arm.
2. **Noise-aware placebo guard** (R2 run-2b lesson): the raw ±10pp
   placebo bar is brittle at small n because some items have ~50/50
   recommend propensity. A fourth arm — **blind-2**, a fresh resample of
   the identical blind prompt on the decisive subset — is pre-registered
   NOW, and the placebo guard is: placebo flip count ≤ blind-2 flip
   count. Raw placebo rate is reported alongside.

## Items

- 12 candidate (constraint, question) pairs per domain (24 total),
  questions phrased so a natural answer violates the constraint.
- STEP 0 violation-decisive gate per domain: an item enters only if the
  BLIND answer violates. A domain with **fewer than 4 decisive items** is
  verdict INSUFFICIENT-YIELD (no claim either way in that domain).
- 6 collateral questions per domain, orthogonal to every constraint,
  answered with one rotated constraint served (R1's shape) — the
  over-blocking guard.
- Placebos are domain-plausible and irrelevant (garden: weatherproof
  plant labels; practice: blue letterhead template).

## Arms (paired, within-item — CHARTER §3)

| Arm | Context |
|---|---|
| A blind | question only |
| B served | question + "Standing constraints: <true constraint>" |
| C placebo | question + the domain's irrelevant constraint |
| E blind-2 | question only, fresh sample (decisive items only) |

Same answerer for all arms (fresh subscription subagent per call).
Judge prompts byte-identical to R1's (`violates` / `substantive`, binary,
quoted-span evidence).

## Pre-registered pass bar (evaluated PER DOMAIN)

- **PASS** in a domain requires ALL of:
  1. Served violation rate ≤ ⅓ × blind rate on the decisive subset.
  2. Placebo flips ≤ blind-2 flips on the decisive subset (noise-aware
     guard).
  3. Collateral substantive rate drops < 10pp vs blind.
- **The cross-domain claim** ("the constraint band generalises beyond
  technical corpora") requires PASS in BOTH domains. One domain passing
  is a partial result, stated as such.
- Judges: subscription first pass; a promising PASS gets the Haiku
  API-judge confirmation on the same transcripts (the established
  `api_judge_confirm.py` pattern) before the claim ships.

## What a PASS buys / what it doesn't

Buys: the domain-general form of the band's strongest content class —
"a knowledge model that knows its own world's standing rules" is then a
claim about gardens and law practices, not just Fabric tenants. Doesn't
buy: anything about capture, and nothing about the other two content
classes (env facts, falsification ledgers) outside software — those
generalisations remain open.

## Artifacts

- `items.json` — both domains' constraints, questions, placebos,
  collateral (committed before any arm runs).
- `run_results/` — per-item per-arm answers + binary verdicts.
- `VERDICT.md` — the pre-registered bar applied per domain, written last.
