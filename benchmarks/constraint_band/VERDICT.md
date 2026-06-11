# R1 Constraint Band — run 1 verdict (2026-06-10)

**PASS on all three pre-registered bars — CONFIRMED by the independent
API judge** (subscription-judge first pass + claude-haiku-4-5 re-score of
the same transcripts, per the design's calibration rule). The claim ships.

## Numbers (run_results/run1_2026-06-10.json; 148 subagents, $0 API)

| Measure | Result | Pre-registered bar |
|---|---|---|
| Blind violation rate (full 24) | **15/24 (62%)** | — (gate yield) |
| Violation-decisive subset | 15 items | — |
| **Served** violation rate (decisive) | **1/15 (7%)** | ≤ 33% → **PASS** |
| **Placebo** violation rate (decisive) | **14/15 (93%)** | ≥ 90% (≈blind) → **PASS** |
| Collateral substantive, blind → served | **10/10 → 10/10** | drop < 10pp → **PASS** |

## Reading

- The 93% relative drop (62% → 7% scaled to the decisive subset:
  100% → 7%) is the constraint's doing, not a compliance artifact: the
  placebo arm (an irrelevant served rule) left 93% of violations intact.
- Zero over-blocking: every collateral question still got a substantive
  answer with a constraint served. The known failure mode of served rules
  did not appear at this serve framing ("standing constraints… hard
  requirements").
- The one served-arm violation (i06, C1: a report-writeback answer that
  offered the still-in-preview list slicer among the trigger options) is
  a subtle one — a preview *sub-feature* inside an otherwise-GA flow;
  worth keeping as a probe item in future runs.
- Gate yield 62% confirms the item design: a majority of natural Fabric
  answers DO violate standing constraints blind — the content class is
  load-bearing, not hypothetical.

## What this buys

The serve-all, no-selection design works at zero collateral cost: ~5
standing rules conditioned 15 otherwise-wrong answers. Combined with the
locked env-premise proof, the user-world band now has two proven content
classes (environment facts + standing constraints). Next per the design:
API-judge confirmation of these transcripts, then the capture side
(constraints are exactly what the dormant attribute miner's GATE 1/3
target).

## API-judge confirmation (same transcripts, independent judge)

`api_judge_confirm.py` re-scored all 74 transcripts with
`claude-haiku-4-5` using byte-identical judge prompts
(`run_results/api_confirm_run1.json`; ~$0.10 of the $20 review budget).
Bars are computed against the API judge's **own** blind reference on the
unchanged pre-registered decisive subset:

| Measure | Subscription | API (Haiku) | Bar under API blind |
|---|---|---|---|
| Blind violation (decisive 15) | 15/15 (100%) | 10/15 (67%) | reference |
| Served violation | 1/15 (7%) | **0/15 (0%)** | ≤ 22% → **PASS** |
| Placebo violation | 14/15 (93%) | 9/15 (60%) | within ±10pp of 67% → **PASS** |
| Collateral substantive | 10/10 → 10/10 | 10/10 → 10/10 | drop < 10pp → **PASS** |

Reading:

- **The paired contrast is judge-robust.** Under both judges, serving the
  true constraint collapses violations (sub: 100%→7%; API: 67%→0%) while
  the placebo leaves them ~at the blind rate (sub: 93%; API: 60%, within
  7pp of its 67% reference). The effect is the constraint's, not the
  judge's.
- **Post-merge correction (Codex review, 2026-06-11): the placebo bar's
  API reading is subset-sensitive.** The table above keeps the
  pre-registered 15-item subset and the API judge's blind rate on it.
  Computed instead WITHIN the API judge (only the 10 items its own blind
  verdicts find violating — the within-judge form the later R1-X/R2
  confirms standardised on): served **0/10**, placebo **7/10 (70%)** vs a
  100% blind reference — a 30pp deviation that breaches the ±10pp form of
  the guard. Same picture as R1-X: the served-collapse and collateral
  bars are judge-robust; the placebo guard specifically is
  **judge-sensitive** (Haiku systematically reads placebo-arm answers as
  less violating). The R1 claim stands on the subscription primary; the
  API confirm confirms the core bars and qualifies the placebo guard.
- **Per-item agreement**: served 14/15, collateral 20/20, blind 16/24,
  placebo 10/15 — 14 disagreements in two modes. (a) C1 GA/preview
  strictness flips on non-decisive items (i01/i03/i05): the API judge
  can't browse and its training predates several 2025/26 GA dates, so it
  flags features the web-checking subscription judge verified GA. These
  touch no bar. (b) Leniency on borderline violation calls — 5 blind
  flips on the decisive subset (i06/C1, i10/C2, i14/C3, i15/C3, i20/C5)
  and 5 placebo flips. Mode (b) lowers the API judge's blind and placebo
  references *together*, so the confirm is conservative: the served-arm
  collapse to 0/15 stands against a weaker reference, not a stronger one.
- The one subscription-flagged served violation (i06, the list-slicer
  preview mention) was the sole served-arm disagreement — the API judge
  read it as compliant, consistent with the list slicer's preview status
  being post-cutoff world knowledge it lacks.

## Provenance

- Design + items committed before any arm ran (`DESIGN.md`, `items.json`,
  commit 7d9d3dc9).
- Answerer + judges: fresh subscription subagents (Workflow run
  wf_fcf0f69e); binary verdicts with quoted-span evidence.
- Caveat honestly stated: subscription judges are aggregate-faithful but
  coarser per-item (feedback_subscription_judge_viability); the API
  confirm is the calibration step, not a formality.
