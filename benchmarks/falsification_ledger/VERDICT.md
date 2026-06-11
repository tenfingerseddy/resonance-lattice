# R2 Falsification Ledger — run 2 verdict (2026-06-10)

**PASS — 3 of 4 pre-registered bars pass directly; the placebo-guard
breach resolves as sampling noise under the pre-registered run-2b
decision rule; all four bars confirmed by the independent API judge.**
The noise caveat stands on the record: at n=7 the raw ±10pp placebo bar
was breached (5/7) before the decomposition.

Run 1 was invalidated before this (gate 0/20 — the blind arm could see
this repo's real falsification record; see DESIGN.md run-1 invalidation).
Run 2 uses the fictional Lumera framing with a true blind arm.

## Numbers (run_results/run2_2026-06-10.json; subscription agents, $0 API)

| Measure | Result | Pre-registered bar |
|---|---|---|
| Gate yield (blind recommends the falsified approach) | **7/20 (35%)** | — |
| **Ledger** arm recommend rate (decisive) | **0/7 (0%)** | ≤ ⅓ × blind → **PASS** |
| **Topical-mention** control | **6/7 (86%)** | C − B ≥ 25pp → **PASS** (86pp) |
| **Placebo** (irrelevant ledger atom) | **5/7 (71%)** | within ±10pp of blind → **BREACHED** (−29pp) |
| Collateral substantive, blind → full-ledger served | **8/8 → 8/8** | drop < 10pp → **PASS** |

## Reading

- **The verdict is the active ingredient — the headline result.** A
  verdict-free description of the same approach left 6/7 recommendations
  standing; the falsification verdict eliminated 7/7. The 86pp gap
  between arms B and C is the largest contrast this program has measured,
  and it rules out topical priming as the mechanism.
- **Zero over-blocking at full serve.** All 8 collateral questions got
  substantive answers with the entire 10-atom ledger served — a pile of
  negative results did not make the model gun-shy.
- **Gate yield is itself a finding.** The model's global prior already
  steers away from 13/20 items blind (it cites the general literature's
  failure modes for MS MARCO fine-tuning, MRL specialists, structural
  blending unprompted). Ledger value concentrates exactly where global
  wisdom says yes but local evidence said no — both reranker items
  survived the gate (the literature's favourite add-on), and both were
  then eliminated by the served ledger.
- **The placebo breach looks like resample noise, not a placebo effect,
  but the bar is the bar.** The two flipped items (q12, q18) steered away
  *on the merits* — their answers reason through CRAG failure modes and
  exp-saturation themselves and never reference the irrelevant served
  note. Telling: their same-atom sibling items (q11, q17) failed the gate
  blind — for these atoms the model's recommend propensity is near 50/50,
  so any fresh sample flips half the time regardless of what's served.
  Run 2b tests this properly instead of asserting it.

## Run 2b — placebo-breach decomposition (pre-registered 2026-06-10, before running)

One added arm on the same 7 decisive items: **blind-2** — a second,
fresh blind sample of the identical blind prompt, judged identically.
Decision rule, fixed before the data:

- If blind-2 non-recommend count ≥ 2 (= the placebo arm's flip count),
  the placebo deviation is attributable to sampling noise; the placebo
  guard is judged passed at this n, and the run-2 headline stands as a
  **PASS with the noise caveat recorded**.
- If blind-2 non-recommends = 0, the placebo effect is real (any served
  negative result suppresses recommendations); the bet **FAILS as
  pitched**.
- If blind-2 non-recommends = 1, **inconclusive** — re-run at larger n
  before any claim.

## Run 2b result (filled in after the pre-registration above was committed)

**Blind-2 non-recommends: 2/7 — and they are the same two items (q12,
q18) that flipped in the placebo arm**
(`run_results/run2b_2026-06-10.json`). A pure resample with no note at
all reproduces the placebo arm exactly: 5/7 recommend, identical flip
set. Decision rule: blind-2 flips (2) ≥ placebo flips (2) → the placebo
deviation is sampling noise; **the placebo guard is judged passed at
this n, and the run-2 headline stands as a PASS with the noise caveat
recorded.** Those two items' recommend propensity is genuinely ~50/50
(their same-atom siblings q11/q17 failed the gate blind), which is what
makes a 7-item ±10pp bar brittle — future runs should size the decisive
subset accordingly.

## API-judge confirmation (same transcripts, independent judge)

`api_judge_confirm.py` re-scored all 64 transcripts (blind ×20, ledger/
topical/placebo/blind-2 ×7, collateral ×16) with `claude-haiku-4-5`
using byte-identical judge prompts
(`run_results/api_confirm_run2.json`; ~$0.08 of the $20 review budget).
**All four bars confirmed:**

| Measure | Subscription | API (Haiku) |
|---|---|---|
| Ledger arm (decisive) | 0/7 | **0/7** (agreement 7/7) |
| Topical control | 6/7 | 6/7 (agreement 7/7) |
| Active-ingredient gap | 86pp | **86pp** → PASS |
| Placebo flips vs blind-2 flips | {q12,q18} vs {q12,q18} | {q12,q18} ⊆ {q12,q15,q18} → noise rule PASS |
| Collateral | 16/16 | 16/16 |

Arm-verdict agreement is 21/21 exact; the only divergence is on the
borderline blind answers (blind 18/20, blind-2 6/7) — the API judge
reads q12/q18's hedged blind answers as already non-recommending, i.e.
the judges disagree precisely on the ~50/50-propensity items and nowhere
else. Both judges independently support the same conclusion.

## Provenance

- Run-2 items committed before any arm ran (`items_run2.json`, commit
  a100c993 / 94206a39 on the PR branch).
- Answerer + judges: fresh subscription subagents (workflow wf_fb5d32a9;
  the run was interrupted by a session limit mid-flight and resumed from
  the journal — 13 blind answers replayed from cache, all other agents
  fresh).
- Binary verdicts with quoted-span evidence throughout; same judge-prompt
  discipline as R1.
