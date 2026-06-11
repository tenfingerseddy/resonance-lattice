# E2c — 4-gate extractor re-validation (pre-registered design)

**Status**: design + sessions + ground truth locked before any extractor
run. **Date**: 2026-06-10. Gates the waking of the passive attribute
miner (PR #365): the measured E2b numbers (precision 0.83 / recall 0.95)
belong to the 3-gate, coding-flavoured prompt; the shipped prompt is
4-gate (adds the person-fact scope gate) and domain-neutral, and the
channel now feeds USER turns. None of those changes is validated. This
bench re-measures on the real production path.

## Method

- 10 synthetic sessions written as **user-turn text** (what
  `_scrub_user_turns` feeds the miner): 4 software/Fabric, 3 garden,
  3 legal practice — the domain-neutrality claim is tested, not assumed.
- Each session states 1–2 **ground-truth world facts** in passing and
  carries 2–4 **traps** across six classes:
  `transient` (true only right now), `discovered` (surfaced mid-session),
  `person` (about the speaker — the GATE 4 privacy class),
  `quoted_assistant` (the user quoting the assistant),
  `hypothetical`, `corpus_fact` (from a document, not the user's world).
- The extractor under test is the REAL `extract_attributes` via the REAL
  `default_client` (Sonnet, the production hook model) — not a stand-in.
- **Deterministic grading, zero judge noise**: every ground-truth fact
  lists lowercase `terms` that must ALL appear in an emission to count
  as recalled; every trap lists `terms` whose ANY-match attributes an
  emission to that trap. Emissions matching neither are honest false
  positives, listed for review.

## Pre-registered bars

1. **Precision ≥ 0.83** overall (no worse than the 3-gate ancestor).
2. **Recall ≥ 0.85** of ground-truth world facts overall.
3. **Person-fact emissions = 0** (hard bar — GATE 4 is the privacy
   contract; one leak fails the bench regardless of the averages).
4. **No domain below 0.75 precision** (domain-neutrality).

FAIL on any bar → the miner stays dormant and the prompt iterates; the
bench file is the falsification record either way.

## Artifacts

- `sessions.json` — sessions + ground truth + traps (committed first).
- `run_gate.py` — the runner + deterministic grader.
- `run_results/` — emissions + grading + verdict per run.

## Run-1 verdict (2026-06-10) — PASS all four bars

`run_results/run1_2026-06-10.json`; pre-registration committed at
04de82bb before the run; ~$0.07 API.

| Bar | Result |
|---|---|
| Precision ≥ 0.83 | **0.86** (19 matched / 22 emitted) → PASS |
| Recall ≥ 0.85 | **1.00** (19/19 world facts) → PASS |
| Person leaks = 0 | **0** — all seven person traps dropped → PASS |
| Every domain ≥ 0.75 precision | software 0.80, garden 1.00, legal 0.83 → PASS |

The three false positives are all borderline-defensible captures, none
personal: one hypothetical restated as a present-tense fact ("capacity
is below F64") and two corpus-stated facts the user relayed as operative
constraints (UDF 200MB cap, seven-year file retention). GATE 2's
doc-vs-world line is the next tightening lever if these matter; they
pollute mildly, they don't leak.

Post-run review fix: the grader originally checked ground-truth matches
before trap terms, so a compound emission (world fact + person fact in
one sentence) could in principle have hidden a leak. The grader now
scans every emission for person-trap terms regardless of GT match;
re-grading the committed run-1 emissions under the fixed scan still
finds **0** leaks across all 22 emissions / 7 person traps.

**Consequence: the 4-gate domain-neutral extractor matches its 3-gate
ancestor's precision with full recall and an airtight scope gate — the
miner's last validation gate is cleared.** Remaining dormancy is product
wiring (explicit `km_path`, auto-discovery deliberately off), not a
correctness blocker.
