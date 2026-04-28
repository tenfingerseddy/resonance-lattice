# Bench 2 v3 — distractor failure root cause analysis

**Date**: 2026-04-27
**Source data**: `benchmarks/results/user_bench/hallucination_fabric_v3.json`,
12 distractor + 51 answerable questions on Microsoft Fabric corpus.

## The headline question

Bench 2 v3 surfaced **33.3% distractor hallucination across all rlat
lanes** (4 of 12 distractor questions answered when the right
response was refusal). The roadmap flagged this as a v2.0.1 follow-up:
"explore whether a top-1-score floor on distractor-style queries is
the right gate."

This doc captures the analysis that proposal needs. The headline:
**a top1_score or top1_top2_gap floor cannot discriminate distractors
from answerable questions on this corpus.** A different mechanism is
needed.

## Per-distractor retrieval profile

For each distractor question, the live `rlat search --top-k 5` against
`fabric-docs-v2.rlat`:

| ID | Subtype | top1 | top1_top2_gap | rlat outcome (any lane wrong?) | Notes |
|---|---|---:|---:|---|---|
| fb31 | nonexistent (Quantum Data Lakes) | 0.811 | 0.013 | no — refused all 3 rlat lanes | gap small ✓ |
| fb32 | nonexistent (F4096 SKU) | 0.802 | 0.002 | no — refused all 3 | gap small ✓ |
| fb33 | adjacent_product (Personal Gateway) | 0.847 | 0.001 | no — refused all 3 | gap small ✓ |
| fb34 | adjacent_product (Synapse Dedicated SQL Pools) | 0.857 | (no parse) | knowledge=partial | top-1 = migration page |
| fb35 | nonexistent (Spark Vector Engine) | 0.766 | 0.017 | no — refused all 3 | gap small ✓ |
| fb36 | renamed (Power BI Premium per Capacity) | 0.888 | 0.003 | no — correct refusal-with-explanation | gap small ✓ |
| **fb37** | **nonexistent (Materialized View Express → MLV swap)** | **0.787** | **0.018** | **YES — ALL 3 rlat lanes hallucinated `wrong`** | name-aliasing trap |
| fb39 | nonexistent (Fabric "cartridge" terminology) | 0.857 | 0.019 | no — refused all 3 | |
| fb40 | nonexistent (specific F1 pricing) | 0.777 | 0.000 | no — refused all 3 | gap zero ✓ |
| fb61 | nonexistent (Snowflake Mirror Bundle) | 0.761 | 0.008 | no — refused all 3 | gap small ✓ |
| fb62 | adjacent_product (Service Connect wizard) | 0.858 | 0.027 | no — refused all 3 | |
| fb63 | nonexistent (F2048 SKU) | 0.797 | (no parse) | no — refused all 3 | |

**The four hallucinated trials** in the bench-2 v3 distractor count
(per `by_approach.distractor_hallucinated`) all map to **fb37 across
the three rlat lanes** — augment, knowledge, constrain_verified all
gave the MLV-default-action answer to the MVE-default-action question.

## Why gap-based gating doesn't separate distractors from answerables

The pre-existing v2.0.1 hypothesis was: "distractors have low
top1_top2_gap because no passage uniquely matches; answerables have
high gap because one passage is clearly the answer." Threshold the
gap and refuse below it.

This is **wrong on a paraphrase-rich documentation corpus**. Measured
distributions:

| Set | n | mean | median | min | max |
|---|---:|---:|---:|---:|---:|
| Answerable | 51 | 0.024 | 0.015 | 0.001 | 0.095 |
| Distractor | 10 (parseable) | 0.011 | 0.011 | 0.000 | 0.027 |

Threshold sensitivity (refuse when gap < threshold):

| Threshold | Answerables refused | Distractors refused |
|---:|---:|---:|
| 0.005 | 18% | 33% |
| 0.010 | 37% | 50% |
| 0.020 | 59% | 92% |
| 0.030 | 71% | 100% |
| 0.050 | 84% | 100% |

**No threshold separates the populations cleanly.** Catching all 12
distractors (gap < 0.030) would refuse 36 of 51 answerables —
collapsing answerable accuracy from 55% to 18%. Catching even half
the distractors would refuse 19+ answerables.

**Why**: Microsoft Fabric documentation has many similarly-phrased
passages on overlapping topics. Even a correct answerable question
finds 3–5 passages that all phrase the relevant fact slightly
differently and score within 0.01–0.05 of each other. The "tight gap"
signal that should mean "uncertain retrieval" instead means "the
corpus has redundant / paraphrased content" — which on a real-world
documentation corpus is a *feature*, not a bug.

## Why top1_score absolute floor doesn't work either

| Set | top1 mean | top1 median | top1 min | top1 max |
|---|---:|---:|---:|---:|
| Answerable | (computed live) | ~0.83 | ~0.65 | ~0.92 |
| Distractor | ~0.82 | ~0.82 | 0.76 | 0.89 |

Distractor top-1 scores **overlap completely** with answerable top-1
scores. The retrieval thinks distractor questions match the corpus
just as well as answerable ones, because:
- "Materialized View Express" semantically matches "Materialized Lake View"
- "Power BI Premium per Capacity" semantically matches "Fabric capacity"
- "Quantum Data Lakes" matches generic Fabric overview content

A `top1_score < 0.50` floor would catch zero distractors here.

## The actual fix: name-verification

The pattern across hallucinated distractors:

1. The question contains a **distinctive proper noun or acronym** that
   doesn't exist in the corpus: `MVE`, `Quantum Data Lakes`, `F4096`,
   `Snowflake Mirror Bundle`, etc.
2. The corpus surfaces passages that share the **head noun** (`Materialized`,
   `Lakes`, `F-series`, `Snowflake Mirror`) at high similarity.
3. The LLM, reading those passages, answers as if the question used
   the corpus's name.
4. fb37 is the case where this paid out as a hallucination because MLV's
   default-action documentation is *definitive* and the LLM happily
   parroted it without checking that "MVE" and "MLV" aren't the same
   thing.

The fix isn't a retrieval threshold. It's a **prompt-level
disambiguation step** in the grounding directive. Concrete proposal:

### Augment / knowledge / constrain header addendum (v2.0.1)

Add to every grounding-mode header:

> **Name verification**: before answering, check whether any
> distinctive proper noun, acronym, or product name from the
> question appears verbatim in the retrieved passages. If the
> question references a specific named entity (`X`) but the passages
> only describe an adjacent entity (`Y`), refuse — say "the corpus
> covers `Y` but not `X`; the question may be about a different
> product."

### Why this works (hypothesis to test)

The four trials that hallucinated fb37 ALL produced answers
mentioning "MLV" (the corpus name) without quoting "MVE" (the
question's name). A prompt-level name-check would catch that the
question's distinctive token isn't in the passages and force a
refusal. Cost: zero LLM calls, zero retrieval work — just a richer
system directive.

### Cost of being wrong

If the name-check is too aggressive it could refuse legitimate
synonyms ("Microsoft Fabric Lakehouse" vs corpus's "Fabric
Lakehouse"). Mitigation: scope the check to **acronyms and
non-stopword proper nouns that don't appear as substrings in the
passages**, not the full question. This is a small Python helper, not
a model call.

### Cheap empirical test

To test pre-deployment, build a string-matching distractor-floor lane
in the bench harness:
1. Run rlat retrieval as usual.
2. Extract distinctive tokens from the question (capitalised words,
   ALL-CAPS acronyms, alphanumeric IDs like `F4096`).
3. Check if each distinctive token appears as a substring in any
   retrieved passage.
4. If a distinctive token from the question is missing from all
   retrieved passages: prepend a hard refusal directive.
5. Re-run bench 2.

Estimated lift on the distractor axis: **catches 9-10 of 10
nonexistent distractors (those using fake product names) without
touching answerable accuracy**. fb37 (MVE→MLV) is the canary case
that proves the design.

### What this doesn't fix

- **fb34 (Synapse Dedicated SQL Pools)**: real Microsoft product, the
  question phrasing is plausible. The corpus correctly returns
  migration pages; the LLM correctly explains the migration path. The
  judge graded `partial` because the answer doesn't say "Fabric
  doesn't have dedicated SQL pools, you migrate to Warehouse." This
  is a borderline case where the *correct* user-facing answer is
  what augment said. We may want to mark this trial `correct` on
  re-judge — it's a question-set / rubric issue, not a hallucination.

- **fb36 (Power BI Premium per Capacity)**: similar. Real legacy
  product name, corpus has migration content, all rlat lanes
  correctly explained the rename. Already graded `correct`.

## Recommendation

Ship the prompt-level name-verification check in v2.0.1 as part of
all three grounding modes. The threshold-based distractor floor
hypothesis is falsified by the data; document the falsification and
move on. Test the name-check via a new bench-2 lane.

**Effort**: ~50 lines of Python (token extraction + presence check) +
~30 lines of system-prompt addendum. ~2 hours.
**Cost**: $0 to deploy, ~$1 to re-bench against this Fabric test set.

## Open question for the question set

fb34 and fb36 are arguably question-set issues, not rlat issues. The
"correct" response for "How do I configure Synapse Dedicated SQL
Pools in Fabric?" is to redirect to migration — which is what augment
did. Should we re-grade these as `correct` retrospectively? Doing so
would lift the distractor-correct-refusal rate from 67% → 75% on the
existing data, without changing any code.
