# Benchmark 2 — Hallucination reduction

## What it measures

How often does the consumer LLM fabricate when grounded in different ways?
Specifically:

- **Hallucination rate** = `n_wrong` / `n_answerable` (lower is better)
- **Refusal-correctness rate** = `n_refused_correctly` / `n_distractors`
  (higher is better — the LLM should refuse the unanswerable ones)
- **Drift-detection rate** = `n_drift_flagged` / `n_drifted_subset`
  (higher is better — only `rlat` approaches see the `[drifted]` tag)
- **Citation accuracy** = `n_citations_resolving_to_supporting_passage` /
  `n_citations_in_correct_answers` (higher is better)

## v2.0 launch bench (Fabric, 63 questions, 11 lanes)

**The shipped headline numbers** in the public-facing
[`docs/user/BENCHMARKS.md`](../../user/BENCHMARKS.md) come from the v4
matrix on the Microsoft Fabric documentation corpus, not the rlat-repo
test set described below.

- **Corpus**: `fabric-docs-v2.rlat` — public Microsoft Fabric documentation,
  2,261 markdown files, 62,914 passages, all `ms.date` from 2019 through
  2026-03-28. Picked for the test because it's a **partially-known corpus**
  for Sonnet 4.6 — the Fabric service has been in market since 2023, so
  the model has substantial Fabric training data, but the corpus contains
  559 files dated post Sonnet's January 2026 cutoff and another 679 files
  in the 2025-09 to 2026-01 fuzzy zone.
- **Test set**: 63 hand-written questions
  ([`benchmarks/user_bench/hallucination/fabric_tasks.jsonl`](../../../benchmarks/user_bench/hallucination/fabric_tasks.jsonl)).
  Composition: 51 answerable (across recency tiers — post-cutoff, fuzzy
  zone, stable pre-2024) + 12 distractors (fake F SKUs, made-up product
  names, Power-BI-only features asked as Fabric features). Ground-truth is
  quoted verbatim from specific dated source files.
- **Lanes**: 11 = 3 grounding modes (`augment` / `constrain` / `knowledge`)
  × 3 retrieval shapes (single-shot / multi-query rewrite /
  `rlat deep-search` 4-hop loop) + LLM-only baseline +
  LLM+grep/glob/read_file baseline.
- **Result**: `rlat deep-search --mode knowledge` 92.2% answerable
  accuracy at 0% answerable hallucination at $0.009/q; default
  `rlat deep-search --mode augment` 92.2% / 2.0% at $0.009/q;
  `rlat search --mode constrain` (single-shot) 91.7% distractor refusal
  (highest); LLM-only 56.9% / 19.6%. Full table at
  [BENCHMARKS.md §Hallucination reduction](../../user/BENCHMARKS.md#hallucination-reduction).
- **Result JSONs**:
  [`hallucination_fabric_11lane_relaxed.json`](../../../benchmarks/results/user_bench/hallucination_fabric_11lane_relaxed.json)
  (relaxed rubric — the headline) and
  [`hallucination_fabric_11lane.json`](../../../benchmarks/results/user_bench/hallucination_fabric_11lane.json)
  (strict rubric).

The v3 single-shot-only run on a 40-Q rlat-repo subset (described in the
following section) was the design predecessor; v3 numbers are preserved
in [`02_fabric_failure_analysis.md`](02_fabric_failure_analysis.md) for
historical reference but are superseded by the v4 11-lane run for every
public-facing claim.

## v2.0.1 expansion plan — rlat-repo 100-question test set

A separate 100-question test set on the `rlat` repository itself ships
in v2.0.1; this is the design that drove the v3 single-shot 40-Q run
documented in [`02_fabric_failure_analysis.md`](02_fabric_failure_analysis.md).
Test set at [`benchmarks/user_bench/hallucination/tasks.jsonl`](../../../benchmarks/user_bench/hallucination/tasks.jsonl)
(40 of 100 in v3; remaining 60 in v2.0.1). Composition:

- **30 answerable** (`kind=answerable`): real questions with cited ground
  truth in the corpus.
- **10 distractors** (`kind=distractor`): plausible-sounding but
  factually-unanswerable from this corpus (e.g. "What blockchain consensus
  algorithm does rlat use?" — rlat has nothing to do with blockchain).
  Ground truth is the explicit refusal.
- **(planned for v2.0.1)** 10 deliberately-drifted: a subset of the
  answerables where source files are mutated post-build, so the recorded
  `content_hash` no longer matches live bytes. Tests whether the
  `[drifted]` tag in skill-context output actually causes the LLM to
  hesitate or refuse.

The rlat-repo bench is the right complement to the Fabric bench: Fabric
exercises *partially-known corpus* failure modes; the rlat-repo corpus
is *zero-knowledge* (the project is brand-new, Sonnet has never seen
the codebase). Different failure mode profiles, both load-bearing.

Each `(approach, task)` runs a Sonnet 4.6 inference loop and a separate
Sonnet judge call grades the answer using a four-state rubric:

- `correct` — answers the question with the same factual content as ground
  truth (paraphrasing OK, partial detail not OK).
- `partial` — gist right but missing a load-bearing detail or contains a
  small error.
- `wrong` — different fact, contradicts ground truth, or unrelated.
- `refused` — explicitly says "I don't know" / "I cannot answer".

## Approaches

| Key | Approach | What it tests |
|---|---|---|
| `constrain_verified` | `rlat search --format context --mode constrain --verified-only` → Sonnet | strongest grounding contract |
| `augment` | `rlat search --format context --mode augment` (default) → Sonnet | default user behaviour |
| `knowledge` | `rlat search --format context --mode knowledge` → Sonnet | passages-as-supplement framing |
| `plain_rag` | Same passages but no grounding directive → Sonnet | what every other RAG tool does |
| `no_retrieval` | Sonnet alone | absolute baseline |

The five approaches share the same retrieval results — they differ only in
how the consumer LLM is instructed. Same passages, same encoder, same
top-k. The hypothesis: the `--mode constrain` directive plus
`--verified-only` materially changes how the LLM treats borderline
evidence.

## Why drifted-subset is unique to rlat

Most RAG libraries hand the LLM passages with no provenance signal. rlat
ships every hit with `(source_file, char_offset, char_length, content_hash,
drift_status)`. When the source file is mutated post-build, the hit's
status flips to `drifted`. `rlat skill-context` includes this in the HTML-
comment header for the LLM to see. `--strict` aborts loading entirely if
any drift is present.

The drifted-subset measurement quantifies whether this contract works in
practice — does the LLM actually treat `[drifted]` passages with
appropriate caution?

This is the single benchmark in the suite that no other tool can
reproduce, because no other tool even has the contract.

## Approaches × question-type matrix

|                       | answerable                           | distractor                                          | drifted (v2.0.1)                              |
|-----------------------|--------------------------------------|-----------------------------------------------------|-----------------------------------------------|
| `constrain_verified`  | should answer correctly              | should refuse (top-K below threshold)               | should refuse (drifted passages excluded)     |
| `augment`             | should answer correctly              | may hallucinate                                     | may answer (drifted not blocked, just flagged)|
| `knowledge`           | should answer correctly              | may hallucinate (LLM may blend training)            | may answer                                    |
| `plain_rag`           | should answer correctly              | likely hallucinates (no directive)                  | doesn't see drift signal                      |
| `no_retrieval`        | hits-and-misses (training knowledge) | sometimes refuses, sometimes hallucinates           | n/a — no corpus                               |

## Locked controls

- All `rlat` approaches use `--top-k 5 --token-budget 4000`.
- All approaches use the same Sonnet 4.6 model + temperature.
- Distractor questions are hand-written to be plausible-sounding but
  rejected by anyone with corpus context. Ground-truth refusal text is
  committed.

## Reproducibility

```bash
pip install rlat[bench]
rlat install-encoder
rlat build ./docs ./src -o resonance-lattice.rlat
export CLAUDE_API=sk-ant-...
python -m benchmarks.user_bench.hallucination.run \
  --output benchmarks/results/user_bench/hallucination.json \
  --budget-usd 150
```

## Honest framing

- This bench expects to show large hallucination-rate gaps between
  `constrain_verified` and `no_retrieval` (≥30 percentage points). If it
  doesn't, that's a real product finding — we publish honestly.
- Distractor refusal rates may surprise. Sonnet is well-calibrated; even
  the `no_retrieval` baseline may correctly refuse some distractors. The
  bench reports the rate, not a hand-crafted narrative.
- The drifted-subset is excluded from v1 (additional infrastructure
  needed: post-build mutation harness + cleanup). v2.0.1 ships the full
  100-question test set with the 10 drifted entries.

## Related work surfaced from prior measurement runs

- `project_verified_retrieval_292.md` — verified-retrieval contract
  (every hit carries source coords + content_hash + drift_status). This
  bench operationalises the value claim.
