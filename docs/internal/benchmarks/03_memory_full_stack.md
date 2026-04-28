# Benchmark 3 — Memory full-stack

> **Status: deferred to v2.0.1.** Methodology locked here; harness +
> committed numbers ship in v2.0.1. v1's user-doc surfaces this bench
> as a "coming soon" link that points here.

## What it measures

End-to-end task accuracy when the rlat memory subsystem is in the loop —
not retrieval recall, **task completion**. Bridges the gap between the
existing LongMemEval v14 measurement (R@5 0.924 / MRR 0.919) and what
users actually care about: did the assistant answer correctly, given
prior session context?

Two evaluation surfaces:

1. **LongMemEval task accuracy** — re-run the v14 split with an LLM-
   answering layer wrapped around retrieval. Existing harness measures
   recall; this bench measures answer correctness via a locked rubric.
2. **Multi-session resume** — 20 hand-designed multi-day workflows where
   the assistant must resume context across sessions. Measures whether
   the layered-memory + retention + consolidation + primer machinery
   actually helps real cross-session work.

Headline metrics:

- LongMemEval task accuracy (full 500 split)
- Multi-session resume accuracy
- Primer-driven token savings at session start: tokens-to-correct-answer
  with vs without `.claude/memory-primer.md` loaded

## Approaches

| Key | Approach |
|---|---|
| `full_stack` | working/episodic/semantic + retention + consolidation + primer at session start |
| `primer_only` | primer at session start; no recall queries during session |
| `chatgpt_memory` | OpenAI memory API where allowed (skipped if API restrictions block reproduction) |
| `naive_append` | concatenate every prior conversation into context |
| `no_memory` | session starts cold |

## Why deferred to v2.0.1

The hand-designed 20 multi-session workflows are the bulk of the work — a
multi-session workflow is a 5-10 turn conversation where the user goal
spans days. Each takes ~30 min to design, validate, and rubric. 20 × 30
min = 10 hours of design alone. Plus the LongMemEval LLM-answering wrap
needs careful integration with the existing v14 harness.

Estimated total: 5 days dev + ~$200-300 API. Doesn't fit Stage 1's
~7-day window without compromising bench 1 + 2 test-set quality.

## Locked references for the v2.0.1 build

- LongMemEval v14 ship recipe: `project_longmemeval_ku_filter.md`
  (R@5 0.924 / MRR 0.919 on stratified 500). Locked because every
  attempt to lift it (multi-granularity session vectors, post-hoc KU
  filter) regressed.
- LoCoMo cartridge baseline: 68.2% (full 10 conversations) per
  `project_locomo_feature_transfer.md`. "Full-stack" features
  (subgraph, CE, diversify) catastrophically regressed; cartridge-only
  is the ship recipe.
- Existing harnesses to wrap, not reimplement:
  - `benchmarks/bench_longmemeval.py` (1073 lines)
  - `benchmarks/bench_locomo.py` (1440 lines)

## Test-set design protocol for the multi-session 20

Each workflow carries:

- Workflow name + ground-truth completion criterion
- 5–10 turns spanning ≥2 simulated sessions (timestamped)
- 2-3 distractor turns in between (unrelated questions to verify the
  memory system isn't blindly recalling)
- Expected behaviour: assistant correctly resumes the goal in turn N,
  citing the original turn

Workflow-design pattern (one example):

> **Goal: "I started refactoring the auth middleware in session 1; I'm
> back two days later — pick up where we left off."**
> - Session 1 turns 1–4: refactor planning, file paths, decisions
> - Session 1 turn 5: "Got pulled away, will continue tomorrow"
> - Session 2 turn 1: unrelated question (database schema)
> - Session 2 turn 2: unrelated question (CSS bug)
> - Session 3 turn 1: "Where were we on the auth refactor?"
> - Session 3 turn 2: "What was that file path you suggested?"
>
> Assistant must produce session-1 file paths + decision rationale.
> `full_stack` should ace this; `naive_append` may bury the relevant
> context; `no_memory` fails.

## Reproducibility (when it ships)

```bash
pip install rlat[bench]
rlat install-encoder
export CLAUDE_API=sk-ant-...
python -m benchmarks.user_bench.memory_full_stack.run_longmemeval \
  --output benchmarks/results/user_bench/memory_longmemeval.json \
  --budget-usd 200
python -m benchmarks.user_bench.memory_full_stack.run_multi_session \
  --output benchmarks/results/user_bench/memory_multi_session.json \
  --budget-usd 100
```

## Honest framing

- LongMemEval task accuracy may be lower than retrieval R@5 — even with
  perfect retrieval, the LLM can mishandle the context. We report both.
- The multi-session bench is genuinely hard to make reproducible because
  it depends on the assistant model's sampling. Variance is higher than
  single-shot retrieval benches; we'll report seed + temperature locks
  and run 3 trials per workflow with median scoring.
- ChatGPT-memory-API comparison may be impossible to reproduce externally
  due to the API's privacy model; we'll note where comparison falls back
  to single-vendor.
