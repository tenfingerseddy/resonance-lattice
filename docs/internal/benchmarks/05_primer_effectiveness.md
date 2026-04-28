# Benchmark 5 — Session-start primer effectiveness

> **Status: shipped MVP scope.** 25-scenario / 5-lane harness committed
> at [`benchmarks/user_bench/primer_effectiveness/`](../../../benchmarks/user_bench/primer_effectiveness/).
> Result JSON at
> [`benchmarks/results/user_bench/primer_effectiveness.json`](../../../benchmarks/results/user_bench/primer_effectiveness.json).
> User-facing summary lives at
> [`docs/user/BENCHMARKS.md` § Session-start primer](../../user/BENCHMARKS.md#session-start-primer).

## What it measures

At the moment a session starts, which of `rlat`'s session-start
affordances actually moves the needle? `rlat` ships two complementary
primer surfaces and one always-available retrieval verb:

- **Code-base primer** — `rlat summary <km> -o
  .claude/resonance-context.md`. Extractive: top-N passages closest
  to the corpus centroid (Landscape) + per-source-file passage-count
  table (Structure). ~3 KB. Generated once per build.
- **Memory primer** — `rlat memory primer ./memory/ -o
  .claude/memory-primer.md`. Clustered semantic-tier entries from the
  cross-session memory store. ~2 KB. Regenerated when memory
  consolidates.
- **`rlat search`** — per-turn retrieval, top-k passages back as the
  Sonnet system prompt's context block.

Headline metrics:

- **Turn-1 correct rate** — fraction of opening questions answered
  correctly with a single answer model call.
- **Turn-2 correct rate** — fraction of follow-up questions answered
  correctly given the turn-1 context.
- **Both-correct rate** — fraction of conversations where both turns
  judged correct.
- **$ / question** — measured Sonnet 4.6 spend per scenario at $3/$15
  per 1M in/out tokens (one answer call per turn × 2 turns +
  one judge call per turn = 4 calls per scenario per lane).
- **Mean wall** — end-to-end seconds.

## Lanes

| Key | What's loaded |
|---|---|
| `cold` | No primer, no tools (control floor) |
| `primer_loaded` | Code-base primer (`.claude/resonance-context.md`) injected as system context |
| `memory_primer_loaded` | Memory primer (`.claude/memory-primer.md`) injected as system context |
| `both_primers` | Both primers concatenated as system context |
| `rlat_search_v1` | No primer; per-turn `rlat search --format context --top-k 5 --mode augment` |

`full_context` (whole-corpus dump) intentionally **omitted** —
[Bench 1](01_token_usage.md) established it as 67× pricier than
`rlat skill-context constrain` without changing the conclusion. Re-running
it here would have burned ~$30 of the $4 budget for no new finding.

## Test set: 25 hand-designed scenarios

Each scenario:

```jsonl
{
  "id": "pe01",
  "tier": 1,
  "topic": "project orientation",
  "q1": "What is rlat at a high level?",
  "gt1": "...",
  "q2": "Follow-up that builds on q1...",
  "gt2": "...",
  "source_files": ["docs/user/CORE_FEATURES.md", "..."]
}
```

Tier breakdown:

| Tier | Type | n | Where the answer lives | Predicted winner |
|---|---|---:|---|---|
| 1 | Project orientation | 5 | Code-base primer Landscape + Structure | `primer_loaded` |
| 2 | Specific factual | 10 | Deep in corpus, primer can't fit | `rlat_search_v1` |
| 3 | Cross-reference | 5 | Spans multiple files | mixed |
| 4 | Memory recall | 5 | Memory primer semantic entries only | `memory_primer_loaded` |

Test set committed at
[`benchmarks/user_bench/primer_effectiveness/tasks.jsonl`](../../../benchmarks/user_bench/primer_effectiveness/tasks.jsonl).

## Memory fixture

Tier 4 (memory recall) requires a populated memory store. The bench
ships one at
[`benchmarks/user_bench/primer_effectiveness/fixtures/memory/`](../../../benchmarks/user_bench/primer_effectiveness/fixtures/memory/):
10 hand-curated semantic-tier entries covering BSL-1.1 license,
Sonnet 4.6 judge model, the dropped cross-encoder reranker, the
OpenVINO Intel runtime, the encoder revision pin, the three memory
tiers, the cartridge → knowledge-model rename, the BEIR fiqa
optimise regression, the API/skill bench-equivalence, and the
simplify→codex→harness→board cadence.

The memory primer is generated from this fixture at build time:

```bash
rlat memory primer benchmarks/user_bench/primer_effectiveness/fixtures/memory \
  -o benchmarks/user_bench/primer_effectiveness/fixtures/memory_primer.md
```

The fixture-vs-live-memory split is deliberate: tier-4 ground truth
must not drift as Kane's actual `.claude/memory/` evolves day to day.

## Judge

Sonnet 4.6 grades each turn on a 4-state rubric:

- **correct** — load-bearing fact captured (relaxed) / fully-detailed
  (strict).
- **partial** — substantively right but missing a critical detail.
- **wrong** — different fact, contradicts ground truth, or invents.
- **refused** — explicit "I don't know" / "the corpus doesn't cover
  this".

For this bench we report the **relaxed** rubric (closer to user-
perceived usefulness). The strict-rubric grades are stored in the
result JSON for users with stricter evaluation needs.

## Cost discipline

- Budget cap: `--budget-usd 4` (target ~$2.50 actual, came in at
  $2.31).
- Incremental checkpoint: result JSON written after every trial with
  `partial: true` flag. Bench can be killed and reasoned about at any
  point.
- 25 scenarios × 5 lanes × (2 answer calls + 2 judge calls) = 500
  Sonnet calls. Mean spend: ~$0.005/call.

## Reproducibility

```bash
pip install rlat[bench]
rlat install-encoder
rlat build ./docs ./src -o resonance-lattice.rlat
rlat summary resonance-lattice.rlat -o .claude/resonance-context.md
rlat memory primer benchmarks/user_bench/primer_effectiveness/fixtures/memory \
  -o benchmarks/user_bench/primer_effectiveness/fixtures/memory_primer.md
export CLAUDE_API=sk-ant-...
python -m benchmarks.user_bench.primer_effectiveness.run \
  --km resonance-lattice.rlat \
  --primer .claude/resonance-context.md \
  --memory-primer benchmarks/user_bench/primer_effectiveness/fixtures/memory_primer.md \
  --tasks-file benchmarks/user_bench/primer_effectiveness/tasks.jsonl \
  --output benchmarks/results/user_bench/primer_effectiveness.json \
  --budget-usd 4
```

The harness re-runs the entire 5-lane × 25-scenario matrix; numbers
should reproduce within ±10pp on individual lanes (Sonnet judge
variance dominates at n=25).

## Headline finding

**Every primer type has a coverage profile**: it shines on the tier
its content was designed for and degrades to roughly cold elsewhere.

| Tier (turn 1) | cold | code primer | memory primer | both primers | rlat search |
|---|---:|---:|---:|---:|---:|
| 1 — orientation | 0/5 | **3/5** | 0/5 | **3/5** | 0/5 |
| 2 — specific factual | 0/10 | 2/10 | 2/10 | 3/10 | **8/10** |
| 3 — cross-reference | 0/5 | 0/5 | 1/5 | 1/5 | **2/5** |
| 4 — memory recall | 0/5 | 0/5 | **5/5** | **5/5** | 4/5 |

Aggregate (turn-1 correct): cold 0% / primer_loaded 20% /
memory_primer_loaded 32% / both_primers 48% / rlat_search 56%.

Combined-stack reading: load both primers at session start (free,
~5 KB) **and** keep `rlat search` available. Primers carry
orientation + memory recall; per-turn search picks up specific facts
the primers can't fit. For session-start questions that turn out to
need synthesis, `rlat deep-search` (the [hallucination bench](02_hallucination.md)
lane that hit 92.2%) is the right escalation.

## Token cost of each approach (measured)

Mean Sonnet 4.6 input/output tokens per call, averaged across the 25
scenarios:

| Approach | Input t1 | Input t2 | Output total | Tokens / scenario |
|---|---:|---:|---:|---:|
| `cold` | 90 | 280 | 295 | 664 |
| `rlat_search_v1` | 794 | 919 | 428 | 2,141 |
| `memory_primer_loaded` | 836 | 1,028 | 320 | 2,184 |
| `primer_loaded` | 1,798 | 2,089 | 487 | 4,374 |
| `both_primers` | 2,544 | 2,788 | 408 | 5,740 |

Primer-only input cost (input above the cold baseline of ~90 tokens
of base system prompt + question):

- Code primer ≈ **1,708 tokens** per call (~3 KB markdown).
- Memory primer ≈ **746 tokens** per call (~2 KB markdown).
- Both concatenated ≈ **2,454 tokens** per call.
- `rlat search` top-5 ≈ **704 tokens of passages** per call.

A user paying Sonnet 4.6 prices ($3/$15 per 1M in/out) can budget
session-start primers at:

- Code primer alone: ~$0.005 per call × ~10 calls/session = ~$0.05/session.
- Memory primer alone: ~$0.002 per call × ~10 calls/session = ~$0.02/session.
- Both primers: ~$0.007 per call × ~10 calls/session = ~$0.07/session.

These are flat overheads — they don't scale with corpus size. Compare
against [Bench 1](01_token_usage.md): a full-corpus dump for the
Fabric corpus is ~3.4M tokens (~$10 per call). The primer surface is
**~1,400× smaller** than a corpus dump for the corpus shape we test.

## Honest framing

- **MVP scope.** 25 scenarios, half the v1 plan's 50. Tier-level n=5
  is small; per-tier numbers are directional, not precise.
  Confidence intervals are wider than the visible gaps in tier 3
  (`memory_primer_loaded` 1/5 vs `cold` 0/5 is one-trial noise).
- **Judge variance.** Sonnet 4.6 grading itself; relaxed rubric.
  Strict-rubric grades ship in the result JSON.
- **One corpus.** Numbers measured on `resonance-lattice.rlat` (this
  project's docs + src, 3,506 passages, 126 files, 4f04867c). The
  *shape* of the result — primers cover orientation + memory
  recall, search covers specific facts, cold is hopeless — should
  generalise; exact magnitudes are corpus-specific.
- **Refresh discipline.** The bench was killed at $1.46 sunk because
  the primer was a day stale relative to the KM. The primer-vs-KM
  freshness invariant matters for any session-start measurement.
  Always regenerate `.claude/resonance-context.md` after `rlat
  refresh`.

## Locked references

- `feedback_rlat_benchmark.md` established that rlat is "an 18× faster
  context primer" vs Grep/Read for high-level orientation. Bench 5
  formalises that with reproducible numbers — and surfaces the
  per-tier coverage profile that the rough heuristic missed.
- The primer surfaces are documented in [docs/user/CLI.md § rlat
  summary](../../user/CLI.md) and [docs/user/CLI.md § rlat memory
  primer](../../user/CLI.md). The bench measures what those commands
  produce, not the commands themselves.
