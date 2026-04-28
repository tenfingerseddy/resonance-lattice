# Benchmark 1 — Token usage

## What it measures

Tokens, $ cost, and wall-time to a correct grounded answer. Headline metrics:

- **Tokens per correct answer** = (sum input + output tokens) / n_correct
- **$ per correct answer** = sum cost_usd / n_correct
- **Wall seconds per correct answer** = sum wall_seconds / n_correct

## Methodology

Fixed task set of 20 questions about the Resonance Lattice repository, each
with a hand-written ground-truth answer + cited source files (test set at
[`benchmarks/user_bench/token_usage/tasks.jsonl`](../../../benchmarks/user_bench/token_usage/tasks.jsonl);
20 of 50 planned in the v1 release; remaining 30 in v2.0.1). Every task is
answerable from the corpus.

Each `(approach, task)` runs a real Claude Sonnet 4.6 inference loop. We
record the full token telemetry from the Anthropic API (`usage.input_tokens`,
`usage.output_tokens`) and convert to $ via the published rate card
($3 / $15 per million for in / out).

After all approaches answer all tasks, a separate Sonnet 4.6 judge call grades
each candidate answer against the ground truth on a four-state rubric
(`correct` / `partial` / `wrong` / `refused`). Judge prompt is committed at
`benchmarks/user_bench/token_usage/run.py:JUDGE_PROMPT`. 10% of judge calls
are spot-validated by hand to confirm inter-rater agreement.

## Approaches

| Key | Approach | LLM calls per task |
|---|---|---|
| `rlat` | rlat skill-context → 1 Sonnet call | 1 |
| `grep_read` | Sonnet given grep + Read tools, multi-turn loop, capped at 6 turns | 1–6 |
| `full_context` | Concat all corpus text into the system prompt, 1 Sonnet call | 1 |
| `no_retrieval` | Ask Sonnet alone, no corpus | 1 |

### `rlat`

Calls `rlat skill-context resonance-lattice.rlat --query "<task>" --top-k 5
--token-budget 4000` via subprocess. Captures the markdown output (passages
+ HTML-comment citation anchors + grounding-mode header) and feeds it into a
single Sonnet call that asks the question and instructs the model to cite
passages. One inference, bounded input length.

### `grep_read`

Sonnet receives two tool definitions:

- `grep(pattern, path_glob?)` — runs `rg -n -S -m 25 <pattern> <path>` and
  returns up to 25 matching lines.
- `read(path)` — returns the full UTF-8 contents of a file relative to the
  corpus root, capped at 50K chars per call.

Sonnet alternates tool calls and assistant turns until it produces a final
text answer (no more tool-use blocks) OR hits the `max_turns=6` cap. Real
multi-turn agent loop — both input and output tokens accumulate across turns.

### `full_context`

Concatenates every text file under `docs/`, `src/`, plus `README.md` and
`CLAUDE.md` into a single system prompt block, capped at 600K chars
(~150K tokens, well within Sonnet 4.6's 200K context). Order: docs first,
then src — same priority a developer would skim. Tasks that need source-
code-level detail may not fit if the cap evicts the relevant file.

### `no_retrieval`

Sonnet answers from training-knowledge alone. Baseline that quantifies how
much corpus context actually adds (vs the LLM happening to know the
answer).

## Why this corpus

Resonance Lattice's own repository is the corpus because it's:

- **Reproducible** — anyone with the public repo can re-run and verify.
- **Out-of-distribution for Sonnet** — the project is recent, the test
  questions ask about specific revisions, locked numbers, and v2.0-specific
  flags that do not appear in public training data verbatim.
- **Fully covered by all four approaches** — small enough for `full_context`,
  searchable by `grep_read`, indexable by `rlat`.

Bench 1 is intentionally a single corpus. Cross-corpus generalisation is out
of scope; the question is "does rlat skill-context burn fewer tokens to get
to the same answer on the same corpus?", not "does rlat work on every
domain?". Cross-corpus signal is in benchmark 7 (BEIR fiqa) and the Fabric
optimised probe.

## Locked controls

- Encoder revision pinned (`e7f32e3c00f91d699e8c43b53106206bcc72bb22` —
  matches BENCHMARK_GATE.md).
- `rlat skill-context --top-k 5 --token-budget 4000` — match SKILLS.md
  example so users see the same numbers when reproducing.
- `max_turns=6` for `grep_read` — caps loop divergence; matches the average
  Claude Code session for "find the answer in this repo" tasks.
- `max_chars=600_000` for `full_context` — empirical cap that fits ~150K
  tokens with headroom; matches what a developer would prepend.
- Judge model = answer model = Sonnet 4.6. We accept the LLM-as-judge bias
  trade-off; mitigated by inter-rater 10% spot-validation.

## Reproducibility

```bash
pip install rlat[bench]
rlat install-encoder
rlat build ./docs ./src -o resonance-lattice.rlat
export CLAUDE_API=sk-ant-...
python -m benchmarks.user_bench.token_usage.run \
  --output benchmarks/results/user_bench/token_usage.json \
  --budget-usd 100
```

Pilot run with 5 tasks for harness validation (~$5):

```bash
python -m benchmarks.user_bench.token_usage.run --n-tasks 5 --budget-usd 5
```

## Tolerance for re-runs

LLM-judge variance and Sonnet sampling produce ±5% noise on
`tokens_per_correct` and `$_per_correct`. Treat any committed number as
"true within 5%"; significant ratio differences (1.5× or larger between
approaches) survive that noise comfortably.

## Honest framing

- `rlat` should win on tokens-per-correct on this corpus by a large factor
  vs `full_context` and `grep_read`. We expect the result; this benchmark
  is the receipt.
- `no_retrieval` may surprise — Sonnet 4.6 has read enough open-source code
  that it can answer some rlat-meta questions from training. Where it can,
  the bench reports honestly. Where the answer is wrong, it shows up as
  `wrong` (not `refused`) in the rubric.
- We don't measure cross-encoder rerankers or hosted vector DBs (Pinecone,
  Weaviate). The first is dropped from rlat's default path; the second is
  network-bound and not directly comparable. Both can be added by users
  who want to extend.

## Related work surfaced from prior measurement runs

- `feedback_rlat_benchmark.md` — prior measurement: rlat is an 18× faster
  context primer vs Grep/Read for high-level orientation. This bench
  formalises that comparison.
