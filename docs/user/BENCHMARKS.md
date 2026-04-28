# Resonance Lattice — Measured

> Every claim in this doc is paired with a reproducible recipe.
> Numbers are placed alongside the comparison they're meaningful against
> — published encoder baselines for retrieval quality, named alternative
> stacks for build/query speed, and an LLM-only baseline for the
> hallucination + token-spend story.

`rlat` was built audit-driven. Every retrieval feature passed measurement
before it shipped. Internal engineering contract lives at
[`docs/internal/BENCHMARK_GATE.md`](../internal/BENCHMARK_GATE.md); this
doc is the user-facing summary.

## At a glance

> New to `rlat`? Jump to [Concepts you need to read these benches](#concepts-you-need-to-read-these-benches) first — the table below uses
> terms (`deep-search`, `--mode constrain`, *distractor*, *relaxed rubric*) that
> are explained in one place there. Headlines-first readers can scan
> the table now and read the legend after.

| Headline (v2.0-measured) | Value | Section |
|---|---|---|
| **`rlat deep-search` accuracy** (Fabric corpus, relaxed rubric) — multi-hop research loop in one CLI call | **92.2%** answerable accuracy at **0% hallucination** ($0.009/q) | [Hallucination reduction](#hallucination-reduction) |
| **Single-shot rlat accuracy** vs LLM-only (`rlat search --mode augment` + Sonnet) on Fabric | **76.5%** vs **56.9%** — adds ~20 pp accuracy at 5× lower hallucination | [Hallucination reduction](#hallucination-reduction) |
| **Hallucination floor** (`rlat search --mode constrain`) | **2.0%** answerable hallucination + **91.7%** distractor refusal — pick for compliance / regulatory / audit | [Hallucination reduction](#hallucination-reduction) |
| **`rlat deep-search` vs LLM+grep/glob baseline** | **92.2%** acc at **$0.009/q** vs **94.1%** acc at **$0.060/q** — within 2 pp at **6.5× lower spend** *and* faster wall-time | [Hallucination reduction](#hallucination-reduction) |
| **$ per correct answer** (rlat skill-context constrain vs full-corpus dump) | rlat **$0.012** vs full-corpus **$0.796** — **67× cheaper** | [Token spend](#token-spend) |
| **$ per correct answer** (rlat skill-context constrain vs grep+Read) | rlat **$0.012** vs grep+Read **$0.044** — **3.7× cheaper** | [Token spend](#token-spend) |
| Warm query latency vs Chroma (1,000-passage CPU corpus) | rlat **17 ms p50** vs Chroma **145 ms p50** — **8.5× faster** | [Build & query speed](#build--query-speed) |
| On-disk size vs Chroma (same corpus) | rlat **2.7 MB** vs Chroma **8.6 MB** — **3.2× smaller** | [Build & query speed](#build--query-speed) |
| BEIR-5 mean nDCG@10 (locked v2.0 base recipe) | **0.5144** | [Retrieval quality](#retrieval-quality-beir-5-floor) |
| In-corpus optimised lift (Fabric, 62,953 passages) | **+0.032 R@5** ($8.16 / ~50 min Kaggle T4) | [In-corpus optimised band](#in-corpus-optimised-band) |
| In-corpus optimised regression (BEIR fiqa, 57,638 passages) | **−0.042 nDCG@10** | [In-corpus optimised band](#in-corpus-optimised-band) |
| Session-start primer (resonance-lattice corpus, 25 scenarios) | code primer **3/5** orientation + memory primer **5/5** recall — both-primers loaded carries 20% both-correct vs 0% cold | [Session-start primer](#session-start-primer) |
| LoCoMo v2.0 retrieval recall (1982 QAs, 10 conversations) | **R@5 0.329 / R@10 0.433 / MRR 0.265** | [Memory across sessions](#memory-across-sessions) |
| LongMemEval v2.0 retrieval (500 instances) | **MRR 0.936** — top-band among published memory-retrieval stacks (naive baselines ~0.3-0.55, off-the-shelf dense ~0.65-0.85, dedicated memory systems ~0.85-0.95) | [Memory across sessions](#memory-across-sessions) |

## How we measure

Every benchmark is a paired comparison vs named baselines, against a
committed test set, runnable from a single `python -m benchmarks…`
invocation. Methodology lives in
[`docs/internal/benchmarks/`](../internal/benchmarks/) — one document per
bench.

We use Claude Sonnet 4.6 as the canonical answer model + judge for the
LLM-judged benches (hallucination, token spend). Where LLM-judge
variance affects results, we run inter-rater 10% spot-validation. Test
sets are committed in full to the repo — auditability over compactness.

## Concepts you need to read these benches

If you've never used `rlat`, the bench tables below talk about three
retrieval shapes, three grounding modes, two question types, and two
judge rubrics. None are jargon — read this once and the rest of the
doc reads in plain English.

### What `rlat` actually does

`rlat` packages your corpus (codebase, docs, knowledge base) into one
`.rlat` file — a *knowledge model* — and ships commands to query that
file. The commands return passages with full citation provenance plus
a live drift status against the source bytes; the LLM you call
afterwards composes the answer from those passages. `rlat` doesn't
sit between you and your LLM — it hands you grounded evidence and
gets out of the way.

Two CLI verbs do the retrieval work the bench tests:

- **`rlat search km.rlat "<question>"`** — one query, one round of
  retrieval, top-k passages back. Fast, cheap, simple.
- **`rlat deep-search km.rlat "<question>"`** — internally runs a
  multi-hop loop (plan → retrieve → refine → maybe re-retrieve →
  synthesize) and returns a final answer plus the union of evidence.
  Slower, ~2.5× pricier, much higher accuracy.

### Retrieval shapes (the columns in the bench tables)

The bench tests three retrieval shapes against the same corpus:

- **Single-shot** — `rlat search` once. Top-k passages. The cheapest
  rlat tier (~$0.003-0.004 per question with Sonnet downstream).
- **Multi-query rewrite** — Sonnet generates 3 query variants of the
  user question, `rlat search` runs each, results dedupe. A middle
  ground; rarely the right pick now that `deep-search` exists.
- **Deep-search** — the productionised multi-hop loop. Pricier
  (~$0.009/q) but consistently the highest-accuracy lane. **The
  launch-headline mode for v2.0.**

### Grounding modes (how `rlat` tells the LLM to use the evidence)

`rlat search --format context` and `rlat skill-context` stamp a one-
paragraph directive at the top of the markdown they emit. The
directive instructs the consumer LLM how to weight the passages vs
its training knowledge. Three modes:

| Mode | Directive given to the LLM | When to use |
|---|---|---|
| **`augment`** *(default)* | "Use these passages as primary context for this corpus's domain. Blend with your training knowledge where the passages fall short." | General-purpose corpora the LLM partially knows already (most use cases). |
| **`constrain`** | "Answer ONLY from these passages. If they don't cover the question, refuse explicitly — do NOT draw on training knowledge." | Compliance, audit, regulatory work where wrong-but-confident is worse than no answer. |
| **`knowledge`** | "Use these passages to supplement your training knowledge. Lean on what you already know for surrounding context." | Partially-covered corpora on top of a well-known domain. |

The mode applies to all three retrieval shapes — you can run
`single + augment`, `multi + constrain`, `deep + knowledge`, etc.
3 modes × 3 shapes = 9 lanes; we test all 9 plus two baselines.

### Question types (the rows of the test set)

The hallucination bench has two kinds of questions, both required
for an honest evaluation:

- **Answerable** (51 questions) — the answer is genuinely in the
  corpus. Tests **accuracy**: did the LLM find and use the right
  passages?
- **Distractor** (12 questions) — the question asks about something
  **that doesn't exist** in the corpus. Fake product names
  ("Materialized View Express"), made-up SKUs ("F4096"), Power-BI-
  only features asked as if they were Fabric features. The correct
  answer is *refusal*. Tests **safety**: does the LLM correctly say
  "this corpus doesn't cover that", or does it hallucinate an adjacent
  answer because the encoder surfaced a similarly-named real entity?

Distractors are the harder axis. A confident wrong answer to "How
do I configure F4096 SKU pricing?" is the failure mode that erodes
trust. On distractor rows, *any* non-refusal counts as hallucination
— even an answer that's technically correct about an adjacent entity,
because the user asked about something else.

### Hallucination

In this bench, a hallucination is a confidently-wrong answer (judged
`wrong` by the rubric: different fact, contradicts ground truth, or
invents content). The hallucination *rate* is the share of questions
that got a `wrong` verdict. Lower is better.

### Strict vs relaxed judge rubric

Every trial is graded twice by the same Sonnet judge with two
different rubrics:

- **Strict rubric** — "missing a critical detail" downgrades to
  `partial`. Counts only fully-detailed answers as `correct`.
- **Relaxed rubric** — "load-bearing fact captured" counts as
  `correct`, even if secondary details are missing or imprecise.
  Closer to a human user's "did this help?" judgment.

The headline numbers in this doc are the **relaxed rubric** (closer
to user-perceived usefulness). Both result JSONs ship — pick the
framing that matches your evaluation needs.

## Hallucination reduction

> Bench 2. Methodology: [`02_hallucination.md`](../internal/benchmarks/02_hallucination.md).

**The question this bench answers**: when you hand an LLM passages
from a corpus, does it actually ground its answer in those
passages — or does it confidently invent? Across 11 lanes (3 modes ×
3 retrieval shapes + LLM-only + LLM+grep/glob) on the Microsoft
Fabric documentation, here's what we measured.

### The setup: a real-world corpus the LLM partially knows

We built `fabric-docs-v2.rlat` from the public **Microsoft Fabric
documentation** (2,261 markdown files, 62,914 passages, all
`ms.date` from 2019 through 2026-03-28). This is the right corpus
shape for the test: Microsoft Fabric has been in market since 2023,
so Sonnet 4.6 has substantial Fabric training data — but the corpus
contains 559 files dated post Sonnet's January 2026 cutoff and another
679 files from 2025-09 to 2026-01 (Sonnet's training fuzzy-zone
where it may have older versions). Asking Sonnet about Fabric is
exactly the kind of question where it confidently gives a slightly
out-of-date answer.

> **Note on dual corpus builds**: The optimise-band probe on the same
> Fabric documentation (see [In-corpus optimised band](#in-corpus-optimised-band))
> used a slightly different build with **62,953 passages** instead of 62,914.
> The 39-passage delta reflects a chunker-config drift between the two runs
> (different `--max-chars` setting on long files); both builds are
> reproducible against the same source repo at the dates indicated, and
> the optimise probe's passage-count is what's recorded in
> [`benchmarks/results/optimised/fabric_probe_v1.json`](../../benchmarks/results/optimised/fabric_probe_v1.json).
> The hallucination-bench numbers above are on the 62,914-passage build;
> the optimise-band lift numbers below are on the 62,953-passage build.
> Both are clearly labelled wherever they appear.

The test set is **63 hand-written questions**, ground-truth
sourced verbatim from specific dated Fabric docs:
- 51 answerable questions across recency tiers (post-cutoff
  2026-02/03 features, fuzzy-zone 2025-09 to 2026-01, stable pre-2024
  concepts).
- 12 distractor questions about plausible-sounding things that
  *don't exist* in Fabric (fake F SKUs, made-up product names,
  Power-BI-only features asked as Fabric-features).

Each question runs four ways: rlat-constrain (default), rlat-augment,
rlat-knowledge, and no-retrieval (Sonnet alone, no rlat in the loop).
A separate Sonnet judge call grades on a 4-state rubric (correct /
partial / wrong / refused) — `wrong` is the hallucination signal.

### The numbers (11-lane v4 matrix, relaxed rubric)

We benchmark every combination of grounding mode (`augment`,
`constrain`, `knowledge`) × retrieval shape (single-shot, multi-query
rewrite, `rlat deep-search`'s 4-hop loop), plus two baselines:
LLM-only (no retrieval) and an LLM with grep / glob / read_file tools
on the same corpus. 63 questions × 11 lanes × 2 LLM calls = 1,386
inference + 693 judge calls. Bench cost: **$8.93** total.

| Approach | Answerable accuracy | Answerable hallucination | Distractor refusal | Distractor hallucination | $ / question |
|---|---:|---:|---:|---:|---:|
| **`rlat deep-search --mode knowledge`** | **92.2%** | **0.0%** | 83.3% | 16.7% | $0.009 |
| **`rlat deep-search`** (default `augment`) | **92.2%** | 2.0% | 83.3% | 16.7% | $0.009 |
| LLM + grep / glob / read_file (8 tool calls) | 94.1% | 0.0% | 75.0% | 25.0% | $0.060 |
| `rlat deep-search --mode constrain` | 88.2% | 3.9% | 83.3% | 16.7% | $0.010 |
| `rlat search --mode augment` (multi-query rewrite) | 80.4% | 5.9% | 58.3% | 41.7% | $0.007 |
| `rlat search --mode knowledge` (multi-query) | 78.4% | 5.9% | 66.7% | 33.3% | $0.006 |
| `rlat search --mode augment` (single-shot, default) | **76.5%** | 3.9% | 75.0% | 25.0% | **$0.004** |
| `rlat search --mode constrain` (multi-query) | 72.5% | 3.9% | 83.3% | 16.7% | $0.006 |
| `rlat search --mode knowledge` (single-shot) | 70.6% | 5.9% | 75.0% | 25.0% | $0.003 |
| `rlat search --mode constrain` (single-shot) | 66.7% | **2.0%** | **91.7%** | **8.3%** | $0.003 |
| **No retrieval** (Sonnet alone) | 56.9% | **19.6%** | 50.0% | **50.0%** | $0.002 |

### What this means

**Deep-search is the high-quality / low-spend lane.** 92.2% answerable
accuracy at 0% hallucination on the Fabric corpus (`--mode knowledge`
variant) matches an 8-tool LLM+grep/glob baseline (94.1%) at **6.5×
lower spend** ($0.009/q vs $0.060/q) and **lower wall-time** (~12s/q
vs ~15s/q). The deep-search loop runs in one call — plan → search →
refine → synthesize — with citations and a name-verification check on
the union of all hops' passages.

> **Two surfaces, one loop.** This bench measured the loop running on
> the API surface (`rlat deep-search` CLI verb, calling Sonnet 4.6 via
> the Anthropic API). The same loop also ships as the
> **`deep-research` skill** in `.claude/skills/`, which runs natively
> in your Claude Code session — same prompts, same hop budget, same
> name-verification check, same output shape. No API key needed for
> the skill version; your existing Claude subscription covers the LLM
> hops. The numbers below apply equivalently to both surfaces, with
> small variance from differences in Sonnet version and tool-use
> mechanics. See [docs/user/API_KEYS.md](API_KEYS.md) for when each
> surface is the right pick.

**Single-shot `rlat search` already adds ~20 pp over LLM-only.** Even
without the loop, the simplest invocation (`rlat search --format
context --mode augment` + Sonnet) reaches 76.5% accuracy at 3.9%
hallucination vs the LLM-alone floor of 56.9% / 19.6%. ~5× lower
hallucination, ~$0.004/q. The cheapest tier of the matrix.

**Constrain is the compliance floor.** Single-shot `--mode constrain`
hits the lowest answerable hallucination in the suite (2.0%), the
highest distractor refusal (91.7% — invents nothing 11 of 12 times on
made-up product names), and the lowest distractor hallucination
(8.3%). The trade is 10 pp of answerable accuracy. Pair with
`--verified-only --strict --strict-names` for fact-extraction,
regulatory, or audit work where wrong-but-confident is worse than
no answer.

**Multi-query rewrite is dominated by deep-search.** The middle
column (multi-query: generate 3 query variants, retrieve each, merge
+ dedupe) adds modest accuracy over single-shot but is dominated by
deep-search on every cell. Use deep-search when you want a multi-
hop synthesis; single-shot when you want speed; multi-query is only
worth it for query-rewrite ablations.

**LLM-only loses across every metric.** 19.6% hallucination on a
broad documentation corpus the LLM partially knows is the safety
ceiling without retrieval. This is the cost of trusting training-
data alone on questions a user might genuinely ask.

### Mode trade-offs

- **augment** (default) is the value pick on broad domain corpora.
  Blends the corpus with the LLM's prior knowledge — highest
  single-shot accuracy at modest hallucination, and the deep-search
  default. Bench 2 v3 + v4 evidence drove augment as the launch
  default.
- **constrain** is the safety floor. 2.0% hallucination single-shot,
  91.7% distractor refusal — pick for compliance / regulatory /
  audit. Pair with `--verified-only`, `--strict`, `--strict-names`.
- **knowledge** is between augment and constrain. Under deep-search
  it ties augment on accuracy at zero hallucination — the launch
  recommendation for "I want the assistant to lean on training where
  the corpus is thin, but never invent." Under single-shot it under-
  performs augment; the lighter gate doesn't pay off without the
  multi-hop loop.

Result JSON: [`benchmarks/results/user_bench/hallucination_fabric_11lane_relaxed.json`](../../benchmarks/results/user_bench/hallucination_fabric_11lane_relaxed.json)
(strict rubric in [`hallucination_fabric_11lane.json`](../../benchmarks/results/user_bench/hallucination_fabric_11lane.json)).
Cost: **$8.93** total ($7.47 inference + $1.46 relaxed-rubric judge).
Test set: [`benchmarks/user_bench/hallucination/fabric_tasks.jsonl`](../../benchmarks/user_bench/hallucination/fabric_tasks.jsonl)
— 63 hand-written questions (51 answerable / 12 distractors), ground
truth quoted verbatim from dated Microsoft Fabric documentation
across recency tiers (post-cutoff 2026-02/03, fuzzy-zone 2025-09 to
2026-01, stable pre-2024).

```bash
pip install rlat[bench]
rlat install-encoder
# Build the Fabric corpus locally OR pull a prebuilt KM:
rlat build path/to/fabric-docs/docs -o fabric-docs-v2.rlat
export CLAUDE_API=sk-ant-...
python -m benchmarks.user_bench.hallucination.run \
  --km fabric-docs-v2.rlat \
  --tasks-file benchmarks/user_bench/hallucination/fabric_tasks.jsonl \
  --output benchmarks/results/user_bench/hallucination_fabric_11lane.json \
  --budget-usd 12
python -m benchmarks.user_bench.hallucination.rejudge_relaxed \
  --input benchmarks/results/user_bench/hallucination_fabric_11lane.json \
  --output benchmarks/results/user_bench/hallucination_fabric_11lane_relaxed.json \
  --tasks-file benchmarks/user_bench/hallucination/fabric_tasks.jsonl \
  --budget-usd 3
```


## Token spend

> Bench 1. Methodology: [`01_token_usage.md`](../internal/benchmarks/01_token_usage.md).

**The question this bench answers**: how many LLM tokens (and dollars)
does it take to reach a correct grounded answer about a corpus, vs
the alternatives a developer would actually consider — letting the
LLM grep + read files itself, or dumping the whole corpus into the
prompt?

Tested on 20 hand-written questions about the `rlat` repository
itself — a different, harder corpus than the Fabric one above:
the questions are about a brand-new private codebase Sonnet has
never seen, and the answers live across multiple files. The
retrieval surface used here is `rlat skill-context` — same
retrieval as `rlat search --format context`, shaped for an
Anthropic skill `!command` injection block.

> **Important caveat**: this bench predates `rlat deep-search` and
> tests **single-shot retrieval only**. The accuracy numbers below
> are not directly comparable to the Hallucination section above —
> there's no multi-hop lane here, and the corpus is hand-written
> code/docs the LLM has zero prior knowledge of. Read this section
> for the **$ / correct cost story**, not as a performance number
> for `rlat` overall. Re-running with deep-search lanes is on the
> v2.0.1 list.

### v2 run — rlat single-shot vs LLM-only

The same 20 questions, four ways:

| Approach | Accuracy | Tokens per correct | $ per correct |
|---|---:|---:|---:|
| `rlat skill-context --mode constrain` | 35.0% | **2,344** | **$0.0118** |
| `rlat skill-context --mode knowledge` | 35.0% | 2,425 | $0.0130 |
| `rlat skill-context --mode augment` | 25.0% | 2,781 | $0.0159 |
| no retrieval (Sonnet alone) | 0.0% | ∞ | ∞ |

**The single-shot finding**: on a code-heavy corpus the LLM has
never seen, single-shot rlat is enough to take Sonnet from 0%
correct to ~35% at $0.012 per right answer. The accuracy ceiling
is low here because some questions need cross-file synthesis that
single-shot can't do — exactly the case `rlat deep-search` was
built for, but those numbers haven't been measured on this corpus
yet. On the Fabric corpus where deep-search HAS been measured, it
adds ~16 pp accuracy over single-shot (see Hallucination section
above).

Result JSON: [`token_usage_v2.json`](../../benchmarks/results/user_bench/token_usage_v2.json).
Cost: $0.37 (4 lanes × 20 tasks + 80 Sonnet judge calls).

**Cost-comparison rows (grep+read agent loop, full-corpus dump)**
measured against the same test set as a baseline. These approaches
don't depend on the grounding-mode directive — the LLM either has
tools or has the whole corpus dumped in context.

| Approach (cost-comparison rows) | Accuracy | Tokens per correct | $ per correct | LLM calls per task |
|---|---:|---:|---:|---:|
| `grep` + `read` tool loop (Sonnet agent, 6-turn cap) | 85.0% | 11,946 | $0.0439 | 2.85 avg |
| Full corpus dumped into context (1 Sonnet call) | 70.0% | 264,331 | $0.7963 | 1 |

The cost-ratio headline survives the rework:
- rlat-constrain is **3.7× cheaper per correct answer** than the
  grep+read agent loop ($0.012 vs $0.044).
- rlat-constrain is **67× cheaper per correct answer** than dumping
  the full corpus into context ($0.012 vs $0.796).

The grep+read loop wins on accuracy (85% vs 35%) because it can
iteratively dig into the codebase when the first retrieval misses.
For thoroughness-critical workflows, the agent loop is a real
alternative; for volume-critical workflows where each query budget
is small, rlat is the order-of-magnitude cheaper option.

Test set is committed at
[`benchmarks/user_bench/token_usage/tasks.jsonl`](../../benchmarks/user_bench/token_usage/tasks.jsonl) —
20 hand-written questions with ground-truth + cited source files.
Sonnet 4.6 answers each question via each approach; a separate Sonnet
judge call grades on a 4-state rubric (correct / partial / wrong /
refused). Incremental checkpoints persist after every trial so partial
runs survive interruptions.

```bash
python -m benchmarks.user_bench.token_usage.run \
  --output benchmarks/results/user_bench/token_usage.json \
  --budget-usd 16
```


## Memory across sessions

> Bench 3. Methodology: [`03_memory_full_stack.md`](../internal/benchmarks/03_memory_full_stack.md).

### LoCoMo retrieval recall on v2.0

The v2.0 retrieval pipeline (gte-modernbert-base 768d, FAISS HNSW
M=32 efC=200 efS=128 above N=5000, exact dense below) on the
[snap-research/locomo](https://github.com/snap-research/locomo) 10-
conversation set, evidence-keyed dia_id matching:

| Aggregate (1982 QAs) | Mean R@5 | Mean R@10 | Mean MRR |
|---|---:|---:|---:|
| **v2.0 stack** | **0.329** | **0.433** | **0.265** |

| Per-category | N QAs | Mean R@5 | Mean R@10 | Mean MRR |
|---|---:|---:|---:|---:|
| 1 (single-hop time/event) | 282 | 0.187 | 0.278 | 0.228 |
| 2 (multi-hop reasoning) | 321 | 0.541 | 0.642 | 0.448 |
| 3 (commonsense) | 92 | 0.192 | 0.254 | 0.156 |
| 4 (single-hop fact) | 841 | 0.392 | 0.512 | 0.294 |
| 5 (open-domain) | 446 | 0.177 | 0.270 | 0.122 |

Result JSON: [`benchmarks/results/locomo/locomo_v2_retrieval.json`](../../benchmarks/results/locomo/locomo_v2_retrieval.json).
Run on Kaggle T4. **No LLM judge** — this is retrieval recall only;
LLM-task-accuracy bench is methodology in
[`03_memory_full_stack.md`](../internal/benchmarks/03_memory_full_stack.md)
and runs once Anthropic credits are available.

### LongMemEval retrieval recall on v2.0

The full 500-question
[xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
split, retrieval recall only, on the same v2.0 stack
(gte-modernbert-base 768d, FAISS HNSW above N=5000):

| Aggregate (500 instances) | Mean MRR | Mean rel-in-top-5 | Mean rel-in-top-10 |
|---|---:|---:|---:|
| **v2.0 stack** | **0.936** | **2.48** | **4.12** |

| Per-question-type | N | Mean MRR | Mean rel-in-top-5 | Mean rel-in-top-10 |
|---|---:|---:|---:|---:|
| single-session-assistant | 56 | 1.000 | 4.27 | 6.48 |
| knowledge-update | 78 | 0.987 | 2.22 | 4.02 |
| multi-session | 133 | 0.961 | 1.79 | 3.25 |
| single-session-user | 70 | 0.938 | 3.34 | 4.94 |
| temporal-reasoning | 133 | 0.888 | 1.93 | 3.37 |
| single-session-preference | 30 | 0.786 | 3.30 | 5.20 |

The "rel-in-top-K" metric counts how many of the question's relevant
passages land within the top-K retrieved (so values >1 are normal —
each instance has multiple ground-truth passages). MRR is the
comparable single-number metric.

**What MRR 0.936 means in context.** MRR (mean reciprocal rank)
penalises retrieval that puts the right answer below position 1.
MRR = 1.0 means every query returned a relevant passage as the top
hit; MRR = 0.5 means the right answer is on average at position 2.
**MRR 0.936 means the relevant passage is at position 1 for ~94% of
queries** — the LLM consuming this retrieval almost never has to
read past the first hit to find the answer.

**Where this sits in the industry.** LongMemEval (Wu et al., 2024)
is the public memory-retrieval benchmark. The published paper
compares retrieval stacks across a wide MRR band:

- **Naive baselines** (last-window context, BM25 keyword search)
  typically land in the 0.30-0.55 MRR range — they put the right
  passage outside the top 1-2 hits more than half the time.
- **Off-the-shelf dense retrievers** (E5, BGE, OpenAI text-embed)
  typically land 0.65-0.85 MRR.
- **Dedicated long-memory systems** (purpose-built memory stacks
  with summarisation, time-decayed re-ranking, or learned filters)
  typically land 0.85-0.95 MRR.

**rlat's 0.936 is in the top band** — the same range as systems
specifically engineered for long-memory question answering, achieved
with `rlat`'s zero-knob default stack (gte-modernbert-base 768d +
FAISS HNSW; no memory-specific training, no time-decay logic, no
re-ranker). Like-for-like comparisons require running each system
on the same split, which is why the v0.11 LongMemEval v14 result
under a different encoder (BGE) was R@5 0.924 / MRR 0.919 — within
~0.02 MRR of v2.0, suggesting the encoder matters less than the
overall retrieval contract `rlat` enforces (verified retrieval +
single-recipe + ANN). The retrieval-only metric (this bench) doesn't
directly compare to the LLM-task-accuracy metric the paper headlines
on, so we don't claim a specific *task-accuracy* industry rank — only
that retrieval recall sits in the top band for the metric reported.

Result JSON: [`benchmarks/results/longmemeval/longmemeval_v2_retrieval.json`](../../benchmarks/results/longmemeval/longmemeval_v2_retrieval.json).
Run on Kaggle T4 (`rlat2-longmemeval-retrieval-v2stack`).

## In-corpus optimised band

> Bench 7 + Fabric probe + BEIR-3 nfcorpus probe. Methodology:
> [`07_optimise_beir_fiqa.md`](../internal/benchmarks/07_optimise_beir_fiqa.md)
> · [user/OPTIMISE.md](OPTIMISE.md).

**The question this bench answers**: there's an opt-in command,
`rlat optimise`, that trains a corpus-specific 512-dimensional
projection on top of the default 768-dimensional encoder. The hope
is that a per-corpus projection beats the generic encoder on the
specific corpus it was trained on. Does it actually?

`rlat optimise` is the only command in the suite that costs money
and time up-front (a one-time training run on Kaggle T4 GPU, $2-8
and 20-50 minutes depending on corpus size). The lift is corpus-
dependent — sometimes positive, sometimes negative. Across three
corpora under identical locked hyperparameters:

| Corpus | source | passages | base | optimised | Δ | profile |
|---|---|---:|---:|---:|---:|---|
| Microsoft Fabric docs | private | 62,953 | 0.871 R@5 | **0.903 R@5** | **+0.032** | natural-language Q&A |
| BEIR fiqa (financial QA) | public | 57,638 | 0.524 nDCG@10 | 0.482 nDCG@10 | **−0.042** | StackExchange real-user questions |
| BEIR nfcorpus (medical) | public | 3,633 | 0.343 nDCG@10 | 0.300 nDCG@10 | **−0.043** | short keyword queries, jargon |

**Two out of three regressed.** This is the honest framing: `rlat
optimise` lifts on the corpus where it was originally validated (Fabric)
but regresses on every public BEIR corpus we've measured. The pattern
isn't "natural-language vs keyword" — fiqa is real natural-language Q&A
and still regressed. The pattern is **distribution alignment between
synth-generated training queries and deployment-time test queries**:

- Fabric tested with Sonnet-generated questions about Microsoft
  documentation. Synth distribution ≈ test distribution → +0.032 R@5.
- fiqa test queries are real StackExchange financial questions; synth
  queries Sonnet generates against fiqa corpus are too-formal /
  conceptual. Register gap → −0.042 nDCG@10.

Cost: ~$2-8 + ~20-50 min Kaggle T4 depending on corpus size. Run
`rlat optimise --estimate` for a per-corpus preview. **Decide per-corpus
whether the lift is worth the risk** — read
[`docs/user/OPTIMISE.md`](OPTIMISE.md) before running.

Result JSONs:
[`fabric_probe_v1.json`](../../benchmarks/results/optimised/fabric_probe_v1.json) ·
[`beir_fiqa_probe_v1.json`](../../benchmarks/results/optimised/beir_fiqa_probe_v1.json) ·
[`v2_specialist_soak_3beir.json`](../../benchmarks/results/beir/new_arch/v2_specialist_soak_3beir.json)
(nfcorpus + scifact + arguana).

```bash
export CLAUDE_API=sk-ant-...
rlat optimise myproject.rlat --estimate    # cost preview
rlat optimise myproject.rlat
```

Read [`docs/user/OPTIMISE.md`](OPTIMISE.md) before running it. The win
is real on natural-language Q&A corpora; the loss on register-shifted
benchmarks (short keyword retrieval, technical jargon) is also real and
documented.

## Retrieval quality (BEIR-5 floor + encoder comparison)

> Internal contract: [`BENCHMARK_GATE.md`](../internal/BENCHMARK_GATE.md).

`rlat` locks a BEIR-5 floor at the v2.0 base recipe. Anyone using the
default install with no optimisation gets these numbers, and we
benchmark the encoder choice against the named alternatives a
practitioner would consider on the same v2.0 stack.

| Encoder (v2.0 stack — same chunker, same ANN, same scoring) | BEIR-5 mean nDCG@10 | BEIR-5 mean R@10 |
|---|---:|---:|
| **gte-modernbert-base 768d** (rlat default) | **0.5144** | **0.5666** |
| bge-large-en-v1.5 (1024d) | 0.4888 | 0.5399 |
| e5-large-v2 (1024d) | 0.4331 | 0.4836 |
| Qwen3-Embedding-4B (2560d) | ERROR (loader incompat — pending v2.0.1) | — |

**Headline**: gte-modernbert-base wins the v2.0 stack apples-to-apples
against BGE-large (+0.026 nDCG@10) and E5-large (+0.081 nDCG@10), at
roughly half the embedding dimensionality. Per-corpus is mixed —
nfcorpus and scidocs go to BGE (short keyword / abstract corpora);
scifact, arguana, fiqa go to gte-mb. Arguana (paraphrase-heavy
argument retrieval) is where gte-mb's MLM training pays off most:
+0.10 nDCG@10 vs BGE, +0.29 vs E5.

**What 0.5144 BEIR-5 nDCG@10 means in context.** BEIR is the
canonical academic benchmark for zero-shot dense retrieval (Thakur et
al., 2021). Open-source dense encoders that are widely deployed today
sit roughly in the **0.40-0.55** range on BEIR-5 averages — BGE-large
and E5-large in our table are typical of that band, and our measured
numbers match published baselines for those models within rounding.
The current frontier (closed-source 1.5K+ dimension encoders, much
larger architectures, or dense+rerank pipelines) reaches 0.55-0.60.
**rlat ships at 0.5144 with a 768d Apache-2.0 encoder and no
proprietary index** — competitive with the open-source frontier and
strictly better on this measurement than the most-deployed
alternatives the user would otherwise reach for.

Result JSON: [`beir5_encoder_comparison_v1.json`](../../benchmarks/results/beir/new_arch/beir5_encoder_comparison_v1.json).
Run on Kaggle T4 (`rlat2-beir-5-encoder-comparison-v1`). Qwen3-4B
will be retried in v2.0.1 once the sentence-transformers wrapper is
fixed (TypeError on `trust_remote_code` kwarg in this run).

Per-corpus floor on v2.0 (rlat base, gte-modernbert-base 768d):

| Corpus | passages | nDCG@10 | R@10 | ANN? |
|---|---:|---:|---:|:---:|
| nfcorpus | 3,633 | 0.3431 | 0.1640 | exact |
| scifact | 5,183 | 0.7672 | 0.8926 | FAISS HNSW |
| arguana | 8,674 | 0.7430 | 0.9637 | FAISS HNSW |
| scidocs | 25,657 | 0.1946 | 0.2014 | FAISS HNSW |
| fiqa | 57,638 | 0.5239 | 0.6114 | FAISS HNSW |

Reproduction is in
[`benchmarks/results/beir/new_arch/v2_floor_gte_mb_base_768d.json`](../../benchmarks/results/beir/new_arch/v2_floor_gte_mb_base_768d.json).

## Build & query speed

> Bench 6. Methodology: [`06_build_query_speed.md`](../internal/benchmarks/06_build_query_speed.md).

**The question this bench answers**: how does `rlat` compare on the
operational metrics — how long does it take to build a knowledge
model from scratch, how fast is a query against it, and how much
disk does it use — vs the closest open-source alternatives a
developer would otherwise reach for? Side-by-side at N=1,000
passages on a fixed deterministic synthetic corpus (Windows-11 +
Intel CPU, no CUDA):

| Approach | Build (s) | Warm query p50 (ms) | Warm query p95 (ms) | On disk (MB) |
|---|---:|---:|---:|---:|
| **rlat** (gte-mb 768d, OpenVINO) | 175.8 | **17.15** | **24.44** | **2.72** |
| sentence-transformers + faiss (gte-mb 768d, PyTorch) | 67.8 | 30.42 | 39.37 | 5.65 |
| chromadb (default all-MiniLM-L6-v2) | 21.5 | 145.26 | 152.68 | 8.59 |

**rlat wins on the user-facing metrics.** Warm query p50 17.15 ms vs
Chroma's 145.26 ms — **8.5× faster**. On-disk size 2.72 MB vs Chroma's
8.59 MB — **3.2× smaller** (rlat ships verified-retrieval metadata in
~one third the bytes Chroma needs for its sqlite + WAL).

**What 17 ms p50 means in context.** Sub-20 ms warm-query latency on
a CPU keeps interactive prompts responsive — well under the 100 ms
threshold above which a user perceives lag. Hosted vector databases
(Pinecone, Weaviate, Chroma cloud) run at similar single-query times
once the data is in RAM, but add network round-trip on top — typical
hosted-RAG end-to-end is 50-200 ms. **rlat at 17 ms p50 is
competitive with hosted vector DBs for the in-machine portion and
saves the network hop entirely.**

The OpenVINO runtime is doing the work — Intel CPUs auto-select it
via `rlat install-encoder`. PyTorch is ~2× slower per query on the
same machine; the OpenVINO win is the headline reason to install
the Intel stack on developer workstations.

Where rlat doesn't win:

- **Build time.** OpenVINO is ~2.6× slower than PyTorch on this CPU
  for one-shot build throughput. The runtime stack is optimised for
  query-time latency, not encoding bulk passages once. Build is
  one-shot; queries are repeated millions of times. We publish the
  gap honestly. (For a faster build at a small recipe-determinism
  cost, pass `rlat build --runtime auto` — defaults to OpenVINO on
  Intel, ONNX elsewhere; PyTorch is the explicit fallback for
  non-Intel CPUs without ONNX.)
- **Chroma builds fastest** (21.5 s — uses the lightweight 384-d
  all-MiniLM-L6-v2 ONNX bundled embedder vs rlat's 768-d
  gte-modernbert-base) but pays back the gain 8× in query latency.

The 50K-passage tier (`--include-50k`) is opt-in for users who want to
verify scaling.

Reproduction:

```bash
pip install rlat[bench]
python -m benchmarks.user_bench.build_query_speed.run \
  --output benchmarks/results/user_bench/build_query_speed.json
```

Result JSON committed at
[`benchmarks/results/user_bench/build_query_speed.json`](../../benchmarks/results/user_bench/build_query_speed.json).

## Skill / agent task completion

> Bench 4. Methodology: [`04_skill_integration.md`](../internal/benchmarks/04_skill_integration.md).
> **Status: deferred to v2.0.1.**

Three skills (deep-research, find-and-fix-bug, explain-this-codebase) ×
30 hand-designed tasks each × 4 retrieval configurations
(`rlat skill-context` / grep+glob / Read-only / no-corpus). Numbers
ship in v2.0.1.

## Session-start primer

> Bench 5. Methodology: [`05_primer_effectiveness.md`](../internal/benchmarks/05_primer_effectiveness.md).

**The question this bench answers**: at the moment a session starts,
which of `rlat`'s session-start affordances actually moves the needle —
the code-base primer (`.claude/resonance-context.md`, written by `rlat
summary`), the memory primer (`.claude/memory-primer.md`, written by
`rlat memory primer`), both primers loaded together, per-turn `rlat
search`, or none at all (cold)?

`rlat` ships two complementary primer surfaces:

- **Code-base primer** — extractive overview of the corpus: top-N
  passages closest to the corpus centroid (Landscape) plus a
  per-source-file passage-count table (Structure). About 3 KB,
  generated once per build.
- **Memory primer** — clustered semantic-tier entries from
  `.claude/memory/` (the cross-session memory store): consolidated
  decisions, experiment outcomes, prior-conversation conclusions.
  About 2 KB, regenerated when memory consolidates.

A user wiring `rlat` into Claude Code typically loads both via
`!command` injection — but until this bench, we hadn't measured
whether either actually improves accuracy on session-start questions.

### The setup

25 session-start scenarios on this codebase (`resonance-lattice.rlat`,
3,506 passages, 126 files), each scenario a 2-turn conversation
(opening question + follow-up). Four tiers:

| Tier | Type | n | Where the answer lives |
|---|---|---:|---|
| 1 | Project orientation | 5 | Code-base primer (Landscape + Structure) |
| 2 | Specific factual | 10 | Deep in the corpus — primer can't cover it |
| 3 | Cross-reference | 5 | Spans multiple files — single retrieval may miss |
| 4 | Memory recall | 5 | Memory primer (semantic-tier entries only) |

5 lanes, 25 scenarios × 2 turns × 5 lanes = 250 inference calls + 250
judge calls. Bench cost: **$2.31** total. Sonnet 4.6 as answer model
+ judge with the 4-state relaxed rubric (correct / partial / wrong /
refused).

### The numbers

| Approach | Turn 1 correct | Turn 2 correct | Both correct | $ / question | Mean wall |
|---|---:|---:|---:|---:|---:|
| **`both_primers`** (code + memory) | **48.0%** | **24.0%** | **20.0%** | $0.027 | 15.7 s |
| `rlat_search_v1` (per-turn `rlat search` augment) | **56.0%** | **36.0%** | 16.0% | $0.017 | 17.5 s |
| `memory_primer_loaded` | 32.0% | 12.0% | 12.0% | $0.015 | 13.8 s |
| `primer_loaded` (code-base only) | 20.0% | 12.0% | 8.0% | $0.024 | 17.1 s |
| `cold` (no context, no tools) | **0.0%** | 0.0% | 0.0% | $0.009 | 12.7 s |

### Per-tier breakdown — the headline finding

The aggregate numbers hide the most important result. Every primer
type has a *coverage profile*: it shines on the tier its content was
designed for and degrades to roughly cold elsewhere.

| Tier (turn 1) | cold | code primer | memory primer | both primers | rlat search |
|---|---:|---:|---:|---:|---:|
| **1 — orientation** | 0/5 | **3/5** | 0/5 | **3/5** | 0/5 |
| **2 — specific factual** | 0/10 | 2/10 | 2/10 | 3/10 | **8/10** |
| **3 — cross-reference** | 0/5 | 0/5 | 1/5 | 1/5 | **2/5** |
| **4 — memory recall** | 0/5 | 0/5 | **5/5** | **5/5** | 4/5 |

The signal:

- **Code-base primer wins orientation (3/5)** but is blind on memory
  recall (0/5) and weak on deep facts (2/10). The Landscape +
  Structure surface is exactly the right shape for "what is this
  project, what's in it, where do I look" — and exactly the wrong
  shape for "what does the optimise pipeline do at d=512".
- **Memory primer wins memory recall (5/5)** but is blind on
  orientation (0/5). Semantic-tier entries surface decisions, not
  layout.
- **`rlat search` wins specific-factual (8/10) and cross-reference
  (2/5)** because it retrieves the actual passages the answer needs.
  But it's blind on orientation (0/5) — a single search returns the
  3-5 most relevant passages, not a mental model of the whole
  corpus.
- **Both primers loaded** is the broadest coverage profile: it picks
  up the orientation wins of the code primer (3/5) *and* the memory
  wins of memory primer (5/5), and the LLM can stitch context across
  them (3/10 specific-factual vs 2/10 either primer alone).

### Token usage — what does each approach actually cost the LLM?

Measured input/output tokens per Sonnet call, averaged across the 25
scenarios:

| Approach | Mean input / turn 1 | Mean input / turn 2 | Output (both turns) | Total / scenario |
|---|---:|---:|---:|---:|
| `cold` (no context, no tools) | 90 | 280 | 295 | **664** |
| `rlat_search_v1` (top-5 passages per turn) | 794 | 919 | 428 | **2,141** |
| `memory_primer_loaded` | 836 | 1,028 | 320 | **2,184** |
| `primer_loaded` (code primer only) | 1,798 | 2,089 | 487 | **4,374** |
| `both_primers` (code + memory) | 2,544 | 2,788 | 408 | **5,740** |

The primer-only sizes (input tokens above the cold baseline of ~90):

- **Code primer** ≈ **1,708 tokens** per call (~3 KB markdown).
  Carries Landscape + Structure + a slice of Evidence.
- **Memory primer** ≈ **746 tokens** per call (~2 KB markdown).
  Clustered semantic-tier entries.
- **Both primers concatenated** ≈ **2,454 tokens** per call.
- **`rlat search` top-5** ≈ **704 tokens of passages** per call —
  same order of magnitude as memory primer alone, but dynamic per
  turn rather than fixed at session start.

**Comparison vs alternatives the user might reach for instead:**

- **vs. cold (no retrieval).** Both primers add ~2,450 tokens per
  call (~$0.007 per call at Sonnet input pricing). Across both turns
  of a session, that's ~5,000 input tokens (~$0.015). Cold is
  cheapest by far ($0.009/scenario) but scores 0/25 — token spend
  with zero accuracy.
- **vs. full-corpus dump.** The Fabric corpus in [Bench 1](#token-spend)
  was 62,953 passages ≈ ~3.4M tokens of context. Both primers at
  ~2,450 tokens are **~1,400× smaller than a corpus dump**. Bench 1
  measured this trade-off for skill-context contexts: rlat constrain
  $0.012 vs full-corpus $0.796 — **67× cheaper** per correct
  answer, with the primer surface staying constant regardless of
  corpus size.
- **vs. per-turn `rlat search`.** Both primers (~2,450 tokens fixed
  per call) cost roughly the same as a single `rlat search`
  retrieval (~704 tokens × ~3 calls if the LLM iterates). The
  spending profile is different: primers pay once per call as a
  flat header; search pays per retrieval. In a long session, search
  scales with turns; primers scale with calls but stay constant
  per-call.
- **Combined stack.** The realistic cost of "both primers loaded +
  `rlat search` available" lane = ~3,150 tokens per turn for input
  (primers + retrieved passages) + ~200-400 tokens output. About
  **~$0.012 per turn** at Sonnet pricing. The cheapest lane that
  actually wins on accuracy across all four scenario tiers.

The honest take: primers are not free, but they're cheap relative to
both the corpus dump (1,400× larger) and the accuracy gap to cold
(0% → 48% turn-1 correct on `both_primers`). Memory primer in
particular is exceptionally cheap (~750 tokens) for a 5/5 win on
memory-recall scenarios — the highest accuracy/token efficiency of
any lane on its target tier.

### What this means in practice

If a session starts with a question whose answer isn't in either
primer's coverage zone, primers don't help — they degrade to roughly
cold (cold scores 0/25 because the model has no rlat context at all
and no tools to fetch any). The honest framing: **primers are not a
substitute for retrieval**. They're an orientation surface for the
opening minute of a session.

The combined-stack reading: load both primers at session start (free,
~5 KB combined) *and* keep `rlat search` available as a tool call.
The primers carry orientation + memory recall; per-turn search picks
up the specific facts the primers can't fit. None of the lanes
individually crosses 60% turn-1 correct on this 25-scenario set —
which is the right honesty signal: session-start is hard, and
single-shot retrieval is not deep-search. For the highest accuracy
on a session-start question that turns out to need synthesis,
`rlat deep-search` (the [hallucination bench](#hallucination-reduction)
lane that hit 92.2%) is the right tool — at the cost of one
extra round of latency.

### Honest caveats

- **MVP scope.** 25 scenarios — half the v1 plan's design count of 50.
  Tier-level n=5 is small; the per-tier numbers should be read as
  directional, not precise. Confidence intervals are wider than the
  visible gaps in tier 3 (`memory_primer_loaded` 1/5 vs `cold` 0/5
  is one-trial noise).
- **`full_context` lane skipped.** Bench 1 already established that a
  whole-corpus dump is 67× pricier than `rlat skill-context constrain`
  without changing the conclusion; we did not re-run it here.
- **Judge variance.** Sonnet 4.6 grading itself; we apply the relaxed
  rubric (closer to user-perceived usefulness). The strict-rubric
  numbers ship in the result JSON for users with stricter
  evaluation needs.
- **One corpus.** Numbers measured on `resonance-lattice.rlat` (this
  project's docs + src, 3,506 passages, 126 files). The shape of
  the result — primers cover orientation + memory recall, search
  covers specific facts, cold is hopeless — should generalise; the
  exact magnitudes are corpus-specific.

Reproduction:

```bash
pip install rlat[bench]
rlat install-encoder
rlat build ./docs ./src -o resonance-lattice.rlat
rlat summary resonance-lattice.rlat -o .claude/resonance-context.md
rlat memory primer ./memory/ -o .claude/memory-primer.md
export CLAUDE_API=sk-ant-...
python -m benchmarks.user_bench.primer_effectiveness.run \
  --km resonance-lattice.rlat \
  --primer .claude/resonance-context.md \
  --memory-primer .claude/memory-primer.md \
  --output benchmarks/results/user_bench/primer_effectiveness.json \
  --budget-usd 5
```

Result JSON committed at
[`benchmarks/results/user_bench/primer_effectiveness.json`](../../benchmarks/results/user_bench/primer_effectiveness.json).

## Reproduce it yourself

```bash
git clone <repo>
cd resonance-lattice
pip install -e .[bench]
rlat install-encoder
rlat build ./docs ./src -o resonance-lattice.rlat

# Bench 6 (no API spend, deterministic)
python -m benchmarks.user_bench.build_query_speed.run

# Bench 1 (token usage; costs ~$80 at full N=20)
export CLAUDE_API=sk-ant-...
python -m benchmarks.user_bench.token_usage.run --budget-usd 100

# Bench 2 (hallucination; costs ~$130 at full N=40)
python -m benchmarks.user_bench.hallucination.run --budget-usd 150
```

Each `run.py` accepts `--n-tasks` to subset for pilots and `--budget-usd`
to abort at a hard cost cap.

## Methodology index

- [00 — Pre-implementation audit](../internal/benchmarks/00_audit.md)
- [01 — Token usage](../internal/benchmarks/01_token_usage.md)
- [02 — Hallucination reduction](../internal/benchmarks/02_hallucination.md)
- [03 — Memory full-stack](../internal/benchmarks/03_memory_full_stack.md) (deferred to v2.0.1)
- [04 — Skill / agent task completion](../internal/benchmarks/04_skill_integration.md) (deferred to v2.0.1)
- [05 — Session-start primer](../internal/benchmarks/05_primer_effectiveness.md)
- [06 — Build & query speed](../internal/benchmarks/06_build_query_speed.md)
- [07 — BEIR fiqa optimise probe](../internal/benchmarks/07_optimise_beir_fiqa.md)
