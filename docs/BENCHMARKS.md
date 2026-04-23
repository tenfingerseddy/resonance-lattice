---
title: Benchmarks
slug: benchmarks
description: The main evidence page for Resonance Lattice: what was measured, what the results show, and what not to overclaim.
nav_group: Deep Dives
nav_order: 30
aliases:
---

# Benchmarks

## What This Page Is For

This is the main evidence page for repo-first evaluation. Use it when you want to know what the strongest current results are, what they actually mean for adoption, and which claims are well supported versus still early.

For the product thesis, see [Overview](/docs/overview). For operational benchmark details, thresholds, and baseline policy, see [Benchmark Runbook](/docs/benchmark-runbook).

## Headline Results

These are the strongest benchmark-backed claims the repo can make today:

| Benchmark family | Headline result | What it means |
|------------------|-----------------|---------------|
| Internal retrieval quality | `Recall@5 = 1.00`, `MRR = 0.93`, `0%` failed retrieval on the internal Fabric benchmark | the full retrieval pipeline is strong on technical documentation corpora |
| Token efficiency | `24.6x` fewer tokens than a grep-plus-read workflow | RL can give assistants much denser, more structured context than whole-file stuffing |
| LLM grounding | hallucination rate `0.78 -> 0.16`, fact recall `0.27 -> 0.91` | the grounded context materially improves answer quality on the internal evaluation |
| BEIR / cross-corpus | production pipeline beat flat E5 on `3 of 5` tested datasets | the workflow generalizes beyond the internal corpus, but not uniformly |
| LongMemEval / long-horizon memory | `Recall@5 = 0.924`, `MRR = 0.919` on 500 LongMemEval_s questions | the pipeline retrieves the correct session in top-5 for 92% of long-horizon conversational-memory queries across six question types |
| LoCoMo / end-to-end conversational QA | `66.23%` leaderboard-style (excl. adversarial) / `71.85%` overall on 1986 QA across 10 conversations | the full retrieve-read-judge pipeline lands rlat in Mem0 / Letta territory on the standard long-conversation QA benchmark (Sonnet 4.6 judge; GPT-4o judge pending) |
| Comparative / Obsidian | `Recall@5 1.00` vs `0.81`, `MRR 0.929` vs `0.714` on the published comparison | RL outperformed the best tested Obsidian LLM wiki workflow as a retrieval layer for grounded assistant use |

If the question is "why should I spend time setting this up when I already have a wiki, skills, or memory?", the most relevant sections on this page are [Comparative / Obsidian](#comparative-obsidian), [Token Efficiency](#token-efficiency), and [LLM Grounding](#llm-grounding). Those three together answer whether RL is just another context wrapper or a materially better retrieval layer. For Obsidian specifically, the claim is that RL is the stronger retrieval system for assistant grounding, not merely a different UI preference.

## Internal Retrieval Quality

### What this benchmark tests

This is the main internal retrieval benchmark against a 24,635-chunk Microsoft Fabric documentation corpus with 100 evaluation questions. It measures whether the full RL pipeline returns the right evidence with the right ranking.

### Headline numbers

| | rlat (reranked) | Hybrid RRF | Flat E5 | BM25 |
|---|---|---|---|---|
| **Recall@5** | **1.00** | 0.94 | 0.93 | 0.84 |
| **MRR** | **0.93** | 0.77 | 0.80 | 0.72 |
| **Failed retrieval** | **0%** | 6% | 7% | 16% |

### Strongest valid takeaway

The full RL retrieval pipeline is strong on factual and technical corpora. It outperforms the published flat E5 and BM25 baselines on the internal evaluation, and it does so with zero failed retrieval on the reported run.

### Limits and caveats

- this is an internal corpus, not a universal benchmark
- the result is for the full pipeline, not the dense field alone
- the numbers should be read as evidence of workflow quality, not proof that RL wins every retrieval task

### Raw outputs

See `benchmarks/results/retrieval_gate/` and the procedure in [Benchmark Runbook](/docs/benchmark-runbook).

## Pipeline Ablation

### What this benchmark tests

The ablation work asks where the observed quality actually comes from: dense semantic retrieval alone, hybrid retrieval, or the full reranked pipeline.

### Headline read

The dense field alone is not the whole story. The main gains come from combining:

- dense semantic retrieval
- lexical evidence injection
- reranking

### Strongest valid takeaway

The product is best understood as a knowledge-model-plus-pipeline system, not as "just an embedding model" and not as "just a field." The measured quality comes from the full retrieval path.

### Limits and caveats

- the ablation story explains where quality comes from, but it is not a substitute for end-to-end benchmark runs
- dense-only and reranked modes can behave differently depending on corpus type

### Raw outputs

See `benchmarks/results/ablation/` and `benchmarks/results/retrieval_gate/`.

## Token Efficiency

### What this benchmark tests

This benchmark compares RL context output against a common fallback workflow: keyword search followed by reading whole files into the assistant.

### Headline numbers

| Approach | Tokens per query | What you get |
|----------|-----------------|-------------|
| **grep + read top 5 files** | 37,154 | raw file text with no ranking, structure, or coverage signal |
| **rlat search (top 10)** | 1,518 | ranked passages with coverage, related topics, and source attribution |

That is `24.6x` fewer tokens with structured context. Median compression on the run was `19.9x`.

### Strongest valid takeaway

RL is not only about finding relevant text. It is also about making the retrieved knowledge usable for assistants without wasting most of the prompt on irrelevant file content.

### Limits and caveats

- this is not a universal token-efficiency law; it is a benchmark against a concrete fallback workflow
- the comparison is strongest when the alternative is whole-file reading, not when the alternative is a highly tuned retrieval system

### Raw outputs

See `benchmarks/results/token_efficiency/`.

## LLM Grounding

### What this benchmark tests

This benchmark compares an LLM answering questions without RL context versus the same model answering with RL-supplied context. The judged dimensions include accuracy, completeness, groundedness, hallucination rate, and fact recall.

### Headline numbers

- hallucination rate: `0.78 -> 0.16`
- fact recall: `0.27 -> 0.91`

### Strongest valid takeaway

When the assistant is grounded in RL context, answer quality improves materially on the current internal evaluation. This is one of the strongest practical reasons to adopt the knowledge model workflow.

### Limits and caveats

- this is an answer-quality benchmark, not a pure retrieval benchmark
- the result depends on both the retrieval path and the LLM/judge setup
- the numbers support "RL improves grounding in this evaluation," not "RL eliminates hallucinations"

### Raw outputs

See `benchmarks/results/llm_judge/`.

## BEIR and Cross-Corpus

### What this benchmark tests

These runs test whether the production pipeline generalizes beyond the internal corpus on standard retrieval datasets.

### Headline numbers

We ship three encoder tiers — each measured against the same 5-BEIR harness. None wins every corpus; the honest answer is workload-dependent. See [Encoder Choice](/docs/encoder-choice) for the per-workload decision guide.

**5-BEIR averages (nDCG@10, 2026-04-22 rebench):**

| Encoder | Tier | 5-BEIR avg | Notes |
|---------|------|-----------|-------|
| **`qwen3-8b`** | Frontier (16 GB GPU) | **0.500** | Block E final; roughly 1 pt below `text-embedding-3-large`. Launch gate (≥0.46) passes on this tier. |
| **`e5-large-v2`** | Portable alternative | 0.455 | Strongest on counter-argument retrieval (ArguAna-like). |
| **`bge-large-en-v1.5`** | Portable default (since 2026-04-20) | **0.445** (provisional, Block D) | Wins SciFact, NFCorpus, FiQA, SciDocs vs E5; loses ArguAna by ~9.7 pts. Net −1.0 pt vs E5 on the 5-corpus average; below the 0.46 gate. |

**Per-corpus breakdown** (historic `E5-large-v2` run with the production pipeline, reranker-on where applicable):

| BEIR dataset | rlat (best) | Mode | Flat E5 | BM25 |
|---|---|---|---|---|
| SciFact (5K) | **0.713** | reranked | 0.735 | 0.665 |
| NFCorpus (3.6K) | **0.360** | reranked | 0.337 | 0.325 |
| FiQA (57K) | **0.393** | reranked | 0.350 | 0.236 |
| ArguAna (8.7K) | **0.492** | dense | 0.501 | 0.315 |
| SciDocs (25K) | **0.189** | dense | 0.158 | 0.158 |

### Strongest valid takeaway

The production pipeline exceeds flat E5 on 3 of 5 BEIR corpora under the historic E5-large-v2 run, and on the 2026-04-22 rebench the default `bge-large-en-v1.5` wins 4 of 5 corpora outright versus E5 (losing only ArguAna). Because `bge-large-en-v1.5` is ~1 pt below the 0.46 launch gate on the 5-BEIR average — driven entirely by the ArguAna regression — the v1.0.0 product position is **publish full per-encoder numbers and recommend `qwen3-8b` for users who need the extra headroom**, rather than gate-failing the default. See [Honest Claims](/docs/honest-claims#measurement-state) for the full Block D/E story.

### Limits and caveats

- the pipeline does not win every dataset under every encoder
- reranking helps on factual and technical corpora, but dense-only can be stronger on argument-style retrieval
- BGE's 0.445 is provisional (Block D) pending the final rebench; E5 (0.455) and Qwen3-8B (0.500) are finalized
- the `qwen3-8b` result uses last-token pooling (required for Qwen3-Embedding); a broken mean-pool run previously collapsed to 0.250 on the same harness — the Qwen3 preset pins the correct pooling automatically
- LongMemEval (below) was measured with `E5-large-v2` and auto-routed retrieval; a BGE / Qwen3-8B rerun is tracked work
- corpus fit matters more than any single leaderboard headline

### Raw outputs

See `benchmarks/results/beir/`.

## LongMemEval

### What this benchmark tests

[LongMemEval](https://arxiv.org/abs/2410.10813) is a long-horizon conversational memory benchmark. Each instance has 30–70 prior sessions (~200k tokens of past dialogue) and a question that references facts stated in specific session(s). The evaluator measures whether retrieval surfaces the correct source session out of that haystack.

We report on LongMemEval_s (the 500-question variant), across six question types:

- single-session-user, single-session-assistant, single-session-preference — fact stated once in one session
- multi-session — fact synthesized across sessions
- knowledge-update — superseded fact (latest statement is not always the right answer)
- temporal-reasoning — fact dependent on *when* it was said

### Headline numbers

v14 config on LongMemEval_s (500 questions, 800/50 conversation chunking, E5-large-v2, auto-routed retrieval):

| | v14 (auto) |
|---|---|
| **Recall@5** | **0.924** |
| **Recall@10** | 0.961 |
| **MRR** | 0.919 |
| **Failed retrieval** | 2.8% |

### Strongest valid takeaway

The full RL pipeline is strong on multi-session conversational memory retrieval: it finds the correct session in the top-5 for 92% of long-horizon memory queries, across a mix of single-session, multi-session, knowledge-update, and temporal question types. The result comes from raw E5-large-v2 with no trained heads, combined with the adaptive query router (per-query-class routing of lexical, rerank, temporal, and recency knobs).

### Limits and caveats

- this is a **retrieval** benchmark; it is not directly comparable to Mem0's or Zep's published LongMemEval QA accuracy numbers (those are end-to-end GPT-4o-judged answer correctness, which depends on both retrieval and the answering LLM)
- the 800/50 chunk geometry is tuned for conversation-style corpora and differs from the default document chunker
- knowledge-update is the weakest category; session-amplification experiments (multi-granularity session vectors) gave partial wins on multi-session and preference but did not resolve the KU regression, so they remain research-only

### Raw outputs

See `benchmarks/results/longmemeval/v14_full500_800.json`.

## LoCoMo

### What this benchmark tests

[LoCoMo](https://github.com/snap-research/locomo) is an end-to-end conversational memory QA benchmark. Each of 10 conversations contains 19–32 dialogue sessions between two speakers spanning 2–3 months, with 105–260 questions per conversation grounded in that history. Unlike LongMemEval (which measures retrieval only), LoCoMo measures the full retrieve → read → judge pipeline against an LLM judge: can the assistant answer correctly from its memory of the conversation?

Five question categories:

- **single-hop** — fact stated once in one session
- **multi-hop** — fact synthesised across sessions
- **temporal** — fact dependent on *when* it was said
- **open-domain** — inference from conversation context
- **adversarial** — **unanswerable from the conversation**. Correct answer is the reader refusing to commit ("I don't have enough information")

Published leader numbers (leaderboard-style, excl. adversarial, GPT-4o judge): Mem0 ~67%, Letta ~68%, MemMachine ~73%, Zep ~79%.

### Headline numbers

Phase 2 baseline on 10 conversations, 1986 QA total, E5-large-v2, auto-routed retrieval, Claude Haiku 4.5 reader, cartridge-per-conversation:

| | Claude Sonnet 4.6 judge | Claude Haiku 4.5 judge |
|---|---|---|
| **Overall (incl. adversarial)** | **71.85%** | 68.23% |
| **Leaderboard-style (excl. adversarial)** | **66.23%** | — |

Per-category (Sonnet 4.6 judge):

| Category | n | Accuracy |
|---|---|---|
| **adversarial** (unanswerable) | 446 | **91.3%** |
| single-hop | 841 | 82.0% |
| temporal | 321 | 57.3% |
| multi-hop | 282 | 42.2% |
| open-domain | 96 | 28.1% |

### Honest "I don't know" is a measured capability, not a claim

The 91.3% on adversarial questions is the direct measurement of whether the reader correctly refuses to answer when the conversation doesn't contain the information. This is a specific failure mode for most memory systems — they commit to a best-guess rather than decline, because refusing looks like a capability gap. On LoCoMo's adversarial set, the rlat reader correctly says "I don't have enough information" on **91 of every 100** unanswerable questions. The grounded-by-default posture isn't just a prompt-engineering preference; it's a measured, verifiable behaviour.

This is also why the leaderboard convention *excludes* adversarial: a system that aggressively refuses inflates its overall score without solving the harder answerable categories. The 66.23% leaderboard number is the one to compare against Mem0/Letta/Zep.

### Strongest valid takeaway

End-to-end on LoCoMo, rlat is in mid-tier memory-system territory (Mem0 / Letta class) with a measured strength in refusal accuracy that most systems don't report. Single-hop is well-handled (82%); multi-hop and open-domain are the clear weakness, as with the published leaders.

### Limits and caveats

- **Judge model matters.** The numbers above use Claude Sonnet 4.6 as the LLM judge. The LoCoMo leaderboard convention is GPT-4o; a GPT-4o rejudge is pending and likely shifts the number by 2–4 pt (Claude judges tend to be slightly stricter than GPT judges on this task). Do not cite the 66.23% against a published leader number without noting the judge difference.
- **Reader matters.** The reader was Claude Haiku 4.5 direct-prompted (no citation-style injection; the built-in APIReader's `[N]` citation template hurts LoCoMo scores by ~15 pt because it instructs the reader to hedge, and we bypass it for this benchmark).
- **LME-proven retrieval features don't transfer.** On LoCoMo's per-conversation cartridge shape (~150 chunks), the LME "full-stack" retrieval features regress hard: `subgraph` spectral expansion −30 pt (catastrophic — dilutes top-k in small cartridges), `diversify_by_session` −3.3 pt (small sessions carry complementary adjacency, not redundancy). `prefer_recent` and cross-encoder rerank are neutral. The shipped LoCoMo config uses the cartridge baseline + auto-routed retrieval without these features.
- **Tier-weighted LayeredMemory regresses here too** (−11 pt aggregate on a 2-conversation smoke, with −27 pt craters on multi-hop and temporal for conv-30 — same 30d/180d tier-weight cliff the LME P1-tier hit). Since LoCoMo conversations span 2–3 months, the recency tier cliff down-weights older-session evidence that is the actual answer. The LoCoMo ship number uses cartridge retrieval, not tier-weighted recall.
- **Fact extraction (LLM-assisted, `--extract-facts`) was tested and deferred.** The current fact-row ingest displaces raw chunks in top-k, cratering single-hop on some conversations. The path forward is facts-as-ranking-signal rather than facts-as-context-rows; that rewrite is post-launch work.

### Raw outputs

- `benchmarks/results/locomo/phase2_cartridge_e5_all10.json` — Haiku-judged baseline
- `benchmarks/results/locomo/phase2_cartridge_e5_all10_sonnet_judge.json` — Sonnet-judged rejudge
- `benchmarks/results/locomo/smoke/s1_baseline.json`, `s2_prefer_recent.json`, `s3_diversify.json`, `s4_cross_encoder.json`, `s5_subgraph.json`, `s6_combo_pr_ce.json` — per-feature ablation smokes
- `benchmarks/results/locomo/smoke/s7_facts_5qa.json`, `s8_facts_30qa.json`, `s9_facts_topk20.json` — LLM-assisted fact-extraction smokes (deferred path)
- `benchmarks/results/locomo/smoke/s10_layered.json` — LayeredMemory tiered-retrieval smoke

## Comparative / Obsidian

### What this benchmark tests

This comparison asks whether RL beats a strong Obsidian LLM wiki workflow, not a straw-man baseline. The tested vault was enriched with summaries, keywords, aliases, and 11,000+ wikilinks.

### Headline numbers

| | Obsidian (best) | rlat (reranked) |
|---|---|---|
| **Recall@5** | 0.81 | 1.00 |
| **MRR** | 0.714 | 0.929 |
| **Failed retrieval** | 19% | 0% |

### Strongest valid takeaway

RL outperformed the best tested Obsidian workflow on the published comparison. For grounded assistant retrieval over this corpus, the practical conclusion is that RL is the stronger system.

### Limits and caveats

- this is a retrieval claim, not a claim about every use of Obsidian as a note-taking environment
- the benchmark is strongest as evidence about grounded assistant retrieval, not every possible graph-centric workflow
- an exploratory 10-question multi-hop add-on exists in one comparative result payload, but it is not the canonical product-positioning benchmark and should not override the main published comparison

### Raw outputs

See `benchmarks/results/comparative/`.

## Claims We Can Defend Today

These claims are strong enough to use in repo-first positioning:

- RL gives you a portable semantic knowledge artifact, not just retrieval plumbing
- the full RL pipeline is strong on technical retrieval workloads
- RL can reduce assistant token waste dramatically compared with whole-file reading workflows
- RL materially improves grounding on the current answer-quality evaluation
- the system has meaningful cross-corpus evidence, even if the gains are not uniform across every dataset
- RL retrieves the correct session in the top-5 for 92% of LongMemEval_s long-horizon conversational-memory queries (retrieval metric, not end-to-end QA accuracy)
- End-to-end on LoCoMo (retrieve → read → judge, 1986 QA), rlat scores 66.2% leaderboard-style / 71.9% overall with a Claude Sonnet 4.6 judge — mid-tier memory-system territory alongside Mem0 and Letta
- The LoCoMo adversarial subset measures correct refusal on unanswerable questions — rlat scores 91.3%, evidence that grounded-by-default refusal is a measured capability, not a prompt-engineering preference

## How To Read These Results

These benchmarks support a strong case for Resonance Lattice, but they still need to be read with the right scope:

- the results show that RL is strong on the evaluated technical retrieval and grounding workloads
- they do not mean RL will beat every retrieval system, every corpus, or every benchmark
- the strongest gains come from the full RL pipeline, not dense retrieval in isolation
- advanced diagnostics such as contradiction signals are useful aids, not perfect truth detectors
- experimental encoder or memory paths should be treated as research until they are backed by the same level of benchmark evidence as the production path

## Where The Raw Evidence Lives

The repo preserves scripts and outputs under:

- `benchmarks/results/retrieval_gate/`
- `benchmarks/results/ablation/`
- `benchmarks/results/token_efficiency/`
- `benchmarks/results/llm_judge/`
- `benchmarks/results/beir/`
- `benchmarks/results/longmemeval/`
- `benchmarks/results/locomo/`
- `benchmarks/results/comparative/`

Use [Benchmark Runbook](/docs/benchmark-runbook) when you need reproduction steps, thresholds, baseline policy, or CI gating guidance.
