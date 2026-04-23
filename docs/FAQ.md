---
title: FAQ
slug: faq
description: Practical answers about what Resonance Lattice is, how it compares to alternatives, and where it is strongest today.
nav_group: Reference and Support
nav_order: 10
aliases:
---

# FAQ

This is the long-form support page for repo visitors evaluating Resonance Lattice.

Use it when you want:

- the longer explanation of what the product is and why it matters
- direct comparisons with grep, standard RAG, vector databases, and Obsidian
- the practical answer to "I already have an Obsidian wiki, skills, or memory. Why bother?"
- practical answers about adoption, integration, and limits
- the fuller benchmark interpretation behind the headline claims
- the longer trust and limitations story behind the repo-first positioning

For the product thesis, see [Overview](/docs/overview). For the main evidence page, see [Benchmarks](/docs/benchmarks).

---

## What It Is and Why It Matters

### What is the one-sentence version?

Portable semantic context for AI assistants. Frontier retrieval without retrieval engineering.

### What does this solve and why does it matter?

Most knowledge retrieval today relies on an LLM in one of two ways:

- stuff files into a prompt and hope for the best
- bolt vector search onto a generative model and trust the output

Both approaches create the same problems:

- limited control over what context is actually being used
- poor visibility into what was retrieved and why
- brittle workflows that depend on a specific model stack
- hallucination rates that make outputs hard to trust

Resonance Lattice takes a different approach. It builds a portable semantic model of your corpus as a single `.rlat` file you own as a local file. You control what goes in, you can inspect what the model appears to cover, and you get ranked evidence passages grounded in your actual sources.

A strong `README`, `CLAUDE.md`, or project guide is still valuable. Resonance Lattice is not trying to replace that human-written layer. It solves a different problem: making the wider corpus queryable, inspectable, and reusable as grounded assistant context.

In internal evaluation, feeding rlat context to an LLM reduced hallucinations from `78%` to `16%` and lifted fact recall from `0.27` to `0.91`. The retrieval pipeline itself achieves `Recall@5 = 1.00` with `0%` failed retrieval on the internal Fabric benchmark. The practical takeaway is that Resonance Lattice is designed to give you retrieval and assistant context that is easier to inspect, reuse, and trust than ad-hoc prompt stuffing or dense-only search on its own.

For the fuller product thesis, see [Overview](/docs/overview). For the benchmark backing behind these claims, see [Benchmarks](/docs/benchmarks).

### How should I think about Resonance Lattice?

A useful way to think about it is:

**Portable semantic knowledge layer for assistants**

Resonance Lattice does not fit neatly into the standard buckets:

- it is not a vector database
- it is not an embedding model provider
- it is not a hosted RAG platform
- it is not an LLM wrapper

Those products usually focus on infrastructure, model access, or managed pipelines. Resonance Lattice gives you something different: a portable, inspectable semantic artefact you can query, compare, compose, and plug into assistant workflows. It is the abstraction layer that makes backbone power simple, portable, and swappable.

Another useful mental model is:

- your `README` explains the project to humans
- your `.rlat` knowledge model makes the broader corpus queryable to tools and assistants

They work best together, not as substitutes for each other.

### How does it work?

Resonance Lattice is a **three-layer semantic router**:

1. **Field routes** — a fixed-size latent tensor (~80 MB) models semantic structure. Given a query, the field returns the handful of chunks most likely to be relevant.
2. **Lossless store serves** — those chunks are resolved through a lossless store that reads the actual source file. The store is authoritative; the field is a fast router over it.
3. **Reader synthesizes** — an LLM (Claude, etc.) composes the final answer from the served passages, if you want synthesis. The router works fine without one.

`rlat build` chunks your files, encodes each chunk with a local encoder, and packages everything into a single `.rlat` knowledge model. When you query, your question is encoded the same way, resonated against the field, and the store resolves the top matches back to source text.

After a build, the practical result is one file you can use to:

- ask semantic questions and get ranked evidence
- inspect what the corpus appears to cover
- compare versions of project knowledge
- plug grounded retrieval into assistants through CLI, MCP, or HTTP

You choose how the knowledge model is packaged. The store comes in three serving topologies — one abstraction, three backends:

| Store mode | What you get | Pick when |
|-----------|-------------|-----------|
| **`local`** (default) | Thin `.rlat`; source files live on disk and are resolved via `--source-root` at query time | Developing against a working copy; large corpora where you don't want to bundle source |
| **`bundled`** | Self-contained `.rlat` with raw source files packed inside (zstd frames) | Shipping a self-contained artifact — HF Hub, CI, offline, a release with provenance |
| **`remote`** | Thin `.rlat` that points at a public GitHub repo, SHA-pinned with a local cache | Pointing at an upstream repo you don't own; query/load never touches the network |

The historical name **`external`** is still accepted as a synonym for `local`. Legacy **`embedded`** mode (pre-chunked SQLite store) is deprecated and will be removed in v2.0.0 — it is *not* the same as `bundled`: `bundled` stores whole files and preserves every retrieval feature, `embedded` stored pre-chunked text and lost the whole-file view.

```bash
# Default — thin knowledge model, source files on disk
rlat build ./docs ./src -o project.rlat

# Self-contained artifact
rlat build ./docs ./src -o project.rlat --store-mode bundled

# Remote — points at an upstream GitHub repo
rlat build https://github.com/MicrosoftDocs/fabric-docs -o fabric-docs.rlat
rlat freshness fabric-docs.rlat
rlat sync fabric-docs.rlat
```

For more on the three modes — format details, freshness model, when to pick each — see [Storage Modes](/docs/storage-modes).

### What does normal usage look like?

For most users, the day-one workflow is:

1. build one knowledge model from your docs and code
2. ask a few real questions with `rlat search`
3. inspect coverage with `rlat profile` or `rlat xray` if you want to understand the corpus better
4. use `rlat resonate` or MCP when you want assistant-ready context
5. refresh the knowledge model with `rlat build` or `rlat sync` as the project changes

If you already have a good `README` or `CLAUDE.md`, keep it. `summary` is best thought of as a supplemental machine-generated context file for assistants, not a replacement for human-written project documentation.

### Why does portability matter so much?

Portability is not just a packaging detail. It changes how you can work with the knowledge layer day to day.

If your knowledge layer is a file:

- you can version it
- you can diff it
- you can archive it
- you can move it between machines
- you can attach it to branches, releases, or projects
- you can share it without standing up a service

That is materially different from the usual retrieval story, where the "knowledge base" lives inside a hosted system, a database, or a pipeline you have to recreate.

### What do you mean by "semantic model"?

Not a keyword index. Not a vector database. A model of what your corpus appears to know — one you can query for evidence, but also inspect (`rlat profile`), compare across versions (`rlat compare`), and compose with algebra (`merge`, `diff`, `forget`).

The distinction is important:

- a search index answers queries
- a semantic model can also describe its own shape, coverage gaps, and changes over time

That is why Resonance Lattice can expose profiling, x-ray, locate, probe, compare, and algebra operations instead of only "top-k chunks." In user terms, you do not just get answers. You can also inspect whether the corpus seems thin, saturated, contradictory, or unevenly documented.

### Why is it built on maths instead of an LLM?

Resonance Lattice is fundamentally a mathematical system, not a generative one. For users, that means the knowledge model behaves predictably when you add, remove, merge, or compare sources. The field has algebraic structure, and every operation — adding a source, removing a source, merging two knowledge models, diffing versions — is a defined mathematical operation with known properties.

This matters because maths gives you guarantees that probabilistic systems cannot:

- **Exactness** — removing a file returns the field to the exact state it was in before that file was added. No approximation, no retraining, no drift.
- **Composability** — merge is order-independent. The sequence you combine knowledge models does not change the result.
- **Inspectability** — coverage, signal quality, saturation, and gaps are properties of the field itself, not metadata bolted on later.
- **Determinism** — the same inputs always produce the same field. No temperature, no sampling, no stochastic variation.
- **No LLM dependency** — build, query, profile, compare, and algebra operations run locally with a small encoder model.

Most retrieval systems treat embeddings as opaque vectors in a list. Resonance Lattice treats the field as a mathematical object you can reason about, and that is what makes operations like `forget`, `diff`, `profile`, and `xray` possible at all.

### What can I learn about the model itself?

If you only care about retrieval, you can ignore this at first. But these commands are part of what makes Resonance Lattice more than search. They expose the field's internal structure — insights no vector database can produce because the field has algebraic structure, not just a list of vectors:

| Command | What it tells you |
|---------|------------------|
| `rlat xray corpus.rlat` | Corpus-level health: signal quality, saturation estimate, and diagnostic flags |
| `rlat locate corpus.rlat "query"` | Where a query sits in the knowledge landscape: what the field knows, what it *doesn't* know, and where to look next |
| `rlat probe corpus.rlat <recipe>` | Quick insight recipes answering specific questions about novelty, saturation, coverage gaps, and more |

These work because the field carries intrinsic semantic properties of the corpus — not metadata bolted on after the fact.

### How should I think about backbones and encoder choice?

Most users do not need to overthink this on day one. The simplest way to think about it is:

- the encoder matters
- the pipeline matters more than dense-only search
- the knowledge model workflow should survive encoder changes

Resonance Lattice ships with **named encoder presets** so you can switch backbones without tracking HuggingFace model IDs, prefix conventions, or pooling strategies. Run `rlat encoders` to see available presets, or pass any preset name to `--encoder`:

```bash
rlat build ./docs -o project.rlat --encoder qwen3-0.6b
rlat build ./docs -o project.rlat --encoder arctic-embed-2
```

Available presets cover a range of trade-offs:

| Tier | Preset | Params | Context | Notes |
|------|--------|--------|---------|-------|
| **Default** | `bge-large-en-v1.5` | 335M | 512 | CLI default (since 2026-04-20); portable, CPU-friendly, strong on docs/science/code |
| **Alternative** | `e5-large-v2` | 335M | 512 | Opt-in; wins on ArguAna-like counter-argument corpora |
| **Fast** | `nomic-v2` | 305M active | 8K | MoE efficiency, longer context |
| **Balanced** | `arctic-embed-2` | 335M | 8K | Strong retrieval per parameter |
| **Balanced** | `bge-m3` | 568M | 8K | Multilingual workhorse, 1000+ languages |
| **Quality** | `qwen3-0.6b` | 600M | 32K | Best small model, Apache 2.0 |
| **Quality** | `nemotron-1b` | 1B | 8K | Strong open-weight, NVIDIA backed |
| **Quality** | `qwen3-4b` | 4B | 32K | Serious quality jump |
| **Max** | `qwen3-8b` | 8B | 32K | Frontier quality (BEIR-5 avg 0.500), needs ~16 GB GPU |

You can also pass any raw HuggingFace model ID to `--encoder` for unlisted models.

Three encoders are well-supported and benchmarked — BGE-large-en-v1.5 (default, portable), E5-large-v2 (opt-in, strongest on counter-argument retrieval), and Qwen3-Embedding-8B (opt-in, frontier quality at 16 GB GPU). See [Encoder Choice](/docs/encoder-choice) for the per-workload decision guide. All three use pretrained weights with random projection heads — trained heads were tested and rejected because they broke build/query parity.

The practical mindset is not "pick the perfect model forever." It is:

- start with the default
- keep the same `.rlat` workflow
- upgrade quality, multilinguality, or latency later by switching presets

### What is the current recommended encoder setup?

The default encoder is `BAAI/bge-large-en-v1.5` (flipped from E5-large-v2 as the CLI default on 2026-04-20). That is the safe starting point — it is local, fast, well-tested, and runs on ordinary hardware.

Three encoders are well-supported and have measured BEIR coverage:

- **BGE-large-en-v1.5** (default) — best general choice across docs, science, and code
- **E5-large-v2** — opt-in, strongest on counter-argument corpora (ArguAna-like)
- **Qwen3-Embedding-8B** — opt-in, frontier quality, needs ~16 GB GPU

Switch explicitly when you have a reason to:

```bash
# Frontier quality, 16 GB GPU required
rlat build ./docs -o project.rlat --encoder qwen3-8b

# Multilingual workhorse, 1000+ languages
rlat build ./docs -o project.rlat --encoder bge-m3
```

Run `rlat encoders` to see all available presets. See [Encoder Choice](/docs/encoder-choice) for the full decision guide.

### What do I use this for?

- **Grounding LLMs in your data** — feed controlled, evidence-backed context to any LLM so it reasons over your sources instead of hallucinating from training data
- **CLI search** — query your docs and code semantically without leaving the terminal
- **Version comparison** — compare the knowledge in two knowledge models to see what changed
- **Assistant context packaging** — generate prompt-ready or assistant-ready context when you want machine-generated support beside your human-written docs
- **Coverage profiling** — see what your documentation actually covers and where it is thin
- **Portable project memory** — carry a project's semantic knowledge as one local artefact instead of a hosted retrieval stack

### What output formats are available?

The `--format` flag controls how results come back, so the same query serves different workflows:

| Format | What you get | Use it for |
|--------|-------------|-----------|
| **text** | Human-readable passages with source paths, coverage, and related topics | Reading in the terminal |
| **json** | Structured object with scores, passages, metadata, and coverage arrays | Scripts, pipelines, and programmatic consumption |
| **prompt** | Pre-formatted context block ready to paste into an LLM conversation | Manual LLM workflows |
| **context** | Compressed context with injection framing (used by `resonate`) | Automated assistant integration |

```bash
rlat search project.rlat "how does auth work?" --format text
rlat search project.rlat "how does auth work?" --format json
rlat resonate project.rlat "how does auth work?" --format context
```

### What are the unique features?

- **Knowledge Model algebra** — merge, diff, and forget knowledge artifacts with mathematical guarantees
- **Semantic profiling** — inspect a knowledge model's health, coverage, and structure
- **Field X-Ray** — corpus-level diagnostic showing signal quality, saturation, and knowledge gaps
- **Query positioning** — see where a question sits in the knowledge landscape and what the field does not cover
- **RQL insight recipes** — curated pipelines answering specific questions about your corpus
- **Enriched query** — one call returns passages, coverage, related topics, and contradiction signals
- **Assistant context packaging** — generate supplemental assistant context from the same model you search
- **Fully local after first encoder download** — no cloud service, no API keys, no LLM required for the core workflow
- **Single-file portable** — one `.rlat` knowledge model you can move like any other file

### Who is this for?

The best early fit is:

- solo developers who want better assistant grounding
- local-first power users
- people working with private or sensitive code and docs
- agent builders who want an inspectable retrieval layer
- teams that want one portable artefact instead of a service dependency
- people who care about context quality but do not want to become retrieval engineers

The CLI is approachable, but the underlying model has enough depth to support more technical users who care about profiling, algebra, encoder choices, and evaluation.

### Who is it not for yet?

Resonance Lattice is a weaker fit if your primary need is:

- internet-scale hosted retrieval infrastructure
- turnkey multi-tenant SaaS retrieval
- multimodal production retrieval across images/audio/video today
- frontier multilingual retrieval as the default out of the box
- exact symbol lookup as the only problem you need to solve

### Why was this created?

Resonance Lattice started from a simple question: is there a better way to retrieve and manage project knowledge than "throw chunks into a model and hope the prompt works"?

The result is a system where the model itself is inspectable, composable, and portable — not just a lookup table. That solves a different class of problem:

- context bloat
- opaque coverage
- service dependency
- throwaway embeddings
- assistant workflows that break every time the model stack changes

### Where does it work best today?

Resonance Lattice is strongest today when you want:

- grounded local/private context for AI assistants
- semantic search over code and docs
- evidence-backed retrieval instead of whole-file stuffing
- inspectable corpus diagnostics such as profile, xray, locate, and compare
- a portable knowledge artefact you can version, move, diff, and reuse

It is a weaker fit when you primarily need:

- internet-scale hosted retrieval infrastructure
- exact symbol lookup with no semantic layer
- frontier multilingual retrieval as the default out of the box
- multimodal production retrieval across images, audio, and video today

---

## How It Compares

Use this section when the main question is "why this instead of the obvious alternative?" These answers are the objection-handling companion to [Overview](/docs/overview), with benchmark references where the comparison depends on measured results.

### Does it replace skills, CLAUDE.md, and memory?

No. They are complementary layers:

- **CLAUDE.md** gives an assistant rules and project conventions
- **Skills** give it executable capabilities
- **Memory** gives it persistent facts across conversations
- **Resonance Lattice** gives it grounded project knowledge — the actual content of your codebase and docs, semantically indexed

Resonance Lattice fills the "what does this project actually contain" gap. It does not replace the other layers; it feeds them better context.

A useful way to think about it: skills today load the same static document regardless of the query. Most of that context is irrelevant to any given request. With RL **knowledge-model-backing**, a skill declares which knowledge models it draws from, and context adapts to the specific question:

| Tier | Source | What it provides |
|------|--------|-----------------|
| **1. Static** | SKILL.md header | Workflow structure — always loaded |
| **2. Foundational** | Skill-authored queries | Baseline domain knowledge the skill always needs |
| **3. Specific** | User query | Context unique to this request |
| **4. Derived** | Caller-supplied queries | Implicit needs the user didn't know to ask for |

Tiers 1-3 are automatic. Tier 4 accepts additional queries from the caller (e.g., LLM-generated search terms passed via `--derived`) — this lets an orchestrating agent surface knowledge the user didn't express. Mode-aware gating controls injection: `augment` suppresses when the model already knows the topic, `knowledge` uses a softer threshold, `constrain` always injects (for compliance or regulatory skills where the model must not improvise). Skills without `cartridge-*` fields work exactly as they do today.

We also ship an **rlat skill for Claude Code** that teaches Claude when and how to use rlat automatically. When installed, Claude will use `rlat search` for conceptual questions and fall back to grep for exact symbol lookup.

```bash
rlat search project.rlat "how does auth work?" --format json
```

### Why not just use grep? What's the difference?

Grep and Resonance Lattice are complementary — they solve different problems and work well together. Grep finds exact text matches: give it a keyword, it finds every line that contains it. That is powerful when you already know what you are looking for.

What grep cannot do is understand how knowledge relates to itself. It does not know that "authentication" and "login flow" are about the same thing, or that a design decision in one file contradicts guidance in another. It searches keywords blindly, with no awareness of meaning.

Resonance Lattice fills that gap. It finds conceptually relevant content even when the words do not match, and returns semantic context — coverage, related topics, contradictions — alongside the results.

Use grep when you need a specific string. Use Resonance Lattice when you need to understand what a corpus knows about a topic.

There is also a practical cost difference. Grep gives you entire files — and when you feed those to an LLM, most of the tokens are noise. We benchmarked this directly:

| Approach | Tokens per query | What you get |
|----------|-----------------|-------------|
| **grep + read top 5 files** | 37,154 | Raw file text — no ranking, no structure, no coverage signal |
| **rlat search (top 10)** | 1,518 | Ranked passages with per-band coverage, related topics, and source attribution |

That is **24.6x fewer tokens** with structured context — ranked passages with coverage signals and source attribution, instead of raw file text.

**Benchmark methodology**: 50 evaluation questions against a 2,266-file Fabric documentation corpus. For each question, we extracted keywords and grep-matched the top 5 files by keyword count, then read those files in full and counted tokens. `rlat search` ran `rlat search --format prompt --top-k 10` on the same questions against a knowledge model built from the same corpus. Token counts via tiktoken (`cl100k_base`). Median compression was `19.9x`; all 50 questions produced valid comparisons. Full results live in the repo's `benchmarks/results/token_efficiency/` directory.

### How is it different from standard RAG?

Standard RAG is a retrieval step bolted onto an LLM: chunk, embed, cosine search, stuff into prompt. Resonance Lattice is a semantic model you can query, profile, compare, and compose — with or without an LLM in the loop.

The retrieval pipeline is also benchmark-proven stronger on the internal benchmark:

| | rlat (reranked) | Hybrid RRF | Flat E5 | BM25 |
|---|---|---|---|---|
| **Recall@5** | **1.00** | 0.94 | 0.93 | 0.84 |
| **MRR** | **0.93** | 0.77 | 0.80 | 0.72 |
| **Failed retrieval** | **0%** | 6% | 7% | 16% |

The knowledge model is a portable file, not a service dependency. And you get operations standard RAG setups usually do not offer cleanly:

- merge two knowledge bases
- diff them
- selectively forget a source
- profile what the corpus appears to cover

The full pipeline is what makes the difference. Each stage adds measurable value:

The dense field trades some retrieval quality for a semantic model you can inspect, compose, and profile. The pipeline (lexical injection + reranking) recovers that cost and then some.

### How does it compare to vector databases?

Vector databases and Resonance Lattice solve related but different problems.

Vector databases primarily give you:

- scalable vector storage
- similarity search infrastructure
- hosted or embedded retrieval plumbing

Resonance Lattice primarily gives you:

- a portable knowledge artefact
- inspectable semantic structure
- deterministic knowledge operations
- local workflows that fit assistant usage well

That means the most useful comparison is not "database versus database." It is:

- infrastructure versus portable knowledge artefact

In some setups they are complementary. You might still use a vector DB for scale or serving and use Resonance Lattice when portability, inspectability, or assistant context quality is the higher-value property.

### Why not just ask an LLM directly?

LLMs are trained on the internet — a broad but unreliable source of truth. They do not know your codebase, your internal docs, or your project's current state. When they guess, they hallucinate confidently.

The better approach is to keep their reasoning and action capabilities, but ground them in your own data. LLMs are good at thinking, refactoring, synthesis, and action. The knowledge itself should be yours: curated, portable, and under your control.

That is what Resonance Lattice does. It packages your actual knowledge into a model you own as a local file. You control what goes in, what gets queried, and what gets fed to any LLM. The LLM reasons over your sources instead of inventing answers from internet training data.

In internal evaluation, feeding rlat context to an LLM reduced hallucinations from `78%` to `16%` and lifted fact recall from `0.27` to `0.91`.

You also control *how* the LLM uses your knowledge. The `--mode` flag controls how that context is framed:

| Mode | What it tells the LLM |
|------|----------------------|
| **augment** | Use your own knowledge, but add detail and citations from these sources |
| **constrain** | Answer ONLY from the provided sources — if it's not covered, say so |
| **knowledge** | Base your answer primarily on this context; be transparent about gaps |
| **custom** | Your own system prompt, your rules |

```bash
rlat resonate project.rlat "how does auth work?" --mode constrain --format context
```

### Why not just create an Obsidian vault?

Obsidian is a graph-based knowledge tool — a well-curated vault with wikilinks between documents acts like a graph database for your notes. It is excellent when you want manual structure, note-taking, and graph navigation.

Resonance Lattice is better when the goal is grounded assistant retrieval. It builds a semantic model automatically from your existing files — no manual linking, curation, or LLM required. The model understands meaning, not just keywords or graph structure, and it runs entirely locally with no generative AI in the loop.

We benchmarked both approaches on the same 2,246-document Fabric corpus in the main published comparison. The Obsidian vault was enriched with summaries, keywords, aliases, and 11,000+ wikilinks — a strong best-case baseline for an Obsidian LLM wiki workflow:

| | Obsidian (best) | rlat (reranked) |
|---|---|---|
| **Recall@5** | 0.81 | 1.00 |
| **MRR** | 0.714 | 0.929 |
| **Failed retrieval** | 19% | 0% |

Short version:

- Resonance Lattice is the stronger retrieval layer for grounded assistant use
- Obsidian can still be a good workspace, but the benchmark case here is that RL beats the Obsidian LLM wiki approach on retrieval quality

If you like Obsidian as the interface, use the [Obsidian plugin](https://github.com/tenfingerseddy/resonance-lattice/tree/main/obsidian-plugin) and let RL provide the retrieval layer underneath.

### How much of the quality comes from the backbone versus the pipeline?

The backbone matters, but the practical result you experience comes from the full system:

- dense semantic retrieval
- lexical evidence injection
- reranking
- knowledge model structure
- assistant-friendly warm-path usage

That matters because the dense field alone is not the whole story. If you are evaluating Resonance Lattice, focus on answer quality, evidence quality, inspectability, and workflow fit rather than only asking whether one encoder beats another on a narrow benchmark.

---

## Benchmarks And Evidence

This section summarizes the benchmark story in FAQ form. For the dedicated evidence page, raw result families, and a clearer split between defendable claims and caveats, see [Benchmarks](/docs/benchmarks).

### What do the internal retrieval benchmarks show?

On a 24,635-chunk Microsoft Fabric documentation corpus with 100 evaluation questions:

| | rlat (reranked) | Hybrid RRF | Flat E5 | BM25 |
|---|---|---|---|---|
| **Recall@5** | **1.00** | 0.94 | 0.93 | 0.84 |
| **MRR** | **0.93** | 0.77 | 0.80 | 0.72 |
| **Failed retrieval** | **0%** | 6% | 7% | 16% |

The practical read is:

- the full pipeline is strong
- the dense field alone is not the whole story
- pipeline quality matters more than any single dense component

### What does the pipeline ablation show?

The ablation shows that quality does not come from the dense field alone. The retrieval quality comes from combining:

- dense semantic retrieval
- lexical evidence injection
- reranking

That is important because it shows where the quality actually comes from. The main gain comes from the knowledge model + pipeline + assistant workflow, not from dense retrieval alone.

### What should I make of the BEIR / cross-corpus results?

The five-BEIR coverage (SciFact, NFCorpus, FiQA, ArguAna, SciDocs) is how we measure cross-corpus generalisation. None of the three well-supported encoders wins everywhere, and we report all three — that's the honest version.

**BEIR-5 best-mode averages (nDCG@10, 2026-04-22 rebench — Qwen3-8B final, BGE figures provisional pending final rebench):**

| Encoder | BEIR-5 avg | Notes |
|---------|-----------|-------|
| **Qwen3-Embedding-8B** | **0.500** | Frontier-adjacent; ~1 pt below `text-embedding-3-large`. Needs ~16 GB GPU. |
| **E5-large-v2** | 0.455 | Strongest on counter-argument retrieval (ArguAna). |
| **BGE-large-en-v1.5** (default) | 0.445 | Wins 4/5 corpora vs E5; loses ArguAna by ~9.7 pts — net −1.0 pt vs E5 on the 5-corpus average. |

**Per-corpus breakdown** (historic E5-large-v2 run, reranker-on where applicable):

| BEIR Dataset | rlat (best) | Mode | Flat E5 | BM25 |
|---|---|---|---|---|
| SciFact (5K) | **0.713** | reranked | 0.735 | 0.665 |
| NFCorpus (3.6K) | **0.360** | reranked | 0.337 | 0.325 |
| FiQA (57K) | **0.393** | reranked | 0.350 | 0.236 |
| ArguAna (8.7K) | **0.492** | dense | 0.501 | 0.315 |
| SciDocs (25K) | **0.189** | dense | 0.158 | 0.158 |

The right way to read this:

- do **not** treat it as evidence that Resonance Lattice will win every external benchmark
- do treat it as evidence that the pipeline exceeds flat E5 on 3 of 5 datasets under E5-large-v2
- the reranker helps on factual/technical corpora (SciFact, NFCorpus, FiQA) but can hurt on argument-style retrieval (ArguAna, SciDocs) where dense-only is stronger
- the default encoder (BGE) is the right starting point for docs, science, and code corpora — switch to E5 if your workload is ArguAna-like counter-argument retrieval, or Qwen3-8B if you have a 16 GB GPU and want SOTA quality
- fit for your own corpus matters more than any single benchmark headline

All per-corpus numbers in the breakdown table use pretrained encoder weights only — no trained heads, no checkpoints.

### How does rlat compare to Mem0, Letta, Zep, and MemMachine on memory benchmarks?

The standard memory-benchmark headline those systems quote is [LoCoMo](https://github.com/snap-research/locomo) — 10 multi-month conversations, 1986 questions across five categories (single-hop, multi-hop, temporal, open-domain, adversarial). It measures the full retrieve → read → judge pipeline, not just retrieval, so the whole memory system is under test.

On Phase 2 baseline with Claude Sonnet 4.6 as the LLM judge:

| System | Leaderboard-style (excl. adversarial) |
|---|---|
| Mem0 | ~67% |
| Letta | ~68% |
| **rlat (Phase 2 baseline)** | **66.23%** |
| MemMachine | ~73% |
| Zep | ~79% |

rlat lands squarely in Mem0 / Letta territory. MemMachine and Zep are meaningfully ahead. The published leaders use GPT-4o as judge; our number is Sonnet 4.6 and a GPT-4o rejudge is pending — expect ±2-4 pt shift when that lands.

What is not in the table: rlat scores **91.3% on the adversarial category** (LoCoMo's unanswerable-question subset) — the reader correctly refuses to commit to an answer on 91 out of every 100 unanswerable questions. That is a direct measurement of the "honest I-don't-know" capability, and it is higher than any other category. It is also why the leaderboard convention excludes adversarial: a system that aggressively refuses inflates the overall number without solving the harder answerable categories. Our overall-incl-adversarial score is 71.85%.

For the full per-category breakdown, feature ablations (which LME-proven retrieval features transfer and which crater), and the raw output files, see [Benchmarks — LoCoMo](/docs/benchmarks#locomo). For the memory-architecture positioning alongside these results, see [LLM Memory](/docs/llm-memory).

### Does rlat know when it doesn't know? What if I ask something my knowledge model can't answer?

Yes, and it is measured.

The reader is instructed to refuse when the retrieved evidence is thin — to say "I don't have enough information" instead of inventing an answer. This is the opposite of the default LLM failure mode (fabricate a plausible-sounding answer to preserve fluency). The refusal posture is enforced two ways:

1. **Reader system prompt.** `rlat ask` uses a grounded-only synthesis prompt that tells the reader to decline when the evidence doesn't cover the question. Deterministic, prompt-level behaviour.
2. **Citation verification.** The reader's `[N]` citations are parsed, validated against the evidence set, and (with `build_bundle`) quote-checked against the source bytes. Fabricated references can't reach the answer, and quote drift surfaces with a `✗` marker.

Measured evidence on LoCoMo: **91.3% correct refusal** on the 446-question adversarial subset. The questions are deliberately unanswerable from the conversation; the reader correctly declines on 91 out of every 100 of them. See [Honest Claims — Honest refusal on unanswerable questions is measured](/docs/honest-claims#honest-refusal-on-unanswerable-questions-is-measured).

Practical consequence: if your corpus doesn't cover a topic, you get a clean "I don't have enough information" instead of a confident-sounding wrong answer. For legal, medical, regulatory, and compliance workflows where a wrong answer is worse than no answer, this is a meaningful default.

### Where does the LLM grounding number come from?

The answer-quality benchmark compares an LLM answering questions without rlat context versus the same model answering with rlat-supplied context. The judge scores outputs on:

- accuracy
- completeness
- groundedness
- hallucination rate
- fact recall

On the current internal run, adding rlat context reduced hallucination rate from `0.78` to `0.16` and lifted fact recall from `0.27` to `0.91`.

### How should I interpret the Obsidian comparison?

The Obsidian comparison is useful because it is not a trivial baseline. The tested vault was enriched with:

- summaries
- keywords
- aliases
- wikilinks

That makes it a stronger-than-average Obsidian LLM wiki baseline. The point of the comparison is that RL outperformed that stronger baseline on grounded retrieval. Treat the main published comparison as the canonical message. Do not let the exploratory 10-question multi-hop add-on override it.

### Where can I inspect the raw benchmark outputs?

The repo contains benchmark outputs under:

- `benchmarks/results/ablation/`
- `benchmarks/results/token_efficiency/`
- `benchmarks/results/llm_judge/`
- `benchmarks/results/beir/`
- `benchmarks/results/comparative/`

The FAQ summarizes them. The repo preserves the raw JSON and benchmark scripts. For the consolidated evidence page and the benchmark contract, see [Benchmarks](/docs/benchmarks) and [Benchmark Runbook](/docs/benchmark-runbook).

---

## Practical Usage

### How does it work if I change a file?

It is not a full rebuild. The knowledge model tracks every file by content hash — when you run `rlat build` or `rlat sync`, only changed, added, or removed files are processed. Unchanged files are skipped entirely.

Under the hood, adding or removing a file is an exact algebraic operation on the field. The maths guarantees that removing a file returns the field to the same state as if that file had never been added. There is no drift, no approximation, and no need to retrain or rebuild.

Three commands handle different update workflows:

| Command | What it does |
|---------|-------------|
| `rlat build` | Incremental by default — skips unchanged files via content hash |
| `rlat add` | Add new sources to an existing knowledge model without touching anything else |
| `rlat sync` | Three-phase update: detect deleted files and remove them, then add new and updated files |

If the encoder changes (new model or checkpoint), the manifest detects the mismatch and re-encodes affected files automatically. For projects that change frequently, `rlat sync` can be scripted into your workflow — it handles the full add/update/remove lifecycle in one pass.

### Does it need an LLM? Is it fast? Does it work offline?

No LLM is required. Build, query, profile, compare, and algebra operations run locally with a small encoder model — not a generative LLM.

Important nuance:

- the first `rlat build` downloads the default encoder (~1.2 GB) from HuggingFace and caches it locally
- after that, everything works offline

You can optionally feed Resonance Lattice output into an LLM for synthesis, but the tool itself is LLM-free.

If you do want LLM synthesis, pairing rlat with a local model (Ollama, llama.cpp, etc.) gives you a fully private stack after the encoder is cached.

There are two performance paths:

- **cold** — first query, loads the knowledge model and encoder from disk
- **warm** — subsequent queries with everything already in memory

| Operation | Cold | Warm |
|-----------|------|------|
| Field resonance (raw) | ~8 ms | ~8 ms |
| Dense retrieval | ~220 ms | ~78 ms |
| Full hybrid + reranked pipeline | ~800 ms | ~80 ms |

The first query pays a one-time load cost. After that, a background worker keeps everything in memory, so subsequent queries hit the warm path automatically.

### What is the "warm path" and why does it matter?

Warm path means:

- the knowledge model is already loaded
- the encoder is already loaded
- the background worker is already running

That matters in assistant workflows because repeated queries inside one coding session should feel interactive. This is one reason the MCP server matters: it keeps the knowledge model hot instead of paying repeated startup cost.

### What are store modes?

The canonical CLI flag is `--store-mode {bundled,local,remote}`:

| Mode | What's in the `.rlat` | Pick when |
|------|------------------------|-----------|
| `local` (default) | Field + registry; source files resolved from disk via `--source-root` | Working copy development; large corpora; you don't want to bundle source |
| `bundled` | Field + registry + raw source files packed inside (zstd frames) | Self-contained artifact — HF Hub, CI, offline, release bundles |
| `remote` | Field + registry + HTTP pointer to a SHA-pinned upstream GitHub repo + local cache | Pointing at an upstream repo you don't own; SHA-pinned freshness via `rlat freshness` / `rlat sync` |

`external` is the historical name for `local` and is still accepted. `embedded` (legacy pre-chunked SQLite store) is **deprecated** and removed in v2.0.0 — rebuild old `embedded` knowledge models with `--store-mode bundled` or `--store-mode local`. See [Storage Modes](/docs/storage-modes) for the bundled-vs-embedded distinction.

Local mode is useful when you want:

- a smaller knowledge model
- to avoid embedding source text
- to keep evidence current at query time from local files
- to share semantic structure without shipping the evidence store

Bundled mode is the right answer when the knowledge model itself needs to be the distributable — a release artifact, a Hugging Face Hub upload, or a self-contained offline demo. Remote mode is the right answer when the corpus is an upstream repo you don't own: `rlat freshness` and `rlat sync` manage upgrades lockfile-style without ever touching the network at query time.

### What are the bands?

The field is organised into multiple semantic bands. They represent different levels of semantic abstraction, roughly moving from broad subject area through to more lexical or surface-level matches.

In the CLI reference, the working labels are:

- `domain`
- `topic`
- `relations`
- `entity`
- `verbatim`

These are useful labels, not guaranteed ontology categories. They are working descriptions of how the bands behave, not a promise that every corpus will separate cleanly into those concepts.

### How does encoder selection work?

Resolution order is:

1. `--encoder` flag (preset name or HuggingFace model ID)
2. stored encoder in the knowledge model
3. default encoder (`bge-large-en-v1.5`)

When `--encoder` is a known preset name (like `qwen3-0.6b` or `arctic-embed-2`), Resonance Lattice automatically applies the correct prefix, pooling, and token-length settings for that backbone. If the name is not a preset, it is treated as a raw HuggingFace model ID with default settings.

Most users never need `--encoder`. The knowledge model stores its encoder at build time, and queries restore it automatically.

The practical guidance is:

- use the stored encoder by default
- only override when you intentionally want a different backbone
- treat encoder swaps as benchmarked changes, not casual tweaks
- run `rlat encoders` to see available presets

### What happens if I change encoders or checkpoints?

Changing the encoder is a real semantic change, not just a config toggle. If the encoder metadata no longer matches, the affected files must be re-encoded.

That is why the workflow should be:

- build with a deliberate encoder choice
- keep the knowledge model bound to that encoder (the encoder is stamped in the `.rlat`; querying auto-restores the same preset, so you cannot accidentally query a BGE knowledge model with E5)
- benchmark new encoder choices before adopting them

For most users today, the recommended setup is the default BGE-large-en-v1.5 encoder with pretrained weights only — no trained heads, no checkpoints.

### What are the practical scaling limits today?

The main constraints worth knowing are:

- **initial build cost** — first build is CPU-intensive because files must be chunked and encoded
- **default local backbone** — `bge-large-en-v1.5` is the default; stronger backbones (Qwen3-8B for frontier quality, bge-m3 for multilingual) are available via presets (`rlat encoders`)
- **dense-only mode** — faster, but weaker than the full hybrid + reranked pipeline
- **language support** — the default encoder is strongest in English; multilingual presets like `bge-m3` and `qwen3-8b` are available
- **very large corpora** — the dense field has practical scaling limits; other backends trade memory and retrieval characteristics differently

The important point is that these are backbone and pipeline trade-offs, not workflow trade-offs. The `.rlat` knowledge model remains the stable user-facing abstraction.

### Does it support binary file types?

Yes, with optional dependencies for common formats.

From the CLI reference:

- `.pdf` via `pdfplumber` or `PyPDF2`
- `.docx` via `python-docx`
- `.xlsx` / `.xls` via `openpyxl`

Install them only if you need them:

```bash
pip install pdfplumber python-docx openpyxl
```

### Can I speed up CPU encoding?

Yes. ONNX acceleration provides roughly `2-5x` CPU encoding speedup.

The CLI supports `--onnx`, and it can also auto-detect an ONNX directory alongside the knowledge model in common cases.

### Can I compress the registry?

Yes. Registry quantization gives a useful size/quality trade-off:

- `--quantize-registry 8` gives roughly `~50%` compression with minimal quality loss
- `--quantize-registry 4` gives roughly `~87%` compression with a larger quality trade-off

This is a good option when knowledge model size matters more than the last bit of retrieval fidelity.

### Can I compress the knowledge model file itself?

Yes. The build command supports knowledge model compression:

- `--compression zstd` for better compression ratio
- `--compression lz4` for faster decompression

### Can I share a knowledge model without sharing source text?

Yes. Use field-only export:

```bash
rlat export project.rlat -o shared.rlat --field-only
```

That exports the field and registry without the evidence store. The receiver can query the semantic model, but they cannot read the original embedded source text.

### How do contradictions work?

Contradiction detection is an opt-in part of the search experience. It is useful when you want to surface conflicting guidance or changes in how a concept is described across sources.

Relevant commands and flags:

- `rlat search ... --with-contradictions`
- `rlat contradictions ...`

Use it as:

- useful signal for investigation
- not a claim of perfect logical contradiction detection
- best used as an extra diagnostic layer, not the only decision mechanism

### How do I get started?

```bash
pip install resonance-lattice
rlat build ./docs ./src -o project.rlat
rlat search project.rlat "how does auth work?" --format text
```

That is the shortest path. For a fuller walkthrough — profiling, comparison, assistant context files, MCP integration, and HTTP serving — see [Getting Started](/docs/getting-started).

---

## Assistant Integration

### Which integration path should I choose?

There are three main paths:

| Path | Best for | Trade-off |
|------|----------|-----------|
| **CLI** | universal fallback, scripting, agents with terminal access | no persistent warm process by default |
| **MCP server** | repeated assistant queries in one session | needs config, but gives the smoothest assistant experience |
| **Summary file** | adding a supplemental context layer for assistants beside your existing docs | static snapshot, not live retrieval; a supplement, not a replacement for human-written guidance |

The short advice is:

- use **MCP** for repeated live assistant use
- use **CLI** when you need portability or scripting
- use **summary** when you want an extra machine-generated context file beside your human-written project docs

### How do I integrate with Claude Code?

Two strong approaches:

**Approach 1: Dual primer files**  
Best when you want a machine-generated context layer for assistants alongside your existing project docs and instructions. Resonance Lattice ships two complementary primers:

- **Code primer** (`rlat summary`) — captures what the project *is*: structure, conventions, patterns
- **Memory primer** (`rlat memory primer`) — captures how the work has unfolded: settled decisions, reversals, active threads. Reads from `./memory/` (see [LLM Memory](/docs/llm-memory))

```bash
rlat summary project.rlat -o .claude/resonance-context.md
rlat memory primer ./memory/ -o .claude/memory-primer.md
```

Then reference both from `CLAUDE.md`:

```markdown
@.claude/resonance-context.md
@.claude/memory-primer.md
```

The two primers de-duplicate against each other — topics covered in one are skipped in the other, so you don't pay for the same context twice. Use this as a supplement, not a replacement, for a good `README` or `CLAUDE.md`. Human-written project docs are still the primary place for conventions, architecture intent, and onboarding guidance.

**Approach 2: MCP server**  
Best when you want live semantic search, profile, and compare inside the conversation.

```json
{
  "mcpServers": {
    "rlat": {
      "command": "rlat",
      "args": ["mcp", "project.rlat"]
    }
  }
}
```

Claude Code config location:

- `.mcp.json` in project root

This exposes 19 MCP tools grouped by purpose — search and context (`rlat_search`, `rlat_resonate`, `rlat_compose_search`, `rlat_ask`), info and diagnostics (`rlat_info`, `rlat_profile`, `rlat_compare`, `rlat_locate`, `rlat_xray`, `rlat_health`, `rlat_negotiate`), discovery and freshness (`rlat_discover`, `rlat_freshness`, `rlat_switch`), skill routing (`rlat_skill_route`, `rlat_skill_inject`), and layered memory (`rlat_memory_recall`, `rlat_memory_save`, `rlat_memory_forget`). See [MCP](/docs/mcp#current-tool-surface) for the full table with descriptions.

### Can I use it with GitHub Copilot, Cursor, or other AI tools?

Yes.

- **VS Code / GitHub Copilot** — use `.vscode/mcp.json`
- **Cursor** — use `cursor/mcp.json` or the MCP settings UI
- **Any tool with terminal access** — call `rlat` directly
- **Any tool that wants HTTP** — use `rlat serve`

The CLI and `--format json` output make it usable even without MCP support.

### Is `summary` meant to replace `README.md` or project instructions?

No.

`summary` is useful when you want a machine-generated context file that reflects the wider corpus and can be refreshed automatically. It is helpful as an assistant supplement, especially for large or fast-changing projects.

What it is **not**:

- not a better replacement for a strong `README.md`
- not a substitute for `CLAUDE.md`, `.cursorrules`, or other instruction files
- not the main reason to adopt Resonance Lattice

The primary value remains the knowledge model and query workflow:

- `search` for grounded semantic retrieval
- `resonate` for compact assistant-ready context
- `profile`, `xray`, `locate`, and `compare` for inspection and analysis

### When should I use `search`, `ask`, `resonate`, `summary`, `profile`, or `compare`?

Use this rule of thumb:

- **`search`** — primary query surface; start here when you know you want ranked passages
- **`ask`** — auto-selects the best retrieval lens (search, locate, profile, compare, compose) from the question's intent; use it when you're not sure which command to run
- **`resonate`** — compact context for prompt injection
- **`summary`** — generate the *code primer* (a supplemental assistant context file describing what the project *is*)
- **`memory primer`** — generate the *memory primer* (a supplemental context file describing how the work has unfolded — settled decisions, reversals, active threads). Code primer + memory primer together are the dual-primer system
- **`profile`** — inspect what the knowledge model appears to know
- **`compare`** — compare semantic shape across two knowledge models
- **`xray` / `locate` / `probe`** — diagnostics, positioning, and insight workflows

### Can I use it without MCP?

Yes. MCP is useful, but not required.

Any assistant that can run shell commands can call:

```bash
rlat search project.rlat "question" --format json
```

That means the universal fallback is always the CLI.

### Can I use it with HTTP or apps?

Yes. The HTTP server exposes a REST interface:

```bash
rlat serve project.rlat --port 8080
```

This is useful when you want to integrate the semantic model into an application rather than only into a coding assistant.

### What is a knowledge-model-backed skill?

A regular skill is a static document — the same context loads every time. A knowledge-model-backed skill has a live knowledge connection. It declares which `.rlat` knowledge models it draws from, and RL adapts the injected context to each request.

The skill author adds a few frontmatter fields to SKILL.md:

```yaml
---
name: my-skill
knowledge models:
  - .rlat/project-docs.rlat
knowledge model-queries:
  - "What are the core design patterns in this project"
knowledge model-mode: augment
knowledge model-budget: 2000
---
```

Adding `knowledge models:` alone enables user-query search. Adding `knowledge model-queries:` enables foundational context that loads every trigger. Tier 4 (derived queries) accepts caller-supplied search terms via `--derived` to surface implicit needs. Adoption is incremental — skills without `knowledge model-*` fields work exactly as before. See [Skill Integration](/docs/skill-integration) for the full schema.

### How do foundational queries work?

Foundational queries are the skill author's answer to: "what does this skill always need to know, regardless of the user's question?"

They are defined in the `knowledge model-queries` frontmatter field. When the skill triggers, these queries resonate against the declared knowledge models and pull ranked passages — the same retrieval pipeline as `rlat search`, but automated and budget-capped (40% of the token budget by default).

For example, a notebook-creation skill might always need Fabric API patterns, pyspark conventions, and workspace auth — whether the user asks about CSV ingestion or REST API ingestion. The foundational queries guarantee that baseline context, while the user's actual question (Tier 3, 30%) and any caller-supplied derived queries (Tier 4, 30%) fill the remaining budget with request-specific context.

The practical effect: skill authors guarantee baseline domain knowledge without bloating SKILL.md with thousands of tokens of reference material. The knowledge model provides it dynamically, ranked, and within budget.

---

## Status And Limits

### What is the project status?

Current status is:

**Beta (`0.11.0`)** — targeting **v1.0.0 on 2026-06-08**

That should be read as:

- technically serious
- benchmarked
- feature-rich
- converging on v1.0.0 — the core CLI and three-layer semantic router are stable; docs, evaluation coverage, and ergonomics are still tightening for launch

### What are the main limitations today?

The main things to keep in mind are:

- the project is in beta, so workflows are stable but still being refined ahead of the v1.0.0 cut
- the default local encoder is strongest in English; multilingual presets like `bge-m3` and `qwen3-8b` are available via `--encoder`
- the first build downloads and caches the encoder before fully offline use
- best-quality retrieval usually comes from the full hybrid + reranked path, though dense-only can outperform reranking on some corpus types (e.g. argument-style retrieval)
- very large corpora introduce practical memory and scaling trade-offs
- cross-corpus evaluation (BEIR) shows the pipeline exceeding flat E5 on 3 of 5 datasets — these are backbone and pipeline trade-offs, not workflow trade-offs

---

## Context Composition and Control

### What is context composition?

Context composition lets you combine multiple `.rlat` knowledge models at search time without merging them into a single file. Think of it like mixing audio tracks: each knowledge model stays independent, but your query hears them all together.

Build one knowledge model for your docs, another for your codebase, another for compliance policies, and mix them at query time however the situation demands. No rebuild required.

```bash
rlat search docs.rlat "how does auth work?" --with code.rlat
```

### How do I search across multiple knowledge bases at once?

Use the `--with` flag to layer additional knowledge models into a single search. Results come back tagged with which knowledge model they came from.

```bash
rlat search docs.rlat "auth flow" --with code.rlat --with runbooks.rlat
```

Each result includes a knowledge model label, so you always know whether an answer about OAuth middleware came from the documentation, the source code, or an operational runbook.

### Can I remove a topic from search results on the fly?

Yes. The `--suppress` flag performs a mathematical subtraction on the semantic field. This is not keyword filtering -- it removes an entire semantic direction. Even paraphrased or indirect references to the suppressed topic are dampened.

```bash
rlat search docs.rlat "recent developments" --suppress "politics"
rlat search docs.rlat "project status" --suppress "budget" --suppress "personnel"
```

Because this operates on the field's geometry rather than matching words, suppressing "politics" also reduces results about "partisan," "legislation," and "campaign" -- even if those words never appear in your suppress term.

### What does "project through" mean?

Projection lets you view one knowledge base through the lens of another. "Show me code, but only the parts relevant to compliance" is a natural-language way of saying "project my codebase through my compliance knowledge model."

```bash
rlat search code.rlat "data handling" --through compliance.rlat
```

Under the hood, this uses orthogonal projection: the compliance knowledge model defines a subspace of meaning, and the code knowledge model's results are projected into that subspace. Anything in the code that has no compliance relevance falls to near-zero.

### Can I see what changed between two versions?

Yes. The `--diff` flag creates a queryable semantic diff between two knowledge models. This is not a file-level diff -- it is a field you can ask questions about.

```bash
rlat search current.rlat "what changed?" --diff baseline.rlat
rlat search v2.rlat "did the auth model change?" --diff v1.rlat
```

The diff field captures what was added, removed, or shifted in semantic weight. You can ask "what changed about authentication?" and get meaningful answers even if the word "authentication" never appeared in any changelog.

### Where do two knowledge bases disagree?

The `^` operator creates a contradiction field -- a semantic space that is strong where two knowledge models diverge and weak where they agree.

```bash
rlat compose "docs.rlat ^ code.rlat" "authentication"
```

This is useful for catching drift between what docs say and what code actually does. If your docs describe one auth flow but the codebase implements a different one, the contradiction field surfaces that divergence.

### Can different knowledge models have different LLM injection modes?

Yes. Each knowledge model in a composition can control how its knowledge interacts with the LLM.

You might want your general docs to use **augment** mode (the LLM can blend its own knowledge with rlat results) but require your compliance knowledge model to use **constrain** mode (the LLM must only use what rlat provides).

In augment mode, rlat context enriches the LLM's existing knowledge. In constrain mode, the injection framing instructs the LLM to answer only from the provided sources. Enforcement depends on the LLM's compliance with the framing -- this is a prompt-level control, not a cryptographic guarantee.

### What are Knowledge Lenses?

A Knowledge Lens is a named, reusable semantic perspective. It reshapes what the field reveals -- it does not just filter keywords, it amplifies connections that are relevant to the lens.

```bash
rlat search docs.rlat "user input" --with code.rlat --lens sharpen
rlat search docs.rlat "configuration" --lens denoise
rlat search code.rlat "error handling" --lens sharpen
```

Pre-built lenses include `sharpen` (boost distinctive results), `flatten` (equalise band emphasis), and `denoise` (suppress boilerplate). You can also build custom lenses from exemplar texts and save them as `.rlens` files.

A security lens does not just find documents containing the word "security." It amplifies security-relevant semantic structure -- input validation, trust boundaries, privilege escalation paths -- even in code that never uses the word "security."

### What's the difference between `--lens sharpen` and `--sharpen`?

They're both corpus transforms, but use different math and apply at different intensities.

- `--lens sharpen` — named `LensBuilder` transform from the original lens system. Moderate reshaping via subspace projection.
- `--sharpen <strength>` — EML self-sharpening (v0.11.0+): `expm(F) - logm(F)` per band, scale-invariant. Exponentially amplifies dominant topics, compresses noise. Parametric strength (try `0.5`–`2.0`).

Rule of thumb: start with `--tune focus` (uses `--sharpen` internally as a preset). Fall back to `--lens sharpen` if you need the older lens-subspace behaviour.

### What are the EML corpus transforms?

Nonlinear spectral transforms (v0.11.0+) that reshape the whole field *before* retrieval. Unlike `--boost`/`--suppress` (rank-1, topic-specific), EML transforms operate on the entire eigenvalue spectrum without requiring you to name topics.

- `--sharpen <strength>` — unsupervised contrast enhancement for more precise results
- `--soften <strength>` — logarithmic flattening for broader exploration
- `--contrast <background.rlat>` — asymmetric REML contrast against a baseline knowledge model ("what do my docs know that theirs don't?")
- `--tune focus|explore|denoise` — task-matched presets (one word, no tuning knobs)

All scale-invariant (normalise eigenvalues before filtering, rescale after), so they work on any corpus size.

### Can I track how knowledge evolves over time?

Build knowledge models at regular intervals (weekly, monthly, per-release) and query the evolution. The temporal diff chain shows semantic shifts at each step. The knowledge trend query tells you whether a topic is growing, stable, or shrinking across the sequence.

```bash
rlat search current.rlat "auth changes" --diff last_month.rlat
```

### What about role-based access control?

Algebraic access control creates role-specific views of a knowledge model using orthogonal projection. This is auditable: the projection produces a certificate that proves exactly what was hidden. The math guarantees that `visible + hidden = original`, so nothing is lost -- it is partitioned, not deleted.

### How do I set up a reusable composition?

Save your composition as a `.rctx` context file in YAML format:

```yaml
name: team-context
knowledge models:
  docs: ./docs.rlat
  code: ./code.rlat
weights:
  docs: 0.7
  code: 0.3
suppress: ["meeting notes"]
injection_modes:
  docs: augment
  code: constrain
```

Then query it like a single knowledge model:

```bash
rlat compose team.rctx "how does auth work?"
```

Context files make compositions reproducible and shareable. Check them into your repo so the whole team uses the same knowledge configuration. YAML format requires `pyyaml` (`pip install pyyaml`); JSON context files work without extra dependencies.

### What is the performance impact of composition?

Composition is designed to stay within interactive latency budgets:

| Operation | Additional latency |
|-----------|-------------------|
| 2-knowledge model warm query | ~5-25ms over single knowledge model |
| 5-knowledge model warm query | ~15-60ms over single knowledge model |
| Topic sculpting (suppress) | Negligible (single rank-1 update) |
| Cold first query per knowledge model | ~200-500ms for load, then warm |

The cold-load cost is a one-time penalty per knowledge model per session. Once loaded, subsequent queries add only milliseconds.

### How do I set up automatic knowledge model discovery for my assistant?

Run `init-project --auto-integrate` to wire everything in one step:

```bash
rlat init-project --auto-integrate
```

This builds a knowledge model, generates a primer, creates a `.rlat/manifest.json` (machine-readable index of all knowledge models), updates `.mcp.json` to load the MCP server, and injects a knowledge model section into `CLAUDE.md` with query recipes.

Once set up, your assistant can call `rlat_discover` (MCP tool) to list available knowledge models with freshness status, and `rlat_freshness` to check if a rebuild is needed. The assistant no longer needs to guess which knowledge model to use — it reads the manifest automatically.


## Where Should I Go Next?

Start with [Overview](/docs/overview) and [Getting Started](/docs/getting-started), then move into [CLI](/docs/cli), [MCP](/docs/mcp), [Context Control](/docs/context-control), or [Benchmarks](/docs/benchmarks) depending on what you need.
