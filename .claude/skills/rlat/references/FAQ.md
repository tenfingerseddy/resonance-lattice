# Resonance Lattice FAQ

Self-contained Q&A reference. All claims are evidence-backed or explicitly marked as architectural.

## Contents

- [What Is Resonance Lattice?](#what-is-resonance-lattice) — core thesis, three layers, what a `.rlat` is
- [How It Compares](#how-it-compares) — vs grep, vs standard RAG, vs vector DBs, vs static skills
- [Practical Questions](#practical-questions) — encoder choice, store modes, rebuilding, memory
- [Integration](#integration) — CLI, MCP, assistant setup

---

## What Is Resonance Lattice?

### What does it solve?

Most knowledge retrieval relies on an LLM — stuff files into a prompt and hope, or bolt vector search onto a generative model. Both give limited control over context, no visibility into what was retrieved, and hallucination rates that make output hard to trust.

Resonance Lattice builds a portable semantic model of your corpus as a single `.rlat` file — no LLM and no cloud service. You control what goes in, inspect what the model covers, and get ranked evidence passages grounded in actual sources.

### How does it work?

`rlat build` chunks your files, encodes each chunk with a local encoder, and packages everything into a knowledge model with three layers:

- **Field** — models semantic structure (what the corpus appears to know)
- **Registry** — maps semantic hits back to ranked source files
- **Store** — returns evidence text, passages, and metadata

When you query, your question is encoded the same way, resonated against the field, and the registry resolves top matches back to source text.

### What is a "semantic model"?

Not a keyword index. Not a vector database. A model of what your corpus appears to know — one you can query for evidence, but also inspect (`rlat profile`), compare across versions (`rlat compare`), and compose with algebra (`merge`, `diff`, `forget`).

A search index answers queries. A semantic model can also describe its own shape, coverage gaps, and how it differs from another model.

### Why maths instead of an LLM?

The field has algebraic structure. Every operation is a defined mathematical operation with known properties:

- **Exactness** — removing a file returns the field to the exact pre-addition state. No approximation, no drift.
- **Composability** — merge is order-independent. Sequence doesn't change the result.
- **Inspectability** — coverage, signal quality, saturation, gaps are intrinsic properties of the maths.
- **Determinism** — same inputs always produce the same field. No temperature, no sampling.
- **No LLM dependency** — build, query, profile, compare all run locally with a small encoder model.

### What can I learn about the model itself?

Three commands expose the field's internal structure:

| Command | What it tells you |
|---------|------------------|
| `rlat xray` | Corpus-level health: signal quality, saturation, diagnostics |
| `rlat locate <query>` | Where a query sits in the knowledge landscape and what the field doesn't know |
| `rlat probe <recipe>` | Quick insight recipes: novelty, saturation, coverage gaps, band flow |

### What are the unique features?

- **Knowledge Model algebra** — merge, diff, forget, intersect with mathematical guarantees
- **Semantic profiling** — inspect knowledge model health, coverage, structure
- **Field X-Ray** — corpus-level diagnostics (signal quality, saturation, gaps)
- **Query positioning** — see where a question sits in the knowledge landscape
- **Enriched query** — one call returns passages, coverage, related topics, contradictions
- **Corpus primers** — synthesize assistant context from the same model (strongest on documentation corpora)
- **Fully local** — no cloud, no API keys. One-time encoder download on first build, then fully offline
- **Single-file portable** — one `.rlat` knowledge model, share like any file

### What output formats are available?

| Format | What you get | Use it for |
|--------|-------------|-----------|
| `text` | Human-readable with source paths, coverage, topics | Terminal reading |
| `json` | Full metadata: scores, passages, latency | Scripts, pipelines |
| `prompt` | Rich markdown ready for LLM paste | Manual LLM workflows |
| `context` | Compressed line-per-passage with injection framing | Automated assistant integration |

---

## How It Compares

### Does it replace CLAUDE.md, skills, and memory?

No. They are complementary layers:

- **CLAUDE.md** gives an assistant rules and project conventions
- **Skills** give it executable capabilities
- **Memory** gives it persistent facts across conversations
- **Resonance Lattice** gives it grounded project knowledge — the actual content of your codebase and docs, semantically indexed

Resonance Lattice fills the "what does this project actually contain" gap.

### Why not just use grep?

Grep and rlat are complementary. Grep finds exact text matches — powerful when you know what you're looking for.

What grep can't do: understand that "authentication" and "login flow" are about the same thing, or that a design decision in one file contradicts guidance in another. Grep searches keywords blindly, with no awareness of meaning.

Resonance Lattice finds conceptually relevant content even when words don't match, and returns semantic context (coverage, related topics, contradictions) alongside results.

**Use grep** when you need a specific string. **Use rlat** when you need to understand what a corpus knows about a topic.

### Why is it better than standard RAG?

Standard RAG is a retrieval step bolted onto an LLM: embed chunks, cosine search, stuff into prompt. Resonance Lattice is a semantic model you can query, profile, compare, and compose.

The retrieval pipeline is benchmark-proven stronger:

| | rlat (reranked) | Best baseline (hybrid RRF) |
|---|---|---|
| **Recall@5** | 1.00 | 0.94 |
| **MRR** | 0.92 | 0.77 |
| **Failed retrieval** | 0% | 6% |

The knowledge model is a portable file, not a service dependency. And you get operations RAG doesn't offer: merge, diff, forget, profile.

**Honest caveat**: the full pipeline is the differentiator — the dense field alone underperforms flat cosine. These benchmarks are from one corpus; cross-corpus evaluation is ongoing.

### Why not just ask an LLM directly?

LLMs don't know your codebase, internal docs, or project state. When they guess, they hallucinate confidently.

Better approach: keep LLM reasoning, but ground it in your data. Resonance Lattice packages your knowledge into a model you own. The LLM reasons over your sources instead of inventing answers.

Feeding rlat context to an LLM reduced hallucinations from 78% to 16% and lifted fact recall from 0.27 to 0.91.

You control *how* the LLM uses your knowledge with `--mode`:

| Mode | What it tells the LLM |
|------|----------------------|
| `augment` | Use own knowledge + cite sources for detail |
| `constrain` | Answer ONLY from sources, cite [1][2] |
| `knowledge` | Base answer on this context, be transparent about gaps |
| `custom` | Your own system prompt |

### Why not just create an Obsidian vault?

Obsidian is a graph-based knowledge tool — wikilinks between documents for navigation. Resonance Lattice builds a semantic model automatically from existing files, no manual linking required.

**Retrieval benchmark** (same corpus, 2,246 documents, 100 questions):

| | Obsidian (best) | rlat (reranked) |
|---|---|---|
| **Recall@5** | 0.81 | 1.00 |
| **MRR** | 0.714 | 0.929 |
| **Failed retrieval** | 19% | 0% |

**End-to-end answer quality** (40 questions incl. 10 multi-hop, LLM judge):

| | rlat | Obsidian | Graph (AST) |
|---|---|---|---|
| **Accuracy** | 3.98 | 3.88 | 2.80 |
| **Completeness** | 3.48 | 3.08 | 1.15 |
| **Groundedness** | 4.25 | 3.83 | 4.05 |
| **Hallucinations/Q** | 0.85 | 1.23 | 0.28 |

On multi-hop questions (2-4 documents), rlat leads on accuracy (3.7 vs 3.3) and hallucinates 60% less than Obsidian (1.0 vs 2.5/Q).

**Methodology**: Retrieval uses BM25 over vault markdown with metadata-enriched and graph-boosted variants. Answer quality uses 40 questions (structural/semantic/hybrid/multi-hop) judged by Claude Sonnet on accuracy, completeness, groundedness, and hallucination rate. Each condition feeds context to the same answer model; judge scores blind to condition.

rlat leads on every metric across both single-topic and multi-hop questions.

---

## Practical Questions

### How does it handle file changes?

Not a full rebuild. The knowledge model tracks every file by content hash. Only changed, added, or removed files are processed.

| Command | What it does |
|---------|-------------|
| `rlat build` | Incremental by default — skips unchanged files |
| `rlat add` | Add new sources without touching existing ones |
| `rlat sync` | Full lifecycle: detect deleted, add new, re-encode modified |

Removing a file is algebraically exact — returns the field to the same state as if that file was never added. No drift, no approximation.

### Does it need an LLM? Is it fast? Does it work offline?

No LLM required. Everything runs locally with a small encoder model. No API keys, no network calls, no token costs. Works fully offline after first encoder download.

| Operation | Cold | Warm |
|-----------|------|------|
| Field resonance (raw) | ~8 ms | ~8 ms |
| Dense retrieval | ~220 ms | ~78 ms |
| Full hybrid + reranked | ~800 ms | ~80 ms |

First query pays a one-time load cost. Background worker keeps everything in memory for subsequent queries.

### What are store modes?

| Mode | What's in the .rlat | Trade-off |
|------|---------------------|-----------|
| `embedded` (default) | Field + registry + all evidence text | Portable, self-contained |
| `external` | Field + registry only | Smaller; pass `--source-root` at query time |

External mode is useful for reducing knowledge model size, avoiding embedded PII, or sharing the semantic model without source text.

### What are the bands?

The field is organised into multiple semantic bands, each capturing a different level of meaning — from broad subject area through to close lexical matches. These are working labels, not guaranteed ontology categories.

### How does the encoder work?

Resolution order: (1) `--encoder` flag, (2) stored encoder in knowledge model, (3) default.

Most users never need `--encoder`. The knowledge model stores its encoder at build time and queries restore it automatically. First-time download of the default encoder takes ~30 seconds.

ONNX export provides 2-5x CPU encoding speedup. The CLI auto-detects an ONNX directory alongside the knowledge model.

### Can I compress the registry?

Yes. `--quantize-registry 8` gives ~50% compression with minimal quality loss. `--quantize-registry 4` gives ~87% compression with slight quality loss. No codebook training needed.

### Can I compress the knowledge model file?

Yes. `--compression zstd` (best ratio, requires zstd library) or `--compression lz4` (fastest decompression).

---

## Integration

### How do I integrate with Claude Code?

Two approaches:

**Approach 1: Corpus primer** (simplest, best for documentation corpora)
```bash
rlat summary project.rlat -o .claude/resonance-context.md
```
Then in CLAUDE.md: `@.claude/resonance-context.md`

Note: the summary primer is strongest on text-dense corpora (doc sites, knowledge bases, note vaults). For code repos with an existing README, the README itself is typically better orientation.

**Approach 2: MCP server** (live queries)
```bash
rlat mcp project.rlat
```
Add to Claude Code settings:
```json
{ "mcpServers": { "resonance": { "command": "rlat", "args": ["mcp", "project.rlat"] } } }
```

### Can I use it with other AI tools?

Yes. The HTTP server (`rlat serve`) exposes a REST API. The `--format json` output works with any tool that consumes JSON. The `--format context` output is designed for injection into any LLM prompt.

### Can I share a knowledge model without sharing source text?

Yes. `rlat export project.rlat -o shared.rlat --field-only` exports the field and registry without the evidence store. The receiver can query the semantic model but cannot read original source text.
